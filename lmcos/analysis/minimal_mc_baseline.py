"""
Analysis 0b: minimal MC baseline — tree stats → MLP → halt/continue.

Replaces the GNN encoder with a hand-crafted scalar feature vector (per snapshot)
and trains a small MLP with the same architecture as the MC head to predict
sign(target_advantage). Tests whether the GNN adds value over raw tree statistics.

Features per snapshot:
  best_q       — best root-child Q (WDL win) at this expansion step
  wdl_var      — variance across evaluated root children at this step
  t            — expansion step index (normalised by budget)
  remaining_budget — budget left

Target: sign(target_advantage) ∈ {−1, +1} (continue vs halt)

Figures saved to lmcos/analysis/figures/:
  minimal_mc_sign_accuracy.png   — sign accuracy vs GNN+MC baseline + by bucket
  minimal_mc_weights.png         — first-layer weight magnitudes (feature importance)
  minimal_mc_timing.png          — inference time comparison

Usage (from lmcos/):
    python analysis/minimal_mc_baseline.py
    python analysis/minimal_mc_baseline.py --n-trees 2000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "human_analytics"))

from src.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    DEFAULT_BUDGET_BUCKETS,
    compute_budgeted_oracle,
)
from utils.helpers import apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR, PHASE_COLORS

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()

# Primary budget for per-snapshot training
_PRIMARY_BUDGET = 43

# GNN+MC baseline sign accuracy (from training logs, subtree_weighting_root run)
_GNN_MC_SIGN_ACCURACY = 0.901


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MinimalMC(nn.Module):
    """MLP matching the MC head architecture, takes raw tree stats instead of z_root."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, hidden_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature + target extraction
# ---------------------------------------------------------------------------

def extract_snapshot_features(t: dict, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) for all snapshot steps up to budget from one tree.

    X columns: [best_q, wdl_var_over_evaluated, t_normalised, remaining_budget_normalised]
    y: sign(target_advantage) ∈ {−1, +1}
    """
    q = t["oracle_root_q_trace"]       # [T, n_children]
    best_idx = t["oracle_best_move_index"]
    T = min(len(q), budget)

    halt_rewards = [float(q[s, best_idx[s].item()].item()) for s in range(T)]
    tree_sizes = [1] * T

    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, T, _CONFIG)
    advantages = np.array(policy.target_advantages, dtype=np.float32)
    y = np.sign(advantages).astype(np.float32)
    y[y == 0] = 1.0  # ties → halt

    X = np.zeros((T, 4), dtype=np.float32)
    for s in range(T):
        q_row = q[s]
        evaluated = q_row[q_row != 0]
        best_q = float(q[s, best_idx[s].item()].item())
        wdl_var = float(evaluated.var().item()) if len(evaluated) > 1 else 0.0
        X[s, 0] = best_q
        X[s, 1] = wdl_var
        X[s, 2] = s / max(T - 1, 1)            # t normalised
        X[s, 3] = (T - s) / T                  # remaining_budget normalised

    return X, y


def compute_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of snapshots where sign(pred) matches sign(y_true)."""
    return float(np.mean(np.sign(y_pred) == y_true))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(trees_root: str, n_trees: int, budget: int, seed: int = 42
                 ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load snapshot features + targets from n_trees filtered_shard trees."""
    dirs = sorted(d for d in os.listdir(trees_root) if d.startswith("filtered_shard"))
    files: list[str] = []
    for d in dirs:
        p = os.path.join(trees_root, d)
        files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))

    np.random.default_rng(seed).shuffle(files)
    all_X, all_y, episode_lengths = [], [], []
    for path in tqdm(files[:n_trees], desc="Extracting features"):
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            X, y = extract_snapshot_features(t, budget)
            if len(X) > 0:
                all_X.append(X); all_y.append(y)
                episode_lengths.append(len(X))
        except Exception:
            pass
    return np.concatenate(all_X), np.concatenate(all_y), episode_lengths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_minimal_mc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4096,
    seed: int = 0,
) -> MinimalMC:
    torch.manual_seed(seed)
    model = MinimalMC(input_dim=input_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # train on raw advantage (sign computed at eval time)

    X = torch.from_numpy(X_train)
    y = torch.from_numpy(y_train).unsqueeze(1)
    n = len(X)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0:
            acc = compute_sign_accuracy(y_train, model(X).detach().numpy().ravel())
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  sign_acc={acc:.4f}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sign_accuracy(train_acc: float, val_acc: float, output_path: str) -> None:
    apply_poster_style()
    fig, ax = plt.subplots(figsize=(14, 9))
    models = ["GNN+MC\n(baseline)", "Minimal MLC\n(train)", "Minimal MLC\n(val)"]
    accs = [_GNN_MC_SIGN_ACCURACY, train_acc, val_acc]
    colors = [PHASE_COLORS[1], PHASE_COLORS[2], PHASE_COLORS[3]]
    bars = ax.bar(models, accs, color=colors, alpha=0.8, width=0.5)
    ax.axhline(_GNN_MC_SIGN_ACCURACY, color="black", linestyle="--", lw=2, label="GNN+MC baseline")
    ax.axhline(0.80, color="red", linestyle=":", lw=2, label="Acceptance threshold (80%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.005, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=FONT_SIZE_TICKS, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Sign accuracy", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Minimal MC baseline vs GNN+MC", fontsize=FONT_SIZE_LABEL, pad=12)
    ax.legend(fontsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def plot_weights(model: MinimalMC, feature_names: list[str], output_path: str) -> None:
    apply_poster_style()
    W = model.net[0].weight.detach().abs().mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(feature_names, W, color=MAIN_COLOR, alpha=0.8)
    ax.set_ylabel("Mean |weight| (first layer)", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Feature importance in minimal MC baseline", fontsize=FONT_SIZE_LABEL, pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def measure_inference_time(model: MinimalMC, n_snapshots: int = 10000) -> float:
    """Return mean ms per snapshot for the minimal MLC."""
    model.eval()
    x = torch.randn(n_snapshots, 4)
    with torch.no_grad():
        # warm-up
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(10):
            model(x)
        elapsed = (time.perf_counter() - t0) / 10
    return elapsed / n_snapshots * 1000  # ms per snapshot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees-root", default=_TREES_ROOT)
    p.add_argument("--n-trees", type=int, default=2000, help="Trees for training (default: 2000)")
    p.add_argument("--n-val-trees", type=int, default=500, help="Trees for validation (default: 500)")
    p.add_argument("--budget", type=int, default=_PRIMARY_BUDGET)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(_FIGURES_DIR))
    args = p.parse_args(argv)

    print(f"Loading training data ({args.n_trees:,} trees, budget={args.budget})…")
    X_train, y_train, _ = load_dataset(args.trees_root, args.n_trees, args.budget, seed=args.seed)
    print(f"  Train snapshots: {len(X_train):,}")

    print(f"Loading validation data ({args.n_val_trees:,} trees)…")
    X_val, y_val, _ = load_dataset(args.trees_root, args.n_val_trees, args.budget, seed=args.seed + 1)
    print(f"  Val snapshots:   {len(X_val):,}")

    print("\nTraining minimal MC baseline (4 features → MLP → advantage)…")
    model = train_minimal_mc(X_train, y_train, input_dim=4, hidden_dim=64, hidden_layers=2)

    train_acc = compute_sign_accuracy(y_train, model(torch.from_numpy(X_train)).detach().numpy().ravel())
    val_acc = compute_sign_accuracy(y_val, model(torch.from_numpy(X_val)).detach().numpy().ravel())
    ms_per_snap = measure_inference_time(model)

    gnn_mc_note = "✓ CLOSE TO BASELINE" if val_acc >= 0.80 else "✗ BELOW THRESHOLD"
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  GNN+MC baseline sign accuracy:  {_GNN_MC_SIGN_ACCURACY:.3f}")
    print(f"  Minimal MLC train sign accuracy: {train_acc:.3f}")
    print(f"  Minimal MLC val  sign accuracy:  {val_acc:.3f}  ← {gnn_mc_note}")
    print(f"  Inference: {ms_per_snap:.4f} ms/snapshot")
    print(f"{'='*60}\n")

    out = args.output_dir
    feature_names = ["best_q", "wdl_var", "t_norm", "budget_rem_norm"]
    plot_sign_accuracy(train_acc, val_acc, os.path.join(out, "minimal_mc_sign_accuracy.png"))
    plot_weights(model, feature_names, os.path.join(out, "minimal_mc_weights.png"))


if __name__ == "__main__":
    main()
