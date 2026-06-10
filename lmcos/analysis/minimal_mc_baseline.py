"""
Analysis 0b: minimal MC baseline — tree stats → MLP → halt/continue.

Replaces the GNN encoder with a hand-crafted scalar feature vector (per snapshot)
and trains a small MLP with the same architecture as the MC head to predict
target_advantage. Tests whether the GNN adds value over raw tree statistics.

Oracle labels from ``pack.py``:
  trajectory["halt_rewards"], target_advantages, oracle_stop_step = budgeted_oracle_from_trajectory(...)

Features per snapshot:
  best_q       — teacher Q for the MCTS-recommended action at this step (Q_final[best_idx[s]])
  wdl_var      — variance across evaluated root children at this step
  t            — expansion step index (normalised by budget)
  remaining_budget — budget left

Target: target_advantage from compute_budgeted_oracle (same as packed GNN training).

Evaluation:
  - Per-snapshot sign accuracy (advantage > 0 vs continue)
  - Episode exact_stop_step_accuracy: predicted_stop == oracle_stop_step
  - Pearson r(predicted_stop, oracle_stop_step)

Figures saved to lmcos/analysis/figures/:
  minimal_mc_sign_accuracy.png   — sign accuracy vs GNN+MC baseline
  minimal_mc_weights.png         — first-layer weight magnitudes (feature importance)

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._data import load_shard_trees
from analysis._plots import analysis_style, save_fig
from src.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    predicted_stop_from_advantages,
)
from src.data.preprocess_mc.pack import (
    budgeted_oracle_from_trajectory,
    build_compact_trajectory_from_payload,
)

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()

# Primary budget for per-snapshot training
_PRIMARY_BUDGET = 43

# GNN+MC baseline from packed controller training (subtree_weighting_root; labels always correct)
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

def extract_snapshot_features(t: dict, budget: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return (X, y, oracle_stop_step) for all snapshot steps up to budget from one tree.

    X columns: [best_q, wdl_var_over_evaluated, t_normalised, remaining_budget_normalised]
    y: target_advantage (continuous, from compute_budgeted_oracle)
    oracle_stop_step: policy.optimal_stop_step for this episode
    """
    q = t["oracle_root_q_trace"]       # [T, n_children]
    best_idx = t["oracle_best_move_index"]
    q_final = t["oracle_final_root_q_values"]
    num_steps = min(len(q), budget)

    halt_rewards = [float(q_final[best_idx[s].item()].item()) for s in range(num_steps)]
    tree_sizes = [1] * num_steps

    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, num_steps, _CONFIG)
    advantages = np.array(policy.target_advantages, dtype=np.float32)
    oracle_stop_step = policy.optimal_stop_step

    X = np.zeros((num_steps, 4), dtype=np.float32)
    for s in range(num_steps):
        q_row = q[s]
        evaluated = q_row[q_row != 0]
        best_q = halt_rewards[s]
        wdl_var = float(evaluated.var().item()) if len(evaluated) > 1 else 0.0
        X[s, 0] = best_q
        X[s, 1] = wdl_var
        X[s, 2] = s / max(num_steps - 1, 1)            # t normalised
        X[s, 3] = (num_steps - s) / num_steps          # remaining_budget normalised

    return X, advantages, oracle_stop_step


def compute_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of snapshots where sign(pred) matches sign(y_true)."""
    return float(np.mean((y_pred > 0) == (y_true > 0)))


def evaluate_stop_steps(
    model: MinimalMC,
    X: np.ndarray,
    episode_lengths: list[int],
    oracle_stops: list[int],
) -> tuple[float, float]:
    """Return (exact_stop_step_accuracy, pearson_r) on episode-level stops."""
    predicted_stops: list[int] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for length in episode_lengths:
            preds = model(torch.from_numpy(X[offset:offset + length])).numpy().ravel()
            predicted_stops.append(predicted_stop_from_advantages(preds))
            offset += length

    predicted = np.asarray(predicted_stops, dtype=np.float64)
    oracle = np.asarray(oracle_stops, dtype=np.float64)
    exact = float(np.mean(predicted == oracle))
    if len(predicted) < 2 or np.std(predicted) == 0 or np.std(oracle) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(predicted, oracle)[0, 1])
    return exact, pearson


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    trees_root: str,
    n_trees: int,
    budget: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Load snapshot features + targets from n_trees filtered_shard trees."""
    def _process(t: dict):
        X, y, stop_step = extract_snapshot_features(t, budget)
        return (X, y, stop_step) if len(X) > 0 else None

    triples = load_shard_trees(trees_root, n_trees, _process, seed=seed, desc="Extracting features")
    all_X = [r[0] for r in triples]
    all_y = [r[1] for r in triples]
    episode_lengths = [len(r[0]) for r in triples]
    oracle_stops = [r[2] for r in triples]
    return np.concatenate(all_X), np.concatenate(all_y), episode_lengths, oracle_stops


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
    loss_fn = nn.MSELoss()

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
    analysis_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["GNN+MC\n(baseline)", "Minimal MLC\n(train)", "Minimal MLC\n(val)"]
    accs = [_GNN_MC_SIGN_ACCURACY, train_acc, val_acc]
    colors = ["#6366f1", "#2563EB", "#16a085"]
    bars = ax.bar(models, accs, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    ax.axhline(_GNN_MC_SIGN_ACCURACY, color="black", linestyle="--", lw=1.5,
               label=f"GNN+MC baseline ({_GNN_MC_SIGN_ACCURACY:.1%})")
    ax.axhline(0.80, color="red", linestyle=":", lw=1.5, label="Threshold (80%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.003, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Sign accuracy")
    ax.set_title("Minimal MLC vs GNN+MC sign accuracy")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    save_fig(output_path)


def plot_weights(model: MinimalMC, feature_names: list[str], output_path: str) -> None:
    analysis_style()
    W = model.net[0].weight.detach().abs().mean(dim=0).numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(feature_names, W, color="#2563EB", alpha=0.8, edgecolor="white")
    ax.set_ylabel("Mean |weight| (first layer)")
    ax.set_title("Feature importance — minimal MLC")
    plt.tight_layout()
    save_fig(output_path)


def plot_stop_step_scatter(
    model: MinimalMC,
    X: np.ndarray,
    episode_lengths: list[int],
    oracle_stops: list[int],
    output_path: str,
    label: str = "val",
) -> None:
    """Scatter oracle_stop_step vs predicted_stop_step, one point per episode."""
    predicted: list[int] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for length in episode_lengths:
            preds = model(torch.from_numpy(X[offset:offset + length])).numpy().ravel()
            predicted.append(predicted_stop_from_advantages(preds))
            offset += length

    oracle = np.asarray(oracle_stops, dtype=np.float64)
    pred_arr = np.asarray(predicted, dtype=np.float64)
    r = float(np.corrcoef(oracle, pred_arr)[0, 1]) if np.std(pred_arr) > 0 else float("nan")
    budget = max(oracle_stops) if oracle_stops else 1

    analysis_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(oracle, pred_arr, alpha=0.25, s=8, color="#2563EB", rasterized=True)
    lo, hi = 0, budget + 1
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.5, linestyle="--", label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("oracle_stop_step")
    ax.set_ylabel("predicted_stop_step (MLP greedy)")
    ax.set_title(f"A0b {label}: oracle vs predicted stop  r={r:+.3f}  n={len(oracle):,}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    save_fig(output_path)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def measure_inference_time(model: MinimalMC, n_snapshots: int = 10000) -> float:
    """Return mean ms per snapshot for the minimal MLC."""
    model.eval()
    x = torch.randn(n_snapshots, 4)
    with torch.no_grad():
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
    X_train, y_train, train_lengths, train_stops = load_dataset(
        args.trees_root, args.n_trees, args.budget, seed=args.seed,
    )
    print(f"  Train snapshots: {len(X_train):,}  episodes: {len(train_lengths):,}")

    print(f"Loading validation data ({args.n_val_trees:,} trees)…")
    X_val, y_val, val_lengths, val_stops = load_dataset(
        args.trees_root, args.n_val_trees, args.budget, seed=args.seed + 1,
    )
    print(f"  Val snapshots:   {len(X_val):,}  episodes: {len(val_lengths):,}")

    print("\nTraining minimal MC baseline (4 features → MLP → target_advantage)…")
    model = train_minimal_mc(X_train, y_train, input_dim=4, hidden_dim=64, hidden_layers=2)

    with torch.no_grad():
        train_preds = model(torch.from_numpy(X_train)).numpy().ravel()
        val_preds = model(torch.from_numpy(X_val)).numpy().ravel()

    train_acc = compute_sign_accuracy(y_train, train_preds)
    val_acc = compute_sign_accuracy(y_val, val_preds)
    train_exact, train_r = evaluate_stop_steps(model, X_train, train_lengths, train_stops)
    val_exact, val_r = evaluate_stop_steps(model, X_val, val_lengths, val_stops)
    ms_per_snap = measure_inference_time(model)

    gnn_mc_note = "✓ CLOSE TO BASELINE" if val_acc >= 0.80 else "✗ BELOW THRESHOLD"
    print(f"\n{'='*60}")
    print("Results (A0a-correct oracle labels):")
    print(f"  GNN+MC baseline sign accuracy:     {_GNN_MC_SIGN_ACCURACY:.3f}")
    print(f"  Minimal MLC train sign accuracy:   {train_acc:.3f}")
    print(f"  Minimal MLC val  sign accuracy:    {val_acc:.3f}  ← {gnn_mc_note}")
    print(f"  Minimal MLC train exact stop acc:  {train_exact:.3f}")
    print(f"  Minimal MLC val  exact stop acc:   {val_exact:.3f}")
    print(f"  Minimal MLC val  r(pred, oracle): {val_r:+.3f}  (n={len(val_stops)})")
    print(f"  Inference: {ms_per_snap:.4f} ms/snapshot")
    print(f"{'='*60}\n")

    out = args.output_dir
    feature_names = ["best_q", "wdl_var", "t_norm", "budget_rem_norm"]
    plot_sign_accuracy(train_acc, val_acc, os.path.join(out, "minimal_mc_sign_accuracy.png"))
    plot_weights(model, feature_names, os.path.join(out, "minimal_mc_weights.png"))
    plot_stop_step_scatter(model, X_val, val_lengths, val_stops,
                           os.path.join(out, "minimal_mc_stop_scatter.png"), label="val")


if __name__ == "__main__":
    main()
