"""
Analysis 2.1: Tree-Statistic Summary MC Baseline.

Extracts richer tree-level statistics from prefixes (snapshot trees) and trains
a tabular baseline model (MLP) to predict target advantages. Compares results
with the GNN+MC baseline (90.1%) and the A0b Minimal MC baseline (86.4%).

Features per snapshot:
  1. tree_size             - total nodes in prefix tree
  2. expanded_nodes        - number of expanded nodes in prefix tree
  3. max_depth             - maximum node depth in prefix tree
  4. mean_depth            - mean node depth in prefix tree
  5. terminal_ratio        - ratio of terminal nodes in prefix tree
  6. avg_value             - mean scalar value across prefix tree nodes
  7. var_value             - variance of scalar value across prefix tree nodes
  8. root_value            - static value of the root node
  9. root_wdl_var          - variance of WDL win across root children
  10. root_best_q          - best Q-value of root children
  11. t                    - normalized expansion step
  12. remaining_budget     - normalized remaining budget

Usage:
    python analysis/tree_stats_baseline.py --n-trees 2000
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
    compute_budgeted_oracle,
)
from src.core.tree import SearchTree
from src.data.preprocess_gnn.teacher_targets import PretrainExample
from utils.helpers import apply_poster_style

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()

_PRIMARY_BUDGET = 43
_GNN_MC_SIGN_ACCURACY = 0.901
_A0B_SIGN_ACCURACY = 0.864


class TreeStatsMC(nn.Module):
    """MLP that takes the 12 tree statistics and outputs a scalar continue advantage."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, hidden_layers: int = 2):
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


def extract_tree_stats_from_prefix(prefix_tree: SearchTree, q_row: torch.Tensor, best_idx: int, budget: int) -> list[float]:
    """Compute the 12 tree statistics for a given prefix tree snapshot, normalized where appropriate."""
    nodes = list(prefix_tree.iter_nodes())
    tree_size = len(nodes)
    expanded_nodes = sum(1 for n in nodes if n.is_expanded)
    max_depth = max(n.depth for n in nodes)
    mean_depth = sum(n.depth for n in nodes) / tree_size
    terminal_ratio = sum(1 for n in nodes if n.is_terminal) / tree_size
    
    values = [n.scalar_features.get("value", 0.0) for n in nodes]
    avg_value = sum(values) / tree_size
    var_value = float(np.var(values)) if tree_size > 1 else 0.0
    
    root_value = prefix_tree.get_node(0).scalar_features.get("value", 0.0)
    
    # Root children statistics
    evaluated = q_row[q_row != 0]
    root_wdl_var = float(evaluated.var().item()) if len(evaluated) > 1 else 0.0
    root_best_q = float(q_row[best_idx].item())
    
    # Normalize count and depth features to prevent scaling issues in the MLP
    return [
        float(tree_size) / budget,
        float(expanded_nodes) / budget,
        float(max_depth) / 4.0,
        float(mean_depth) / 4.0,
        float(terminal_ratio),
        float(avg_value),
        float(var_value),
        float(root_value),
        float(root_wdl_var),
        float(root_best_q),
    ]


def extract_snapshot_features(t: dict, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a tree record, reconstruct prefixes, and compute features/targets.
    
    Returns:
      X: shape (T, 12) feature matrix
      y: shape (T,) targets (advantage sign)
    """
    # Rehydrate the tree from the record format
    # Using RawPretrainExampleRecord if it's saved in that format
    from src.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
    
    # Check if tree key is present or if we need to load via Record
    if "parent_index" in t:
        record = RawPretrainExampleRecord.from_payload(t)
        example = record.to_pretrain_example()
        tree = example.tree
    else:
        # Legacy/direct dict
        tree = t["tree"]

    q = t["oracle_root_q_trace"]       # [T, n_children]
    best_idx = t["oracle_best_move_index"]
    T = min(len(q), budget)

    halt_rewards = [float(q[s, best_idx[s].item()].item()) for s in range(T)]
    tree_sizes = [1] * T

    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, T, _CONFIG)
    advantages = np.array(policy.target_advantages, dtype=np.float32)
    y = np.sign(advantages).astype(np.float32)
    y[y == 0] = 1.0  # ties -> halt

    X = np.zeros((T, 12), dtype=np.float32)
    for s in range(T):
        # Reconstruct prefix tree at step s
        # In build_tree, step s corresponds to s expansions
        prefix_tree = tree.clone_expansion_prefix(s)
        
        # Get 10 statistics
        stats = extract_tree_stats_from_prefix(prefix_tree, q[s], best_idx[s].item(), T)
        
        X[s, :10] = stats
        X[s, 10] = s / max(T - 1, 1)            # t normalised
        X[s, 11] = (T - s) / T                  # remaining_budget normalised

    return X, y


def load_dataset(trees_root: str, n_trees: int, budget: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load tree-statistic features + targets from n_trees filtered_shard trees."""
    dirs = sorted(d for d in os.listdir(trees_root) if d.startswith("filtered_shard"))
    files: list[str] = []
    for d in dirs:
        p = os.path.join(trees_root, d)
        files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))

    np.random.default_rng(seed).shuffle(files)
    all_X, all_y = [], []
    for path in tqdm(files[:n_trees], desc="Extracting features"):
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            X, y = extract_snapshot_features(t, budget)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            pass
    return np.concatenate(all_X), np.concatenate(all_y)


def compute_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == y_true))


def train_stats_mc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4096,
    seed: int = 0,
) -> TreeStatsMC:
    torch.manual_seed(seed)
    model = TreeStatsMC(input_dim=12, hidden_dim=64, hidden_layers=2)
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


def plot_comparison_bar(train_acc: float, val_acc: float, output_path: str) -> None:
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["GNN+MC\n(baseline)", "A0b Minimal\n(MLP, 4 stats)", "A2.1 Rich Tree\n(MLP, 12 stats)"]
    accs = [_GNN_MC_SIGN_ACCURACY, _A0B_SIGN_ACCURACY, val_acc]
    colors = ["#6366f1", "#f59e0b", "#10b981"]
    bars = ax.bar(models, accs, color=colors, alpha=0.85, width=0.5, edgecolor="white")
    ax.axhline(_GNN_MC_SIGN_ACCURACY, color="black", linestyle="--", lw=1.5,
               label=f"GNN+MC baseline ({_GNN_MC_SIGN_ACCURACY:.1%})")
    ax.axhline(0.80, color="red", linestyle=":", lw=1.5, label="Threshold (80%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.003, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Sign accuracy")
    ax.set_title("A2.1: Tree-Statistic Baseline Sign Accuracy")
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-root", default=_TREES_ROOT)
    parser.add_argument("--n-trees", type=int, default=1000)
    parser.add_argument("--n-val-trees", type=int, default=300)
    parser.add_argument("--budget", type=int, default=_PRIMARY_BUDGET)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading training trees ({args.n_trees} trees)...")
    X_train, y_train = load_dataset(args.trees_root, args.n_trees, args.budget, seed=args.seed)
    print(f"  Loaded {X_train.shape[0]} train snapshots.")

    print(f"Loading validation trees ({args.n_val_trees} trees)...")
    X_val, y_val = load_dataset(args.trees_root, args.n_val_trees, args.budget, seed=args.seed + 1)
    print(f"  Loaded {X_val.shape[0]} val snapshots.")

    print("\nTraining richer Tree-Stats MLP MC model...")
    model = train_stats_mc(X_train, y_train)

    train_acc = compute_sign_accuracy(y_train, model(torch.from_numpy(X_train)).detach().numpy().ravel())
    val_acc = compute_sign_accuracy(y_val, model(torch.from_numpy(X_val)).detach().numpy().ravel())

    print(f"\n{'='*60}")
    print(f"A2.1 Rich Tree Statistics Baseline Results:")
    print(f"  GNN+MC baseline sign accuracy: {_GNN_MC_SIGN_ACCURACY:.3f}")
    print(f"  A0b Minimal (4 stats) sign acc: {_A0B_SIGN_ACCURACY:.3f}")
    print(f"  A2.1 Rich Tree (12 stats) val acc: {val_acc:.3f}")
    print(f"{'='*60}\n")

    plot_comparison_bar(train_acc, val_acc, os.path.join(str(_FIGURES_DIR), "tree_stats_baseline_accuracy.png"))


if __name__ == "__main__":
    main()
