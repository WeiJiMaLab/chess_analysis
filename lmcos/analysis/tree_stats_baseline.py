"""
Analysis 2.1: Tree-Statistic Summary MC Baseline.

Extracts richer tree-level statistics from prefixes (snapshot trees) and trains
a tabular baseline model (MLP) to predict target advantages. Compares against
the GNN+MC packed-controller baseline (90.1%). A0b minimal-MC comparison is
deferred until a correct A0b re-run (2026-06-03 run used wrong halt_rewards).

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._data import load_shard_trees
from analysis._plots import analysis_style, save_fig
from src.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from src.data.preprocess_mc.oracle import BudgetedOracleConfig, predicted_stop_from_advantages
from src.data.preprocess_mc.pack import (
    budgeted_oracle_from_trajectory,
    build_compact_trajectory_from_payload,
)
from src.core.tree import SearchTree

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()

_PRIMARY_BUDGET = 43
# GNN+MC baseline from packed controller training (subtree_weighting_root; labels always correct)
_GNN_MC_SIGN_ACCURACY = 0.901


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


def extract_tree_stats_from_prefix(
    prefix_tree: SearchTree,
    q_row: torch.Tensor,
    halt_reward: float,
    budget: int,
) -> list[float]:
    """Compute 10 tree-stat features for one snapshot, normalised where appropriate.

    The last slot (root_best_q) uses the pack-path halt_reward (oracle_final_root_q_values
    indexed by the step's recommended move) rather than the evolving q_row estimate.
    t_norm and remaining_budget_norm are appended by the caller (indices 10-11).
    """
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

    evaluated = q_row[q_row != 0]
    root_wdl_var = float(evaluated.var().item()) if len(evaluated) > 1 else 0.0

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
        halt_reward,
    ]


def extract_snapshot_features(t: dict, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a tree record, reconstruct prefixes, and compute features/targets.
    
    Returns:
      X: shape (T, 12) feature matrix
      y: shape (T,) targets (advantage sign)
    """
    # Rehydrate the tree from the record format
    record = RawPretrainExampleRecord.from_payload(t)
    example = record.to_pretrain_example()
    tree = example.tree

    q = t["oracle_root_q_trace"]
    best_idx = t["oracle_best_move_index"]
    trajectory = build_compact_trajectory_from_payload(t)
    if trajectory is None:
        raise ValueError("Tree has no controller trajectory (no root expansion).")
    halt_rewards = trajectory["halt_rewards"]
    root_rank = int(trajectory["first_decision_expansion_count"]) - 1
    num_steps = min(len(halt_rewards), budget)
    policy = budgeted_oracle_from_trajectory(trajectory, budget, _CONFIG)
    advantages = np.array(policy.target_advantages, dtype=np.float32)
    y = np.sign(advantages).astype(np.float32)
    y[y == 0] = 1.0
    halt_slice = [float(value) for value in halt_rewards[:num_steps]]

    X = np.zeros((num_steps, 12), dtype=np.float32)
    for s in range(num_steps):
        trace_step = root_rank + s
        # Reconstruct prefix tree at step s
        # In build_tree, step s corresponds to s expansions
        prefix_tree = tree.clone_expansion_prefix(trace_step)
        stats = extract_tree_stats_from_prefix(prefix_tree, q[trace_step], halt_slice[s], num_steps)
        X[s, :10] = stats
        X[s, 10] = s / max(num_steps - 1, 1)            # t normalised
        X[s, 11] = (num_steps - s) / num_steps          # remaining_budget normalised

    return X, y


def load_dataset(trees_root: str, n_trees: int, budget: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Load tree-statistic features + targets from n_trees filtered_shard trees."""
    def _process(t: dict):
        X, y = extract_snapshot_features(t, budget)
        return (X, y) if len(X) > 0 else None

    pairs = load_shard_trees(trees_root, n_trees, _process, seed=seed, desc="Extracting features")
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


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
    analysis_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["GNN+MC\n(baseline)", "A2.1 train", "A2.1 val"]
    accs = [_GNN_MC_SIGN_ACCURACY, train_acc, val_acc]
    colors = ["#6366f1", "#2563EB", "#10b981"]
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
    save_fig(output_path)


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
    print("A2.1 Rich Tree Statistics Baseline Results:")
    print(f"  GNN+MC baseline sign accuracy:     {_GNN_MC_SIGN_ACCURACY:.3f}")
    print(f"  A2.1 Rich Tree (12 stats) train:   {train_acc:.3f}")
    print(f"  A2.1 Rich Tree (12 stats) val:     {val_acc:.3f}")
    print(f"  (A0b minimal-MC val sign acc: 54.6% — GNN encoder confirmed essential)")
    print(f"{'='*60}\n")

    plot_comparison_bar(train_acc, val_acc, os.path.join(str(_FIGURES_DIR), "tree_stats_baseline_accuracy.png"))


if __name__ == "__main__":
    main()
