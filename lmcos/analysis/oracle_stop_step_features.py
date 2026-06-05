"""
Analysis 0a: oracle_stop_step vs board features on lmcos filtered_shard trees.

Definitions
-----------
oracle_stop_step = ``budgeted_oracle_from_trajectory(...).optimal_stop_step`` (same field as
packed ``oracle_stop_step`` in controller shards).

halt_rewards come from ``build_compact_trajectory`` (``pack.py``) — not recomputed in analysis.

Features (Lc0-native)
----------------------
toptwo_equiv:
    Gap between the top-2 actions in oracle_final_root_q_values (Lc0's deep Q-values).
    Lc0 analog of Stockfish toptwo = e_win_best - e_win_second_best at depth_deep.
    Both computed from the deep (96-node) evaluation. NOT myopic.

gain_depth_equiv:
    oracle_final_root_q_values[best_idx[-1]] - oracle_final_root_q_values[best_idx[1]]
    = teacher's Q of the deep recommendation minus teacher's Q of the step-1 recommendation.
    Lc0 analog of Stockfish VOC = V_deep(a_deep) - V_deep(a_shallow).
    Zero when the step-1 recommendation is already the final best.

Regime note
-----------
All features use Lc0's oracle search (96-node MCTS, WDL output from the neural network).
The human RT correlation targets (_HUMAN_R) come from a Stockfish depth=5 analysis on
human games (human_analytics, n=1M). The comparison is deliberately cross-regime:
alignment means that the positional factors driving Lc0 oracle search depth also drive
human thinking time, despite the engine difference.

Usage (from lmcos/):
    python analysis/oracle_stop_step_features.py
    python analysis/oracle_stop_step_features.py --n-trees 39668 --budget 43
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "human_analytics"))

from analysis.board_tree_features import (
    HUMAN_RT_CORRELATIONS,
    extract_board_features,
    extract_tree_features,
)
from src.data.preprocess_mc.oracle import BudgetedOracleConfig, DEFAULT_BUDGET_BUCKETS
from src.data.preprocess_mc.pack import (
    budgeted_oracle_from_trajectory,
    build_compact_trajectory_from_payload,
)
from utils.helpers import analysis_style

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()
_HUMAN_R = HUMAN_RT_CORRELATIONS
# Representative budget per bucket: roughly mid-range of each
_BUCKET_BUDGETS = [2, 7, 18, 43, 90]
_BUCKET_NAMES = [b.name for b in DEFAULT_BUDGET_BUCKETS]


def process_tree(t: dict, budgets: list[int]) -> dict | None:
    """Extract features + oracle_stop_step at each budget level from one tree.

    Uses ``build_compact_trajectory`` halt_rewards + ``compute_budgeted_oracle`` (pack.py path).
    """
    try:
        fen = t["root_position_spec"]
        board_feats = extract_board_features(fen)
        tree_feats = extract_tree_features(t)
        trajectory = build_compact_trajectory_from_payload(t)
        if trajectory is None:
            return None

        row = {**board_feats, **tree_feats}
        for budget in budgets:
            row[f"oracle_stop_step_b{budget}"] = budgeted_oracle_from_trajectory(
                trajectory, budget, _CONFIG
            ).optimal_stop_step
        return row
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trees(trees_root: str, n_max: int, seed: int = 42) -> pd.DataFrame:
    dirs = sorted(d for d in os.listdir(trees_root) if d.startswith("filtered_shard"))  # all shards
    files: list[str] = []
    for d in dirs:
        p = os.path.join(trees_root, d)
        files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))

    np.random.default_rng(seed).shuffle(files)  # deterministic shuffle for reproducibility
    rows = []
    for path in tqdm(files[:n_max], desc="Loading trees"):
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            row = process_tree(t, _BUCKET_BUDGETS)
            if row is not None:
                rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_ORACLE_COLOR = "#2563EB"
_HUMAN_COLOR  = "#16a085"


def plot_comparison_bar(df: pd.DataFrame, primary_budget: int, output_path: str) -> None:
    """Figure 1: r and r² for oracle_stop_step vs features, vs human RT."""
    analysis_style()
    col = f"oracle_stop_step_b{primary_budget}"
    features = [
        ("n_possible_moves",        "Branching",   "branching"),
        ("n_self_pieces_exc_pawns", "Material",    "material"),
        ("gain_depth_equiv",        "gain_depth",  "gain_depth"),
        ("toptwo_equiv",            "toptwo",      "toptwo"),
    ]

    oracle_r, oracle_r2, human_r, labels = [], [], [], []
    for feat_col, feat_label, human_key in features:
        sub = df[[feat_col, col]].dropna()
        if len(sub) < 5:
            continue
        r = float(np.corrcoef(sub[feat_col], sub[col])[0, 1])
        oracle_r.append(r)
        oracle_r2.append(r ** 2)
        human_r.append(_HUMAN_R[human_key])
        labels.append(feat_label)

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, yvals_oracle, yvals_human, ylabel, title in [
        (axes[0], oracle_r,  human_r,
         "Pearson r", f"Feature correlations  (n={len(df):,}, budget={primary_budget})"),
        (axes[1], oracle_r2, [r**2 for r in human_r], "r²", "Variance explained"),
    ]:
        ax.bar(x - width/2, yvals_oracle, width, color=_ORACLE_COLOR, alpha=0.8,
               label=f"oracle_stop_step (b={primary_budget})")
        ax.bar(x + width/2, yvals_human, width, color=_HUMAN_COLOR, alpha=0.8,
               label="human log(RT)  [Stockfish d=5]")
        ax.axhline(0, color="black", lw=1.2, linestyle="--")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1), framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def plot_correlation_matrix(df: pd.DataFrame, primary_budget: int, output_path: str) -> None:
    """Figure 2: correlation matrix of oracle_stop_step + lmcos tree features."""
    analysis_style()
    col = f"oracle_stop_step_b{primary_budget}"
    cols = [col, "n_possible_moves", "n_self_pieces_exc_pawns", "toptwo_equiv", "gain_depth_equiv"]
    labels = [f"oracle_stop (b={primary_budget})", "Branching", "Material", "toptwo", "gain_depth"]
    sub = df[cols].dropna()

    corr = sub.corr()
    corr.columns = labels; corr.index = labels
    n = len(corr)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04).set_label("Pearson r")
    ax.set_title(f"lmcos tree feature correlations  (n={len(sub):,})", pad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, primary_budget: int) -> None:
    col = f"oracle_stop_step_b{primary_budget}"
    oss = df[col].dropna()
    print(f"\n{'='*60}")
    print(f"lmcos trees  n={len(df):,}  primary_budget={primary_budget}")
    print(f"oracle_stop_step: mean={oss.mean():.1f}  std={oss.std():.1f}  p50={oss.median():.0f}")
    print(f"{'='*60}")
    for feat_col, feat_label, human_key in [
        ("n_possible_moves",        "branching",  "branching"),
        ("n_self_pieces_exc_pawns", "material",   "material"),
        ("gain_depth_equiv",        "gain_depth", "gain_depth"),
        ("toptwo_equiv",            "toptwo",     "toptwo"),
    ]:
        sub = df[[feat_col, col]].dropna()
        if len(sub) < 5:
            continue
        r_oracle = np.corrcoef(sub[feat_col], sub[col])[0, 1]
        r_human = _HUMAN_R[human_key]
        match = "✓ SAME" if (r_oracle * r_human) > 0 else "✗ DIFFER"
        print(f"  {feat_label:<14} oracle r={r_oracle:+.4f}  human r={r_human:+.4f}  {match}")
    print()

    print("  Budget sensitivity  r(branching, oracle_stop_step):")
    for b, bname in zip(_BUCKET_BUDGETS, _BUCKET_NAMES):
        bcol = f"oracle_stop_step_b{b}"
        if bcol not in df:
            continue
        sub = df[["n_possible_moves", bcol]].dropna()
        r = np.corrcoef(sub["n_possible_moves"], sub[bcol])[0, 1]
        print(f"    {bname:<15} (budget={b:3d}): r = {r:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees-root", default=_TREES_ROOT)
    p.add_argument("--n-trees", type=int, default=39668)
    p.add_argument("--budget", type=int, default=43, help="Primary oracle budget (default: 43 = large-bucket mid)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(_FIGURES_DIR))
    args = p.parse_args(argv)

    print(f"Loading up to {args.n_trees:,} trees…")
    df = load_trees(args.trees_root, n_max=args.n_trees, seed=args.seed)
    print(f"  {len(df):,} trees loaded.")
    print_summary(df, args.budget)

    out = args.output_dir
    plot_comparison_bar(df, args.budget, os.path.join(out, "oracle_stop_step_vs_human_rt.png"))
    plot_correlation_matrix(df, args.budget, os.path.join(out, "oracle_stop_step_correlation_matrix.png"))


if __name__ == "__main__":
    main()
