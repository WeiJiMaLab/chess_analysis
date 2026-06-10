"""
Analysis: converged_expansions vs board features.

converged_expansions(best_idx) = first step t >= 1 at which oracle_best_move_index[t] == final_best
                                 AND oracle_best_move_index[s] == final_best for all s > t.
                                 = 1 + last step where the MCTS deviated from its final recommendation.

Step 0 is excluded — at step 0 all Q-values are zero, so argmax = 0 is an
initialisation artifact, not a meaningful evaluation.

Relationship to oracle_stop_step:
    oracle_stop_step <= converged_expansions always.
    Equal iff no MCTS oscillation after first correct recommendation (~43% of trees).
    See tests/test_oracle_stop_vs_min_expansions.py.

Usage (from lmcos/):
    python analysis/converged_expansions_analysis.py --n-trees 5000
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis._data import load_shard_trees
from analysis._plots import analysis_style, save_fig
from analysis.board_tree_features import HUMAN_RT_CORRELATIONS, extract_board_features, extract_tree_features

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"

def converged_expansions(best_idx: torch.Tensor) -> int:
    """First step t >= 1 at which oracle_best_move_index permanently equals final_best.

    = 1 + last step where the MCTS deviated from its final recommendation.
    Excludes step 0 (all-zero Q-trace → argmax = 0 is an initialisation artifact).
    Returns 1 if the oracle immediately committed to final_best at step 1 and never changed.
    """
    if len(best_idx) < 2:
        return 1

    final = best_idx[-1].item()
    relevant = best_idx[1:]                              # shape [T-1], skip step 0
    wrong_steps = (relevant != final).nonzero(as_tuple=True)[0]

    if len(wrong_steps) == 0:
        return 1

    last_wrong_actual = int(wrong_steps[-1].item()) + 1   # convert back to step index
    return last_wrong_actual + 1


def process_tree(t: dict) -> dict | None:
    """Extract board/tree features + converged_expansions from one filtered_shard tree."""
    try:
        fen = t["root_position_spec"]
        board_feats = extract_board_features(fen)
        tree_feats = extract_tree_features(t)

        best_idx = t["oracle_best_move_index"]           # [T]
        ce = converged_expansions(best_idx)
        T = len(best_idx)

        final_best = best_idx[-1].item()

        # Count steps where the final best was the CURRENT best (how stable was it?)
        steps_correct = int((best_idx[1:] == final_best).sum().item())

        return {
            **board_feats,
            **tree_feats,
            "converged_expansions": ce,
            "budget": T,
            "steps_correct": steps_correct,
            "frac_correct": steps_correct / max(T - 1, 1),
        }
    except Exception:
        return None


def load_trees(trees_root: str, n_max: int, seed: int = 42) -> pd.DataFrame:
    return pd.DataFrame(load_shard_trees(trees_root, n_max, process_tree, seed=seed))


def print_summary(df: pd.DataFrame) -> None:
    ce = df["converged_expansions"].dropna()
    budget = df["budget"].dropna()
    print(f"\n{'='*65}")
    print(f"converged_expansions analysis  n = {len(df):,}")
    print(f"  converged_expansions: mean={ce.mean():.1f}  std={ce.std():.1f}  p50={ce.median():.0f}")
    print(f"  budget:               mean={budget.mean():.1f}  (always 96 for these trees)")
    print(f"  frac_correct:         mean={df['frac_correct'].mean():.3f}  "
          f"(fraction of steps where best == final)")
    print(f"{'='*65}")

    features = [
        ("n_possible_moves",        "branching",    "branching"),
        ("n_self_pieces_exc_pawns", "material",     "material"),
        ("gain_depth_equiv",        "gain_depth",   "gain_depth"),
        ("toptwo_equiv",            "toptwo",       "toptwo"),
    ]
    for feat_col, feat_label, human_key in features:
        sub = df[[feat_col, "converged_expansions"]].dropna()
        if len(sub) < 5:
            continue
        r = np.corrcoef(sub[feat_col], sub["converged_expansions"])[0, 1]
        r_human = HUMAN_RT_CORRELATIONS[human_key]
        match = "✓ SAME" if (r * r_human) > 0 else "✗ DIFFER"
        print(f"  {feat_label:<14} conv_exp r={r:+.4f}   human r={r_human:+.4f}   {match}")

    sub = df[["gain_depth_equiv", "converged_expansions", "frac_correct"]].dropna()
    print(f"\n  r(gain_depth, frac_correct) = "
          f"{np.corrcoef(sub['gain_depth_equiv'], sub['frac_correct'])[0,1]:+.4f}")
    print("  → Negative: high gain_depth positions keep the same best move for fewer steps")
    print("  → Positive: high gain_depth = oracle consistently correct about the best move\n")


_CE_COLOR    = "#6366f1"   # indigo for converged_expansions
_HUMAN_COLOR = "#16a085"   # teal for human


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    analysis_style()
    os.makedirs(output_dir, exist_ok=True)

    features = [
        ("n_possible_moves",        "Branching",   "branching"),
        ("n_self_pieces_exc_pawns", "Material",    "material"),
        ("gain_depth_equiv",        "gain_depth",  "gain_depth"),
        ("toptwo_equiv",            "toptwo",      "toptwo"),
    ]

    ce_r, human_r, labels = [], [], []
    for feat_col, feat_label, human_key in features:
        sub = df[[feat_col, "converged_expansions"]].dropna()
        if len(sub) < 5:
            continue
        ce_r.append(np.corrcoef(sub[feat_col], sub["converged_expansions"])[0, 1])
        human_r.append(HUMAN_RT_CORRELATIONS[human_key])
        labels.append(feat_label)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, ce_r, width, label="converged_expansions",
           color=_CE_COLOR, alpha=0.8)
    ax.bar(x + width/2, human_r, width, label="human log(RT)",
           color=_HUMAN_COLOR, alpha=0.8)
    ax.axhline(0, color="black", lw=1.2, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"converged_expansions vs human log(RT)  (n={len(df):,})")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1), framealpha=0.9)
    plt.tight_layout()
    save_fig(os.path.join(output_dir, "converged_expansions_vs_human_rt.png"))

    sub = df[["gain_depth_equiv", "converged_expansions", "frac_correct"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    analysis_style()

    for ax, y_col, y_label in [
        (axes[0], "converged_expansions", "converged_expansions"),
        (axes[1], "frac_correct",         "frac. steps where best == final"),
    ]:
        r = np.corrcoef(sub["gain_depth_equiv"], sub[y_col])[0, 1]
        labels_bins = pd.qcut(sub["gain_depth_equiv"], q=20, labels=False, duplicates="drop")
        trend = sub.groupby(labels_bins)["gain_depth_equiv"].mean().values
        y_trend = sub.groupby(labels_bins)[y_col].mean().values
        ax.scatter(sub["gain_depth_equiv"], sub[y_col], color=_CE_COLOR, alpha=0.08, s=3)
        ax.plot(trend, y_trend, color="black", lw=2.5)
        ax.set_xlabel("gain_depth_equiv")
        ax.set_ylabel(y_label)
        ax.set_title(f"r = {r:+.3f}")

    fig.suptitle("gain_depth vs converged_expansions", fontsize=13, y=1.03)
    plt.tight_layout()
    save_fig(os.path.join(output_dir, "converged_expansions_gain_depth_diagnostic.png"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees-root", default=_TREES_ROOT)
    p.add_argument("--n-trees", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(_FIGURES_DIR))
    args = p.parse_args(argv)

    print(f"Loading up to {args.n_trees:,} trees…")
    df = load_trees(args.trees_root, n_max=args.n_trees, seed=args.seed)
    print(f"  {len(df):,} trees loaded.")
    print_summary(df)
    plot_results(df, args.output_dir)


if __name__ == "__main__":
    main()
