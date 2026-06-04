"""
Analysis: min_expansions vs board features.

min_expansions(t) = first step t >= 1 at which oracle_best_move_index[t] == final_best
                    AND oracle_best_move_index[s] == final_best for all s > t.

Step 0 is excluded — at step 0 all Q-values are zero, so argmax = 0 is an
initialisation artifact, not a meaningful evaluation.

Key question: why is min_expansions *negatively* correlated with gain_depth_equiv?

Usage (from lmcos/):
    python analysis/min_expansions_analysis.py --n-trees 5000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "human_analytics"))

from analysis.oracle_stop_step_features import (
    extract_board_features,
    extract_tree_features,
)
from utils.helpers import apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR, PHASE_COLORS

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"

# Human RT correlations for comparison
_HUMAN_R = {
    "branching": +0.195,
    "material":  +0.039,
    "gain_depth": +0.096,
    "toptwo":    -0.064,
}


def min_expansions(best_idx: torch.Tensor) -> int:
    """
    First step t >= 1 at which oracle_best_move_index[t] == final_best
    and never deviates from final_best thereafter.

    Excludes step 0 (all-zero Q-trace → argmax = 0 is an artifact).

    Returns 1 if the oracle immediately committed to the final best at step 1
    and never changed its mind.
    """
    if len(best_idx) < 2:
        return 1

    final = best_idx[-1].item()
    # Consider only steps 1..T-1 (skip step 0)
    relevant = best_idx[1:]                              # shape [T-1]
    wrong_steps = (relevant != final).nonzero(as_tuple=True)[0]

    if len(wrong_steps) == 0:
        # Already correct at step 1, never deviates
        return 1

    # Last wrong step in relevant is at index wrong_steps[-1],
    # which corresponds to actual step wrong_steps[-1] + 1.
    # First stable step = last wrong step + 1.
    last_wrong_actual = int(wrong_steps[-1].item()) + 1   # convert back to step index
    return last_wrong_actual + 1


def process_tree(t: dict) -> dict | None:
    """Extract board/tree features + min_expansions from one filtered_shard tree."""
    try:
        fen = t["root_position_spec"]
        board_feats = extract_board_features(fen)
        tree_feats = extract_tree_features(t)

        best_idx = t["oracle_best_move_index"]           # [T]
        me = min_expansions(best_idx)
        T = len(best_idx)

        # Also store final best move's Q and num evaluations as diagnostics
        final_best = best_idx[-1].item()
        q = t["oracle_root_q_trace"]
        q_best_over_time = q[:, final_best]              # Q of the final best move at each step

        # Count steps where the final best was the CURRENT best (how stable was it?)
        steps_correct = int((best_idx[1:] == final_best).sum().item())

        return {
            **board_feats,
            **tree_feats,
            "min_expansions": me,
            "budget": T,
            "steps_correct": steps_correct,   # steps where best == final_best (excluding step 0)
            "frac_correct": steps_correct / max(T - 1, 1),
        }
    except Exception:
        return None


def load_trees(trees_root: str, n_max: int, seed: int = 42) -> pd.DataFrame:
    dirs = sorted(d for d in os.listdir(trees_root) if d.startswith("filtered_shard"))
    files: list[str] = []
    for d in dirs:
        p = os.path.join(trees_root, d)
        files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))

    np.random.default_rng(seed).shuffle(files)
    rows = []
    for path in tqdm(files[:n_max], desc="Loading trees"):
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            row = process_tree(t)
            if row is not None:
                rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    me = df["min_expansions"].dropna()
    budget = df["budget"].dropna()
    print(f"\n{'='*65}")
    print(f"min_expansions analysis  n = {len(df):,}")
    print(f"  min_expansions: mean={me.mean():.1f}  std={me.std():.1f}  p50={me.median():.0f}")
    print(f"  budget:         mean={budget.mean():.1f}  (always 96 for these trees)")
    print(f"  frac_correct:   mean={df['frac_correct'].mean():.3f}  "
          f"(fraction of steps where best == final)")
    print(f"{'='*65}")

    features = [
        ("n_possible_moves",        "branching",    "branching"),
        ("n_self_pieces_exc_pawns", "material",     "material"),
        ("gain_depth_equiv",        "gain_depth",   "gain_depth"),
        ("toptwo_equiv",            "toptwo",       "toptwo"),
    ]
    for feat_col, feat_label, human_key in features:
        sub = df[[feat_col, "min_expansions"]].dropna()
        if len(sub) < 5:
            continue
        r = np.corrcoef(sub[feat_col], sub["min_expansions"])[0, 1]
        r_human = _HUMAN_R[human_key]
        match = "✓ SAME" if (r * r_human) > 0 else "✗ DIFFER"
        print(f"  {feat_label:<14} min_exp r={r:+.4f}   human r={r_human:+.4f}   {match}")

    # The key: is the negative gain_depth correlation driven by same-move refinement?
    sub = df[["gain_depth_equiv", "min_expansions", "frac_correct"]].dropna()
    print(f"\n  r(gain_depth, frac_correct) = "
          f"{np.corrcoef(sub['gain_depth_equiv'], sub['frac_correct'])[0,1]:+.4f}")
    print("  → If negative: high gain_depth positions keep the same best move for fewer steps")
    print("  → If positive: high gain_depth = oracle consistently correct about the best move\n")


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    apply_poster_style()
    os.makedirs(output_dir, exist_ok=True)

    features = [
        ("n_possible_moves",        "Branching",   "branching"),
        ("n_self_pieces_exc_pawns", "Material",    "material"),
        ("gain_depth_equiv",        "gain_depth",  "gain_depth"),
        ("toptwo_equiv",            "toptwo",      "toptwo"),
    ]

    # --- Figure 1: comparison bar chart (min_expansions vs human RT) ---
    me_r, human_r, labels = [], [], []
    for feat_col, feat_label, human_key in features:
        sub = df[[feat_col, "min_expansions"]].dropna()
        if len(sub) < 5:
            continue
        me_r.append(np.corrcoef(sub[feat_col], sub["min_expansions"])[0, 1])
        human_r.append(_HUMAN_R[human_key])
        labels.append(feat_label)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.bar(x - width/2, me_r, width, label="min_expansions",
           color=PHASE_COLORS[2], alpha=0.8)
    ax.bar(x + width/2, human_r, width, label="human log(RT)",
           color=PHASE_COLORS[1], alpha=0.8)
    ax.axhline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS)
    ax.set_ylabel("Pearson r", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"Feature correlations: min_expansions vs human log(RT)  (n={len(df):,})",
                 fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "min_expansions_vs_human_rt.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ {path1}")

    # --- Figure 2: gain_depth vs min_expansions scatter + trend ---
    sub = df[["gain_depth_equiv", "min_expansions", "frac_correct"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    apply_poster_style()

    for ax, y_col, y_label in [
        (axes[0], "min_expansions", "min_expansions"),
        (axes[1], "frac_correct",   "frac steps correct (best == final_best)"),
    ]:
        r = np.corrcoef(sub["gain_depth_equiv"], sub[y_col])[0, 1]
        # Binned trend
        labels_bins = pd.qcut(sub["gain_depth_equiv"], q=20, labels=False, duplicates="drop")
        trend = sub.groupby(labels_bins)["gain_depth_equiv"].mean().values
        y_trend = sub.groupby(labels_bins)[y_col].mean().values
        ax.scatter(sub["gain_depth_equiv"], sub[y_col], color=MAIN_COLOR, alpha=0.1, s=4)
        ax.plot(trend, y_trend, color="black", lw=3)
        ax.set_xlabel("gain_depth_equiv", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"r = {r:+.3f}", fontsize=FONT_SIZE_LABEL)

    fig.suptitle("Why is gain_depth negatively correlated with min_expansions?",
                 fontsize=FONT_SIZE_LABEL + 2, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "min_expansions_gain_depth_diagnostic.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ {path2}")


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
