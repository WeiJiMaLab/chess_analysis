"""
Analysis 0a: oracle_stop_step vs board features on existing lmcos trees.

Computes the DP-optimal stopping step for a sample of filtered_shard trees
using compute_budgeted_oracle(), then correlates with board features extracted
from the root FEN. Produces two figures:

  figures/oracle_stop_step_vs_human_rt.png
      Side-by-side r (and r²) for oracle_stop_step vs board features,
      compared against established human RT correlations.

  figures/oracle_stop_step_correlation_matrix.png
      Correlation matrix of {oracle_stop_step, branching, material,
      toptwo_equiv, gain_depth_equiv} on lmcos trees.

Usage (from lmcos/):
    python analysis/oracle_stop_step_features.py
    python analysis/oracle_stop_step_features.py --n-trees 5000 --budget 43
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

# Representative budget per bucket: roughly mid-range of each
_BUCKET_BUDGETS = [2, 7, 18, 43, 90]
_BUCKET_NAMES = [b.name for b in DEFAULT_BUDGET_BUCKETS]

# Human RT correlations from established analysis (human_analytics, n=1M, Stockfish depth=5)
_HUMAN_R = {
    "branching": +0.195,
    "material":  +0.039,
    "gain_depth": +0.096,
    "toptwo":    -0.064,
}

_RE_WHITE = re.compile(r"[RNBQK]")
_RE_BLACK = re.compile(r"[rnbqk]")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def compute_oracle_stop_step(
    halt_rewards: list[float],
    tree_sizes: list[int],
    budget: int,
    config: BudgetedOracleConfig,
) -> int:
    """Run the DP oracle and return the optimal halt step from snapshot 0."""
    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, budget, config)
    return policy.optimal_stop_step


def extract_board_features(fen: str) -> dict:
    """Position features from a root FEN string."""
    parts = fen.split()
    side = parts[1] if len(parts) > 1 else "w"
    placement = parts[0]
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    move_ply = (fullmove - 1) * 2 + (0 if side == "w" else 1)

    board = chess.Board(fen)
    n_possible = board.legal_moves.count()
    flat = placement.replace("/", "")
    n_self = len((_RE_WHITE if side == "w" else _RE_BLACK).findall(flat))

    return {"n_possible_moves": n_possible, "n_self_pieces_exc_pawns": n_self, "move_ply": move_ply}


def extract_tree_features(t: dict) -> dict:
    """toptwo_equiv and gain_depth_equiv from oracle_root_q_trace."""
    q = t["oracle_root_q_trace"]          # [T, n_children]
    final_q = q[-1]
    nonzero = final_q[final_q != 0]

    if len(nonzero) >= 2:
        v = nonzero.topk(2).values
        toptwo = float((v[0] - v[1]).abs().item())
    else:
        toptwo = float("nan")

    first_nz = (q.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
    best_q_final = final_q.max().item()
    if len(first_nz) > 0:
        gain = best_q_final - q[first_nz[0].item()].max().item()
    else:
        gain = float("nan")

    return {"toptwo_equiv": toptwo, "gain_depth_equiv": gain}


def process_tree(t: dict, budgets: list[int]) -> dict | None:
    """Extract all features + oracle_stop_step for each budget from one tree."""
    try:
        fen = t["root_position_spec"]
        board_feats = extract_board_features(fen)
        tree_feats = extract_tree_features(t)

        q = t["oracle_root_q_trace"]
        best_idx = t["oracle_best_move_index"]
        T = len(q)

        halt_rewards = [float(q[s, best_idx[s].item()].item()) for s in range(T)]
        tree_sizes = [1] * T  # maintenance_scale=0 → irrelevant

        row = {**board_feats, **tree_feats}
        for budget in budgets:
            b = min(budget, T)
            row[f"oracle_stop_step_b{budget}"] = compute_oracle_stop_step(
                halt_rewards[:b], tree_sizes[:b], b, _CONFIG
            )
        return row
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
            row = process_tree(t, _BUCKET_BUDGETS)
            if row is not None:
                rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _analysis_style() -> None:
    """Compact style for lmcos analysis figures (not poster scale)."""
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


_ORACLE_COLOR = "#2563EB"   # blue for oracle
_HUMAN_COLOR  = "#16a085"   # teal for human


def plot_comparison_bar(df: pd.DataFrame, primary_budget: int, output_path: str) -> None:
    """Figure 1: r and r² for oracle_stop_step vs board features, vs human RT."""
    _analysis_style()
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
        (axes[0], oracle_r,  human_r,            "Pearson r",
         f"Feature correlations  (n={len(df):,}, budget={primary_budget})"),
        (axes[1], oracle_r2, [r**2 for r in human_r], "r²", "Variance explained"),
    ]:
        b1 = ax.bar(x - width/2, yvals_oracle, width, color=_ORACLE_COLOR, alpha=0.8,
                    label=f"oracle_stop_step (b={primary_budget})")
        b2 = ax.bar(x + width/2, yvals_human, width, color=_HUMAN_COLOR, alpha=0.8,
                    label="human log(RT)")
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
    _analysis_style()
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
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Pearson r")
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
    print(f"lmcos trees  n = {len(df):,}  (primary budget = {primary_budget})")
    print(f"oracle_stop_step: mean={oss.mean():.1f}  std={oss.std():.1f}  p50={oss.median():.0f}")
    print(f"{'='*60}")
    for feat_col, feat_label, human_key in [
        ("n_possible_moves",        "branching",   "branching"),
        ("n_self_pieces_exc_pawns", "material",    "material"),
        ("gain_depth_equiv",        "gain_depth",  "gain_depth"),
        ("toptwo_equiv",            "toptwo",      "toptwo"),
    ]:
        sub = df[[feat_col, col]].dropna()
        if len(sub) < 5:
            continue
        r_oracle = np.corrcoef(sub[feat_col], sub[col])[0, 1]
        r_human = _HUMAN_R[human_key]
        match = "✓ SAME" if (r_oracle * r_human) > 0 else "✗ DIFFER"
        print(f"  {feat_label:<14} oracle r={r_oracle:+.4f}  human r={r_human:+.4f}  {match}")
    print()

    # Budget sensitivity
    print("  Budget sensitivity:")
    for b, bname in zip(_BUCKET_BUDGETS, _BUCKET_NAMES):
        bcol = f"oracle_stop_step_b{b}"
        if bcol not in df:
            continue
        sub = df[["n_possible_moves", bcol]].dropna()
        r = np.corrcoef(sub["n_possible_moves"], sub[bcol])[0, 1]
        print(f"    {bname:<15} (budget={b:3d}): r(branching, oracle_stop_step) = {r:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees-root", default=_TREES_ROOT)
    p.add_argument("--n-trees", type=int, default=5000)
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
