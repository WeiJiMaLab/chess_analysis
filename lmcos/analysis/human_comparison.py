"""
Compare lmcos normative compute allocation against human think-time predictors.

For each tree in the generated_trees dataset, extracts:
  - n_possible_moves  (from root FEN via python-chess)
  - n_self_pieces_exc_pawns  (own non-pawn pieces, same regex as human_analytics)
  - move_ply  (from FEN)
  - min_expansions  (first stable oracle best-move index — the normative opt_think_steps)
  - toptwo_equiv  (Q gap between oracle's top-2 root children at final expansion)
  - gain_depth_equiv  (oracle Q of best move at final step minus step 0)

Produces 3-panel comparison figures alongside the human analytics plots:
  lmcos/analysis/figures/branching_vs_min_expansions.png
  lmcos/analysis/figures/material_vs_min_expansions.png
  lmcos/analysis/figures/gain_depth_vs_min_expansions.png
  lmcos/analysis/figures/toptwo_vs_min_expansions.png

Usage (from chess_analysis/):
    python lmcos/analysis/human_comparison.py
    python lmcos/analysis/human_comparison.py --n-trees 5000
"""

from __future__ import annotations

import argparse
import os
import sys
import re

import chess
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HA = os.path.join(_REPO, "human_analytics")
if _HA not in sys.path:
    sys.path.insert(0, _HA)

from utils.helpers import apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, PHASE_COLORS, MAIN_COLOR

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Regex for own non-pawn pieces (same as human_analytics preprocess.py)
_RE_SELF_WHITE = re.compile(r"[RNBQK]")
_RE_SELF_BLACK = re.compile(r"[rnbqk]")


def _n_self_pieces(board_placement: str, side_to_move: str) -> int:
    if side_to_move == "w":
        return len(_RE_SELF_WHITE.findall(board_placement.replace("/", "")))
    return len(_RE_SELF_BLACK.findall(board_placement.replace("/", "")))


def extract_tree_features(t: dict) -> dict | None:
    """Extract per-tree features from a filtered_shard tree file."""
    try:
        fen = t["root_position_spec"]
        board = chess.Board(fen)
        n_legal = board.legal_moves.count()
        side = fen.split()[1]
        placement = fen.split()[0]
        n_self = _n_self_pieces(placement, side)

        # Move ply from FEN fullmove number and side
        parts = fen.split()
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        move_ply = (fullmove - 1) * 2 + (0 if side == "w" else 1)

        # oracle_best_move_index: shape [budget, n_children] — best move at each expansion
        best_idx = t["oracle_best_move_index"]  # shape [T]
        final_best = best_idx[-1].item()

        # min_expansions: first step where best move == final_best AND stays there
        # Use stable convergence: last flip point + 1
        flip_steps = (best_idx != final_best).nonzero(as_tuple=True)[0]
        if len(flip_steps) == 0:
            min_expansions = 0  # already optimal from step 0
        else:
            min_expansions = flip_steps[-1].item() + 1

        # oracle_root_q_trace: shape [T, n_children]
        q_trace = t["oracle_root_q_trace"]  # [T, n_children]
        final_q = q_trace[-1]  # Q-values of all root children at final step

        # toptwo_equiv: gap between best and second-best at final step
        nonzero_q = final_q[final_q != 0]
        if len(nonzero_q) >= 2:
            top2 = nonzero_q.topk(2).values
            toptwo = (top2[0] - top2[1]).item()
        elif len(nonzero_q) == 1:
            toptwo = float("nan")
        else:
            toptwo = float("nan")

        # gain_depth_equiv: best Q at final step - best Q at step 0
        q_step0 = q_trace[0]
        best_q_final = final_q[final_best].item()
        best_q_step0 = q_step0.max().item() if q_step0.max().item() != 0 else float("nan")
        gain_depth = best_q_final - best_q_step0 if not np.isnan(best_q_step0) else float("nan")

        return {
            "fen": fen,
            "n_possible_moves": n_legal,
            "n_self_pieces_exc_pawns": n_self,
            "move_ply": move_ply,
            "min_expansions": min_expansions,
            "toptwo_equiv": toptwo,
            "gain_depth_equiv": gain_depth,
        }
    except Exception:
        return None


def load_trees(trees_root: str, n_max: int, seed: int = 42) -> pd.DataFrame:
    """Load up to n_max trees from filtered_shards, return DataFrame of features."""
    shard_dirs = sorted(
        d for d in os.listdir(trees_root) if d.startswith("filtered_shard")
    )
    all_files = []
    for sd in shard_dirs:
        p = os.path.join(trees_root, sd)
        all_files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))

    rng = np.random.default_rng(seed)
    rng.shuffle(all_files)
    selected = all_files[:n_max]

    records = []
    for path in tqdm(selected, desc="Loading trees"):
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            row = extract_tree_features(t)
            if row is not None:
                records.append(row)
        except Exception:
            pass

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bin_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 20, min_n: int = 3) -> pd.DataFrame:
    labels = pd.qcut(x, q=min(n_bins, len(x) // max(1, min_n)), labels=False, duplicates="drop")
    rows = []
    for b in sorted(set(labels)):
        mask = labels == b
        n = int(mask.sum())
        if n < min_n:
            continue
        rows.append({
            "mean_x": float(x[mask].mean()),
            "mean_y": float(y[mask].mean()),
            "sem_y": float(y[mask].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n": n,
        })
    return pd.DataFrame(rows)


def _plot_trend(ax, x: np.ndarray, y: np.ndarray, color, label: str | None = None, n_bins: int = 20) -> None:
    trend = _bin_trend(x, y, n_bins=n_bins)
    if trend.empty:
        return
    ci = 1.96 * trend["sem_y"]
    ax.plot(trend["mean_x"], trend["mean_y"], color=color, lw=3, label=label)
    ax.fill_between(trend["mean_x"], trend["mean_y"] - ci, trend["mean_y"] + ci, color=color, alpha=0.2)


def plot_feature_vs_min_expansions(df: pd.DataFrame, x_col: str, x_label: str, output_path: str) -> None:
    """Single-panel: x_col vs min_expansions, stratified by ply tertile."""
    apply_poster_style()
    sub = df.dropna(subset=[x_col, "min_expansions", "move_ply"])
    sub = sub.copy()
    sub["ply_tertile"] = pd.qcut(sub["move_ply"], q=3, labels=[1, 2, 3]).astype(int)

    r = np.corrcoef(sub[x_col], sub["min_expansions"])[0, 1] if len(sub) > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(16, 10))

    for t in [1, 2, 3]:
        s = sub[sub["ply_tertile"] == t]
        if len(s) < 10:
            continue
        color = PHASE_COLORS[t]
        ply_min, ply_max = int(s["move_ply"].min()), int(s["move_ply"].max())
        _plot_trend(ax, s[x_col].to_numpy(float), s["min_expansions"].to_numpy(float),
                    color=color, label=f"Tertile {t} (ply {ply_min}–{ply_max})")

    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("min_expansions (oracle opt_think_steps)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        f"n = {len(sub):,}  ·  r = {r:+.3f}",
        fontsize=FONT_SIZE_LABEL, pad=10,
    )
    ax.legend(fontsize=FONT_SIZE_TICKS)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}   r = {r:+.4f}  (n={len(sub):,})")


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"lmcos trees  n = {len(df):,}")
    print(f"{'='*60}")
    print(f"  min_expansions: mean={df['min_expansions'].mean():.2f}  "
          f"std={df['min_expansions'].std():.2f}  "
          f"p50={df['min_expansions'].median():.1f}")
    for xcol, xlabel in [
        ("n_possible_moves", "branching"),
        ("n_self_pieces_exc_pawns", "own material"),
        ("gain_depth_equiv", "gain_depth_equiv"),
        ("toptwo_equiv", "toptwo_equiv"),
    ]:
        sub = df.dropna(subset=[xcol, "min_expansions"])
        if len(sub) < 5:
            continue
        r = np.corrcoef(sub[xcol], sub["min_expansions"])[0, 1]
        print(f"  r(min_expansions, {xlabel:<20}) = {r:+.4f}  (n={len(sub):,})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees-root", default=_TREES_ROOT)
    parser.add_argument("--n-trees", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=_FIGURES_DIR)
    args = parser.parse_args(argv)

    print(f"Loading up to {args.n_trees:,} trees from {args.trees_root}…")
    df = load_trees(args.trees_root, n_max=args.n_trees, seed=args.seed)
    print(f"  {len(df):,} trees loaded successfully.")
    print_summary(df)

    out = args.output_dir
    for x_col, x_label, fname in [
        ("n_possible_moves",        "Number of legal moves (branching factor)", "branching_vs_min_expansions.png"),
        ("n_self_pieces_exc_pawns", "Own non-pawn pieces",                      "material_vs_min_expansions.png"),
        ("gain_depth_equiv",        "gain_depth_equiv (best Q: final − step 0)","gain_depth_vs_min_expansions.png"),
        ("toptwo_equiv",            "toptwo_equiv (top-2 oracle Q gap)",        "toptwo_vs_min_expansions.png"),
    ]:
        plot_feature_vs_min_expansions(df, x_col, x_label, os.path.join(out, fname))


if __name__ == "__main__":
    main()
