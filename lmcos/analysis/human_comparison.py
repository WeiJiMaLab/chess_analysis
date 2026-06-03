"""
Compare lmcos search depth against human think-time predictors.

For each tree in the generated_trees dataset, extracts:
  - n_possible_moves       (from root FEN via python-chess)
  - n_self_pieces_exc_pawns (own non-pawn pieces, same regex as human_analytics)
  - move_ply               (from FEN)
  - n_expansions           (total expansions in the tree — how much the model searched)
  - toptwo_equiv           (Q gap between top-2 root children at final expansion)
  - gain_depth_equiv       (oracle best Q at final step minus first-evaluation step)

Plots: x_feature vs n_expansions (stratified by ply tertile),
mirroring the human_analytics x_feature vs log(RT) plots.

Figures saved to lmcos/analysis/figures/.

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

from utils.helpers import apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, PHASE_COLORS

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

_RE_SELF_WHITE = re.compile(r"[RNBQK]")
_RE_SELF_BLACK = re.compile(r"[rnbqk]")


def _n_self_pieces(placement: str, side: str) -> int:
    s = placement.replace("/", "")
    return len((_RE_SELF_WHITE if side == "w" else _RE_SELF_BLACK).findall(s))


def extract_tree_features(t: dict) -> dict | None:
    try:
        fen = t["root_position_spec"]
        parts = fen.split()
        side = parts[1]
        placement = parts[0]
        fullmove = int(parts[5]) if len(parts) > 5 else 1

        board = chess.Board(fen)
        n_legal = board.legal_moves.count()
        n_self = _n_self_pieces(placement, side)
        move_ply = (fullmove - 1) * 2 + (0 if side == "w" else 1)

        # n_nodes: total nodes in the Lc0 tree (varies by position even at fixed budget
        # due to tree shape — deeper in tactical positions, wider in equal ones)
        n_nodes = int(t["node_features"].shape[0])

        # toptwo_equiv: top-2 Q gap at final expansion step
        q_trace = t["oracle_root_q_trace"]  # [T, n_children]
        final_q = q_trace[-1]
        nonzero_q = final_q[final_q != 0]
        if len(nonzero_q) >= 2:
            top2 = nonzero_q.topk(2).values
            toptwo = (top2[0] - top2[1]).item()
        else:
            toptwo = float("nan")

        # gain_depth_equiv: best Q at final step minus best Q at first evaluation step
        first_nz = (q_trace.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
        best_q_final = final_q.max().item()
        if len(first_nz) > 0:
            best_q_shallow = q_trace[first_nz[0].item()].max().item()
            gain_depth = best_q_final - best_q_shallow
        else:
            gain_depth = float("nan")

        return {
            "fen": fen,
            "n_possible_moves": n_legal,
            "n_self_pieces_exc_pawns": n_self,
            "move_ply": move_ply,
            "n_expansions": n_nodes,
            "toptwo_equiv": toptwo,
            "gain_depth_equiv": gain_depth,
        }
    except Exception:
        return None


def load_trees(trees_root: str, n_max: int, seed: int = 42) -> pd.DataFrame:
    shard_dirs = sorted(d for d in os.listdir(trees_root) if d.startswith("filtered_shard"))
    all_files = []
    for sd in shard_dirs:
        p = os.path.join(trees_root, sd)
        all_files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))

    rng = np.random.default_rng(seed)
    rng.shuffle(all_files)

    records = []
    for path in tqdm(all_files[:n_max], desc="Loading trees"):
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


def _draw_trend(ax, x: np.ndarray, y: np.ndarray, color, label: str | None, n_bins: int = 20) -> None:
    trend = _bin_trend(x, y, n_bins=n_bins)
    if trend.empty:
        return
    ci = 1.96 * trend["sem_y"]
    ax.plot(trend["mean_x"], trend["mean_y"], color=color, lw=3, label=label)
    ax.fill_between(trend["mean_x"], trend["mean_y"] - ci, trend["mean_y"] + ci, color=color, alpha=0.2)


def plot_feature_vs_n_expansions(df: pd.DataFrame, x_col: str, x_label: str, output_path: str) -> None:
    """x_col vs n_expansions, stratified by ply tertile — mirrors human RT plots."""
    apply_poster_style()
    sub = df.dropna(subset=[x_col, "n_expansions", "move_ply"]).copy()
    sub["ply_tertile"] = pd.Categorical(pd.qcut(sub["move_ply"], q=3, duplicates="drop")).codes + 1

    r = float(np.corrcoef(sub[x_col], sub["n_expansions"])[0, 1]) if len(sub) > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(16, 10))

    for t in sorted(sub["ply_tertile"].unique()):
        s = sub[sub["ply_tertile"] == t]
        if len(s) < 10:
            continue
        color = PHASE_COLORS.get(int(t), "#888888")
        ply_min, ply_max = int(s["move_ply"].min()), int(s["move_ply"].max())
        _draw_trend(ax, s[x_col].to_numpy(float), s["n_expansions"].to_numpy(float),
                    color=color, label=f"Tertile {t} (ply {ply_min}–{ply_max})")

    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("n_nodes (tree size)", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"n = {len(sub):,}  ·  r = {r:+.3f}", fontsize=FONT_SIZE_LABEL, pad=10)
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
    print(f"  n_expansions: mean={df['n_expansions'].mean():.1f}  "
          f"std={df['n_expansions'].std():.1f}  p50={df['n_expansions'].median():.0f}")
    for xcol, xlabel in [
        ("n_possible_moves",       "branching"),
        ("n_self_pieces_exc_pawns","own material"),
        ("gain_depth_equiv",       "gain_depth_equiv"),
        ("toptwo_equiv",           "toptwo_equiv"),
    ]:
        sub = df.dropna(subset=[xcol, "n_expansions"])
        if len(sub) < 5:
            continue
        r = np.corrcoef(sub[xcol], sub["n_expansions"])[0, 1]
        print(f"  r(n_expansions, {xlabel:<22}) = {r:+.4f}  (n={len(sub):,})")
    print()


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
        ("n_possible_moves",        "Number of legal moves (branching factor)", "branching_vs_n_expansions.png"),
        ("n_self_pieces_exc_pawns", "Own non-pawn pieces",                      "material_vs_n_expansions.png"),
        ("gain_depth_equiv",        "gain_depth_equiv (best Q: final − step 1)","gain_depth_vs_n_expansions.png"),
        ("toptwo_equiv",            "toptwo_equiv (top-2 oracle Q gap)",        "toptwo_vs_n_expansions.png"),
    ]:
        plot_feature_vs_n_expansions(df, x_col, x_label, os.path.join(out, fname))


if __name__ == "__main__":
    main()
