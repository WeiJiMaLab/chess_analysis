"""
Replicate human think-time trends using lmcos oracle search trees.

For each tree, the y-axis is oracle_stop_step: the first expansion step at which
the oracle's best-move index has stably converged to its final answer and does not
change again. This is the minimum number of oracle expansions required before the
correct action can be identified — a proxy for the DP-optimal stopping step.

Note: the exact DP-oracle stopping step (as stored in oracle_stop_steps in the
packed episode shards) requires the full halt-reward DP computation and cannot be
reconstructed from filtered_shard tree files alone without re-running the oracle.

Position features extracted from root FEN (same regex as human_analytics):
  - n_possible_moves       branching factor
  - n_self_pieces_exc_pawns  own non-pawn pieces
  - move_ply               game stage

Tree features from oracle_root_q_trace:
  - toptwo_equiv           gap between oracle's top-2 root child Q-values at budget end
  - gain_depth_equiv       best root Q at budget end minus best Q at first evaluation step

Usage (from chess_analysis/):
    python lmcos/analysis/human_comparison.py
    python lmcos/analysis/human_comparison.py --n-trees 5000
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HA = os.path.join(_REPO, "human_analytics")
if _HA not in sys.path:
    sys.path.insert(0, _HA)

from utils.helpers import (
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    MAIN_COLOR,
    PHASE_COLORS,
    apply_poster_style,
)

_TREES_ROOT = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined"
_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

_RE_WHITE = re.compile(r"[RNBQK]")
_RE_BLACK = re.compile(r"[rnbqk]")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _oracle_stop_step(best_idx: torch.Tensor) -> int:
    """
    First stable convergence: last step at which oracle_best_move_index != final
    best, plus one. Equals 0 if the oracle never changes its mind.
    """
    final = best_idx[-1].item()
    flips = (best_idx != final).nonzero(as_tuple=True)[0]
    return 0 if len(flips) == 0 else int(flips[-1].item()) + 1


def extract_features(t: dict) -> dict | None:
    try:
        fen = t["root_position_spec"]
        parts = fen.split()
        side, placement = parts[1], parts[0]
        fullmove = int(parts[5]) if len(parts) > 5 else 1
        move_ply = (fullmove - 1) * 2 + (0 if side == "w" else 1)

        board = chess.Board(fen)
        n_legal = board.legal_moves.count()
        n_self = len((_RE_WHITE if side == "w" else _RE_BLACK).findall(placement.replace("/", "")))

        stop_step = _oracle_stop_step(t["oracle_best_move_index"])

        q = t["oracle_root_q_trace"]           # [T, n_children]
        final_q = q[-1]
        nonzero = final_q[final_q != 0]
        if len(nonzero) >= 2:
            v = nonzero.topk(2).values
            toptwo = float((v[0] - v[1]).abs().item())
        else:
            toptwo = float("nan")

        first_nz = (q.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
        best_q_final = final_q.max().item()
        gain = (best_q_final - q[first_nz[0].item()].max().item()) if len(first_nz) else float("nan")

        return {
            "n_possible_moves": n_legal,
            "n_self_pieces_exc_pawns": n_self,
            "move_ply": move_ply,
            "oracle_stop_step": stop_step,
            "toptwo_equiv": toptwo,
            "gain_depth_equiv": gain,
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
            r = extract_features(t)
            if r is not None:
                rows.append(r)
        except Exception:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting — matches human_analytics poster style
# ---------------------------------------------------------------------------

def _bin_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 20, min_n: int = 5) -> pd.DataFrame:
    labels = pd.qcut(x, q=min(n_bins, max(2, len(x) // min_n)), labels=False, duplicates="drop")
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
        })
    return pd.DataFrame(rows)


def _draw(ax, x: np.ndarray, y: np.ndarray, color, label: str) -> None:
    tr = _bin_trend(x, y)
    if tr.empty:
        return
    ci = 1.96 * tr["sem_y"]
    ax.plot(tr["mean_x"], tr["mean_y"], color=color, lw=3, label=label)
    ax.fill_between(tr["mean_x"], tr["mean_y"] - ci, tr["mean_y"] + ci, color=color, alpha=0.2)


def plot_vs_oracle_stop(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    output_path: str,
) -> None:
    """
    2-panel: raw trend (all data) | by ply tertile.
    y = oracle_stop_step (first stable convergence of oracle best move).
    """
    apply_poster_style()
    sub = df.dropna(subset=[x_col, "oracle_stop_step", "move_ply"]).copy()
    sub["ply_t"] = pd.Categorical(pd.qcut(sub["move_ply"], q=3, duplicates="drop")).codes + 1

    r_all = float(np.corrcoef(sub[x_col], sub["oracle_stop_step"])[0, 1]) if len(sub) > 2 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    # Left: global trend
    _draw(axes[0], sub[x_col].to_numpy(float), sub["oracle_stop_step"].to_numpy(float),
          color=MAIN_COLOR, label=f"r = {r_all:+.3f}")
    axes[0].set_title("Raw trend", fontsize=FONT_SIZE_LABEL, pad=12)
    axes[0].set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel("oracle_stop_step", fontsize=FONT_SIZE_LABEL)
    axes[0].legend(fontsize=FONT_SIZE_TICKS)

    # Right: by ply tertile
    for t in sorted(sub["ply_t"].unique()):
        s = sub[sub["ply_t"] == t]
        if len(s) < 20:
            continue
        color = PHASE_COLORS.get(int(t), MAIN_COLOR)
        pmin, pmax = int(s["move_ply"].min()), int(s["move_ply"].max())
        r_t = float(np.corrcoef(s[x_col], s["oracle_stop_step"])[0, 1])
        _draw(axes[1], s[x_col].to_numpy(float), s["oracle_stop_step"].to_numpy(float),
              color=color, label=f"Tertile {t} (ply {pmin}–{pmax}, r={r_t:+.2f})")
    axes[1].set_title("By ply tertile", fontsize=FONT_SIZE_LABEL, pad=12)
    axes[1].set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel("oracle_stop_step", fontsize=FONT_SIZE_LABEL)
    axes[1].legend(fontsize=FONT_SIZE_TICKS)

    fig.suptitle(f"n = {len(sub):,}", fontsize=FONT_SIZE_LABEL + 4, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}   r = {r_all:+.4f}  (n={len(sub):,})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    oss = df["oracle_stop_step"].dropna()
    print(f"\n{'='*60}")
    print(f"lmcos trees  n = {len(df):,}")
    print(f"  oracle_stop_step: mean={oss.mean():.1f}  std={oss.std():.1f}  p50={oss.median():.0f}")
    print(f"{'='*60}")
    for col, label in [
        ("n_possible_moves",        "branching"),
        ("n_self_pieces_exc_pawns", "own material"),
        ("gain_depth_equiv",        "gain_depth_equiv"),
        ("toptwo_equiv",            "toptwo_equiv"),
    ]:
        s = df.dropna(subset=[col, "oracle_stop_step"])
        if len(s) < 5:
            continue
        r = np.corrcoef(s[col], s["oracle_stop_step"])[0, 1]
        print(f"  r(oracle_stop_step, {label:<22}) = {r:+.4f}  (n={len(s):,})")
    print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trees-root", default=_TREES_ROOT)
    p.add_argument("--n-trees", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=_FIGURES_DIR)
    args = p.parse_args(argv)

    print(f"Loading up to {args.n_trees:,} trees from {args.trees_root}…")
    df = load_trees(args.trees_root, n_max=args.n_trees, seed=args.seed)
    print(f"  {len(df):,} trees loaded.")
    _print_summary(df)

    out = args.output_dir
    for x_col, x_label, fname in [
        ("n_possible_moves",        "Number of legal moves (branching factor)", "branching_vs_oracle_stop.png"),
        ("n_self_pieces_exc_pawns", "Own non-pawn pieces",                      "material_vs_oracle_stop.png"),
        ("gain_depth_equiv",        "gain_depth_equiv",                         "gain_depth_vs_oracle_stop.png"),
        ("toptwo_equiv",            "toptwo_equiv (top-2 oracle Q gap)",        "toptwo_vs_oracle_stop.png"),
    ]:
        plot_vs_oracle_stop(df, x_col, x_label, os.path.join(out, fname))


if __name__ == "__main__":
    main()
