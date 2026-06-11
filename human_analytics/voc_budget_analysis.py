"""
VOC_budget vs VOC_depth comparison.

VOC_depth (current default):
    depth_shallow=1, depth_deep=5 — engine searches to a fixed ply.
    Effective search depth is constant regardless of branching factor.

VOC_budget (this script):
    nodes_shallow=1, nodes_deep=96 — engine uses a fixed node budget.
    In branchy positions, 96 nodes = shallower effective depth;
    in simple positions, 96 nodes = deeper effective depth.
    Hypothesis: branching factor and own material should correlate more
    positively with VOC_budget than VOC_depth, because the engine has less
    budget to resolve complex positions.

Also computes nodes_searched_depth5: how many nodes Stockfish actually
explores at depth=5 (quasi-cost). This should correlate with branching.

Run (from chess_analysis/):
    python human_analytics/voc_budget_analysis.py --n-sample 10000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import chess
import chess.engine
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

_HA = os.path.dirname(os.path.abspath(__file__))
if _HA not in sys.path:
    sys.path.insert(0, _HA)

from engine_analysis import PositionEval, evaluate_position, _win_prob
from utils.helpers import STOCKFISH_SF14_PATH, STOCKFISH_SF14_DIR, apply_poster_style, MAIN_COLOR
from utils.selected_db import SELECTED_DB_DEFAULT

_FIGURES_DIR = os.path.join(_HA, "figures")
_TABLE = "pos_with_engine_eval"

_NODES_DEEP = 96
_NODES_SHALLOW = 1

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------
_w_engine: chess.engine.SimpleEngine | None = None


def _init_worker() -> None:
    global _w_engine
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_SF14_PATH, cwd=STOCKFISH_SF14_DIR)
    engine.configure({"Threads": 1, "Hash": 32})
    _w_engine = engine


def _eval_row_budget(row: dict) -> dict:
    """Compute VOC_budget (nodes=96/1) and nodes_searched_depth5 for one position."""
    global _w_engine
    out = {"fen": row["fen"], "gid": row["gid"], "move_ply": row["move_ply"],
           "n_possible_moves": row["n_possible_moves"],
           "n_self_pieces_exc_pawns": row.get("n_self_pieces_exc_pawns"),
           "voc_depth": row["voc"],
           "move_time": row["move_time"]}

    try:
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["move_uci"])
        if board.is_game_over() or move not in board.legal_moves:
            out["voc_budget"] = float("nan")
            out["nodes_depth5"] = float("nan")
            return out

        _w_engine.configure({"Clear Hash": None})

        # VOC_budget: nodes=96 deep, nodes=1 shallow
        result_budget = evaluate_position(
            board, move, _w_engine,
            nodes_deep=_NODES_DEEP, nodes_shallow=_NODES_SHALLOW,
        )
        out["voc_budget"] = result_budget.voc if result_budget.voc is not None else float("nan")

        # nodes_searched at depth=5 (quasi-cost of the deep search)
        _w_engine.configure({"Clear Hash": None})
        info = _w_engine.analyse(board, chess.engine.Limit(depth=5), multipv=1)
        out["nodes_depth5"] = float(info[0].get("nodes", float("nan"))) if info else float("nan")

    except Exception as exc:  # noqa: BLE001
        out["voc_budget"] = float("nan")
        out["nodes_depth5"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample(db_path: str, n: int, seed: int = 99) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT p.fen, p.gid, p.move_ply, p.move_uci, p.n_possible_moves,
               p.voc, p.toptwo, p.mq, p.move_time,
               pm.n_self_pieces_exc_pawns
        FROM (
            SELECT p.fen, p.gid, p.move_ply, p.move_uci, p.n_possible_moves,
                   p.voc, p.toptwo, p.mq, p.move_time,
                   pm.n_self_pieces_exc_pawns
            FROM {_TABLE} p
            JOIN processed_moves_nonzero pm ON p.gid = pm.gid AND p.move_ply = pm.move_ply
            WHERE p.move_time > 0
        ) filtered
        USING SAMPLE {n} ROWS (RESERVOIR, {seed})
    """).df()
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _corr_matrix_fig(df: pd.DataFrame, cols: list[str], labels: list[str], title: str) -> plt.Figure:
    corr = df[cols].corr()
    corr.columns = labels
    corr.index = labels
    n = len(corr)

    apply_poster_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=10)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            col = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=col,
                    fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    return fig


def plot_voc_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Side-by-side scatter: VOC_depth vs VOC_budget, coloured by branching factor."""
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, col, title in [
        (axes[0], "voc_depth", f"VOC_depth  (depth=5/1)"),
        (axes[1], "voc_budget", f"VOC_budget  (nodes={_NODES_DEEP}/{_NODES_SHALLOW})"),
    ]:
        sub = df.dropna(subset=[col, "n_possible_moves"])
        sc = ax.scatter(sub["n_possible_moves"], sub[col],
                        c=sub["n_possible_moves"], cmap="viridis",
                        alpha=0.3, s=6, linewidths=0)
        ax.set_xlabel("Branching factor (# legal moves)", fontsize=14)
        ax.set_ylabel(f"{col}", fontsize=14)
        ax.set_title(title, fontsize=13)
        r = np.corrcoef(sub["n_possible_moves"], sub[col])[0, 1]
        ax.text(0.05, 0.95, f"r = {r:+.3f}", transform=ax.transAxes,
                fontsize=12, va="top", fontweight="bold")
    plt.colorbar(sc, ax=axes[1]).set_label("Branching factor", fontsize=10)
    plt.tight_layout()
    out = os.path.join(output_dir, "voc_depth_vs_budget_branching.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ {out}")


def plot_nodes_vs_branching(df: pd.DataFrame, output_dir: str) -> None:
    """Nodes searched at depth=5 vs branching factor and own material."""
    apply_poster_style()
    sub = df.dropna(subset=["nodes_depth5", "n_possible_moves", "n_self_pieces_exc_pawns"])
    sub = sub[sub["nodes_depth5"] > 0]
    sub["log_nodes"] = np.log(sub["nodes_depth5"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, xcol, xlabel in [
        (axes[0], "n_possible_moves", "Branching factor (# legal moves)"),
        (axes[1], "n_self_pieces_exc_pawns", "Own non-pawn pieces"),
    ]:
        ax.scatter(sub[xcol], sub["log_nodes"], color=MAIN_COLOR, alpha=0.15, s=4, linewidths=0)
        r = np.corrcoef(sub[xcol], sub["log_nodes"])[0, 1]
        # binned trend
        bins = pd.qcut(sub[xcol], q=20, labels=False, duplicates="drop")
        trend = sub.groupby(bins)["log_nodes"].mean()
        x_trend = sub.groupby(bins)[xcol].mean()
        ax.plot(x_trend.values, trend.values, color="black", lw=2.5)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("log(nodes searched at depth=5)", fontsize=14)
        ax.set_title(f"Tree size quasi-cost  (r = {r:+.3f})", fontsize=13)

    plt.tight_layout()
    out = os.path.join(output_dir, "tree_size_vs_complexity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--n-sample", type=int, default=10_000)
    parser.add_argument("--n-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output-dir", default=_FIGURES_DIR)
    args = parser.parse_args(argv)

    print(f"Sampling {args.n_sample:,} positions…")
    df = _sample(args.db, args.n_sample, seed=args.seed)
    print(f"  {len(df):,} sampled.")

    rows = df.to_dict("records")
    t0 = time.perf_counter()
    with mp.Pool(processes=args.n_workers, initializer=_init_worker) as pool:
        results = list(tqdm(pool.imap(_eval_row_budget, rows, chunksize=max(1, len(rows)//(args.n_workers*4))),
                            total=len(rows), desc=f"VOC_budget + tree size ({args.n_workers} workers)"))
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    res = pd.DataFrame(results)
    res = res.dropna(subset=["voc_budget", "nodes_depth5"])
    print(f"  Valid rows: {len(res):,}")

    # Summary
    print("\n=== VOC_depth vs VOC_budget correlations with complexity ===")
    for xcol, xlabel in [("n_possible_moves", "Branching"), ("n_self_pieces_exc_pawns", "Own material")]:
        sub = res.dropna(subset=[xcol])
        r_d = np.corrcoef(sub[xcol], sub["voc_depth"])[0, 1]
        r_b = np.corrcoef(sub[xcol], sub["voc_budget"])[0, 1]
        print(f"  {xlabel:<16} r(VOC_depth) = {r_d:+.4f}   r(VOC_budget) = {r_b:+.4f}   Δ = {r_b-r_d:+.4f}")

    print("\n=== Tree size (log nodes at depth=5) correlations ===")
    log_nodes = np.log(res["nodes_depth5"].clip(lower=1))
    for xcol, xlabel in [("n_possible_moves", "Branching"), ("n_self_pieces_exc_pawns", "Own material")]:
        sub = res.dropna(subset=[xcol])
        r = np.corrcoef(sub[xcol], np.log(sub["nodes_depth5"].clip(lower=1)))[0, 1]
        print(f"  {xlabel:<16} r(log nodes, {xlabel}) = {r:+.4f}")

    print("\n=== Correlation: VOC_depth vs VOC_budget ===")
    r_vv = np.corrcoef(res["voc_depth"], res["voc_budget"])[0, 1]
    frac_agree = (((res["voc_depth"] > 0.001) == (res["voc_budget"] > 0.001))).mean()
    print(f"  r(VOC_depth, VOC_budget) = {r_vv:+.4f}")
    print(f"  Fraction both non-zero or both zero: {frac_agree:.1%}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_voc_comparison(res, args.output_dir)
    plot_nodes_vs_branching(res, args.output_dir)

    # Correlation matrices side by side
    cols_depth = ["n_possible_moves", "n_self_pieces_exc_pawns", "voc_depth", "toptwo", "mq"]
    cols_budget = ["n_possible_moves", "n_self_pieces_exc_pawns", "voc_budget", "toptwo", "mq"]
    labels = ["Branching", "Own material", "VOC", "toptwo", "MQ"]

    fig_d = _corr_matrix_fig(res, cols_depth, labels, f"VOC_depth  (depth=5/1,  n={len(res):,})")
    fig_b = _corr_matrix_fig(res, cols_budget, labels, f"VOC_budget  (nodes={_NODES_DEEP}/{_NODES_SHALLOW},  n={len(res):,})")

    fig_d.savefig(os.path.join(args.output_dir, "corr_voc_depth.png"), dpi=150, bbox_inches="tight"); plt.close(fig_d)
    fig_b.savefig(os.path.join(args.output_dir, "corr_voc_budget.png"), dpi=150, bbox_inches="tight"); plt.close(fig_b)
    print(f"✅ {os.path.join(args.output_dir, 'corr_voc_depth.png')}")
    print(f"✅ {os.path.join(args.output_dir, 'corr_voc_budget.png')}")


if __name__ == "__main__":
    main()
