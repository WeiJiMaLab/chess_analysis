"""
Exploratory analysis of VOC and Move Quality (MQ) on a sample of positions.

Two quantities computed via engine (default: lc0):
  MQ   = e_win_taken - e_win_best      (≤ 0; 0 = optimal move played)
  VOC  = V_deep(a_deep) - V_deep(a_shallow)  (≥ 0; Russek et al.)

VOC variance analysis: each position is evaluated n_repeats times to measure
per-position estimation uncertainty. For lc0 (multi-threaded MCTS), repeated
calls on the same position produce different WDL estimates due to search-tree
non-determinism; for Stockfish, hash is cleared between repeats.

Plots produced in figures/voc_mq_exploration/:
  mq_vs_clock.png           — MQ as a function of player clock time
  rt_vs_voc.png             — log(move_time) vs mean VOC
  voc_distribution.png      — histogram of per-position mean VOC
  voc_variance_dist.png     — histogram of per-position std(VOC) across repeats
  voc_mean_vs_std.png       — scatter: mean VOC vs std(VOC), coloured by n_possible_moves

Usage (from chess_analysis/):
    python human_analytics/voc_mq_exploration.py
    python human_analytics/voc_mq_exploration.py --n-sample 100 --n-repeats 5 --engine lc0
"""

from __future__ import annotations

import argparse
import os
import sys

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

from engine_analysis import move_quality, voc
from utils.helpers import (
    STOCKFISH_SF14_PATH,
    STOCKFISH_SF14_DIR,
    LC0_PATH,
    LC0_WEIGHTS_PATH,
    apply_poster_style,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    MAIN_COLOR,
    get_lc0_engine,
)
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES_NONZERO

_FIGURES_DIR = os.path.join(_HA, "figures", "voc_mq_analysis")

_N_THREADS_LC0 = 2  # multi-threaded MCTS → natural stochasticity between repeats


def _open_engine(engine_type: str) -> chess.engine.SimpleEngine:
    if engine_type == "lc0":
        return get_lc0_engine(threads=_N_THREADS_LC0)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_SF14_PATH, cwd=STOCKFISH_SF14_DIR)
    engine.configure({"Threads": 1, "Hash": 128})
    return engine


def _reset_between_repeats(engine: chess.engine.SimpleEngine, engine_type: str) -> None:
    """Reset engine state so successive VOC estimates are independent."""
    if engine_type == "stockfish":
        engine.configure({"Clear Hash": None})
    # lc0: MCTS stochasticity with _N_THREADS_LC0 > 1 gives natural variance;
    # no explicit reset needed — each analyse() call uses a fresh search.


def _sample_positions(
    db_path: str, n: int, seed: int = 42, min_ply: int = 15, max_ply: int = 75
) -> pd.DataFrame:
    """
    Sample n middlegame positions (min_ply ≤ move_ply ≤ max_ply, matching Russek et al.
    ply 15–75 window). Dedup on fen. USING SAMPLE must be on the outer subquery.
    """
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT fen, move_uci, player_clock_time, opponent_clock_time, move_time, move_ply, n_possible_moves
        FROM (
            SELECT
                fen, move_uci, player_clock_time, opponent_clock_time, move_time, move_ply, n_possible_moves,
                ROW_NUMBER() OVER (PARTITION BY fen ORDER BY gid) AS _rn
            FROM (
                SELECT
                    pm.fen,
                    m.move_uci,
                    pm.player_clock_time,
                    pm.opponent_clock_time,
                    pm.move_time,
                    pm.move_ply,
                    pm.n_possible_moves,
                    pm.gid
                FROM {TABLE_PROCESSED_MOVES_NONZERO} pm
                JOIN moves m ON pm.gid = m.gid AND pm.move_ply = m.move_ply
                WHERE pm.move_ply BETWEEN {min_ply} AND {max_ply}
            ) filtered
            USING SAMPLE {n} ROWS (RESERVOIR, {seed})
        ) dedup
        WHERE _rn = 1
        LIMIT {n}
    """).df()
    conn.close()
    return df


def _evaluate_mq(
    df: pd.DataFrame,
    engine: chess.engine.SimpleEngine,
    engine_type: str,
    depth_deep: int,
) -> pd.Series:
    """Compute MQ once per position. Returns a Series aligned to df.index."""
    mqs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MQ (1×)"):
        try:
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(row["move_uci"])
            if board.is_game_over() or move not in board.legal_moves:
                mqs.append(float("nan"))
                continue
            _reset_between_repeats(engine, engine_type)
            val = move_quality(board, move, engine, depth=depth_deep)
            mqs.append(val if val is not None else float("nan"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip MQ] {row['fen'][:40]}… — {exc}")
            mqs.append(float("nan"))
    return pd.Series(mqs, index=df.index)


def _evaluate_voc_repeats(
    df: pd.DataFrame,
    engine: chess.engine.SimpleEngine,
    engine_type: str,
    depth_deep: int,
    depth_shallow: int,
    n_repeats: int,
) -> pd.DataFrame:
    """
    For each position, compute VOC n_repeats times.

    Returns a DataFrame with columns:
        fen, voc_mean, voc_std, voc_min, voc_max, voc_n_valid,
        player_clock_time, move_time, n_possible_moves
    """
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"VOC ({n_repeats}×)"):
        try:
            board = chess.Board(row["fen"])
            if board.is_game_over():
                continue
            voc_vals = []
            for _ in range(n_repeats):
                _reset_between_repeats(engine, engine_type)
                v = voc(board, engine, depth_deep=depth_deep, depth_shallow=depth_shallow)
                if v is not None:
                    voc_vals.append(v)
            if not voc_vals:
                continue
            records.append({
                "fen": row["fen"],
                "player_clock_time": float(row["player_clock_time"]),
                "move_time": float(row["move_time"]),
                "n_possible_moves": int(row["n_possible_moves"]),
                "voc_mean": float(np.mean(voc_vals)),
                "voc_std": float(np.std(voc_vals, ddof=1)) if len(voc_vals) > 1 else 0.0,
                "voc_min": float(np.min(voc_vals)),
                "voc_max": float(np.max(voc_vals)),
                "voc_n_valid": len(voc_vals),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip VOC] {row['fen'][:40]}… — {exc}")
    return pd.DataFrame(records)


def _bin_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 20, min_n: int = 2) -> pd.DataFrame:
    """Quantile-bin x, return mean_x/mean_y/sem_y per bin (drops bins with < min_n points)."""
    labels = pd.qcut(x, q=min(n_bins, len(x)), labels=False, duplicates="drop")
    rows = []
    for b in sorted(set(labels)):
        mask = labels == b
        n = int(mask.sum())
        if n < min_n:
            continue
        rows.append({
            "mean_x": x[mask].mean(),
            "mean_y": y[mask].mean(),
            "sem_y": y[mask].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            "n": n,
        })
    return pd.DataFrame(rows)


def _plot_with_trend(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    n_bins: int = 20,
    scatter_n: int = 2000,
    scatter_alpha: float = 0.15,
) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), size=min(scatter_n, len(x)), replace=False)
    ax.scatter(x[idx], y[idx], color=MAIN_COLOR, alpha=scatter_alpha, s=8, linewidths=0)

    trend = _bin_trend(x, y, n_bins=n_bins)
    if not trend.empty:
        ci = 1.96 * trend["sem_y"]
        ax.plot(trend["mean_x"], trend["mean_y"], color=MAIN_COLOR, lw=3, label="Mean")
        ax.fill_between(
            trend["mean_x"], trend["mean_y"] - ci, trend["mean_y"] + ci,
            color=MAIN_COLOR, alpha=0.25, label="95% CI",
        )
        ax.legend(fontsize=FONT_SIZE_TICKS)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_mq_vs_clock(mq: pd.Series, df: pd.DataFrame, output_path: str) -> None:
    """MQ as a function of player clock time left."""
    apply_poster_style()
    sub = df.copy()
    sub["mq"] = mq
    sub = sub.dropna(subset=["mq", "player_clock_time"])
    x = sub["player_clock_time"].to_numpy(dtype=float)
    y = sub["mq"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(18, 12))
    _plot_with_trend(ax, x, y, x_label="Player clock time (s)", y_label="Move Quality (MQ)")
    ax.set_title(
        f"Move Quality vs. Clock Time (n={len(sub):,})\n"
        "Expect: MQ decreases with less time remaining",
        fontsize=FONT_SIZE_LABEL, pad=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def plot_rt_vs_voc(var_df: pd.DataFrame, output_path: str) -> None:
    """log(move_time) as a function of mean VOC."""
    apply_poster_style()
    sub = var_df[var_df["move_time"] > 0].dropna(subset=["voc_mean", "move_time"])
    x = sub["voc_mean"].to_numpy(dtype=float)
    y = np.log(sub["move_time"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(18, 12))
    _plot_with_trend(ax, x, y, x_label="VOC (mean across repeats)", y_label=r"$\ln$(Move Time)")
    ax.set_title(
        f"Response Time vs. VOC (n={len(sub):,})\n"
        "Expect: log RT increases with VOC",
        fontsize=FONT_SIZE_LABEL, pad=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def plot_voc_distribution(var_df: pd.DataFrame, output_path: str) -> None:
    """Histogram of per-position mean VOC."""
    apply_poster_style()
    v = var_df["voc_mean"].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.hist(v, bins=40, color=MAIN_COLOR, alpha=0.7, edgecolor=MAIN_COLOR)
    ax.set_xlabel("VOC (mean)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
    stats_text = (
        f"n={len(v):,}  mean={v.mean():.4f}  std={v.std():.4f}\n"
        f"median={np.median(v):.4f}  p90={np.percentile(v, 90):.4f}  max={v.max():.4f}"
    )
    ax.set_title(f"Distribution of mean VOC\n{stats_text}", fontsize=FONT_SIZE_LABEL, pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def plot_voc_variance_distribution(var_df: pd.DataFrame, n_repeats: int, output_path: str) -> None:
    """Histogram of per-position std(VOC) across n_repeats evaluations."""
    apply_poster_style()
    s = var_df["voc_std"].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.hist(s, bins=40, color=MAIN_COLOR, alpha=0.7, edgecolor=MAIN_COLOR)
    ax.set_xlabel(f"std(VOC)  [{n_repeats} repeats per position]", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_SIZE_LABEL)
    frac_zero = (s < 1e-6).mean()
    stats_text = (
        f"n={len(s):,}  mean={s.mean():.4f}  median={np.median(s):.4f}  "
        f"p90={np.percentile(s, 90):.4f}\n"
        f"fraction with std ≈ 0 (deterministic): {frac_zero:.1%}"
    )
    ax.set_title(
        f"Per-position VOC estimation variance ({n_repeats}× per position)\n{stats_text}",
        fontsize=FONT_SIZE_LABEL, pad=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def plot_voc_mean_vs_std(var_df: pd.DataFrame, output_path: str) -> None:
    """Scatter: mean VOC vs std(VOC), point size/colour by n_possible_moves."""
    apply_poster_style()
    sub = var_df.dropna(subset=["voc_mean", "voc_std"])
    x = sub["voc_mean"].to_numpy(dtype=float)
    y = sub["voc_std"].to_numpy(dtype=float)
    c = sub["n_possible_moves"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(18, 14))
    sc = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.6, s=60, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("# legal moves", fontsize=FONT_SIZE_LABEL - 6)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 6)
    ax.set_xlabel("VOC mean", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("VOC std", fontsize=FONT_SIZE_LABEL)
    ax.set_title(
        f"VOC mean vs. std across repeats (n={len(sub):,})\n"
        "Higher std = more estimation uncertainty",
        fontsize=FONT_SIZE_LABEL, pad=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {output_path}")


def _print_summary(mq: pd.Series, var_df: pd.DataFrame) -> None:
    print("\n=== VOC summary (mean across repeats) ===")
    v = var_df["voc_mean"].dropna()
    print(f"  n={len(v):,}  mean={v.mean():.4f}  std={v.std():.4f}  "
          f"p10={v.quantile(.1):.4f}  p50={v.median():.4f}  p90={v.quantile(.9):.4f}")
    print(f"  Fraction with VOC > 0.001: {(v > 0.001).mean():.1%}")

    print("\n=== VOC per-position estimation std ===")
    s = var_df["voc_std"].dropna()
    print(f"  mean std={s.mean():.4f}  median std={s.median():.4f}  "
          f"p90 std={s.quantile(.9):.4f}  max std={s.max():.4f}")
    print(f"  Fraction deterministic (std ≈ 0): {(s < 1e-6).mean():.1%}")

    print("\n=== MQ summary ===")
    m = mq.dropna()
    print(f"  n={len(m):,}  mean={m.mean():.4f}  std={m.std():.4f}  "
          f"p10={m.quantile(.1):.4f}  p50={m.median():.4f}  min={m.min():.4f}")
    print(f"  Fraction near-optimal (MQ ≥ −0.005): {(m >= -0.005).mean():.1%}")

    print("\n=== Correlations ===")
    sub = var_df[var_df["move_time"] > 0].dropna(subset=["voc_mean", "move_time"])
    if len(sub) > 2:
        r = np.corrcoef(sub["voc_mean"], np.log(sub["move_time"]))[0, 1]
        print(f"  Pearson r(log RT, VOC_mean)  = {r:.4f}  (n={len(sub):,})")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--engine", choices=["lc0", "stockfish"], default="lc0")
    parser.add_argument("--n-sample", type=int, default=100,
                        help="Positions to sample (default: 100)")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="VOC repeats per position for variance estimation (default: 5)")
    parser.add_argument("--depth-deep", type=int, default=5,
                        help="Deep search depth for VOC and MQ (default: 5)")
    parser.add_argument("--depth-shallow", type=int, default=1,
                        help="Shallow search depth for VOC (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=_FIGURES_DIR)
    args = parser.parse_args(argv)

    print(f"Engine: {args.engine}  depth_deep={args.depth_deep}  depth_shallow={args.depth_shallow}  "
          f"n_repeats={args.n_repeats}")
    if args.engine == "lc0":
        print(f"  lc0: {LC0_PATH}")
        print(f"  weights: {LC0_WEIGHTS_PATH}")
        print(f"  threads: {_N_THREADS_LC0}  (stochastic MCTS → natural per-position variance)")

    print(f"\nSampling {args.n_sample} positions from {args.db}…")
    df = _sample_positions(args.db, n=args.n_sample, seed=args.seed)
    print(f"  Sampled {len(df)} unique positions.")

    if df.empty:
        print("ERROR: no positions sampled — check db path and ply filter.")
        return

    engine = _open_engine(args.engine)
    try:
        mq = _evaluate_mq(df, engine, args.engine, depth_deep=args.depth_deep)
        var_df = _evaluate_voc_repeats(
            df, engine, args.engine,
            depth_deep=args.depth_deep,
            depth_shallow=args.depth_shallow,
            n_repeats=args.n_repeats,
        )
    finally:
        engine.quit()

    if var_df.empty:
        print("ERROR: no valid VOC evaluations — check engine path and positions.")
        return

    _print_summary(mq, var_df)

    out = args.output_dir
    plot_mq_vs_clock(mq, df, os.path.join(out, "mq_vs_clock.png"))
    plot_rt_vs_voc(var_df, os.path.join(out, "rt_vs_voc.png"))
    plot_voc_distribution(var_df, os.path.join(out, "voc_distribution.png"))
    plot_voc_variance_distribution(var_df, args.n_repeats, os.path.join(out, "voc_variance_dist.png"))
    plot_voc_mean_vs_std(var_df, os.path.join(out, "voc_mean_vs_std.png"))


if __name__ == "__main__":
    main()
