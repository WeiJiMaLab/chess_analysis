"""
Timing comparison: Stockfish vs lc0 at depth [1, 5, 10] for MQ + VOC.

Samples a small set of middlegame positions (move_ply > 10), evaluates each
with both engines at each depth, and prints a timing table.  Extrapolates to
10K and 100K positions assuming naive serial evaluation and a 10× speedup
from parallelism (conservative; actual SLURM throughput is higher).

Usage (from chess_analysis/):
    python human_analytics/timing_comparison.py
    python human_analytics/timing_comparison.py --n-positions 10 --depths 1 5 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import chess.engine
import duckdb
import pandas as pd
from tqdm import tqdm

_HA = os.path.dirname(os.path.abspath(__file__))
if _HA not in sys.path:
    sys.path.insert(0, _HA)

from engine_analysis import move_quality, voc
from utils.helpers import (
    STOCKFISH_SF14_PATH,
    STOCKFISH_SF14_DIR,
    get_lc0_engine,
)
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES_NONZERO


def _sample(db_path: str, n: int, seed: int = 7) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT fen, move_uci, player_clock_time, move_time
        FROM (
            SELECT fen, move_uci, player_clock_time, move_time, gid,
                   ROW_NUMBER() OVER (PARTITION BY fen ORDER BY gid) AS _rn
            FROM (
                SELECT pm.fen, pm.player_clock_time, pm.move_time, pm.gid, m.move_uci
                FROM {TABLE_PROCESSED_MOVES_NONZERO} pm
                JOIN moves m ON pm.gid = m.gid AND pm.move_ply = m.move_ply
                WHERE pm.move_ply > 10
            ) filtered
            USING SAMPLE {n} ROWS (RESERVOIR, {seed})
        ) dedup
        WHERE _rn = 1
        LIMIT {n}
    """).df()
    conn.close()
    return df


def _open_stockfish() -> chess.engine.SimpleEngine:
    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH_SF14_PATH, cwd=STOCKFISH_SF14_DIR)
    e.configure({"Threads": 1, "Hash": 32})
    return e


def _time_depth(
    rows: list[dict],
    engine: chess.engine.SimpleEngine,
    engine_name: str,
    depth: int,
) -> dict:
    """Evaluate all positions at given depth, return timing stats."""
    times = []
    for row in rows:
        try:
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(row["move_uci"])
            if board.is_game_over() or move not in board.legal_moves:
                continue
            if engine_name == "stockfish":
                engine.configure({"Clear Hash": None})
            t0 = time.perf_counter()
            mq_val = move_quality(board, move, engine, depth=depth)
            voc_val = voc(board, engine, depth_deep=depth, depth_shallow=1)
            elapsed = time.perf_counter() - t0
            if mq_val is not None and voc_val is not None:
                times.append(elapsed)
        except Exception:
            pass
    if not times:
        return {"engine": engine_name, "depth": depth, "n": 0,
                "mean_s": float("nan"), "total_s": float("nan")}
    import statistics
    return {
        "engine": engine_name,
        "depth": depth,
        "n": len(times),
        "mean_s": statistics.mean(times),
        "total_s": sum(times),
    }


def _extrapolate(mean_s: float, parallelism: int = 10) -> tuple[str, str]:
    """Extrapolate to 10K and 100K positions with parallelism factor."""
    t10k = mean_s * 10_000 / parallelism
    t100k = mean_s * 100_000 / parallelism
    def _fmt(s: float) -> str:
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"
    return _fmt(t10k), _fmt(t100k)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--n-positions", type=int, default=10)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--parallelism", type=int, default=10,
                        help="Assumed parallel workers for extrapolation (default: 10)")
    args = parser.parse_args(argv)

    print(f"Sampling {args.n_positions} middlegame positions…")
    df = _sample(args.db, args.n_positions, seed=args.seed)
    print(f"  Got {len(df)} unique positions.")
    rows = df.to_dict("records")

    results = []

    for depth in args.depths:
        print(f"\n--- Stockfish depth={depth} ---")
        sf = _open_stockfish()
        try:
            r = _time_depth(rows, sf, "stockfish", depth)
        finally:
            sf.quit()
        results.append(r)
        print(f"  mean {r['mean_s']:.2f}s/pos  (n={r['n']})")

    for depth in args.depths:
        print(f"\n--- lc0 depth={depth} ---")
        lc0 = get_lc0_engine(threads=1)
        try:
            r = _time_depth(rows, lc0, "lc0", depth)
        finally:
            lc0.quit()
        results.append(r)
        print(f"  mean {r['mean_s']:.2f}s/pos  (n={r['n']})")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Engine':<12} {'Depth':<8} {'n':<6} {'s/pos':<10} "
          f"{'10K ({:d}×par)':<16} {'100K ({:d}×par)':<16}".format(args.parallelism, args.parallelism))
    print("-" * 72)
    for r in results:
        t10k, t100k = _extrapolate(r["mean_s"], args.parallelism)
        print(f"{r['engine']:<12} {r['depth']:<8} {r['n']:<6} "
              f"{r['mean_s']:<10.2f} {t10k:<16} {t100k:<16}")
    print("=" * 72)
    print(f"Parallelism assumption: {args.parallelism}× (e.g. SLURM array with {args.parallelism} workers)")


if __name__ == "__main__":
    main()
