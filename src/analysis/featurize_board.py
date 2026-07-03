"""Featurize the P0 board-correlate battery (plan_final_push.md, P0) — engine-free.

Computes, for every DISTINCT FEN in the ply-windowed analysis set (pmnz_win),
the pre-registered board features via python-chess (no engine anywhere):

  in_check           mover is in check (also a DB column; recomputed for self-containment)
  n_captures_avail   # legal moves that are captures
  n_checks_avail     # legal moves that give check
  self_material      weighted non-pawn material for the side to move (N/B/R/Q = 3/3/5/9)
  opp_material       same for the opponent
  material_imbalance self_material - opp_material (mover POV)

`prev_move_was_capture` is game-context (SQL lag over moves.n_pieces), not a
per-FEN feature — it is derived at analysis time, not here.

Two stages:
  1. dump DISTINCT windowed FENs to <out>/fens_distinct.parquet (skipped if present),
  2. a worker pool featurizes row-group slices and writes <out>/features_NN.parquet shards.

Run (SLURM: slurm/pipeline/featurize_board.slurm):
  python -m analysis.featurize_board --config config_minply15_maxply75.yaml \
      --out-dir /scratch/.../minply15_maxply75/board_features --workers 32
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

# --config early-peek: set $CONFIG before helpers import reads it (same shim as engine.py).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config")
_args, _ = _pre.parse_known_args()
if _args.config:
    os.environ["CONFIG"] = _args.config

import chess  # noqa: E402
import duckdb  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.utils.helpers import CONFIG, create_ply_windowed_views  # noqa: E402
from analysis.utils.selected_db import SELECTED_DB_DEFAULT  # noqa: E402

# Non-pawn material weights (kings excluded). Weighted, not raw count: the
# depth/decidedness hypotheses are about value at stake (Q != N).
_PIECE_VALUES = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

FEATURE_COLUMNS = [
    "in_check", "n_captures_avail", "n_checks_avail",
    "self_material", "opp_material", "material_imbalance",
]


def featurize_fen(fen: str):
    """All P0 features for one FEN (4-field FENs parse fine; ep square honored)."""
    board = chess.Board(fen)
    n_captures = 0
    n_checks = 0
    for move in board.legal_moves:
        if board.is_capture(move):
            n_captures += 1
        if board.gives_check(move):
            n_checks += 1
    self_material = sum(v * len(board.pieces(pt, board.turn)) for pt, v in _PIECE_VALUES.items())
    opp_material = sum(v * len(board.pieces(pt, not board.turn)) for pt, v in _PIECE_VALUES.items())
    return (
        board.is_check(), n_captures, n_checks,
        self_material, opp_material, self_material - opp_material,
    )


def dump_distinct_fens(db_path: str, fens_path: str, limit: int | None) -> int:
    """Stage 1: DISTINCT windowed FENs -> parquet (streamed by DuckDB, low-mem)."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        create_ply_windowed_views(conn)
        lim = f"LIMIT {limit}" if limit else ""
        conn.execute(
            f"COPY (SELECT DISTINCT fen FROM pmnz_win {lim}) TO '{fens_path}' (FORMAT PARQUET)"
        )
        n = conn.execute(f"SELECT count(*) FROM read_parquet('{fens_path}')").fetchone()[0]
    finally:
        conn.close()
    return n


def worker(job) -> str:
    """Featurize this worker's round-robin share of row groups -> one shard parquet."""
    fens_path, out_dir, widx, n_workers = job
    pf = pq.ParquetFile(fens_path)
    rows, bad = [], 0
    t0 = time.time()
    for rg in range(widx, pf.num_row_groups, n_workers):
        for fen in pf.read_row_group(rg, columns=["fen"])["fen"].to_pylist():
            try:
                rows.append((fen, *featurize_fen(fen)))
            except Exception:
                bad += 1  # unparsable FEN: count, don't crash the shard
    df = pd.DataFrame(rows, columns=["fen", *FEATURE_COLUMNS])
    shard = os.path.join(out_dir, f"features_{widx:02d}.parquet")
    df.to_parquet(shard, index=False)
    rate = len(df) / max(time.time() - t0, 1e-9)
    print(f"worker {widx:02d}: {len(df):,} FENs ({bad} unparsable) at {rate:,.0f}/s -> {shard}",
          flush=True)
    return shard


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0 board-correlate featurization (engine-free)")
    parser.add_argument("--config", help="Path to the run config (else $CONFIG or the default).")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    # Default beside the run's other scratch outputs (cache_default = <run_dir>/tree_values_cache).
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(CONFIG["cache_default"]), "board_features"),
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--limit", type=int, help="Smoke-test cap on DISTINCT FENs.")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    fens_path = os.path.join(args.out_dir, "fens_distinct.parquet")

    if os.path.exists(fens_path) and not args.limit:
        print(f"FEN dump exists, reusing: {fens_path}")
    else:
        t0 = time.time()
        n = dump_distinct_fens(args.db, fens_path, args.limit)
        print(f"dumped {n:,} distinct windowed FENs in {time.time() - t0:.1f}s -> {fens_path}")

    jobs = [(fens_path, args.out_dir, w, args.workers) for w in range(args.workers)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        shards = pool.map(worker, jobs)
    print(f"FEATURIZE DONE: {len(shards)} shards in {time.time() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
