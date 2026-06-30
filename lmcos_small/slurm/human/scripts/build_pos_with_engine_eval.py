"""
Build pos_with_engine_eval: engine quantities for every filtered move.

Computes six quantities per (position, move_taken) pair using evaluate_position():
    e_win_best        V_deep(a_deep)          win prob of best move
    e_win_second_best V_deep(rank-2 move)     win prob of second-best
    e_win_taken       V_deep(move_taken)      win prob of move actually played
    voc               V_deep(a_deep) - V_deep(a_shallow)   ≥ 0
    mq                e_win_taken - e_win_best              ≤ 0
    toptwo            e_win_best - e_win_second_best        ≥ 0

Exclusion criteria (Russek et al. 2022 alignment):
    move_ply BETWEEN 15 AND 75
    opponent_clock_time >= 60s

Subcommands:
    eval   — evaluate one SLURM shard, or run a local smoke test
    merge  — combine output parquets → personal.db table pos_with_engine_eval

Examples (from chess_analysis/):

    # Smoke test (10K positions, 10 workers):
    python lmcos_small/slurm/human/scripts/build_pos_with_engine_eval.py eval \\
        --n-total 10000 --n-workers 10

    # SLURM array job (one task per shard):
    python lmcos_small/slurm/human/scripts/build_pos_with_engine_eval.py eval \\
        --shard-id $SLURM_ARRAY_TASK_ID --total-shards 100

    # Merge all shards:
    python lmcos_small/slurm/human/scripts/build_pos_with_engine_eval.py merge
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
from tqdm import tqdm

from _bootstrap import ensure_src

ensure_src()

from engine_analysis import PositionEval, evaluate_position
from utils.helpers import get_engine
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES_NONZERO

# ---------------------------------------------------------------------------
# Exclusion criteria
# ---------------------------------------------------------------------------
_MIN_PLY = 15
_MAX_PLY = 75
_MIN_OPP_CLOCK = 60  # seconds

_OUTPUT_TABLE = "pos_with_engine_eval"
_DEFAULT_OUTPUT_DIR = "/scratch/gpfs/GRIFFITHS/hl4291/tmp/pos_with_engine_eval"

# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------
_worker_engine: chess.engine.SimpleEngine | None = None
_worker_engine_type: str = "stockfish"
_worker_depth_deep: int = 5
_worker_depth_shallow: int = 1
_worker_nodes_deep: int | None = None
_worker_nodes_shallow: int | None = None


def _init_worker(
    engine_type: str,
    depth_deep: int,
    depth_shallow: int,
    nodes_deep: int | None,
    nodes_shallow: int | None,
) -> None:
    global _worker_engine, _worker_engine_type, _worker_depth_deep, _worker_depth_shallow
    global _worker_nodes_deep, _worker_nodes_shallow
    _worker_engine_type = engine_type
    _worker_depth_deep = depth_deep
    _worker_depth_shallow = depth_shallow
    _worker_nodes_deep = nodes_deep
    _worker_nodes_shallow = nodes_shallow
    if engine_type == "stockfish":
        engine = get_engine("stockfish", threads=1, hash_mb=32)
    else:
        engine = get_engine("lc0", threads=1)
    _worker_engine = engine


def _eval_row(row: dict) -> dict:
    global _worker_engine, _worker_engine_type, _worker_depth_deep, _worker_depth_shallow
    global _worker_nodes_deep, _worker_nodes_shallow
    out = {k: row[k] for k in (
        "fen", "gid", "move_ply", "move_uci",
        "player_clock_time", "opponent_clock_time", "move_time", "n_possible_moves",
    )}
    _nan = {k: None for k in ("e_win_best", "e_win_second_best", "e_win_taken", "voc", "mq", "toptwo")}
    try:
        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["move_uci"])
        if board.is_game_over() or move not in board.legal_moves:
            out.update(_nan)
            return out
        if _worker_engine_type == "stockfish":
            _worker_engine.configure({"Clear Hash": None})
        result: PositionEval = evaluate_position(
            board, move, _worker_engine,
            depth_deep=_worker_depth_deep,
            depth_shallow=_worker_depth_shallow,
            nodes_deep=_worker_nodes_deep,
            nodes_shallow=_worker_nodes_shallow,
        )
        out["e_win_best"] = result.e_win_best
        out["e_win_second_best"] = result.e_win_second_best
        out["e_win_taken"] = result.e_win_taken
        out["voc"] = result.voc
        out["mq"] = result.mq
        out["toptwo"] = result.toptwo
    except Exception as exc:  # noqa: BLE001
        out.update(_nan)
        out["_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_shard(db_path: str, shard_id: int, total_shards: int) -> list[dict]:
    """Hash-partitioned shard for SLURM array jobs (no random sampling)."""
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT pm.fen, pm.gid, pm.move_ply, m.move_uci,
               pm.player_clock_time, pm.opponent_clock_time,
               pm.move_time, pm.n_possible_moves
        FROM {TABLE_PROCESSED_MOVES_NONZERO} pm
        JOIN moves m ON pm.gid = m.gid AND pm.move_ply = m.move_ply
        WHERE pm.move_ply BETWEEN {_MIN_PLY} AND {_MAX_PLY}
          AND pm.opponent_clock_time >= {_MIN_OPP_CLOCK}
          AND abs(hash(pm.gid * 1000 + pm.move_ply)) % {total_shards} = {shard_id}
    """).df()
    conn.close()
    return df.to_dict("records")


def _sample(db_path: str, n: int, seed: int = 42) -> list[dict]:
    """Reservoir-sample n positions for smoke tests."""
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    df = conn.execute(f"""
        SELECT fen, gid, move_ply, move_uci,
               player_clock_time, opponent_clock_time, move_time, n_possible_moves
        FROM (
            SELECT pm.fen, pm.gid, pm.move_ply, m.move_uci,
                   pm.player_clock_time, pm.opponent_clock_time,
                   pm.move_time, pm.n_possible_moves
            FROM {TABLE_PROCESSED_MOVES_NONZERO} pm
            JOIN moves m ON pm.gid = m.gid AND pm.move_ply = m.move_ply
            WHERE pm.move_ply BETWEEN {_MIN_PLY} AND {_MAX_PLY}
              AND pm.opponent_clock_time >= {_MIN_OPP_CLOCK}
        ) filtered
        USING SAMPLE {n} ROWS (RESERVOIR, {seed})
    """).df()
    conn.close()
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# Runner and merge
# ---------------------------------------------------------------------------

def run_eval(
    rows: list[dict],
    engine_type: str,
    depth_deep: int,
    depth_shallow: int,
    nodes_deep: int | None,
    nodes_shallow: int | None,
    n_workers: int,
    output_path: str,
) -> None:
    t0 = time.perf_counter()
    chunksize = max(1, len(rows) // (n_workers * 4))
    limit_desc = (
        f"nodes={nodes_deep}/{nodes_shallow if nodes_shallow is not None else 1}"
        if nodes_deep is not None
        else f"depth={depth_deep}/{depth_shallow}"
    )
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(engine_type, depth_deep, depth_shallow, nodes_deep, nodes_shallow),
    ) as pool:
        results = list(tqdm(
            pool.imap(_eval_row, rows, chunksize=chunksize),
            total=len(rows),
            desc=f"{engine_type} {limit_desc} n_workers={n_workers}",
        ))
    elapsed = time.perf_counter() - t0
    n = len(results)
    print(f"  {n:,} positions in {elapsed:.1f}s → {elapsed/n:.4f}s/pos  ({n/elapsed:.1f} pos/s)")

    import pandas as pd
    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    conn = duckdb.connect()
    conn.register("_res", res_df)
    safe = output_path.replace("'", "''")
    conn.execute(f"COPY _res TO '{safe}' (FORMAT PARQUET)")
    conn.close()
    print(f"  Saved → {output_path}")


def run_merge(db_path: str, input_dir: str) -> None:
    """Combine all shard parquets in input_dir → personal.db._OUTPUT_TABLE.

    Adds ply_tertiles (ntile(3) by move_ply) so the Analyzer class can stratify
    plots by game stage without a separate join.
    """
    pattern = os.path.join(os.path.abspath(input_dir), "*.parquet").replace("'", "''")
    conn = duckdb.connect(db_path, read_only=False)
    conn.execute(f"""
        CREATE OR REPLACE TABLE {_OUTPUT_TABLE} AS
        SELECT *,
               ntile(3) OVER (ORDER BY move_ply) AS ply_tertiles
        FROM (
            SELECT fen, gid, move_ply, move_uci,
                   player_clock_time, opponent_clock_time, move_time, n_possible_moves,
                   e_win_best, e_win_second_best, e_win_taken, voc, mq, toptwo
            FROM read_parquet('{pattern}')
        ) raw
    """)
    n = conn.execute(f"SELECT count(*) FROM {_OUTPUT_TABLE}").fetchone()[0]
    conn.close()
    print(f"✅ {_OUTPUT_TABLE}: {n:,} rows → {db_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="Evaluate positions")
    ev.add_argument("--db", default=SELECTED_DB_DEFAULT)
    ev.add_argument("--engine", choices=["stockfish", "lc0"], default="stockfish")
    ev.add_argument("--depth-deep", type=int, default=5)
    ev.add_argument("--depth-shallow", type=int, default=1)
    ev.add_argument("--nodes-deep", type=int, default=None,
                    help="Use MCTS node-budget limits instead of depth for the DEEP search "
                         "(lc0). Defaults to 96 when --engine lc0, matching the tree-gen budget.")
    ev.add_argument("--nodes-shallow", type=int, default=None,
                    help="Node budget for the SHALLOW search. Default 1 = root-only / pure-policy "
                         "a_shallow (lc0's zero-search 'intuition' move).")
    ev.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    # Smoke test
    ev.add_argument("--n-total", type=int, default=None,
                    help="Sample this many positions (smoke test mode)")
    ev.add_argument("--n-workers", type=int, default=10)
    ev.add_argument("--seed", type=int, default=42)
    # SLURM
    ev.add_argument("--shard-id", type=int, default=None)
    ev.add_argument("--total-shards", type=int, default=100)

    mg = sub.add_parser("merge", help="Merge parquets → DB table")
    mg.add_argument("--db", default=SELECTED_DB_DEFAULT)
    mg.add_argument("--input-dir", default=_DEFAULT_OUTPUT_DIR)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    if args.cmd == "eval":
        if args.shard_id is not None and args.n_total is not None:
            raise SystemExit("--shard-id and --n-total are mutually exclusive")

        # lc0 searches are node-budgeted, not depth-budgeted: default to the
        # tree-gen deep budget (96) and a root-only/pure-policy shallow (1)
        # unless the caller overrides. Stockfish keeps its depth limits.
        nodes_deep, nodes_shallow = args.nodes_deep, args.nodes_shallow
        if args.engine == "lc0" and nodes_deep is None:
            nodes_deep, nodes_shallow = 96, 1
        limit_desc = (
            f"nodes={nodes_deep}/{nodes_shallow if nodes_shallow is not None else 1}"
            if nodes_deep is not None
            else f"depth={args.depth_deep}/{args.depth_shallow}"
        )

        if args.shard_id is not None:
            print(f"Shard {args.shard_id}/{args.total_shards}  engine={args.engine}  {limit_desc}")
            rows = _load_shard(args.db, args.shard_id, args.total_shards)
            out = os.path.join(args.output_dir, f"shard_{args.shard_id:04d}.parquet")
            n_workers = 1
        else:
            n = args.n_total or 10_000
            print(f"Smoke test: {n:,} positions  engine={args.engine}  {limit_desc}  workers={args.n_workers}")
            rows = _sample(args.db, n=n, seed=args.seed)
            print(f"  Sampled {len(rows):,} positions")
            out = os.path.join(args.output_dir, "smoke_test.parquet")
            n_workers = args.n_workers

        run_eval(rows, args.engine, args.depth_deep, args.depth_shallow,
                 nodes_deep, nodes_shallow, n_workers, out)

    elif args.cmd == "merge":
        run_merge(args.db, args.input_dir)


if __name__ == "__main__":
    main()
