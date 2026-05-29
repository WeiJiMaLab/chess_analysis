#!/usr/bin/env python3
"""
Smoke-test legacy vs new human-moves pipeline on the same date window.

Writes isolated DuckDB files + parquets under a scratch directory (default
``/scratch/gpfs/GRIFFITHS/hl4291/tmp/pipeline_smoke``), then reports:

- game / move / distinct-gid counts at each stage
- ``gid`` in legacy ``_selected_moves_nonzero_T`` but not ``processed_moves_nonzero`` (new), and vice versa

Legacy path mirrors deleted ``preprocess_data.py`` (406609e^): shard only drops
negative ``move_time`` games; berserk + grant are detected on merged ``moves``;
``preprocess()`` builds ``_selected_moves`` / ``_selected_moves_nonzero_T``.

New path: ``get_games`` → ``preprocess_game_shard`` (neg + berserk + grant in shard
SQL) → ``merge_game_shards`` → ``run_process_moves``.

Run from repo root::

    PYTHONPATH=analysis python3 analysis/slurm/scripts/tests/compare_legacy_new_pipeline_smoke.py \\
        --start-date 2023-10-01 --end-date 2023-10-05

Use a **narrow** ``[start, end)`` window so the test finishes quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

# ``analysis/`` for ``utils`` if needed; ``scripts/`` for ``preprocess``
_scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src = os.path.abspath(os.path.join(_scripts, "..", ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from preprocess import (  # noqa: E402
    duckdb_connect_config,
    get_games,
    merge_game_shards,
    preprocess_game_shard,
    run_process_moves,
)


def _sql_str(s: str) -> str:
    return s.replace("'", "''")


def _gid_bounds_sql(start_date: str, end_date: str) -> str:
    start_yyyymm = int(start_date.replace("-", "")[:6])
    end_yyyymm = int(end_date.replace("-", "")[:6]) + 1
    return f"AND gid >= {start_yyyymm}000000000 AND gid < {end_yyyymm}000000000"


def legacy_build_selected_games(
    conn: duckdb.DuckDBPyConnection,
    lichess_db: str,
    start_date: str,
    end_date: str,
    initial_clock: int,
    clock_increment: int,
    min_elo: int,
) -> int:
    conn.execute(f"ATTACH '{_sql_str(lichess_db)}' AS core (READ_ONLY)")
    s_start, s_end = _sql_str(start_date), _sql_str(end_date)
    gb = _gid_bounds_sql(start_date, end_date)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE selected_games AS
        SELECT gid, utc_datetime
        FROM core.games
        WHERE utc_datetime >= TIMESTAMP '{s_start}'
          AND utc_datetime < TIMESTAMP '{s_end}'
          {gb}
          AND initial_clock = {int(initial_clock)}
          AND clock_increment = {int(clock_increment)}
          AND white_elo >= {int(min_elo)}
          AND black_elo >= {int(min_elo)}
        """
    )
    return int(conn.execute("SELECT count(*) FROM selected_games").fetchone()[0])


def legacy_shard_all_partitions(
    personal_db: str,
    staging_dir: str,
    lichess_db: str,
    moves_root: str,
    threads: int,
    memory_limit: str,
) -> None:
    """Same as legacy ``process_shard`` with ``job_id=0``, ``total_shards=1`` (all keys)."""
    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    conn = duckdb.connect(
        database=personal_db,
        read_only=True,
        config=duckdb_connect_config(staging_dir, threads, memory_limit),
    )
    conn.execute(f"ATTACH '{_sql_str(lichess_db)}' AS core (READ_ONLY)")
    gids = conn.execute("SELECT gid, utc_datetime FROM selected_games").df()
    conn.close()

    gids["partition_tuple"] = gids["gid"].apply(lambda x: (str(x)[:6], str(x)[6:9]))
    partition_tuples = sorted(gids["partition_tuple"].unique().tolist())

    conn = duckdb.connect(
        database=personal_db,
        read_only=True,
        config=duckdb_connect_config(staging_dir, threads, memory_limit),
    )
    for partition_tuple in tqdm(partition_tuples, desc="legacy shards"):
        partition, segment = partition_tuple
        gids_in_partition = gids[gids["partition_tuple"] == partition_tuple]
        gid_list = gids_in_partition["gid"].tolist()
        if not gid_list:
            continue
        parquet_read_path = os.path.join(moves_root, f"partition={partition}", f"{segment}-moves.parquet")
        if not os.path.isfile(parquet_read_path):
            continue
        out_path = os.path.join(staging_dir, f"selected_moves_{partition}_{segment}.parquet")
        pq, op = _sql_str(parquet_read_path), _sql_str(out_path)
        gid_sql = ",".join(str(int(g)) for g in gid_list)
        conn.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{pq}')
                WHERE gid IN ({gid_sql})
                QUALIFY COUNT(*) FILTER (WHERE move_time < 0) OVER (PARTITION BY gid) = 0
            ) TO '{op}' (FORMAT PARQUET)
            """
        )
    conn.close()


def legacy_identify_berserk(conn: duckdb.DuckDBPyConnection, lichess_db: str) -> int:
    conn.execute(f"ATTACH '{_sql_str(lichess_db)}' AS core (READ_ONLY)")
    conn.execute(
        """
        CREATE OR REPLACE TABLE berserk_games AS
        SELECT DISTINCT m.gid
        FROM moves m
        JOIN core.games g ON m.gid = g.gid
        WHERE m.move_ply IN (3, 4)
          AND m.player_clock_time >= (CAST(g.initial_clock AS DOUBLE) / 2) - 5.0
          AND m.player_clock_time <= (CAST(g.initial_clock AS DOUBLE) / 2) + 0.1
        """
    )
    return int(conn.execute("SELECT count(*) FROM berserk_games").fetchone()[0])


def legacy_identify_grant(conn: duckdb.DuckDBPyConnection) -> int:
    # ``core`` must already be attached (``legacy_identify_berserk``).
    conn.execute(
        """
        CREATE OR REPLACE TABLE grant_more_time_games AS
        SELECT DISTINCT gid
        FROM (
            SELECT m.gid, player_clock_time,
                   lag(player_clock_time) OVER (
                       PARTITION BY m.gid, player_white ORDER BY move_ply
                   ) AS prev_clock
            FROM moves m
            JOIN core.games g ON m.gid = g.gid
            WHERE g.clock_increment = 0
        )
        WHERE prev_clock IS NOT NULL
          AND player_clock_time > prev_clock
        """
    )
    return int(conn.execute("SELECT count(*) FROM grant_more_time_games").fetchone()[0])


def legacy_preprocess_feature_tables(conn: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Build ``_selected_moves`` and ``_selected_moves_nonzero_T`` (legacy names)."""
    conn.execute(
        """
        CREATE OR REPLACE TABLE _selected_moves AS
        SELECT
            gid,
            move_ply,
            board_position,
            player_white,
            player_clock_time,
            opponent_clock_time,
            n_possible_moves,
            move_time,
            len(regexp_extract_all(replace(board_position, '/', ''), '[a-zA-Z]')) AS n_pieces_on_board_inc_pawns,
            len(regexp_extract_all(replace(board_position, '/', ''), '[rnbqkRNBQK]')) AS n_pieces_on_board_exc_pawns,
            CASE WHEN player_white THEN len(regexp_extract_all(replace(board_position, '/', ''), '[RNBQK]'))
                 ELSE len(regexp_extract_all(replace(board_position, '/', ''), '[rnbqk]'))
            END AS n_self_pieces_exc_pawns,
            CASE WHEN player_white THEN len(regexp_extract_all(replace(board_position, '/', ''), '[rnbqk]'))
                 ELSE len(regexp_extract_all(replace(board_position, '/', ''), '[RNBQK]'))
            END AS n_opp_pieces_exc_pawns,
            board_position || ' ' ||
            CASE WHEN player_white THEN 'w' ELSE 'b' END || ' ' ||
            COALESCE(castling_rights, '-') || ' ' ||
            COALESCE(en_passant_targets, '-') AS fen,
            ntile(3) OVER (ORDER BY move_ply) AS ply_tertiles
        FROM (
            SELECT
                m.gid, m.move_ply, m.board_position, m.player_white,
                m.player_clock_time, m.opponent_clock_time, m.n_possible_moves,
                m.move_time, m.castling_rights, m.en_passant_targets, m.halfmove_clock
            FROM moves m
            WHERE m.gid NOT IN (SELECT gid FROM berserk_games)
              AND m.gid NOT IN (SELECT gid FROM grant_more_time_games)
        );
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE _selected_moves_nonzero_T AS
        SELECT * FROM _selected_moves
        WHERE move_time > 0
        """
    )
    n_all = int(conn.execute("SELECT count(*) FROM _selected_moves").fetchone()[0])
    n_nz = int(conn.execute("SELECT count(*) FROM _selected_moves_nonzero_T").fetchone()[0])
    return n_all, n_nz


def _print_counts(label: str, conn: duckdb.DuckDBPyConnection) -> None:
    parts = [label]
    for t in ("selected_games", "games", "moves", "berserk_games", "grant_more_time_games"):
        try:
            n = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            parts.append(f"{t}={n:,}")
        except Exception:
            pass
    for t in ("_selected_moves", "_selected_moves_nonzero_T", "processed_moves", "processed_moves_nonzero"):
        try:
            ng = conn.execute(f"SELECT count(distinct gid) FROM {t}").fetchone()[0]
            nr = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            parts.append(f"{t}:gids={ng:,}rows={nr:,}")
        except Exception:
            pass
    print(" | ".join(parts))


def run_smoke(args: argparse.Namespace) -> int:
    base = os.path.abspath(args.scratch_base)
    legacy_dir = os.path.join(base, "legacy")
    new_dir = os.path.join(base, "new")
    legacy_db = os.path.join(legacy_dir, "personal.duckdb")
    new_db = os.path.join(new_dir, "personal.duckdb")
    legacy_staging = os.path.join(legacy_dir, "staging")
    new_staging = os.path.join(new_dir, "staging")
    legacy_work = os.path.join(legacy_dir, "work")
    new_work = os.path.join(new_dir, "work")

    if args.clean and os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(legacy_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)
    os.makedirs(legacy_staging, exist_ok=True)
    os.makedirs(new_staging, exist_ok=True)
    os.makedirs(legacy_work, exist_ok=True)
    os.makedirs(new_work, exist_ok=True)

    print(f"Scratch: {base}")
    print(f"Window: [{args.start_date}, {args.end_date})")

    # --- Legacy ---
    print("\n=== LEGACY pipeline ===")
    if os.path.isfile(legacy_db):
        os.remove(legacy_db)
    conn = duckdb.connect(legacy_db, config=duckdb_connect_config(legacy_work, args.threads, args.memory_limit))
    n_sel = legacy_build_selected_games(
        conn,
        args.lichess_db,
        args.start_date,
        args.end_date,
        args.initial_clock,
        args.clock_increment,
        args.min_elo,
    )
    conn.close()
    print(f"legacy selected_games: {n_sel:,}")

    legacy_shard_all_partitions(
        legacy_db,
        legacy_staging,
        args.lichess_db,
        args.moves_root,
        args.threads,
        args.memory_limit,
    )
    merge_game_shards(legacy_db, legacy_staging, threads=args.threads, memory_limit=args.memory_limit)

    conn = duckdb.connect(legacy_db, config=duckdb_connect_config(legacy_work, args.threads, args.memory_limit))
    n_ber = legacy_identify_berserk(conn, args.lichess_db)
    n_gr = legacy_identify_grant(conn)
    print(f"legacy berserk_games: {n_ber:,} | grant_more_time_games: {n_gr:,}")
    n_all, n_nz = legacy_preprocess_feature_tables(conn)
    print(f"legacy _selected_moves rows: {n_all:,} | _selected_moves_nonzero_T rows: {n_nz:,}")
    legacy_nz_games = int(
        conn.execute("SELECT count(distinct gid) FROM _selected_moves_nonzero_T").fetchone()[0]
    )
    _print_counts("legacy(final)", conn)
    conn.close()

    # --- New ---
    print("\n=== NEW pipeline ===")
    if os.path.isfile(new_db):
        os.remove(new_db)
    get_games(
        personal_db=new_db,
        lichess_db=args.lichess_db,
        start_date=args.start_date,
        end_date=args.end_date,
        work_dir=new_work,
        initial_clock=args.initial_clock,
        clock_increment=args.clock_increment,
        min_elo=args.min_elo,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    preprocess_game_shard(
        personal_db=new_db,
        job_id=0,
        total_jobs=1,
        staging_dir=new_staging,
        moves_root=args.moves_root,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    merge_game_shards(new_db, new_staging, threads=args.threads, memory_limit=args.memory_limit)
    run_process_moves(new_db, new_work, threads=args.threads, memory_limit=args.memory_limit)

    conn = duckdb.connect(new_db, read_only=True)
    new_nz_games = int(conn.execute("SELECT count(distinct gid) FROM processed_moves_nonzero").fetchone()[0])
    _print_counts("new(final)", conn)
    conn.close()

    print("\n=== SUMMARY ===")
    conn = duckdb.connect(legacy_db, read_only=True)
    legacy_move_rows = int(conn.execute("SELECT count(*) FROM moves").fetchone()[0])
    conn.close()
    conn = duckdb.connect(new_db, read_only=True)
    new_move_rows = int(conn.execute("SELECT count(*) FROM moves").fetchone()[0])
    conn.close()
    print(
        "Intermediate ``moves`` row counts differ when legacy keeps berserk/grant games "
        "(removed only in ``_selected_moves``) while new drops them in the shard ``COPY``."
    )
    print(f"  legacy moves rows: {legacy_move_rows:,} | new moves rows: {new_move_rows:,}")
    print(f"distinct gid with move_time>0 — legacy: {legacy_nz_games:,} | new: {new_nz_games:,} | delta: {new_nz_games - legacy_nz_games:+,}")

    conn = duckdb.connect(":memory:")
    conn.execute(f"ATTACH '{_sql_str(legacy_db)}' AS leg (READ_ONLY)")
    conn.execute(f"ATTACH '{_sql_str(new_db)}' AS nw (READ_ONLY)")

    only_new = conn.execute(
        """
        SELECT gid FROM nw.processed_moves_nonzero
        EXCEPT
        SELECT gid FROM leg._selected_moves_nonzero_T
        """
    ).fetchall()
    only_leg = conn.execute(
        """
        SELECT gid FROM leg._selected_moves_nonzero_T
        EXCEPT
        SELECT gid FROM nw.processed_moves_nonzero
        """
    ).fetchall()
    conn.close()

    print(f"\nGids in NEW nonzero only (not in legacy): {len(only_new):,}")
    for row in only_new[:50]:
        print(f"  {row[0]}")
    if len(only_new) > 50:
        print("  ...")

    print(f"\nGids in LEGACY nonzero only (not in new): {len(only_leg):,}")
    for row in only_leg[:50]:
        print(f"  {row[0]}")
    if len(only_leg) > 50:
        print("  ...")

    # Optional: classify why ``only_*`` (needs re-open legacy/new for moves rows)
    if only_new or only_leg:
        _explain_gids(args, legacy_db, new_db, [r[0] for r in only_new[:20]], [r[0] for r in only_leg[:20]])

    summary = {
        "scratch_base": base,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "legacy_nz_games": legacy_nz_games,
        "new_nz_games": new_nz_games,
        "delta_nz_games": new_nz_games - legacy_nz_games,
        "only_new_count": len(only_new),
        "only_leg_count": len(only_leg),
        "only_new_sample": [int(r[0]) for r in only_new[:50]],
        "only_leg_sample": [int(r[0]) for r in only_leg[:50]],
    }
    path = getattr(args, "summary_json", None)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0 if not only_new and not only_leg else 1


def _explain_gids(args, legacy_db: str, new_db: str, gids_new: list, gids_leg: list) -> None:
    """Lightweight reasons: presence in moves / berserk / grant on legacy DB."""
    if not gids_new and not gids_leg:
        return
    conn = duckdb.connect(legacy_db, read_only=True)
    conn.execute(f"ATTACH '{_sql_str(args.lichess_db)}' AS core (READ_ONLY)")
    for label, gids in ("new-only", gids_new), ("legacy-only", gids_leg):
        if not gids:
            continue
        print(f"\n--- {label} sample gid diagnostics (legacy DB for berserk/grant tables) ---")
        gs = ",".join(str(int(g)) for g in gids)
        for gid in gids:
            in_b = conn.execute(f"SELECT count(*) FROM berserk_games WHERE gid = {gid}").fetchone()[0]
            in_g = conn.execute(f"SELECT count(*) FROM grant_more_time_games WHERE gid = {gid}").fetchone()[0]
            in_m = conn.execute(
                f"SELECT count(*), sum(CASE WHEN move_time > 0 THEN 1 ELSE 0 END) FROM moves WHERE gid = {gid}"
            ).fetchone()
            print(
                f"  gid={gid} legacy_moves_rows={in_m[0]} legacy_moves_move_time_gt0={in_m[1]} "
                f"in_berserk={in_b} in_grant={in_g}"
            )
    conn.close()

    conn = duckdb.connect(new_db, read_only=True)
    for gid in gids_new[:10]:
        in_m = conn.execute(
            f"SELECT count(*), sum(CASE WHEN move_time > 0 THEN 1 ELSE 0 END) FROM moves WHERE gid = {gid}"
        ).fetchone()
        print(f"  gid={gid} new_moves_rows={in_m[0]} new_moves_move_time_gt0={in_m[1]}")
    conn.close()


def _run_boundary_suite(cli: argparse.Namespace) -> int:
    """Multiple legacy-vs-new compares: year-end, production end_date, October start bleed."""
    here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    out_combined = os.path.join(here, "boundary_suite_results.json")
    parent = os.path.abspath(cli.scratch_base)
    if not parent.endswith("boundary_suite_runs"):
        parent = os.path.join(parent, "boundary_suite_runs")

    # Keep each window tractable on a login node (avoid full late-Dec → Jan mega-merge unless needed).
    windows = [
        ("oct01_oct05", "2023-10-01", "2023-10-05"),
        ("sep28_oct05", "2023-09-28", "2023-10-05"),
        ("dec28_dec31", "2023-12-28", "2023-12-31"),
        ("dec27_jan01", "2023-12-27", "2024-01-01"),
    ]
    combined: list[dict] = []
    worst = 0
    for label, sd, ed in windows:
        sub = argparse.Namespace(
            scratch_base=os.path.join(parent, label),
            start_date=sd,
            end_date=ed,
            lichess_db=cli.lichess_db,
            moves_root=cli.moves_root,
            initial_clock=cli.initial_clock,
            clock_increment=cli.clock_increment,
            min_elo=cli.min_elo,
            threads=cli.threads,
            memory_limit=cli.memory_limit,
            clean=True,
            summary_json=os.path.join(here, f"boundary_suite_{label}.json"),
        )
        code = run_smoke(sub)
        worst = max(worst, code)
        with open(sub.summary_json, encoding="utf-8") as f:
            combined.append(json.load(f))
    with open(out_combined, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote combined results: {out_combined}")
    return worst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scratch-base",
        default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/pipeline_smoke",
        help="Root directory for legacy/ and new/ trees",
    )
    p.add_argument("--start-date", default="2023-10-01")
    p.add_argument("--end-date", default="2023-10-05", help="Exclusive upper bound (same semantics as preprocess)")
    p.add_argument("--lichess-db", default="/scratch/gpfs/GRIFFITHS/chess-db/lichess.db")
    p.add_argument("--moves-root", default="/scratch/gpfs/GRIFFITHS/chess-db/rawdata")
    p.add_argument("--initial-clock", type=int, default=600)
    p.add_argument("--clock-increment", type=int, default=0)
    p.add_argument("--min-elo", type=int, default=2000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-limit", default="16GB")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove scratch-base before run",
    )
    p.add_argument(
        "--summary-json",
        default=None,
        metavar="PATH",
        help="Write a machine-readable summary (delta, gid samples) to PATH after the run.",
    )
    p.add_argument(
        "--boundary-suite",
        action="store_true",
        help=(
            "Run several windows near calendar / config edges; write combined JSON under "
            "``analysis/slurm/scripts/tests/boundary_suite_results.json`` (and per-window JSON next to it). "
            "Uses ``--scratch-base`` as a parent directory if set, else the default scratch root."
        ),
    )
    args = p.parse_args()
    if args.boundary_suite:
        raise SystemExit(_run_boundary_suite(args))
    raise SystemExit(run_smoke(args))


if __name__ == "__main__":
    main()
