"""
Preprocess Lichess data for move-time analysis.
Includes selection of games, extraction of moves, and feature engineering.

Contract (do not subvert with optional alternate temp paths):
    DuckDB ``temp_directory`` always equals the directory named by ``work_dir`` or ``staging_dir``.
    Shard parquet outputs go under ``staging_dir``. Merge reads ``staging_dir`` only and replaces
    ``moves``. :func:`process_moves` builds ``processed_moves`` and ``processed_moves_nonzero`` from ``moves``;
    ``preprocess.sh`` runs merge then ``process_moves``.

Each function carries its own default ``threads`` / ``memory_limit`` / ``moves_root`` in its signature;
the CLI builds a local ``config`` dict (paths, ``staging_dir``, ``threads``, ``memory_limit``, ``total_shards``).
``threads`` / ``memory_limit`` follow ``DUCKDB_THREADS`` / ``DUCKDB_MEMORY_LIMIT``; ``total_shards`` follows
``PREPROCESS_TOTAL_SHARDS``. The ``shard`` subcommand reads ``job_id`` from ``SLURM_ARRAY_TASK_ID`` only.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import duckdb
from tqdm import tqdm


def duckdb_connect_config(work_dir: str, threads: int, memory_limit: str) -> dict:
    """DuckDB ``connect`` config: spill/sort temp files live in ``work_dir``."""
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    return {
        "threads": int(threads),
        "memory_limit": str(memory_limit),
        "temp_directory": work_dir,
    }


def _sql_str(s: str) -> str:
    """Single-quoted SQL literal fragment."""
    return s.replace("'", "''")


def get_games(
    personal_db: str,
    lichess_db: str,
    start_date: str,
    end_date: str,
    work_dir: str,
    initial_clock: int = 600,
    clock_increment: int = 0,
    min_elo: int = 2000,
    threads: int = 40,
    memory_limit: str = "64GB",
):
    """
    Build table ``games`` in ``personal_db`` from ``core.games`` using fixed selection filters.

    DuckDB temporary files use ``work_dir`` (same constraint as :func:`preprocess_game_shard`).
    """
    print(f"Fetching games and saving to database:\n\t{lichess_db} --> \n\t{personal_db}")
    conn = duckdb.connect(
        database=personal_db,
        read_only=False,
        config=duckdb_connect_config(work_dir, threads, memory_limit),
    )
    conn.sql(f"""ATTACH '{lichess_db}' AS core (READ_ONLY);""")

    s_start = _sql_str(start_date)
    s_end = _sql_str(end_date)

    start_yyyymm = int(start_date.replace("-", "")[:6])
    end_yyyymm = int(end_date.replace("-", "")[:6]) + 1
    gid_bounds = f"AND gid >= {start_yyyymm}000000000 AND gid < {end_yyyymm}000000000"

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE games AS
        SELECT
            gid,
            utc_datetime,
            substr(cast(gid AS VARCHAR), 1, 6) AS partition,
            substr(cast(gid AS VARCHAR), 7, 3) AS segment,
            initial_clock,
            clock_increment
        FROM core.games
        WHERE utc_datetime >= TIMESTAMP '{s_start}'
          AND utc_datetime < TIMESTAMP '{s_end}'
          {gid_bounds}
          AND initial_clock = {int(initial_clock)}
          AND clock_increment = {int(clock_increment)}
          AND white_elo >= {int(min_elo)}
          AND black_elo >= {int(min_elo)}
        """
    )

    count = conn.execute("SELECT count(*) FROM games").fetchone()[0]
    conn.close()
    print(
        "✅ games rebuilt | "
        f"window=[{start_date}, {end_date}) | tc={initial_clock}+{clock_increment} | "
        f"min_elo={min_elo} | n={count:,}"
    )


def preprocess_game_shard(
    personal_db: str,
    job_id: int,
    total_jobs: int,
    staging_dir: str,
    moves_root: str = "/scratch/gpfs/GRIFFITHS/chess-db/rawdata",
    threads: int = 40,
    memory_limit: str = "64GB",
):
    """
    Shard moves parquet by ``(partition, segment)``, join ``games``, then drop bad games.

    Writes ``selected_moves_<partition>_<segment>.parquet`` under ``staging_dir``.
    DuckDB spill uses ``staging_dir`` (same folder). Pass that same path to :func:`merge_game_shards`.
    """
    if total_jobs < 1 or job_id < 0:
        raise ValueError("need total_jobs >= 1 and job_id >= 0")

    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    conn = duckdb.connect(
        database=personal_db,
        read_only=True,
        config=duckdb_connect_config(staging_dir, threads, memory_limit),
    )

    pairs = conn.execute(
        """
        SELECT DISTINCT partition, segment
        FROM games
        ORDER BY partition, segment
        """
    ).fetchall()

    assigned = [
        (str(p), str(s))
        for p, s in pairs[job_id::total_jobs]
    ]
    print(f"preprocess_game_shard job={job_id}/{total_jobs} | shard_pairs={len(pairs)} | assigned={len(assigned)}")

    for partition, segment in tqdm(assigned, desc=f"moves {job_id}"):
        pq_path = os.path.join(moves_root, f"partition={partition}", f"{segment}-moves.parquet")
        assert os.path.isfile(pq_path), f"missing: {pq_path}"
        out = os.path.join(staging_dir, f"selected_moves_{partition}_{segment}.parquet")
        conn.execute(
            f"""
            COPY (
                WITH joined AS (
                    SELECT
                        pq.*,
                        CAST(g.initial_clock AS DOUBLE) AS initial_clock,
                        CAST(g.clock_increment AS BIGINT) AS clock_increment
                    FROM read_parquet('{_sql_str(pq_path)}') pq
                    INNER JOIN games g ON pq.gid = g.gid
                    WHERE g.partition = '{_sql_str(partition)}' AND g.segment = '{_sql_str(segment)}'
                ),
                -- Drop entire games if any move has negative move_time.
                exclude_neg_gid AS (
                    SELECT DISTINCT gid
                    FROM joined
                    WHERE move_time < 0
                ),
                -- Drop entire games if the player chooses to berserk (half the clock time for points)
                exclude_berserk_gid AS (
                    SELECT DISTINCT gid
                    FROM joined
                    -- First full-move window where halved clock shows after berserk.
                    WHERE move_ply BETWEEN 3 AND 4
                      AND player_clock_time >= (initial_clock / 2.0) - 5.0
                      AND player_clock_time <= (initial_clock / 2.0) + 0.1
                ),
                -- Drop entire games if player chooses to grant more time to the opponent (only valid in increment-0 time controls)
                exclude_grant_more_time_gid AS (
                    SELECT DISTINCT gid
                    FROM (
                        SELECT
                            gid,
                            player_clock_time,
                            clock_increment,
                            lag(player_clock_time) OVER (
                                PARTITION BY gid, player_white ORDER BY move_ply
                            ) AS prev_clock
                        FROM joined
                    )
                    WHERE clock_increment = 0
                      AND prev_clock IS NOT NULL
                      AND player_clock_time > prev_clock
                )
                SELECT j.* EXCLUDE (initial_clock, clock_increment)
                FROM joined j
                WHERE j.gid NOT IN (SELECT gid FROM exclude_neg_gid)
                  AND j.gid NOT IN (SELECT gid FROM exclude_berserk_gid)
                  AND j.gid NOT IN (SELECT gid FROM exclude_grant_more_time_gid)
            ) TO '{_sql_str(out)}' (FORMAT PARQUET)
            """
        )
        nm = conn.execute(f"SELECT count(*) FROM read_parquet('{_sql_str(out)}')").fetchone()[0]
        print(f"\t{partition}/{segment} | {nm:,} moves | → {out}")
    conn.close()


def process_moves(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Build ``processed_moves`` (board/fen/ply_tertiles features) and ``processed_moves_nonzero`` (``move_time > 0``).

    Expects table ``moves`` to exist. Matches the former ``_selected_moves`` /
    ``_selected_moves_nonzero_T`` feature set, with bad games already removed at shard time.
    """
    conn.execute(
        """
        CREATE OR REPLACE TABLE processed_moves AS
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
        ) AS _m
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE processed_moves_nonzero AS
        SELECT * FROM processed_moves WHERE move_time > 0
        """
    )
    n_e = conn.execute("SELECT count(*) FROM processed_moves").fetchone()[0]
    n_z = conn.execute("SELECT count(*) FROM processed_moves_nonzero").fetchone()[0]
    print(f"✅ processed_moves: {n_e:,} rows | processed_moves_nonzero: {n_z:,} rows")


def merge_game_shards(
    personal_db: str,
    staging_dir: str,
    threads: int = 40,
    memory_limit: str = "64GB",
):
    """
    Load ``selected_moves_*.parquet`` from ``staging_dir`` into ``moves``.

    DuckDB spill uses ``staging_dir`` (same folder as the parquet glob).
    Run :func:`process_moves` afterward to build ``processed_moves`` / ``processed_moves_nonzero``.
    """
    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    pattern = os.path.join(staging_dir, "selected_moves_*.parquet")

    ts = datetime.now().astimezone().replace(microsecond=0).isoformat()
    print(f"\n{'=' * 72}\n merge_game_shards started {ts}\n read_parquet → {personal_db}.moves\n{'=' * 72}\n", flush=True)
    print(f"merge_game_shards: read_parquet('{pattern}') → {personal_db}.moves", flush=True)
    conn = duckdb.connect(
        database=personal_db,
        read_only=False,
        config=duckdb_connect_config(staging_dir, threads, memory_limit),
    )
    pat = _sql_str(pattern)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE moves AS
        SELECT * FROM read_parquet('{pat}')
        """
    )
    n = conn.execute("SELECT count(*) FROM moves").fetchone()[0]
    print(f"✅ merge_game_shards finished — table moves replaced ({n:,} rows).")
    conn.close()


def run_process_moves(
    personal_db: str,
    work_dir: str,
    threads: int = 40,
    memory_limit: str = "64GB",
) -> None:
    """Open ``personal_db``, run :func:`process_moves` (expects table ``moves``). Spill in ``work_dir``."""
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    ts = datetime.now().astimezone().replace(microsecond=0).isoformat()
    print(f"\n{'=' * 72}\n process_moves started {ts}\n {personal_db} (spill: {work_dir})\n{'=' * 72}\n", flush=True)
    conn = duckdb.connect(
        database=personal_db,
        read_only=False,
        config=duckdb_connect_config(work_dir, threads, memory_limit),
    )
    process_moves(conn)
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("get_games", help="Build games table from Lichess core DB.")
    sub.add_parser("shard", help="One array stride; uses config staging_dir, SLURM_ARRAY_TASK_ID, PREPROCESS_TOTAL_SHARDS.")
    sub.add_parser("merge", help="Parquet shards → table moves only.")
    sub.add_parser(
        "process_moves",
        help="From table moves → processed_moves and processed_moves_nonzero (feature SQL).",
    )

    args = parser.parse_args()
    # end_date is exclusive
    config = {
        "work_dir": "/scratch/gpfs/GRIFFITHS/hl4291/tmp/",
        "staging_dir": "/scratch/gpfs/GRIFFITHS/hl4291/tmp/ld_moves_shard",
        "personal_db": "/scratch/gpfs/GRIFFITHS/hl4291/personal.db",
        "lichess_db": "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db",
        "moves_root": "/scratch/gpfs/GRIFFITHS/chess-db/rawdata",
        "start_date": "2023-10-01",
        "end_date": "2023-12-31",
        "initial_clock": 600,
        "clock_increment": 0,
        "min_elo": 2000,
        "threads": int(os.environ.get("DUCKDB_THREADS", "40")),
        "memory_limit": os.environ.get("DUCKDB_MEMORY_LIMIT", "64GB"),
        "total_shards": int(os.environ.get("PREPROCESS_TOTAL_SHARDS", "11")),
    }

    if args.cmd == "get_games":
        get_games(
            personal_db=config["personal_db"],
            lichess_db=config["lichess_db"],
            work_dir=config["work_dir"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            initial_clock=config["initial_clock"],
            clock_increment=config["clock_increment"],
            min_elo=config["min_elo"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )
    elif args.cmd == "shard":
        job_raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if job_raw is None:
            raise SystemExit("shard requires SLURM_ARRAY_TASK_ID (submit as a Slurm array task).")
        preprocess_game_shard(
            personal_db=config["personal_db"],
            job_id=int(job_raw),
            total_jobs=config["total_shards"],
            staging_dir=config["staging_dir"],
            moves_root=config["moves_root"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )
    elif args.cmd == "merge":
        merge_game_shards(
            personal_db=config["personal_db"],
            staging_dir=config["staging_dir"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )
    elif args.cmd == "process_moves":
        run_process_moves(
            personal_db=config["personal_db"],
            work_dir=config["work_dir"],
            threads=config["threads"],
            memory_limit=config["memory_limit"],
        )


if __name__ == "__main__":
    main()