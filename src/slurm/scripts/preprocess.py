"""
Preprocess Lichess data for move-time analysis.
Includes selection of games, extraction of moves, and feature engineering.

Contract (do not subvert with optional alternate temp paths):
    DuckDB ``temp_directory`` always equals the directory named by ``work_dir`` or ``staging_dir``.
    Shard parquet outputs go under ``staging_dir``. Merge reads ``staging_dir`` only.

Defaults are module constants below; override ``threads`` / ``memory_limit`` only via explicit arguments.
"""

from __future__ import annotations

import os

import duckdb
from tqdm import tqdm

from _bootstrap import ensure_src

ensure_src()

DEFAULT_MOVES_ROOT = "/scratch/gpfs/GRIFFITHS/chess-db/rawdata"
DEFAULT_THREADS = 40
DEFAULT_MEMORY_LIMIT = "64GB"


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
    threads: int = DEFAULT_THREADS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
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
    moves_root: str = DEFAULT_MOVES_ROOT,
    threads: int = DEFAULT_THREADS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
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
                    -- First full-move window where halved clock shows after berserk (see preprocess_data.identify_berserk).
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


def merge_game_shards(
    personal_db: str,
    staging_dir: str,
    threads: int = DEFAULT_THREADS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
):
    """
    Load ``selected_moves_*.parquet`` from ``staging_dir`` into table ``moves``.

    DuckDB spill uses ``staging_dir`` (same folder as the parquet glob).
    """
    staging_dir = os.path.abspath(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)
    pattern = os.path.join(staging_dir, "selected_moves_*.parquet")

    print(f"merge_game_shards: read_parquet('{pattern}') → {personal_db}.moves")
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
    conn.close()
    print(f"✅ merge_game_shards finished — table moves replaced ({n:,} rows).")


if __name__ == "__main__":
    args = argparse.ArgumentParser().parse_args()
    tmpdir = "/scratch/gpfs/GRIFFITHS/hl4291/tmp/"
    personal_db = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
    lichess_db = "/scratch/gpfs/GRIFFITHS/chess-db/lichess.db"

    if args.runtype == "get_games":
        get_games(
            personal_db=personal_db,
            lichess_db=lichess_db,
            work_dir=tmpdir,
            start_date="2023-10-01",
            end_date="2024-01-01",  # exclusive upper bound
            initial_clock=600,
            clock_increment=0,
            min_elo=2000,
            threads=DEFAULT_THREADS,
            memory_limit=DEFAULT_MEMORY_LIMIT,
        )
    elif args.runtype == "shard":
        preprocess_game_shard(
            personal_db=personal_db,
            job_id=args.job_id,
            total_jobs=args.total_shards,
            staging_dir=args.staging_dir,
            threads=DEFAULT_THREADS,
            memory_limit=DEFAULT_MEMORY_LIMIT,
        )
    elif args.runtype == "merge":
        merge_game_shards(
            personal_db=personal_db,
            staging_dir=args.staging_dir,
            threads=DEFAULT_THREADS,
            memory_limit=DEFAULT_MEMORY_LIMIT,
        )
    else:
        raise ValueError(f"unknown runtype: {args.runtype}")