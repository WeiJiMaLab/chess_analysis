"""
Preprocess Lichess data for move-time analysis.
Includes selection of games, extraction of moves, and feature engineering.
"""

import argparse
import duckdb
import pandas as pd
from tqdm import tqdm
import os
import time

# Must match Slurm: #SBATCH --array=0-(TOTAL_SHARDS-1)
TOTAL_SHARDS = 2
DEFAULT_TMPDIR = "/scratch/gpfs/GRIFFITHS/hl4291/tmp/"
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
MOVES_ROOT = "/scratch/gpfs/GRIFFITHS/chess-db/rawdata"
DEFAULT_START_DATE = "2023-10-01"
DEFAULT_END_DATE = "2023-12-31"  # Exclusive upper bound
EPSILON = 1e-6

# Global settings updated by CLI flags
THREADS = 40
MEMORY_LIMIT = "64GB"


def _conn_kw(tmpdir: str) -> dict:
    return {"threads": THREADS, "memory_limit": MEMORY_LIMIT, "temp_directory": tmpdir}


def _sql_str(s: str) -> str:
    """Single-quoted SQL literal fragment."""
    return s.replace("'", "''")


def preprocess_selected_games(
    start_date=DEFAULT_START_DATE,
    end_date=DEFAULT_END_DATE,
    initial_clock=600,
    clock_increment=0,
    min_elo=2000,
    tmpdir=DEFAULT_TMPDIR,
):
    """
    Build selected_games from core.games using fixed selection filters.
    """
    tmpdir = os.path.abspath(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)

    conn = duckdb.connect(database=PERSONAL_DB, read_only=False, config=_conn_kw(tmpdir))
    conn.sql("""ATTACH '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db' AS core (READ_ONLY);""")

    s_start = _sql_str(start_date)
    s_end = _sql_str(end_date)
    
    # Optional optimization: use gid bounds if gid is sorted (likely)
    # GID is YYYYMMSSSIIIIII (6 digits year-month, 3 segment, 6 index)
    try:
        start_yyyymm = int(start_date.replace("-", "")[:6])
        end_yyyymm = int(end_date.replace("-", "")[:6]) + 1
        gid_bounds = f"AND gid >= {start_yyyymm}000000000 AND gid < {end_yyyymm}000000000"
    except Exception:
        gid_bounds = ""

    candidate_count = conn.execute(
        f"""
        SELECT count(*)
        FROM core.games
        WHERE utc_datetime >= TIMESTAMP '{s_start}'
          AND utc_datetime < TIMESTAMP '{s_end}'
          {gid_bounds}
          AND initial_clock = {int(initial_clock)}
          AND clock_increment = {int(clock_increment)}
          AND white_elo >= {int(min_elo)}
          AND black_elo >= {int(min_elo)}
        """
    ).fetchone()[0]

    existing_count = None
    if conn.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'selected_games'").fetchone()[0]:
        existing_count = conn.execute("SELECT count(*) FROM selected_games").fetchone()[0]

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE selected_games AS
        SELECT gid, utc_datetime
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

    new_count = conn.execute("SELECT count(*) FROM selected_games").fetchone()[0]
    min_utc, max_utc = conn.execute("SELECT min(utc_datetime), max(utc_datetime) FROM selected_games").fetchone()
    conn.close()

    print(
        "✅ selected_games rebuilt | "
        f"window=[{start_date}, {end_date}) | tc={initial_clock}+{clock_increment} | "
        f"min_elo={min_elo} | n={new_count:,}"
    )
    print(f"   core.games candidate count: {candidate_count:,}")
    if existing_count is not None:
        print(f"   previous selected_games count: {existing_count:,}")
    print(f"   selected_games min/max utc_datetime: {min_utc} / {max_utc}")


def process_shard(_JOBID, total_shards=TOTAL_SHARDS, tmpdir=DEFAULT_TMPDIR, exclude_negative=True):
    tmpdir = os.path.abspath(tmpdir)
    # Local working database (opened as read-only).
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True, config=_conn_kw(tmpdir))
    conn.sql("""ATTACH '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db' AS core (READ_ONLY);""")

    # assume selected_games is a table in the personal database
    gids = conn.sql("SELECT gid, utc_datetime FROM selected_games;").df()
    gids["partition_tuple"] = gids["gid"].apply(lambda x: (str(x)[:6], str(x)[6:9]))

    sorted_shard_keys = sorted(gids["partition_tuple"].unique().tolist())
    partition_tuples = sorted_shard_keys[_JOBID::total_shards]
    print(f"🧩 Shard keys (job {_JOBID}):", partition_tuples)

    if len(partition_tuples) == 0:
        print("⏭️  No shard keys for this job id — nothing to do.")
        exit()

    os.makedirs(tmpdir, exist_ok=True)

    for partition_tuple in tqdm(partition_tuples, desc=f"♟️ shards job={_JOBID}"):
        partition, segment = partition_tuple
        gids_in_partition = gids[gids["partition_tuple"] == partition_tuple]
        print("♟️  Processing:", _JOBID, partition_tuple, "| games:", len(gids_in_partition))

        time_start = time.time()
        gid_list = gids_in_partition["gid"].tolist()
        if not gid_list:
            print("\t⚠️  skip: no gids for this shard")
            continue
        parquet_read_path = f"{MOVES_ROOT}/partition={partition}/{segment}-moves.parquet"
        out_path = os.path.join(tmpdir, f"selected_moves_{partition}_{segment}.parquet")
        pq, op = _sql_str(parquet_read_path), _sql_str(out_path)
        gid_sql = ",".join(str(int(g)) for g in gid_list)
        qualify_clause = (
            "QUALIFY COUNT(*) FILTER (WHERE move_time < 0) OVER (PARTITION BY gid) = 0"
            if exclude_negative
            else ""
        )
        conn.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{pq}')
                WHERE gid IN ({gid_sql})
                {qualify_clause}
            ) TO '{op}' (FORMAT PARQUET)
            """
        )

        n_moves = conn.execute(f"SELECT count(*) FROM read_parquet('{op}')").fetchone()[0]
        print(f"\t📊 Moves written: {n_moves}")
        print(f"\t💾 Saved: {out_path}")
        print(f"\t⏱️  Elapsed: {time.time() - time_start:.2f}s")

    print(f"✅ Process shard job {_JOBID} finished.")
    conn.close()


def merge_shards(tmpdir=DEFAULT_TMPDIR):
    """Load staging parquet shards into personal.db (single writer)."""
    tmpdir = os.path.abspath(tmpdir)
    pattern = os.path.join(tmpdir, "*.parquet")
    print(f"🔀 Merge: reading {pattern!r} into {PERSONAL_DB}")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False, config=_conn_kw(tmpdir))
    pat = _sql_str(pattern)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE selected_moves AS
        SELECT * FROM read_parquet('{pat}')
        """
    )
    conn.close()
    print("✅ Merge finished — table selected_moves replaced.")


def identify_berserk(tmpdir=DEFAULT_TMPDIR):
    """
    Identify games where at least one player berserked.
    Berserking is detected if player_clock_time = initial_clock / 2 at ply 3 or 4.
    """
    tmpdir = os.path.abspath(tmpdir)
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False, config=_conn_kw(tmpdir))
    conn.sql("""ATTACH '/scratch/gpfs/GRIFFITHS/chess-db/lichess.db' AS core (READ_ONLY);""")

    print(f"🔍 Identifying berserk games in {PERSONAL_DB}...")
    conn.execute(
        """
        CREATE OR REPLACE TABLE berserk_games AS
        SELECT DISTINCT m.gid
        FROM selected_moves m
        JOIN core.games g ON m.gid = g.gid
        WHERE m.move_ply IN (3, 4)
          AND m.player_clock_time = CAST(g.initial_clock AS DOUBLE) / 2
        """
    )

    count = conn.execute("SELECT count(*) FROM berserk_games").fetchone()[0]
    conn.close()
    print(f"✅ Identified {count:,} berserk games. Stored in 'berserk_games' table.")


def identify_grant_more_time(tmpdir=DEFAULT_TMPDIR):
    """
    Identify games where at least one player was granted more time.
    Detected if player_clock_time at ply n > player_clock_time at ply n-2.
    """
    tmpdir = os.path.abspath(tmpdir)
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False, config=_conn_kw(tmpdir))

    print(f"🔍 Identifying 'grant more time' games in {PERSONAL_DB}...")
    conn.execute(
        """
        CREATE OR REPLACE TABLE grant_more_time_games AS
        SELECT DISTINCT m1.gid
        FROM selected_moves m1
        JOIN selected_moves m2 ON m1.gid = m2.gid 
          AND m1.player_white = m2.player_white 
          AND m1.move_ply = m2.move_ply + 2
        WHERE m1.player_clock_time > m2.player_clock_time
        """
    )

    count = conn.execute("SELECT count(*) FROM grant_more_time_games").fetchone()[0]
    conn.close()
    print(f"✅ Identified {count:,} 'grant more time' games. Stored in 'grant_more_time_games' table.")


def preprocess(tmpdir=DEFAULT_TMPDIR, target_table="_selected_moves", limit_clause=""):
    """
    Standard SQL-native preprocessing for chess timing analysis.
    Creates a table with log-transformed variables and quantile bins.
    Excludes games identified as berserk or grant-more-time.
    """
    tmpdir = os.path.abspath(tmpdir)
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False, config=_conn_kw(tmpdir))

    print(f"🛠️  Preprocessing moves into {target_table} (excluding berserk and grant-more-time games)...")
    conn.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT 
            gid,
            move_ply,
            board_position,
            player_white,
            player_clock_time,
            opponent_clock_time,
            n_possible_moves,
            move_time,
            -- Construct FEN (minimal engine-ready state: board, turn, castling, EP)
            board_position || ' ' || 
            CASE WHEN player_white THEN 'w' ELSE 'b' END || ' ' || 
            COALESCE(castling_rights, '-') || ' ' || 
            COALESCE(en_passant_targets, '-') AS fen,
            ntile(3) over (order by move_ply) as game_phase
        FROM (
            SELECT 
                m.gid, m.move_ply, m.board_position, m.player_white, 
                m.player_clock_time, m.opponent_clock_time, m.n_possible_moves, 
                m.move_time, m.castling_rights, m.en_passant_targets, m.halfmove_clock
            FROM selected_moves m
            WHERE m.gid NOT IN (SELECT gid FROM berserk_games)
              AND m.gid NOT IN (SELECT gid FROM grant_more_time_games)
            {limit_clause}
        );
    """)

    # Create companion table with zero-time moves (premoves) filtered out
    conn.execute(f"""
        CREATE OR REPLACE TABLE {target_table}_nonzero_T AS 
        SELECT * FROM {target_table} 
        WHERE move_time > 0;
    """)

    count = conn.execute(f"SELECT count(*) FROM {target_table}").fetchone()[0]
    conn.close()
    print(f"✅ Preprocessing finished. {count:,} moves processed into {target_table}.")


def main():
    p = argparse.ArgumentParser(description="select_games: build selected_games; process_shard: stage parquet by shard; merge: load into personal.db; berserk: identify berserk games; grant_more_time: identify games with time added; preprocess: compute log-transforms/bins")
    p.add_argument("command", choices=("select_games", "process_shard", "merge", "berserk", "grant_more_time", "preprocess"))
    p.add_argument("--total-shards", type=int, default=TOTAL_SHARDS, metavar="N", help="Slurm array width (default %(default)s)")
    p.add_argument("--job-id", type=int, default=None, metavar="I", help="Stride index (default: SLURM_ARRAY_TASK_ID or 0)")
    p.add_argument(
        "--tmpdir",
        default=os.environ.get("LOAD_MOVES_TMPDIR") or DEFAULT_TMPDIR,
        metavar="DIR",
        help="Fresh dir for staging parquet + DuckDB temp (default: env LOAD_MOVES_TMPDIR or %(default)s)",
    )
    p.add_argument(
        "--exclude-negative",
        action="store_true",
        default=True,
        help="Exclude entire games if they contain any negative move_time (default: %(default)s)",
    )
    p.add_argument("--no-exclude-negative", action="store_false", dest="exclude_negative", help="Disable negative move_time exclusion")
    p.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive lower datetime/date for selected_games")
    p.add_argument("--end-date", default=DEFAULT_END_DATE, help="Exclusive upper datetime/date for selected_games")
    p.add_argument("--initial-clock", type=int, default=600, help="Initial clock in seconds (default %(default)s)")
    p.add_argument("--clock-increment", type=int, default=0, help="Clock increment in seconds (default %(default)s)")
    p.add_argument("--min-elo", type=int, default=2000, help="Minimum white/black elo (default %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of moves for smoke testing")
    p.add_argument("--threads", type=int, default=40, help="Number of threads for DuckDB (default %(default)s)")
    p.add_argument("--memory", type=str, default="64GB", help="Memory limit for DuckDB (default %(default)s)")
    args = p.parse_args()
    
    global THREADS, MEMORY_LIMIT
    THREADS = args.threads
    MEMORY_LIMIT = args.memory
    tmpdir = os.path.abspath(args.tmpdir)
    if args.command == "select_games":
        print(
            f"🚀 select_games | window=[{args.start_date}, {args.end_date}) | "
            f"tc={args.initial_clock}+{args.clock_increment} | min_elo={args.min_elo} | tmpdir={tmpdir}"
        )
        preprocess_selected_games(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_clock=args.initial_clock,
            clock_increment=args.clock_increment,
            min_elo=args.min_elo,
            tmpdir=tmpdir,
        )
    elif args.command == "process_shard":
        jid = args.job_id if args.job_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        print(f"🚀 process_shard | job-id={jid} | total-shards={args.total_shards} | tmpdir={tmpdir} | exclude-negative={args.exclude_negative}")
        process_shard(jid, args.total_shards, tmpdir, exclude_negative=args.exclude_negative)
    elif args.command == "berserk":
        print(f"🚀 identify_berserk | tmpdir={tmpdir}")
        identify_berserk(tmpdir)
    elif args.command == "grant_more_time":
        print(f"🚀 identify_grant_more_time | tmpdir={tmpdir}")
        identify_grant_more_time(tmpdir)
    elif args.command == "preprocess":
        print(f"🚀 preprocess | tmpdir={tmpdir} | limit={args.limit}")
        limit_clause = f"LIMIT {args.limit}" if args.limit else ""
        preprocess(tmpdir, limit_clause=limit_clause)
    else:
        print(f"🚀 merge | tmpdir={tmpdir}")
        merge_shards(tmpdir)


if __name__ == "__main__":
    main()
