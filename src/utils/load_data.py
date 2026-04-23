import argparse
import duckdb
import pandas as pd
from tqdm import tqdm
import os
import time

# Must match Slurm: #SBATCH --array=0-(TOTAL_SHARDS-1)
TOTAL_SHARDS = 1
DEFAULT_TMPDIR = "/scratch/gpfs/GRIFFITHS/hl4291/tmp/"
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
MOVES_ROOT = "/scratch/gpfs/GRIFFITHS/chess-db/rawdata"
DEFAULT_START_DATE = "2023-11-01"
DEFAULT_END_DATE = "2023-12-31"  # Exclusive upper bound


def _conn_kw(tmpdir: str) -> dict:
    return {"threads": 10, "memory_limit": "12GB", "temp_directory": tmpdir}


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

    candidate_count = conn.execute(
        f"""
        SELECT count(*)
        FROM core.games
        WHERE utc_datetime >= TIMESTAMP '{s_start}'
          AND utc_datetime < TIMESTAMP '{s_end}'
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


def main():
    p = argparse.ArgumentParser(description="select_games: build selected_games; process: stage parquet by shard; merge: load into personal.db")
    p.add_argument("command", choices=("select_games", "process", "merge"))
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
    args = p.parse_args()
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
    elif args.command == "process":
        jid = args.job_id if args.job_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        print(f"🚀 process | job-id={jid} | total-shards={args.total_shards} | tmpdir={tmpdir} | exclude-negative={args.exclude_negative}")
        process_shard(jid, args.total_shards, tmpdir, exclude_negative=args.exclude_negative)
    else:
        print(f"🚀 merge | tmpdir={tmpdir}")
        merge_shards(tmpdir)


if __name__ == "__main__":
    main()
