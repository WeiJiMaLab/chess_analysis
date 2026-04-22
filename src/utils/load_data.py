import argparse
import duckdb
import pandas as pd
from tqdm import tqdm
import os
import time

# Must match Slurm: #SBATCH --array=0-(TOTAL_SHARDS-1)
TOTAL_SHARDS = 100
STAGING_DIR = "/scratch/gpfs/GRIFFITHS/hl4291/tmp/"
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
MOVES_ROOT = "/scratch/gpfs/GRIFFITHS/chess-db/rawdata"
_CONN_KW = {"threads": 10, "memory_limit": "12GB", "temp_directory": "."}


def process_shard(_JOBID, total_shards=TOTAL_SHARDS):
    # Local working database (opened as read-only).
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True, config=_CONN_KW)
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

    os.makedirs(STAGING_DIR, exist_ok=True)

    for partition_tuple in tqdm(partition_tuples, desc=f"♟️ shards job={_JOBID}"):
        partition, segment = partition_tuple
        gids_in_partition = gids[gids["partition_tuple"] == partition_tuple]
        print("♟️  Processing:", _JOBID, partition_tuple, "| games:", len(gids_in_partition))

        time_start = time.time()
        gid_list = gids_in_partition["gid"].tolist()
        parquet_read_path = f"{MOVES_ROOT}/partition={partition}/{segment}-moves.parquet"
        out_path = os.path.join(STAGING_DIR, f"selected_moves_{partition}_{segment}.parquet")
        placeholders = ",".join(["?"] * len(gid_list))
        conn.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet(?)
                WHERE gid IN ({placeholders})
            ) TO ? (FORMAT PARQUET)
            """,
            [parquet_read_path, *gid_list, out_path],
        )

        n_moves = conn.execute("SELECT count(*) FROM read_parquet(?)", [out_path]).fetchone()[0]
        print(f"\t📊 Moves written: {n_moves}")
        print(f"\t💾 Saved: {out_path}")
        print(f"\t⏱️  Elapsed: {time.time() - time_start:.2f}s")

    print(f"✅ Process shard job {_JOBID} finished.")
    conn.close()


def merge_shards():
    """Load staging parquet shards into personal.db (single writer)."""
    pattern = os.path.join(STAGING_DIR, "*.parquet")
    print(f"🔀 Merge: reading {pattern!r} into {PERSONAL_DB}")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False, config=_CONN_KW)
    conn.execute(
        """
        CREATE OR REPLACE TABLE selected_moves AS
        SELECT * FROM read_parquet(?)
        """,
        [pattern],
    )
    conn.close()
    print("✅ Merge finished — table selected_moves replaced.")


def main():
    p = argparse.ArgumentParser(description="process: staging parquet per Slurm shard; merge: load into personal.db")
    p.add_argument("command", choices=("process", "merge"))
    p.add_argument("--total-shards", type=int, default=TOTAL_SHARDS, metavar="N", help="Slurm array width (default %(default)s)")
    p.add_argument("--job-id", type=int, default=None, metavar="I", help="Stride index (default: SLURM_ARRAY_TASK_ID or 0)")
    args = p.parse_args()
    if args.command == "process":
        jid = args.job_id if args.job_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        print(f"🚀 process | job-id={jid} | total-shards={args.total_shards}")
        process_shard(jid, args.total_shards)
    else:
        print("🚀 merge")
        merge_shards()


if __name__ == "__main__":
    main()
