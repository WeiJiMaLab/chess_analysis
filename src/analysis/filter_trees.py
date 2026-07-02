"""Select the root FENs to generate trees from, for one run.

In this version "filtering" is: take the distinct FENs whose move_ply falls in the
run's [min_ply, max_ply] window (from the preprocessed move DB) and uniformly
subsample N of them, written one-per-line to the treegen input file. This is the
DB-sourced counterpart of the old `shuf fens.txt` — necessary because the raw FEN
pool is truncated (no move counters), so ply lives only in the DB. Downstream
board.py / engine.py apply the SAME window on read, keeping the population
consistent (see the ply-window design note).

    python -m analysis.filter_trees --db DB --out OUT --min-ply 15 --max-ply 75 --n 250000
"""

import argparse
import os

import duckdb


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sample in-window root FENs for tree generation.")
    ap.add_argument("--db", required=True, help="preprocessed move DB (personal.db)")
    ap.add_argument("--out", required=True, help="output FEN list (treegen.fens)")
    ap.add_argument("--min-ply", type=int, required=True)
    ap.add_argument("--max-ply", type=int, required=True)
    ap.add_argument("--n", type=int, default=250000, help="number of distinct FENs to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--table", default="processed_moves", help="source move table")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    conn = duckdb.connect(args.db, read_only=True)
    n_pool = conn.execute(
        f"SELECT count(DISTINCT fen) FROM {args.table} "
        f"WHERE move_ply BETWEEN {args.min_ply} AND {args.max_ply}"
    ).fetchone()[0]
    print(f"filter_trees: {n_pool:,} distinct in-window FENs "
          f"(ply {args.min_ply}-{args.max_ply}); sampling {args.n:,} (seed={args.seed})")

    # Uniform reservoir sample of N distinct in-window FENs.
    rows = conn.execute(
        f"SELECT fen FROM ("
        f"  SELECT DISTINCT fen FROM {args.table} "
        f"  WHERE move_ply BETWEEN {args.min_ply} AND {args.max_ply}"
        f") USING SAMPLE {args.n} ROWS (reservoir, {args.seed})"
    ).fetchall()
    conn.close()

    with open(args.out, "w") as f:
        f.write("".join(f"{r[0]}\n" for r in rows))
    print(f"filter_trees: wrote {len(rows):,} FENs -> {args.out}")


if __name__ == "__main__":
    main()
