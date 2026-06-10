"""
Export all unique FENs from processed_moves_nonzero to a flat text file.

Writes one FEN per line, sorted, to the output path.  The result is used as
the full candidate pool for tree-generation sampling (e.g. human_fens_50k.txt
is a reservoir-sampled subset of this file).

Result: ~110.5 M unique FENs, ~5.4 GB.

Usage (from chess_analysis/):
    python lmcos/fens/export_unique_fens.py
    python lmcos/fens/export_unique_fens.py --db /path/to/personal.db --out /path/to/fens.txt
"""

from __future__ import annotations

import argparse

import duckdb

DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
OUT_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens.txt"


def export(db_path: str, out_path: str) -> None:
    con = duckdb.connect(db_path, read_only=True)
    con.execute(f"""
        COPY (SELECT DISTINCT fen FROM processed_moves_nonzero ORDER BY fen)
        TO '{out_path}' (FORMAT CSV, HEADER false)
    """)
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()
    print(f"exporting unique FENs from {args.db} -> {args.out}")
    export(args.db, args.out)
    print("done")
