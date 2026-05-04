"""
Build move-level table with engine WDL columns joined from *_evaluations on fen.
"""

import argparse

import duckdb

from _bootstrap import ensure_src

ensure_src()

from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES


def main():
    p = argparse.ArgumentParser(
        description=f"Create selected_moves_with_engine from {TABLE_PROCESSED_MOVES} and engine evals"
    )
    p.add_argument("--db", default=SELECTED_DB_DEFAULT, help="Path to personal.db")
    p.add_argument(
        "--engine",
        choices=("stockfish", "lc0"),
        default="stockfish",
        help="Which *evaluations table to join (default: stockfish)",
    )
    args = p.parse_args()

    table_eval = f"{args.engine}_evaluations"

    conn = duckdb.connect(args.db, read_only=False)
    try:
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE selected_moves_with_engine AS
            SELECT
                m.*,
                e.e_win_best,
                e.e_win_second_best,
                e.e_win_move_taken,
                e.n_repeats,
                abs(e.e_win_best - e.e_win_second_best) AS top2_wdl_diff
            FROM {TABLE_PROCESSED_MOVES} m
            INNER JOIN {table_eval} e ON m.fen = e.fen
            """
        )
        n = conn.execute("SELECT count(*) FROM selected_moves_with_engine").fetchone()[0]
        n_base = conn.execute(f"SELECT count(*) FROM {TABLE_PROCESSED_MOVES}").fetchone()[0]
        n_missing = conn.execute(
            f"""
            SELECT count(*)
            FROM {TABLE_PROCESSED_MOVES} s
            LEFT JOIN {table_eval} e ON s.fen = e.fen
            WHERE e.fen IS NULL
            """
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"✅ selected_moves_with_engine: {n:,} rows (joining {TABLE_PROCESSED_MOVES} to {table_eval})")
    print(f"   {TABLE_PROCESSED_MOVES}: {n_base:,} rows; moves with no engine row: {n_missing:,}")


if __name__ == "__main__":
    main()
