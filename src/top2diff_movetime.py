"""
Bivariate analysis: top-2 WDL gap (engine) vs think time, parallel to clock_movetime.
"""

import argparse
import os

import duckdb

from utils import Variable, Analyzer

PERSONAL_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"


def main():
    p = argparse.ArgumentParser(description="Top-2 WDL gap vs log(move time) dashboard")
    p.add_argument("--db", default=PERSONAL_DB_DEFAULT, help="Path to personal.db")
    args = p.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    conn = duckdb.connect(database=args.db, read_only=False)

    x_var = Variable(
        column="top2_wdl_diff",
        is_log=False,
        name="Top-2 WDL gap",
    )
    y_var = Variable(column="move_time", is_log=True, name="T")

    analyzer = Analyzer(
        db_conn=conn,
        table_name="selected_moves_with_engine",
        x_var=x_var,
        y_var=y_var,
        filter_query="move_time > 0",
        title="Top-2 WDL gap vs think time",
    )

    figures_dir = os.path.join(src_dir, "figures", "top2diff_movetime")
    os.makedirs(figures_dir, exist_ok=True)
    analyzer.save_dashboard(os.path.join(figures_dir, "combined.png"))
    conn.close()


if __name__ == "__main__":
    main()
