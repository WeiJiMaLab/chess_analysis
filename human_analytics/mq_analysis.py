"""MQ (move quality) vs move time — full engine-eval dataset.

Reads ``pos_with_engine_eval`` (Stockfish eval of every scored human move) and
plots MQ as a function of log reaction time in the standard quantile-bin dashboard
style (global + by ply tertile). MQ = e_win_taken − e_win_best ≤ 0 (0 = best move).

VOC and Action Gap are no longer produced here — they moved to the tree-derived
"generated values" on the OSS subset (``tree_values_analysis.py``).

Usage (from chess_analysis/):
    PYTHONPATH=human_analytics python human_analytics/mq_analysis.py
"""

from __future__ import annotations

import argparse
import os

import duckdb
import numpy as np

from utils import Variable, Analyzer
from utils.selected_db import SELECTED_DB_DEFAULT

_HA = os.path.dirname(os.path.abspath(__file__))
_FIGURES_DIR = os.path.join(_HA, "figures")
_TABLE = "pos_with_engine_eval"


def plot_mq_vs_logrt(conn: duckdb.DuckDBPyConnection, output_path: str) -> None:
    """MQ vs log reaction time — quantile bins, global + by ply tertile."""
    analyzer = Analyzer(
        db_conn=conn,
        table_name=_TABLE,
        x_var=Variable(column="move_time", is_log=True, name="log RT(s)"),
        y_var=Variable(column="mq", is_log=False, name="MQ"),
        filter_query="move_time > 0",
        title="MQ vs. log Reaction Time",
    )
    analyzer.save_dashboard(output_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--output", default=os.path.join(_FIGURES_DIR, "mq_vs_logrt.png"))
    args = parser.parse_args(argv)

    conn = duckdb.connect(args.db, read_only=False)
    n = conn.execute(f"SELECT count(*) FROM {_TABLE} WHERE move_time > 0").fetchone()[0]
    r = conn.execute(f"SELECT corr(mq, ln(move_time)) FROM {_TABLE} WHERE move_time > 0").fetchone()[0]
    print(f"MQ vs log RT: n={n:,}  Pearson r={r:+.4f}  R²={r**2:.4f}")
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    plot_mq_vs_logrt(conn, args.output)
    conn.close()


if __name__ == "__main__":
    main()
