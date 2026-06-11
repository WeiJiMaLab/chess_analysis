"""
Move time distribution histograms.
Saves figures/movetime_histogram.png and figures/ln_movetime_histogram.png.
"""

import os
import argparse
import math

import duckdb
import matplotlib.pyplot as plt

from utils import apply_poster_style, FONT_SIZE_LABEL, EPSILON
from utils.plots import plot_histogram_from_bins
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES, TABLE_PROCESSED_MOVES_NONZERO

N_BINS = 50


def _histogram_bins(conn, table_name, value_col, out_table):
    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE {out_table} AS
        WITH stats AS (
            SELECT min({value_col}) AS min_v, max({value_col}) AS max_v
            FROM {table_name}
        ),
        bins AS (
            SELECT i AS bin_idx,
                   min_v + (max_v - min_v) * i / {N_BINS} AS bin_left,
                   min_v + (max_v - min_v) * (i + 1) / {N_BINS} AS bin_right
            FROM stats, range({N_BINS}) AS t(i)
        )
        SELECT b.bin_idx, b.bin_left, b.bin_right, count(v.{value_col}) AS n
        FROM bins b
        LEFT JOIN {table_name} v
            ON v.{value_col} >= b.bin_left
           AND (v.{value_col} < b.bin_right OR (b.bin_idx = {N_BINS - 1} AND v.{value_col} <= b.bin_right))
        GROUP BY b.bin_idx, b.bin_left, b.bin_right
        ORDER BY b.bin_idx
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_zeroT", action="store_true")
    args = parser.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    conn = duckdb.connect(database=SELECTED_DB_DEFAULT, read_only=False)

    base_table = TABLE_PROCESSED_MOVES if args.include_zeroT else TABLE_PROCESSED_MOVES_NONZERO
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"n = {n_moves:,} moves")

    conn.execute(
        f"CREATE OR REPLACE TEMPORARY VIEW _summary_view AS "
        f"SELECT *, ln(move_time + {EPSILON}) AS ln_move_time FROM {base_table}"
    )
    _histogram_bins(conn, "_summary_view", "move_time", "_mt_bins")
    _histogram_bins(conn, "_summary_view", "ln_move_time", "_lmt_bins")

    df_mt = conn.execute("SELECT * FROM _mt_bins ORDER BY bin_idx").df()
    df_lmt = conn.execute("SELECT * FROM _lmt_bins ORDER BY bin_idx").df()

    # Weighted mean and median from bins
    def _weighted_mean(df):
        mid = (df["bin_left"] + df["bin_right"]) / 2
        return (mid * df["n"]).sum() / df["n"].sum()

    def _weighted_median(df):
        cumsum = df["n"].cumsum()
        total = df["n"].sum()
        idx = (cumsum >= total / 2).idxmax()
        return float((df["bin_left"].iloc[idx] + df["bin_right"].iloc[idx]) / 2)

    mt_mean = _weighted_mean(df_mt)
    mt_median = _weighted_median(df_mt)
    lmt_mean = _weighted_mean(df_lmt)
    lmt_median = _weighted_median(df_lmt)
    conn.close()

    figures_dir = os.path.join(src_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for df_bins, x_label, fname, mean, median in [
        (df_mt, "Move Time (s)", "movetime_histogram.png", mt_mean, mt_median),
        (df_lmt, "log(Move Time) (s)", "log_movetime_histogram.png", lmt_mean, lmt_median),
    ]:
        apply_poster_style()
        fig, ax = plt.subplots(figsize=(18, 10))
        plot_histogram_from_bins(ax, df_bins, x_label=x_label, mean=mean, median=median)
        ax.set_title(f"n = {n_moves:,} moves", fontsize=FONT_SIZE_LABEL, pad=12)
        plt.tight_layout()
        out = os.path.join(figures_dir, fname)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ {out}")


if __name__ == "__main__":
    main()
