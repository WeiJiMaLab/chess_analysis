"""Move-time distribution: log(MT) histogram (left) + normal QQ plot (right).

One figure on the full move set (``processed_moves_nonzero`` unless
``--include_zeroT``). The QQ plot assesses log-normality of move time: if log(MT)
is normal, the empirical quantiles fall on the reference line. Quantiles and
moments are computed in DuckDB so we never pull the full column into memory.

Usage (from chess_analysis/):
    PYTHONPATH=human_analytics python human_analytics/move_time_summary.py
"""

import argparse
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR
from utils.plots import plot_histogram_from_bins
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES, TABLE_PROCESSED_MOVES_NONZERO

N_BINS = 50
N_QQ = 200  # number of quantile probe points for the QQ plot


def _histogram_bins(conn, view, value_col, out_table):
    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE {out_table} AS
        WITH stats AS (SELECT min({value_col}) AS min_v, max({value_col}) AS max_v FROM {view}),
        bins AS (
            SELECT i AS bin_idx,
                   min_v + (max_v - min_v) * i / {N_BINS} AS bin_left,
                   min_v + (max_v - min_v) * (i + 1) / {N_BINS} AS bin_right
            FROM stats, range({N_BINS}) AS t(i)
        )
        SELECT b.bin_idx, b.bin_left, b.bin_right, count(v.{value_col}) AS n
        FROM bins b
        LEFT JOIN {view} v
            ON v.{value_col} >= b.bin_left
           AND (v.{value_col} < b.bin_right OR (b.bin_idx = {N_BINS - 1} AND v.{value_col} <= b.bin_right))
        GROUP BY b.bin_idx, b.bin_left, b.bin_right
        ORDER BY b.bin_idx
    """)


def _weighted_mean(df):
    mid = (df["bin_left"] + df["bin_right"]) / 2
    return (mid * df["n"]).sum() / df["n"].sum()


def _weighted_median(df):
    cumsum = df["n"].cumsum()
    idx = (cumsum >= df["n"].sum() / 2).idxmax()
    return float((df["bin_left"].iloc[idx] + df["bin_right"].iloc[idx]) / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include_zeroT", action="store_true")
    args = parser.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    conn = duckdb.connect(database=SELECTED_DB_DEFAULT, read_only=False)
    base_table = TABLE_PROCESSED_MOVES if args.include_zeroT else TABLE_PROCESSED_MOVES_NONZERO
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"n = {n_moves:,} moves")

    conn.execute(
        "CREATE OR REPLACE TEMPORARY VIEW _summary_view AS "
        f"SELECT ln(move_time) AS ln_move_time FROM {base_table}"
    )
    _histogram_bins(conn, "_summary_view", "ln_move_time", "_lmt_bins")
    df_lmt = conn.execute("SELECT * FROM _lmt_bins ORDER BY bin_idx").df()

    # Moments + empirical quantiles for the QQ plot (all SQL-side).
    mean, std = conn.execute("SELECT avg(ln_move_time), stddev(ln_move_time) FROM _summary_view").fetchone()
    probs = (np.arange(1, N_QQ + 1)) / (N_QQ + 1)
    emp_q = np.array(conn.execute(
        "SELECT quantile_cont(ln_move_time, ?) FROM _summary_view", [probs.tolist()]
    ).fetchone()[0], dtype=float)
    conn.close()

    theo_q = stats.norm.ppf(probs)  # standard-normal quantiles
    ref = mean + std * theo_q       # reference line if log(MT) ~ Normal(mean, std)

    repo_root = os.path.dirname(src_dir)
    figures_dir = os.path.join(repo_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    apply_poster_style()
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(30, 13.72))

    plot_histogram_from_bins(
        ax_h, df_lmt, x_label="log(Move time)",
        mean=_weighted_mean(df_lmt), median=_weighted_median(df_lmt),
    )

    ax_q.scatter(theo_q, emp_q, color=MAIN_COLOR, s=40, zorder=3)
    ax_q.plot(theo_q, ref, color="#C0392B", lw=3, label="Normal reference")
    ax_q.set_xlabel("Theoretical normal quantile", fontsize=FONT_SIZE_LABEL)
    ax_q.set_ylabel("log(Move time) quantile", fontsize=FONT_SIZE_LABEL)
    ax_q.legend(fontsize=FONT_SIZE_TICKS, loc="upper left", frameon=False)

    fig.suptitle(f"Move time (log) — distribution & normal QQ\nn = {n_moves:,} moves",
                 fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    fig.subplots_adjust(top=0.80)
    out = os.path.join(figures_dir, "movetime_logmt_qq.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
