"""
Summary plots for move time distributions.
Creates side-by-side histograms for move_time and ln_move_time using SQL-binned counts.
"""

import os
import duckdb
import argparse
import matplotlib.pyplot as plt

from utils import apply_poster_style, FONT_SIZE_LABEL, EPSILON
from utils.plots import plot_histogram_from_bins
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES, TABLE_PROCESSED_MOVES_NONZERO
LIMIT_N = None  # Use None for full dataset
N_BINS = 60


def create_histogram_bins(conn, table_name, value_col, out_table, n_bins=N_BINS):
    """
    Build equal-width histogram bins and counts in a TEMPORARY SQL table.
    """
    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE {out_table} AS
        WITH stats AS (
            SELECT
                min({value_col}) AS min_v,
                max({value_col}) AS max_v
            FROM {table_name}
        ),
        bins AS (
            SELECT
                i AS bin_idx,
                min_v + (max_v - min_v) * i / {n_bins} AS bin_left,
                min_v + (max_v - min_v) * (i + 1) / {n_bins} AS bin_right
            FROM stats,
            range({n_bins}) AS t(i)
        )
        SELECT
            b.bin_idx,
            b.bin_left,
            b.bin_right,
            count(v.{value_col}) AS n
        FROM bins b
        LEFT JOIN {table_name} v
            ON (
                v.{value_col} >= b.bin_left
                AND (
                    v.{value_col} < b.bin_right
                    OR (b.bin_idx = {n_bins - 1} AND v.{value_col} <= b.bin_right)
                )
            )
        GROUP BY b.bin_idx, b.bin_left, b.bin_right
        ORDER BY b.bin_idx;
    """)


def main():
    # 1. Setup
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {SELECTED_DB_DEFAULT}...")
    conn = duckdb.connect(database=SELECTED_DB_DEFAULT, read_only=False)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_zeroT', action='store_true', help='Include zero move time moves (premoves)')
    args = parser.parse_args()

    base_table = TABLE_PROCESSED_MOVES if args.include_zeroT else TABLE_PROCESSED_MOVES_NONZERO

    # 3. Get Counts from processed table
    print("Getting processed dataset counts...")
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")

    # 4. SQL histogram binning
    print("Building SQL histogram bins...")
    # Create temporary view with computed log-transform
    conn.execute(f"CREATE OR REPLACE TEMPORARY VIEW _summary_view AS SELECT *, ln(move_time + {EPSILON}) as ln_move_time FROM {base_table}")
    
    create_histogram_bins(conn, "_summary_view", "move_time", "_move_time_hist_bins_tmp")
    create_histogram_bins(conn, "_summary_view", "ln_move_time", "_ln_move_time_hist_bins_tmp")

    # 5. Load binned data
    print("Loading binned histograms...")
    df_move_bins = conn.execute("SELECT * FROM _move_time_hist_bins_tmp ORDER BY bin_idx").df()
    df_ln_move_bins = conn.execute("SELECT * FROM _ln_move_time_hist_bins_tmp ORDER BY bin_idx").df()
    conn.close()

    # 6. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))

    plot_histogram_from_bins(
        axes[0],
        df_move_bins,
        x_label="Move Time (s)",
        y_label="Count",
    )
    plot_histogram_from_bins(
        axes[1],
        df_ln_move_bins,
        x_label="log Move Time (s)",
        y_label="Count",
    )

    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)

    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "move_time_summary")
    os.makedirs(figures_dir, exist_ok=True)

    filename = "combined_include_zeroT.png" if args.include_zeroT else "combined.png"
    output_plot = os.path.join(figures_dir, filename)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Move-time summary plot saved to {output_plot}")


if __name__ == "__main__":
    main()
