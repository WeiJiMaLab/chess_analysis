"""
Analysis of the relationship between player or opponent clock time and move time.
This script visualizes how thinking time changes as the remaining clock budget decreases,
using density hexbins and quantile-binned aggregate statistics.
"""

import os
import json
import duckdb
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from utils import (
    get_db_connection, apply_poster_style, MAIN_COLOR,
    FONT_SIZE_LABEL, FONT_SIZE_TICKS,
    calculate_ols, calculate_plywise_betas,
    EPSILON
)
from utils.plots import plot_qbin_stats, plot_beta_vs_ply, plot_raw_trend, plot_subset_scatterplot

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
LIMIT_N = None # Default to 1M moves for performance; set to None for full dataset
# Min count per distinct log-clock bin for the high-resolution (raw) trend panel
RAW_TREND_MIN_N = 100

def main():
    # 1. Setup
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_zeroT', action='store_true', help='Include zero move time moves (premoves)')
    parser.add_argument('--opp', action='store_true', help='Use opponent clock instead of player clock')
    args = parser.parse_args()

    base_table = "_selected_moves" if args.include_zeroT else "_selected_moves_nonzero_T"
    player = "player" if not args.opp else "opponent"
    
    ln_clock_col = f"ln_{player}_clock_time"
    clock_col = f"{player}_clock_time"
    clock_qbin_col = f"{player}_clock_qbin"
    clock_label = f"{player.capitalize()} Clock"
    

    # 3. Aggregate Statistics
    print("Calculating aggregate statistics...")
    conn.execute(f"""
        CREATE OR REPLACE TABLE _clock_stats AS
        SELECT 
            {ln_clock_col},
            avg({clock_col}) as mean_clock,
            avg(ln_move_time) as mean_y,
            stddev(ln_move_time) as std_y,
            count(*) as n
        FROM {base_table}
        GROUP BY {ln_clock_col};
    """)

    conn.execute(f"""
        CREATE OR REPLACE TABLE _clock_qstats AS
        SELECT 
            {clock_qbin_col},
            avg({clock_col}) as mean_clock,
            avg(ln_move_time) as mean_y,
            stddev(ln_move_time) as std_y,
            count(*) as n
        FROM {base_table}
        GROUP BY {clock_qbin_col};
    """)
    
    # 4. Global Regression (SQL-native)
    print("Calculating global OLS parameters (SQL-native)...")
    slope_global, intercept_global = calculate_ols(conn, base_table, ln_clock_col, "ln_move_time")
    print(f"Global OLS: Slope={slope_global:.4f}, Intercept={intercept_global:.4f}")

    # 5. Per-Ply Regression (SQL-native)
    print("Calculating per-ply OLS slopes (SQL-native)...")
    df_betas = calculate_plywise_betas(conn, base_table, ln_clock_col, "ln_move_time")

    # 6. Get Counts
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    # 7. Load Data for Plotting
    print("Loading data for visualization...")
    df_raw = conn.execute(f"SELECT {ln_clock_col}, ln_move_time FROM {base_table}").df()
    df_stats = conn.execute(f"SELECT * FROM _clock_stats ORDER BY {ln_clock_col}").df()
    df_qstats = conn.execute(f"SELECT * FROM _clock_qstats ORDER BY {clock_qbin_col}").df()
    conn.close()

    # 8. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes = axes.flatten()
    
    plot_raw_trend(
        axes[0], df_stats, 
        x_col='mean_clock', 
        y_col='mean_y', 
        std_col='std_y', 
        n_col='n',
        x_label=f"{clock_label} (s)", 
        y_label="log T (s)",
        min_n=RAW_TREND_MIN_N,
        show_legend=False
    )

    # Plot the same log-log OLS on raw clock axis: y = beta * log(clock) + intercept
    x_raw_min, x_raw_max = df_stats['mean_clock'].min(), df_stats['mean_clock'].max()
    x_raw_lin = np.linspace(x_raw_min, x_raw_max, 300)
    y_pred_raw = slope_global * np.log(x_raw_lin + EPSILON) + intercept_global
    axes[0].plot(x_raw_lin, y_pred_raw, color='red', lw=4, label=f'OLS Fit (β={slope_global:.2f})', linestyle='--')

    plot_qbin_stats(
        axes[1], df_qstats, 
        x_col='mean_clock', 
        x_label=f"Qrank {clock_label}", 
        y_label="log T (s)",
        normalized=False,
        show_legend=False
    )

    plot_subset_scatterplot(
        axes[2], df_raw, ln_clock_col, 'ln_move_time', 
        x_label=f"log {clock_label} (s)", 
        y_label="log T (s)",
        n=10000
    )
    x_log_min, x_log_max = df_raw[ln_clock_col].min(), df_raw[ln_clock_col].max()
    x_log_lin = np.linspace(x_log_min, x_log_max, 300)
    y_pred_log = slope_global * x_log_lin + intercept_global
    axes[2].plot(x_log_lin, y_pred_log, color='red', lw=4, label=f'OLS Fit (β={slope_global:.2f})', linestyle='--')
    axes[2].legend(fontsize=FONT_SIZE_TICKS)

    # 4th Panel: Beta vs Ply
    plot_beta_vs_ply(axes[3], df_betas, title=None)

    # Single-line minimalist title
    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)
    
    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "clock_movetime")
    os.makedirs(figures_dir, exist_ok=True)
    
    name = "combined_include_zeroT" if args.include_zeroT else "combined"
    if args.opp:
        name += "_opp"
    filename = f"{name}.png"
    output_plot = os.path.join(figures_dir, filename)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Combined clock trend plot saved to {output_plot}")

if __name__ == "__main__":
    main()
