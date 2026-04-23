"""
Analysis of the relationship between player clock time and move time.
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
    preprocess, EPSILON
)
from utils.plots import plot_qbin_stats, plot_beta_vs_ply, plot_raw_trend, plot_subset_scatterplot

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
LIMIT_N = None # Default to 1M moves for performance; set to None for full dataset

def main():
    # 1. Setup
    base_table = "_selected_moves"
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--nonzero_T', action='store_true', help='Use nonzero move time variant')
    args = parser.parse_args()
    
    if args.nonzero_T:
        base_table = f"{base_table}_nonzero_T"
    
    # 2. SQL Preprocessing
    if not args.skip_preprocess:
        limit_clause = f"LIMIT {LIMIT_N}" if LIMIT_N is not None else ""
        print(f"Preprocessing moves (limit={LIMIT_N or 'FULL'})...")
        preprocess(conn, target_table=base_table, limit_clause=limit_clause)
    else:
        print("Skipping preprocessing as requested.")

    # 3. Aggregate Statistics
    print("Calculating aggregate statistics...")
    conn.execute(f"""
        CREATE OR REPLACE TABLE _clock_stats AS
        SELECT 
            ln_player_clock_time,
            avg(player_clock_time) as mean_clock,
            avg(ln_move_time) as mean_y,
            stddev(ln_move_time) as std_y,
            count(*) as n
        FROM {base_table}
        GROUP BY ln_player_clock_time;
    """)

    conn.execute(f"""
        CREATE OR REPLACE TABLE _clock_qstats AS
        SELECT 
            player_clock_qbin,
            avg(player_clock_time) as mean_clock,
            avg(ln_move_time) as mean_y,
            stddev(ln_move_time) as std_y,
            count(*) as n
        FROM {base_table}
        GROUP BY player_clock_qbin;
    """)
    
    # 4. Global Regression (SQL-native)
    print("Calculating global OLS parameters (SQL-native)...")
    slope_global, intercept_global = calculate_ols(conn, base_table, "ln_player_clock_time", "ln_move_time")
    print(f"Global OLS: Slope={slope_global:.4f}, Intercept={intercept_global:.4f}")

    # 5. Per-Ply Regression (SQL-native)
    print("Calculating per-ply OLS slopes (SQL-native)...")
    df_betas = calculate_plywise_betas(conn, base_table, "ln_player_clock_time", "ln_move_time")

    # 6. Get Counts
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    # 7. Load Data for Plotting
    print("Loading data for visualization...")
    df_raw = conn.execute(f"SELECT ln_player_clock_time, ln_move_time FROM {base_table}").df()
    df_stats = conn.execute("SELECT * FROM _clock_stats ORDER BY ln_player_clock_time").df()
    df_qstats = conn.execute("SELECT * FROM _clock_qstats ORDER BY player_clock_qbin").df()
    conn.close()

    # 8. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes = axes.flatten()
    
    plot_raw_trend(
        axes[0], df_stats, 
        x_col='ln_player_clock_time', 
        y_col='mean_y', 
        std_col='std_y', 
        n_col='n',
        x_label="log Player Clock (s)", 
        y_label="log T (s)",
        show_legend=False
    )

    # Plot OLS prediction line using SQL-calculated parameters
    x_min, x_max = df_raw['ln_player_clock_time'].min(), df_raw['ln_player_clock_time'].max()
    x_lin = np.linspace(x_min, x_max, 100)
    y_pred = slope_global * x_lin + intercept_global
    axes[0].plot(x_lin, y_pred, color='red', lw=4, label=f'OLS Fit (β={slope_global:.2f})', linestyle='--')

    plot_qbin_stats(
        axes[1], df_qstats, 
        x_col='mean_clock', 
        x_label="Qrank Player Clock", 
        y_label="log T (s)",
        normalized=False,
        show_legend=False
    )

    plot_subset_scatterplot(
        axes[2], df_raw, 'ln_player_clock_time', 'ln_move_time', 
        x_label="log Player Clock (s)", 
        y_label="log T (s)",
        n=10000
    )
    axes[2].plot(x_lin, y_pred, color='red', lw=4, label=f'OLS Fit (β={slope_global:.2f})', linestyle='--')
    axes[2].legend(fontsize=FONT_SIZE_TICKS)

    # 4th Panel: Beta vs Ply
    plot_beta_vs_ply(axes[3], df_betas, title=None)

    # Single-line minimalist title
    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)
    
    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "clock_movetime")
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = "combined_nonzero_T.png" if args.nonzero_T else "combined.png"
    output_plot = os.path.join(figures_dir, filename)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Combined clock trend plot saved to {output_plot}")

if __name__ == "__main__":
    main()
