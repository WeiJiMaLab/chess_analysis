"""
Analysis of the relationship between move ply and move time.
This script calculates and visualizes thinking time trends across
game stages using SQL-precalculated statistics and shaded 95% CIs.
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
    EPSILON
)
from utils.plots import plot_qbin_stats, plot_raw_trend

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
LIMIT_N = None  # Use None for full dataset


def main():
    # 1. Setup
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_zeroT', action='store_true', help='Include zero move time moves (premoves)')
    args = parser.parse_args()
    
    base_table = "_selected_moves" if args.include_zeroT else "_selected_moves_nonzero_T"
    
    # 3. Get Counts from processed table
    print("Getting processed dataset counts...")
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE _ply_stats AS
        SELECT 
            move_ply,
            avg(ln_move_time) as mean_ln_move_time,
            stddev(ln_move_time) as std_ln_move_time,
            count(*) as n
        FROM {base_table}
        GROUP BY move_ply;
    """)
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE _ply_qstats AS
        SELECT 
            move_ply_qbin,
            avg(move_ply) as mean_move_ply,
            avg(ln_move_time) as mean_ln_move_time,
            stddev(ln_move_time) as std_ln_move_time,
            count(*) as n
        FROM {base_table}
        GROUP BY move_ply_qbin;
    """)
    
    # 4. Load Stats for Plotting
    print("Loading data from SQL tables...")
    df_ply_stats = conn.execute("SELECT * FROM _ply_stats").df()
    df_qbin_stats = conn.execute("SELECT * FROM _ply_qstats ORDER BY move_ply_qbin").df()
    conn.close()
    
    # 5. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    
    plot_raw_trend(
        axes[0], df_ply_stats, 
        x_col='move_ply', 
        y_col='mean_ln_move_time', 
        std_col='std_ln_move_time', 
        n_col='n',
        x_label="Move Ply",
        y_label=r"Mean $\log(T)$", 
        show_legend = False
    )

    plot_qbin_stats(
        axes[1], df_qbin_stats,
        x_col='mean_move_ply',
        y_col='mean_ln_move_time',
        std_col='std_ln_move_time',
        x_label="Mean Move Ply in Quantile Bin",
        y_label=r"Mean $\log(T)$",
        show_legend = False
    )
    
    # Single-line minimalist title
    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)
    
    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "ply_movetime")
    os.makedirs(figures_dir, exist_ok=True)
    
    filename = "combined_include_zeroT.png" if args.include_zeroT else "combined.png"
    output_plot = os.path.join(figures_dir, filename)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Combined plot saved to {output_plot}")

if __name__ == "__main__":
    main()
