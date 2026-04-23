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
import matplotlib.pyplot as plt

from utils import (
    get_db_connection, apply_poster_style, MAIN_COLOR,
    FONT_SIZE_LABEL, FONT_SIZE_TICKS
)
from utils.plots import plot_qbin_stats

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
EPSILON = 1e-6
LIMIT_N = 1000000  # Default to 1M moves for performance; set to None for full dataset

def plot_raw_clock_hexbin(df, ax):
    """
    Plot the density of ln_move_time vs ln_player_clock_time using hexbin.
    """
    plt.sca(ax)
    hb = ax.hexbin(df['ln_player_clock_time'], df['ln_move_time'], gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel(r"log Player Clock (s)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"log T (s)", fontsize=FONT_SIZE_LABEL)


def main():
    # 1. Connect
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # 2. SQL Pre-calculation
    limit_clause = f"LIMIT {LIMIT_N}" if LIMIT_N is not None else ""
    print(f"Creating SQL tables _selected_moves and stats (limit={LIMIT_N or 'FULL'})...")
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE _selected_moves AS
        SELECT 
            gid,
            move_ply,
            player_clock_time,
            ln(player_clock_time + {EPSILON}) as ln_player_clock_time,
            ln(move_time + {EPSILON}) as ln_move_time,
            ntile(10) over (order by player_clock_time) as clock_qbin
        FROM (
            SELECT gid, move_ply, player_clock_time, move_time
            FROM selected_moves
            {limit_clause}
        );
    """)
    
    # 3. Aggregate Statistics
    print("Calculating aggregate statistics...")
    conn.execute("""
        CREATE OR REPLACE TABLE _clock_stats AS
        SELECT 
            clock_qbin,
            avg(player_clock_time) as mean_clock,
            avg(ln_move_time) as mean_y,
            stddev(ln_move_time) as std_y,
            count(*) as n
        FROM _selected_moves
        GROUP BY clock_qbin;
    """)
    
    # 4. Get Counts
    n_games = conn.execute("SELECT count(distinct gid) FROM _selected_moves").fetchone()[0]
    n_moves = conn.execute("SELECT count(*) FROM _selected_moves").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    # 5. Load Data for Plotting
    print("Loading data for visualization...")
    # For the hexbin, we sample if the table is very large to avoid memory issues in Pandas
    df_raw = conn.execute("SELECT ln_player_clock_time, ln_move_time FROM _selected_moves USING SAMPLE 100000").df()
    df_stats = conn.execute("SELECT * FROM _clock_stats ORDER BY clock_qbin").df()
    conn.close()
    
    # 6. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    plot_raw_clock_hexbin(df_raw, axes[0])
    plot_qbin_stats(
        axes[1], df_stats, 
        x_col='mean_clock', 
        x_label="Qbin Player Clock (s)", 
        y_label="log T (s)"
    )

    plot_qbin_stats(
        axes[2], df_stats, 
        x_col='mean_clock', 
        x_label="Qrank Player Clock", 
        y_label="log T (s)",
        normalized=True
    )

    # Single-line minimalist title
    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)
    
    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "clock_movetime")
    os.makedirs(figures_dir, exist_ok=True)
    output_plot = os.path.join(figures_dir, "combined.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Combined clock trend plot saved to {output_plot}")

if __name__ == "__main__":
    main()
