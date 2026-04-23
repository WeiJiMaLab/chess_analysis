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
import matplotlib.pyplot as plt

from utils import (
    get_db_connection, apply_poster_style, MAIN_COLOR,
    FONT_SIZE_LABEL, FONT_SIZE_TICKS
)

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
EPSILON = 1e-6
LIMIT_N = None  # Use None for full dataset

def plot_raw_ply_trend(stats, ax, min_samples=30):
    """
    Plot the raw ply trend using precalculated stats with shaded 95% CI.
    """
    plt.sca(ax)
    
    # stats has ['move_ply', 'mean_ln_move_time', 'std_ln_move_time', 'n']
    stats = stats[stats['n'] >= min_samples].copy()
    stats = stats.sort_values('move_ply')
    
    stats['sem'] = stats['std_ln_move_time'] / np.sqrt(stats['n'])
    stats['ci_y'] = 1.96 * stats['sem']
    
    y_mean = stats['mean_ln_move_time']
    y_lower = y_mean - stats['ci_y']
    y_upper = y_mean + stats['ci_y']
    
    ax.plot(stats['move_ply'], y_mean, color=MAIN_COLOR, lw=3, label="Mean")
    ax.fill_between(stats['move_ply'], y_lower, y_upper, color=MAIN_COLOR, alpha=0.2, label="95% CI")
    
    ax.set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"Mean $\log(T)$", fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_TICKS)

def plot_qbin_ply_trend(stats, ax):
    """
    Plot the quantile-binned trend using precalculated stats with shaded 95% CI.
    """
    plt.sca(ax)
    
    # stats has ['qbin', 'mean_move_ply', 'mean_ln_move_time', 'std_ln_move_time', 'n']
    stats = stats.sort_values('qbin')
    stats['sem'] = stats['std_ln_move_time'] / np.sqrt(stats['n'])
    stats['ci_y'] = 1.96 * stats['sem']
    
    y_mean = stats['mean_ln_move_time']
    y_lower = y_mean - stats['ci_y']
    y_upper = y_mean + stats['ci_y']
    
    ax.plot(stats['mean_move_ply'], y_mean, marker='o', color=MAIN_COLOR, lw=3, markersize=12, label="Mean")
    ax.fill_between(stats['mean_move_ply'], y_lower, y_upper, color=MAIN_COLOR, alpha=0.2, label="95% CI")
    
    ax.set_xlabel("Mean Move Ply in Quantile Bin", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"Mean $\log(T)$", fontsize=FONT_SIZE_LABEL)
    ax.legend(fontsize=FONT_SIZE_TICKS)

def main():
    # 1. Connect
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # 2. Get Counts
    print("Getting dataset counts...")
    n_games = conn.execute("SELECT count(distinct gid) FROM selected_moves").fetchone()[0]
    n_moves = conn.execute("SELECT count(*) FROM selected_moves").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    # 3. SQL Pre-calculation
    limit_clause = f"LIMIT {LIMIT_N}" if LIMIT_N is not None else ""
    print(f"Creating SQL tables _selected_moves and stats (limit={LIMIT_N or 'FULL'})...")
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE _selected_moves AS
        SELECT 
            move_ply,
            log(move_time + {EPSILON}) as ln_move_time,
            ntile(10) over (order by move_ply) as qbin
        FROM (
            SELECT move_ply, move_time, gid
            FROM selected_moves
            {limit_clause}
        );
    """)
    
    conn.execute("""
        CREATE OR REPLACE TABLE _ply_stats AS
        SELECT 
            move_ply,
            avg(ln_move_time) as mean_ln_move_time,
            stddev(ln_move_time) as std_ln_move_time,
            count(*) as n
        FROM _selected_moves
        GROUP BY move_ply;
    """)
    
    conn.execute("""
        CREATE OR REPLACE TABLE _qbin_stats AS
        SELECT 
            qbin,
            avg(move_ply) as mean_move_ply,
            avg(ln_move_time) as mean_ln_move_time,
            stddev(ln_move_time) as std_ln_move_time,
            count(*) as n
        FROM _selected_moves
        GROUP BY qbin;
    """)
    
    # 4. Load Stats for Plotting
    print("Loading data from SQL tables...")
    df_ply_stats = conn.execute("SELECT * FROM _ply_stats").df()
    df_qbin_stats = conn.execute("SELECT * FROM _qbin_stats ORDER BY qbin").df()
    conn.close()
    
    # 5. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    
    plot_raw_ply_trend(df_ply_stats, axes[0], min_samples=30)
    plot_qbin_ply_trend(df_qbin_stats, axes[1])
    
    # Single-line minimalist title
    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)
    
    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "ply_movetime")
    os.makedirs(figures_dir, exist_ok=True)
    output_plot = os.path.join(figures_dir, "combined.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Combined plot saved to {output_plot}")

if __name__ == "__main__":
    main()
