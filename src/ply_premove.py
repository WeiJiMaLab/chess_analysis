"""
Analysis of the probability of pre-moves (move_time == 0) across game stages.
Replicates the structure of ply_movetime.py but for binary pre-move events.
"""

import os
import duckdb
import argparse
import matplotlib.pyplot as plt

from utils import (
    apply_poster_style, FONT_SIZE_LABEL, MAIN_COLOR
)
from utils.plots import plot_qbin_stats, plot_raw_trend

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    # 1. Setup paths
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    base_table = "_selected_moves"
    
    # 2. Get dataset metadata
    print("Getting processed dataset counts...")
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    # 3. Calculate Stats for Raw Ply
    print("Calculating raw ply stats...")
    df_ply_stats = conn.execute(f"""
        SELECT 
            move_ply,
            avg(CASE WHEN move_time = 0 THEN 1.0 ELSE 0.0 END) as mean_premove,
            stddev(CASE WHEN move_time = 0 THEN 1.0 ELSE 0.0 END) as std_premove,
            count(*) as n
        FROM {base_table}
        GROUP BY move_ply
        ORDER BY move_ply;
    """).df()
    
    # 4. Calculate Stats for Quantile Binned Ply
    print("Calculating quantile-binned stats...")
    df_qbin_stats = conn.execute(f"""
        SELECT 
            move_ply_qbin,
            avg(move_ply) as mean_move_ply,
            avg(CASE WHEN move_time = 0 THEN 1.0 ELSE 0.0 END) as mean_premove,
            stddev(CASE WHEN move_time = 0 THEN 1.0 ELSE 0.0 END) as std_premove,
            count(*) as n
        FROM {base_table}
        GROUP BY move_ply_qbin
        ORDER BY move_ply_qbin;
    """).df()
    
    conn.close()
    
    # 5. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    
    plot_raw_trend(
        axes[0], df_ply_stats, 
        x_col='move_ply', 
        y_col='mean_premove', 
        std_col='std_premove', 
        n_col='n',
        x_label="Move Ply",
        y_label="Probability of Pre-move", 
        show_legend=False
    )

    plot_qbin_stats(
        axes[1], df_qbin_stats,
        x_col='mean_move_ply',
        y_col='mean_premove',
        std_col='std_premove',
        n_col='n',
        x_label="Mean Move Ply in Quantile Bin",
        y_label="Probability of Pre-move",
        show_legend=False
    )
    
    # Single-line minimalist title
    title_text = f"Pre-move Probability | {n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)
    
    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "ply_premove")
    os.makedirs(figures_dir, exist_ok=True)
    
    output_plot = os.path.join(figures_dir, "combined.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Pre-move plot saved to {output_plot}")

if __name__ == "__main__":
    main()
