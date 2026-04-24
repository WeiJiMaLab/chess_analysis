"""
Heatmap analysis of the interaction between clock time, game stage (ply), and move time.
Visualizes how thinking time varies across the 2D space of (Clock, Ply).
"""

import os
import sys
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory (src) to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR
)

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    # 1. Setup paths
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    base_table = "_selected_moves_nonzero_T"
    
    # 2. Get dataset metadata
    print("Getting processed dataset counts...")
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")
    
    # 3. Log-scale Heatmap (Raw-ish)
    # We bin log_clock into 0.2 units and ply into 5-unit increments
    print("Aggregating log-scale grid...")
    df_log_grid = conn.execute(f"""
        SELECT 
            floor(ln_player_clock_time * 5) / 5.0 as log_clock_bin,
            floor(move_ply / 5.0) * 5 as ply_bin,
            avg(ln_move_time) as mean_ln_move_time,
            count(*) as n
        FROM {base_table}
        WHERE move_ply <= 150
        GROUP BY 1, 2
        HAVING n >= 100
    """).df()
    
    # 4. Quantile Heatmap
    print("Aggregating quantile grid...")
    df_q_grid = conn.execute(f"""
        SELECT 
            player_clock_qbin,
            move_ply_qbin,
            avg(ln_move_time) as mean_ln_move_time,
            count(*) as n
        FROM {base_table}
        GROUP BY 1, 2
    """).df()
    
    conn.close()
    
    # 5. Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(28, 11))
    
    # Plot 1: Log-scale Heatmap
    pivot_log = df_log_grid.pivot(index='ply_bin', columns='log_clock_bin', values='mean_ln_move_time')
    # Ensure ply labels are integers
    pivot_log.index = pivot_log.index.astype(int)
    # Sort: High Ply top, Low Ply bottom; High Clock left, Low Clock right
    pivot_log = pivot_log.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    
    sns.heatmap(
        pivot_log, 
        ax=axes[0], 
        cmap='Blues', 
        cbar_kws={'label': r'Mean $\log(T)$'}
    )
    # X-axis label with sublabel
    axes[0].set_xlabel("log(Clock Time)", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[0].text(0.5, -0.26, r"Early $\rightarrow$ Late", transform=axes[0].transAxes, 
                 fontsize=FONT_SIZE_TICKS - 5, ha='center', va='top')
    
    # Y-axis label with sublabel
    axes[0].set_ylabel("Move Ply", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[0].text(-0.22, 0.5, r"Early $\rightarrow$ Late", transform=axes[0].transAxes, 
                 fontsize=FONT_SIZE_TICKS - 5, ha='center', va='center', rotation=90)
    
    axes[0].set_title("Log-Scale Interaction", fontsize=FONT_SIZE_LABEL, pad=30)
    
    # Plot 2: Quantile Heatmap
    pivot_q = df_q_grid.pivot(index='move_ply_qbin', columns='player_clock_qbin', values='mean_ln_move_time')
    # Force integer labels for quantiles
    pivot_q.index = pivot_q.index.astype(int)
    pivot_q.columns = pivot_q.columns.astype(int)
    pivot_q = pivot_q.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    
    sns.heatmap(
        pivot_q, 
        ax=axes[1], 
        cmap='Blues', 
        cbar_kws={'label': r'Mean $\log(T)$'}
    )
    # X-axis label with sublabel
    axes[1].set_xlabel("Clock Quantile", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[1].text(0.5, -0.22, r"Early $\rightarrow$ Late", transform=axes[1].transAxes, 
                 fontsize=FONT_SIZE_TICKS - 5, ha='center', va='top')
    
    # Y-axis label with sublabel
    axes[1].set_ylabel("Ply Quantile", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[1].text(-0.15, 0.5, r"Early $\rightarrow$ Late", transform=axes[1].transAxes, 
                 fontsize=FONT_SIZE_TICKS - 5, ha='center', va='center', rotation=90)
    
    axes[1].set_title("Quantile Interaction", fontsize=FONT_SIZE_LABEL, pad=30)
    
    # Single-line minimalist title (metadata only)
    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.05)
    
    plt.subplots_adjust(wspace=0.6)
    # Use tight_layout only for the padding, or manually adjust
    # plt.tight_layout() 
    figures_dir = os.path.join(src_dir, "figures", "exploratory")
    os.makedirs(figures_dir, exist_ok=True)
    
    output_plot = os.path.join(figures_dir, "heatmap_clock_ply.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap plot saved to {output_plot}")
    
    # 6. Save CSVs for 3js Visualization
    data_dir = os.path.join(src_dir, "..", "presentations", "cmc-overview", "public", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    log_csv = os.path.join(data_dir, "heatmap_log.csv")
    q_csv = os.path.join(data_dir, "heatmap_quantile.csv")
    
    pivot_log.to_csv(log_csv)
    pivot_q.to_csv(q_csv)
    
    print(f"📊 CSV data saved to {data_dir}")

if __name__ == "__main__":
    main()
