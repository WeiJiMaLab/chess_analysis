"""
Heatmap analysis of the interaction between clock time, game stage (ply), and move time.
Visualizes how thinking time varies across the 2D space of (Clock, Ply).
Generates two versions: one with linear frequency alpha and one with log-frequency alpha.
"""

import os
import sys
import duckdb
import matplotlib.pyplot as plt

# Add the parent directory (src) to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, EPSILON
)
from utils.plots import get_isoluminant_cmap, plot_heatmap_with_alpha

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def generate_figure(df_log_mean, df_log_counts, df_q_mean, df_q_counts, n_games, n_moves, alpha_mode, output_path):
    """Generate and save a single figure with 2 heatmaps."""
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(34, 11))
    
    iso_cmap = get_isoluminant_cmap()
    
    # Plot 1: Log-scale Heatmap
    plot_heatmap_with_alpha(
        axes[0],
        df_log_mean,
        df_log_counts,
        iso_cmap,
        alpha_mode=alpha_mode,
        value_label="Mean log(T)",
    )
    axes[0].set_xlabel("log Time Left (s)", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[0].text(0.5, -0.2, "More → Less", transform=axes[0].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='top')
    axes[0].set_ylabel("Move Ply", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[0].text(-0.2, 0.5, "Early → Late", transform=axes[0].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='center', rotation=90)
    
    # Plot 2: Quantile Heatmap
    plot_heatmap_with_alpha(
        axes[1],
        df_q_mean,
        df_q_counts,
        iso_cmap,
        alpha_mode=alpha_mode,
        value_label="Mean log(T)",
    )
    axes[1].set_xlabel("Time Left Quantile", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[1].text(0.5, -0.2, "More → Less", transform=axes[1].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='top')
    axes[1].set_ylabel("Ply Quantile", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[1].text(-0.15, 0.5, "Early → Late", transform=axes[1].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='center', rotation=90)
    
    title_text = f"{n_games:,} games | {n_moves:,} moves | Alpha Mode: {alpha_mode.capitalize()}"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.05)
    
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def main():
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    base_table = "_selected_moves_nonzero_T"
    
    print("Getting processed dataset counts...")
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    
    # Aggregation
    print("Aggregating grids...")
    df_log_mean_grid = conn.execute(f"""
        PIVOT (
            SELECT 
                floor(ln(player_clock_time + {EPSILON}) * 5)/5.0 as log_clock_bin, 
                floor(move_ply/5.0)*5 as ply_bin, 
                ln(move_time + {EPSILON}) as ln_move_time 
            FROM {base_table} WHERE move_ply <= 150
        )
        ON log_clock_bin USING avg(ln_move_time) GROUP BY ply_bin
    """).df().set_index('ply_bin')

    df_log_counts_grid = conn.execute(f"""
        PIVOT (
            SELECT 
                floor(ln(player_clock_time + {EPSILON}) * 5)/5.0 as log_clock_bin, 
                floor(move_ply/5.0)*5 as ply_bin 
            FROM {base_table} WHERE move_ply <= 150
        )
        ON log_clock_bin USING count(*) GROUP BY ply_bin
    """).df().set_index('ply_bin')
    
    df_q_mean_grid = conn.execute(f"""
        PIVOT (
            SELECT 
                ntile(20) over (order by player_clock_time) as player_clock_qbin,
                ntile(20) over (order by move_ply) as move_ply_qbin,
                ln(move_time + {EPSILON}) as ln_move_time
            FROM {base_table}
        ) 
        ON player_clock_qbin USING avg(ln_move_time) GROUP BY move_ply_qbin
    """).df().set_index('move_ply_qbin')

    df_q_counts_grid = conn.execute(f"""
        PIVOT (
            SELECT 
                ntile(20) over (order by player_clock_time) as player_clock_qbin,
                ntile(20) over (order by move_ply) as move_ply_qbin
            FROM {base_table}
        ) 
        ON player_clock_qbin USING count(*) GROUP BY move_ply_qbin
    """).df().set_index('move_ply_qbin')
    conn.close()
    
    # Save Figures
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "exploratory")
    os.makedirs(figures_dir, exist_ok=True)
    
    generate_figure(df_log_mean_grid, df_log_counts_grid, df_q_mean_grid, df_q_counts_grid, n_games, n_moves, 'linear', os.path.join(figures_dir, "heatmap_clock_ply_linear.png"))
    generate_figure(df_log_mean_grid, df_log_counts_grid, df_q_mean_grid, df_q_counts_grid, n_games, n_moves, 'log', os.path.join(figures_dir, "heatmap_clock_ply_log.png"))
    
    # Save CSVs
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "presentations", "cmc-overview", "public", "data")
    os.makedirs(data_dir, exist_ok=True)
    df_log_mean_grid.to_csv(os.path.join(data_dir, "heatmap_log.csv"))
    df_q_mean_grid.to_csv(os.path.join(data_dir, "heatmap_quantile.csv"))

if __name__ == "__main__":
    main()
