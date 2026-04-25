"""
Heatmap analysis of the interaction between clock time, game stage (ply), and move time.
Visualizes how thinking time varies across the 2D space of (Clock, Ply).
Generates two versions: one with linear frequency alpha and one with log-frequency alpha.
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
from utils.plots import get_isoluminant_cmap

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def plot_heatmap_with_alpha(ax, pivot_values, pivot_counts, cmap, alpha_mode='log'):
    """
    Plot a heatmap where cell opacity corresponds to the frequency of data points.
    alpha_mode: 'linear' or 'log'
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    
    # Ensure indices and columns are numeric for proper sorting and scaling
    pivot_values.columns = pivot_values.columns.astype(float)
    pivot_values.index = pivot_values.index.astype(float)
    pivot_counts.columns = pivot_counts.columns.astype(float)
    pivot_counts.index = pivot_counts.index.astype(float)

    # Sort: High Ply top (descending index), High Clock left (descending columns)
    pivot_values = pivot_values.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    pivot_counts = pivot_counts.reindex(index=pivot_values.index, columns=pivot_values.columns)

    # Normalize values for colormap
    vals = pivot_values.values
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    
    # Get RGB colors
    rgba = cmap(norm(vals))
    
    # Normalize counts for alpha
    counts = pivot_counts.values
    counts_clean = np.nan_to_num(counts, nan=0.0)
    c_max = np.nanmax(counts_clean)
    
    if c_max > 0:
        if alpha_mode == 'log':
            alpha = np.log1p(counts_clean) / np.log1p(c_max)
            alpha_label = r"$\alpha = \log(1 + \text{freq})$"
        else:
            alpha = counts_clean / c_max
            alpha_label = r"$\alpha = \text{freq}$"
    else:
        alpha = np.zeros_like(counts_clean)
        alpha_label = r"$\alpha = 0$"
    
    # Set Alpha channel
    rgba[..., 3] = alpha
    
    # Calculate extent for proper coordinate scaling
    x_coords = pivot_values.columns
    y_coords = pivot_values.index
    
    dx = abs(x_coords[0] - x_coords[1]) if len(x_coords) > 1 else 1.0
    dy = abs(y_coords[0] - y_coords[1]) if len(y_coords) > 1 else 1.0
    
    extent = [
        x_coords.max() + dx/2, x_coords.min() - dx/2,
        y_coords.min() - dy/2, y_coords.max() + dy/2
    ]
    
    # Plot using imshow
    im = ax.imshow(rgba, extent=extent, aspect='auto', interpolation='nearest')
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cb_label = f"Mean log(T), {alpha_label}"
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cb_label, fontsize=FONT_SIZE_LABEL - 8)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 6)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    
    return im

def generate_figure(df_log_mean, df_log_counts, df_q_mean, df_q_counts, n_games, n_moves, alpha_mode, output_path):
    """Generate and save a single figure with 2 heatmaps."""
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(34, 11))
    
    iso_cmap = get_isoluminant_cmap()
    
    # Plot 1: Log-scale Heatmap
    plot_heatmap_with_alpha(axes[0], df_log_mean, df_log_counts, cmap=iso_cmap, alpha_mode=alpha_mode)
    axes[0].set_xlabel("log Time Left (s)", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[0].text(0.5, -0.2, "More → Less", transform=axes[0].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='top')
    axes[0].set_ylabel("Move Ply", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[0].text(-0.2, 0.5, "Early → Late", transform=axes[0].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='center', rotation=90)
    
    # Plot 2: Quantile Heatmap
    plot_heatmap_with_alpha(axes[1], df_q_mean, df_q_counts, cmap=iso_cmap, alpha_mode=alpha_mode)
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
        PIVOT (SELECT floor(ln_player_clock_time * 5)/5.0 as log_clock_bin, floor(move_ply/5.0)*5 as ply_bin, ln_move_time FROM {base_table} WHERE move_ply <= 150)
        ON log_clock_bin USING avg(ln_move_time) GROUP BY ply_bin
    """).df().set_index('ply_bin')

    df_log_counts_grid = conn.execute(f"""
        PIVOT (SELECT floor(ln_player_clock_time * 5)/5.0 as log_clock_bin, floor(move_ply/5.0)*5 as ply_bin FROM {base_table} WHERE move_ply <= 150)
        ON log_clock_bin USING count(*) GROUP BY ply_bin
    """).df().set_index('ply_bin')
    
    df_q_mean_grid = conn.execute(f"PIVOT {base_table} ON player_clock_qbin USING avg(ln_move_time) GROUP BY move_ply_qbin").df().set_index('move_ply_qbin')
    df_q_counts_grid = conn.execute(f"PIVOT {base_table} ON player_clock_qbin USING count(*) GROUP BY move_ply_qbin").df().set_index('move_ply_qbin')
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
