"""
Heatmap analysis of the interaction between game stage (ply) and the number of possible moves.
Visualizes how thinking time varies as a function of both game phase and available legal moves.
Uses a custom isoluminant colormap with log-frequency alpha.
"""

import os
import sys
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory (src) to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, EPSILON
)
from utils.plots import get_isoluminant_cmap

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def plot_heatmap_with_alpha(ax, pivot_values, pivot_counts, cmap, label=r'Mean $\log(T)$'):
    """Plot a heatmap where cell opacity corresponds to the log-frequency of data points."""
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    
    # Ensure indices and columns are numeric for proper sorting and scaling
    pivot_values.columns = pivot_values.columns.astype(float)
    pivot_values.index = pivot_values.index.astype(float)
    pivot_counts.columns = pivot_counts.columns.astype(float)
    pivot_counts.index = pivot_counts.index.astype(float)

    # Sort: High Ply top (descending index), High Possible Moves right (ascending columns)
    pivot_values = pivot_values.sort_index(ascending=False).sort_index(axis=1, ascending=True)
    pivot_counts = pivot_counts.reindex(index=pivot_values.index, columns=pivot_values.columns)

    # Normalize values for colormap
    vals = pivot_values.values
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    
    rgba = cmap(norm(vals))
    
    # Normalize counts for alpha using log-scale
    counts = pivot_counts.values
    counts_clean = np.nan_to_num(counts, nan=0.0)
    c_max = np.nanmax(counts_clean)
    if c_max > 0:
        alpha = np.log1p(counts_clean) / np.log1p(c_max)
    else:
        alpha = np.zeros_like(counts_clean)
    
    rgba[..., 3] = alpha
    
    x_coords = pivot_values.columns
    y_coords = pivot_values.index
    dx = abs(x_coords[0] - x_coords[1]) if len(x_coords) > 1 else 1.0
    dy = abs(y_coords[0] - y_coords[1]) if len(y_coords) > 1 else 1.0
    
    extent = [
        x_coords.min() - dx/2, x_coords.max() + dx/2,
        y_coords.min() - dy/2, y_coords.max() + dy/2
    ]
    
    im = ax.imshow(rgba, extent=extent, aspect='auto', interpolation='nearest')
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb_label = r"Mean $\log(T)$, $\alpha = \log(1 + \text{freq})$"
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cb_label, fontsize=FONT_SIZE_LABEL - 8)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 6)
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    
    return im

def main():
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    base_table = "_selected_moves_nonzero_T"
    
    print("Getting processed dataset counts...")
    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    
    print("Aggregating grids (Ply vs Possible Moves)...")
    # Raw bins Heatmap
    df_raw_mean = conn.execute(f"""
        PIVOT (
            SELECT 
                floor(n_possible_moves / 2.0) * 2 as moves_bin,
                floor(move_ply / 5.0) * 5 as ply_bin,
                ln(move_time + {EPSILON}) as ln_move_time
            FROM {base_table}
            WHERE move_ply <= 150 AND n_possible_moves <= 100
        )
        ON moves_bin USING avg(ln_move_time) GROUP BY ply_bin
    """).df().set_index('ply_bin')

    df_raw_counts = conn.execute(f"""
        PIVOT (
            SELECT 
                floor(n_possible_moves / 2.0) * 2 as moves_bin,
                floor(move_ply / 5.0) * 5 as ply_bin
            FROM {base_table}
            WHERE move_ply <= 150 AND n_possible_moves <= 100
        )
        ON moves_bin USING count(*) GROUP BY ply_bin
    """).df().set_index('ply_bin')
    
    # Quantile Heatmap
    df_q_mean = conn.execute(f"""
        PIVOT (
            SELECT 
                ntile(20) over (order by n_possible_moves) as n_possible_moves_qbin,
                ntile(20) over (order by move_ply) as move_ply_qbin,
                ln(move_time + {EPSILON}) as ln_move_time
            FROM {base_table}
        )
        ON n_possible_moves_qbin USING avg(ln_move_time) GROUP BY move_ply_qbin
    """).df().set_index('move_ply_qbin')

    df_q_counts = conn.execute(f"""
        PIVOT (
            SELECT 
                ntile(20) over (order by n_possible_moves) as n_possible_moves_qbin,
                ntile(20) over (order by move_ply) as move_ply_qbin
            FROM {base_table}
        )
        ON n_possible_moves_qbin USING count(*) GROUP BY move_ply_qbin
    """).df().set_index('move_ply_qbin')
    conn.close()
    
    # Plotting
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(34, 11))
    iso_cmap = get_isoluminant_cmap()
    
    # Plot 1: Raw Heatmap
    plot_heatmap_with_alpha(axes[0], df_raw_mean, df_raw_counts, cmap=iso_cmap)
    axes[0].set_xlabel("Possible Moves", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[0].text(0.5, -0.2, "Few → Many", transform=axes[0].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='top')
    axes[0].set_ylabel("Move Ply", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[0].text(-0.2, 0.5, "Early → Late", transform=axes[0].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='center', rotation=90)
    
    # Plot 2: Quantile Heatmap
    plot_heatmap_with_alpha(axes[1], df_q_mean, df_q_counts, cmap=iso_cmap)
    axes[1].set_xlabel("Possible Moves Quantile", fontsize=FONT_SIZE_LABEL, labelpad=40)
    axes[1].text(0.5, -0.2, "Few → Many", transform=axes[1].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='top')
    axes[1].set_ylabel("Ply Quantile", fontsize=FONT_SIZE_LABEL, labelpad=60)
    axes[1].text(-0.15, 0.5, "Early → Late", transform=axes[1].transAxes, fontsize=FONT_SIZE_TICKS, ha='center', va='center', rotation=90)
    
    title_text = f"{n_games:,} games | {n_moves:,} moves | Possible Moves × Game Stage"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.05)
    
    plt.subplots_adjust(wspace=0.4)
    figures_dir = os.path.join(src_dir, "figures", "exploratory")
    os.makedirs(figures_dir, exist_ok=True)
    output_plot = os.path.join(figures_dir, "heatmap_ply_nmoves.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Possible moves heatmap saved to {output_plot}")

if __name__ == "__main__":
    main()
