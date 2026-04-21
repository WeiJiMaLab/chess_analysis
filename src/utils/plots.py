import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.formula.api as smf
import colorsys
from .helpers import (
    apply_poster_style, 
    compute_metrics_by_qbin, 
    plot_metrics, 
    FONT_SIZE_LABEL, 
    FONT_SIZE_TICKS,
    MAIN_COLOR
)

def _label_one_line(label):
    """Collapse newlines / runs of whitespace for titles (axis labels may stay multiline)."""
    return " ".join(str(label).split())


def adjust_lightness(color, amount=0.5):
    """
    Adjusts the lightness of the input color by the given amount.
    Input can be hex or named color.
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    rgb = mcolors.hex2color(c)
    hls = colorsys.rgb_to_hls(*rgb)
    rgb_out = colorsys.hls_to_rgb(
        hls[0], max(0, min(1, hls[1] * amount)), hls[2]
    )
    return mcolors.to_hex(rgb_out)

def plot_standard_analysis_quad(
    df, 
    x_var, 
    y_var, 
    x_label, 
    y_label, 
    color=MAIN_COLOR, 
    save_path=None,
    n_bins=20,
    ply_var="move_ply"
):
    """
    Generate a standardized 2x2 quad view analysis plot.
    
    [0,0] - Hexbin Density (x vs y)
    [0,1] - Binned Trend (x vs y, x is raw units)
    [1,0] - Ply Stability (slope of x vs y by game phase)
    [1,1] - Quantile Trend (x vs y, x is rank 0-1)
    """
    apply_poster_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Pre-sort and identify valid data
    df_clean = df.dropna(subset=[x_var, y_var, ply_var]).copy()
    if len(df_clean) < 10:
        print(f"Warning: Not enough data for {x_var} vs {y_var}")
        plt.close()
        return

    # 1. Hexbin [0, 0]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#ffffff", color])
    axes[0, 0].hexbin(df_clean[x_var], df_clean[y_var], gridsize=25, cmap=cmap, mincnt=1)
    sns.regplot(x=x_var, y=y_var, data=df_clean, scatter=False, color=adjust_lightness(color, 0.7), ax=axes[0, 0])
    axes[0, 0].set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    
    # 2. Binned Trend [0, 1] (Raw X)
    df_sorted = df_clean.sort_values(x_var).copy()
    try:
        df_sorted["qbins"], qbin_edges = pd.qcut(df_sorted[x_var], q=n_bins, labels=False, retbins=True, duplicates="drop")
        df_tmp = df_sorted.copy()
        df_tmp["move_time"] = df_tmp[y_var] # helpers use "move_time" key
        metrics = compute_metrics_by_qbin(df_tmp, qbin_edges)
        plot_metrics(metrics, color=color, ax=axes[0, 1])
    except Exception as e:
        print(f"Warning: Binned mapping failed for {x_var}: {e}")
        
    axes[0, 1].set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel(f"Mean {y_label}", fontsize=FONT_SIZE_LABEL)
    
    # 3. Ply Stability [1, 0]
    results = []
    ply_counts = df_clean[ply_var].value_counts()
    # Use plys that have a decent number of samples
    min_samples = 5
    common_plys = ply_counts[ply_counts > min_samples].index
    for ply in sorted(common_plys):
        df_ply = df_clean[df_clean[ply_var] == ply]
        if len(df_ply) > 10:
            try:
                # OLS of y ~ x
                model = smf.ols(f"Q('{y_var}') ~ Q('{x_var}')", data=df_ply).fit()
                results.append({"ply": ply, "coeff": model.params[f"Q('{x_var}')"], "bse": model.bse[f"Q('{x_var}')"]})
            except:
                continue
    
    if results:
        df_res = pd.DataFrame(results)
        axes[1, 0].errorbar(
            df_res["ply"], df_res["coeff"], yerr=1.96 * df_res["bse"], 
            fmt='o', color=color, ecolor=adjust_lightness(color, 1.5), 
            elinewidth=3, capsize=0
        )
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_ylabel(r"Slope ($\beta$)", fontsize=FONT_SIZE_LABEL)
    
    # 4. Quantile Trend [1, 1] (Rank X 0-1)
    if 'metrics' in locals():
        metrics_rank = {k: list(v) for k, v in metrics.items()}
        # Normalize x to 0-1 rank
        metrics_rank["x"] = np.linspace(0, 1, len(metrics_rank["x"]))
        plot_metrics(metrics_rank, color=color, ax=axes[1, 1])
            
    axes[1, 1].set_xlabel(f"Quantile Rank\n({x_label})", fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel(f"Mean {y_label}", fontsize=FONT_SIZE_LABEL)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_distribution_side_by_side(df, raw_col="move_time", log_col="ln_move_time", color=MAIN_COLOR, save_path=None):
    """Replaces analyze_distribution from clocktime_movetime.py"""
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    # line_kws styles the KDE curve; kde_kws is for the internal KDE estimator only (bw, cut, etc.).
    kde_line_kws = {"linewidth": 2.5}
    
    # Raw
    sns.histplot(
        df[raw_col],
        bins=50,
        kde=True,
        color=color,
        alpha=0.4,
        ax=axes[0],
        element="step",
        line_kws=kde_line_kws,
    )
    axes[0].set_xlabel("Move Time (seconds)", fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel("Density", fontsize=FONT_SIZE_LABEL)
    
    # Log
    sns.histplot(
        df[log_col],
        bins=50,
        kde=True,
        color=color,
        alpha=0.4,
        ax=axes[1],
        element="step",
        line_kws=kde_line_kws,
    )
    axes[1].set_xlabel(r"Normalized Move Time: $\log(T)$", fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel("Density", fontsize=FONT_SIZE_LABEL)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
