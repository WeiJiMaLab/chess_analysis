import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.formula.api as smf
import colorsys
try:
    from .helpers import (
        apply_poster_style, 
        compute_metrics_by_qbin, 
        plot_metrics, 
        FONT_SIZE_LABEL, 
        FONT_SIZE_TICKS,
        MAIN_COLOR
    )
except (ImportError, ValueError):
    from helpers import (
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
    min_samples = 5
    common_plys = ply_counts[ply_counts > min_samples].index
    for ply in sorted(common_plys):
        df_ply = df_clean[df_clean[ply_var] == ply]
        if len(df_ply) > 10:
            try:
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
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    kde_line_kws = {"linewidth": 2.5}
    
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

def plot_raw_trend(
    ax,
    df,
    x_col,
    y_col,
    std_col,
    n_col,
    x_label=None,
    y_label=None,
    color=MAIN_COLOR,
    label="Mean",
    min_n=30,
    show_legend=True,
    *,
    ci_legend_label="95% CI",
):
    apply_poster_style()
    df = df[df[n_col] >= min_n].copy()
    df = df.sort_values(x_col)
    sem = df[std_col] / np.sqrt(df[n_col])
    ci_y = 1.96 * sem
    y_mean = df[y_col]
    y_lower = y_mean - ci_y
    y_upper = y_mean + ci_y
    ax.plot(df[x_col], y_mean, color=color, lw=3, label=label)
    fb_kwargs = {"color": color, "alpha": 0.2}
    if ci_legend_label is not None:
        fb_kwargs["label"] = ci_legend_label
    ax.fill_between(df[x_col], y_lower, y_upper, **fb_kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    if show_legend:
        ax.legend(fontsize=FONT_SIZE_TICKS)

def plot_qbin_stats(
    ax,
    df,
    x_col='mean_x',
    y_col='mean_y',
    std_col='std_y',
    n_col='n',
    x_label=None,
    y_label=None,
    color=MAIN_COLOR,
    label="Mean",
    normalized=False,
    show_legend=True,
    *,
    ci_legend_label="95% CI",
):
    apply_poster_style()
    df = df.sort_values(x_col).copy()
    sem = df[std_col] / np.sqrt(df[n_col])
    ci_y = 1.96 * sem
    y_mean = df[y_col]
    y_lower = y_mean - ci_y
    y_upper = y_mean + ci_y
    if normalized:
        x_vals = np.arange(1, len(df) + 1) / len(df)
        if x_label is None:
            x_label = "Quantile Rank"
    else:
        x_vals = df[x_col]
    ax.plot(x_vals, y_mean, marker='o', color=color, lw=3, markersize=12, label=label)
    fb_kwargs = {"color": color, "alpha": 0.2}
    if ci_legend_label is not None:
        fb_kwargs["label"] = ci_legend_label
    ax.fill_between(x_vals, y_lower, y_upper, **fb_kwargs)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    if show_legend:
        ax.legend(fontsize=FONT_SIZE_TICKS)

def plot_beta_vs_ply(ax, df, ply_col='move_ply', beta_col='beta', se_col='beta_se', max_ply=150, title=None):
    apply_poster_style()
    df = df[df[ply_col] <= max_ply].copy()
    ci_beta = 1.96 * df[se_col]
    ax.plot(df[ply_col], df[beta_col], color=MAIN_COLOR, lw=3, label=r"Slope ($\beta$)")
    ax.fill_between(df[ply_col], df[beta_col] - ci_beta, df[beta_col] + ci_beta, color=MAIN_COLOR, alpha=0.2, label="95% CI")
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"Sensitivity ($\beta$)", fontsize=FONT_SIZE_LABEL)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_LABEL)

def plot_subset_scatterplot(ax, df, x_col, y_col, x_label=None, y_label=None, n=10000, color=MAIN_COLOR, alpha=0.1, s=10):
    apply_poster_style()
    if len(df) > n:
        df_sub = df.sample(n=n, random_state=42)
    else:
        df_sub = df
    ax.scatter(df_sub[x_col], df_sub[y_col], color=color, alpha=alpha, s=s)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)

def plot_histogram_from_bins(ax, df_bins, left_col="bin_left", right_col="bin_right", count_col="n", x_label=None, y_label="Count", color=MAIN_COLOR):
    apply_poster_style()
    if df_bins.empty:
        return
    widths = df_bins[right_col] - df_bins[left_col]
    ax.bar(df_bins[left_col], df_bins[count_col], width=widths, align="edge", color=color, alpha=0.5, edgecolor=color, linewidth=1.5)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)


def get_isoluminant_cmap(name="isoluminant_azure", h1=0.58, h2=None, s1=0.0, s2=0.85, lightness=0.6, saturation=None):
    """
    Generate an isoluminant colormap with constant HSL lightness.
    Supports either Hue transitions or Saturation (Gray-to-Color) transitions.
    """
    import colorsys
    from matplotlib.colors import ListedColormap
    
    # Handle backward compatibility and defaults
    if h2 is None: h2 = h1
    if s2 is None: s2 = 0.85
    if saturation is not None:
        s1 = s2 = saturation
        
    hues = np.linspace(h1, h2, 256)
    sats = np.linspace(s1, s2, 256)
    
    colors = [colorsys.hls_to_rgb(h, lightness, s) for h, s in zip(hues, sats)]
    return ListedColormap(colors, name=name)
