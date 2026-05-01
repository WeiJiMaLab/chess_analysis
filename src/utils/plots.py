import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .helpers import FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR, apply_poster_style
except (ImportError, ValueError):
    from helpers import FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR, apply_poster_style


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


def plot_heatmap_with_alpha(
    ax,
    pivot_values,
    pivot_counts,
    cmap,
    *,
    alpha_mode="log",
    value_label="Mean Y",
    imshow_aspect="auto",
):
    """
    Heatmap: cell color encodes ``pivot_values``; alpha encodes ``pivot_counts`` (frequency).
    ``alpha_mode`` is ``\"log\"`` (``log1p`` normalized) or ``\"linear\"``.
    ``imshow_aspect``: passed to ``imshow`` (``\"equal\"`` gives square cells on quantile×quantile grids).
    """
    from matplotlib.cm import ScalarMappable

    pivot_values = pivot_values.copy()
    pivot_counts = pivot_counts.copy()
    pivot_values.columns = pivot_values.columns.astype(float)
    pivot_values.index = pivot_values.index.astype(float)
    pivot_counts.columns = pivot_counts.columns.astype(float)
    pivot_counts.index = pivot_counts.index.astype(float)

    pivot_values = pivot_values.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    pivot_counts = pivot_counts.reindex(index=pivot_values.index, columns=pivot_values.columns)

    vals = pivot_values.values
    v_min, v_max = np.nanmin(vals), np.nanmax(vals)
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

    rgba = cmap(norm(vals))

    counts = pivot_counts.values
    counts_clean = np.nan_to_num(counts, nan=0.0)
    c_max = np.nanmax(counts_clean)

    if c_max > 0:
        if alpha_mode == "log":
            alpha = np.log1p(counts_clean) / np.log1p(c_max)
            alpha_label = r"$\alpha = \log(1 + \text{freq})$"
        else:
            alpha = counts_clean / c_max
            alpha_label = r"$\alpha = \text{freq}$"
    else:
        alpha = np.zeros_like(counts_clean)
        alpha_label = r"$\alpha = 0$"

    rgba[..., 3] = alpha

    x_coords = pivot_values.columns
    y_coords = pivot_values.index

    dx = abs(x_coords[0] - x_coords[1]) if len(x_coords) > 1 else 1.0
    dy = abs(y_coords[0] - y_coords[1]) if len(y_coords) > 1 else 1.0

    extent = [
        x_coords.max() + dx / 2,
        x_coords.min() - dx / 2,
        y_coords.min() - dy / 2,
        y_coords.max() + dy / 2,
    ]

    ax.imshow(rgba, extent=extent, aspect=imshow_aspect, interpolation="nearest")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cb_label = f"{value_label}, {alpha_label}"
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cb_label, fontsize=FONT_SIZE_LABEL - 8)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS - 6)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))

    return sm
