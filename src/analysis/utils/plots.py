"""Plots assume aggregates already computed elsewhere (often SQL-side via ``Analyzer``)."""

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from analysis.utils.helpers import (
    CONFIG, FONT_SIZE_LABEL, FONT_SIZE_TICKS, LEGEND_FONTSIZE, MAIN_COLOR, apply_poster_style,
)


def _wrap_long_label(label: str, threshold: int = 29) -> str:
    """Break a long axis label onto two lines at its first parenthetical.
    At FONT_SIZE_LABEL (52pt) a single-line label past ~29 chars is wider than a
    1x3 dashboard panel and bleeds into the neighboring panel's own label (re-measured
    2026-07-09 against the panels' current, narrower-than-original figsize — e.g.
    "Checks Available (|Imbalance| < 5)" at 34 chars and "Material Imbalance (Excl.
    Recapture)" at 36 chars both visibly collided with their neighbors at the old
    38-char threshold; "Material Imbalance (Absolute)" at 29 chars did not)."""
    if len(label) <= threshold or "(" not in label:
        return label
    head, _, tail = label.partition("(")
    return f"{head.rstrip()}\n({tail}"


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
    mass_col: str = "is_mass",
):
    """Mean and CI band over quantile-bin aggregates (same SEM recipe as ``plot_raw_trend``).

    If ``normalized`` is True, x positions are replaced by ranks ``1..K`` scaled to ``(0, 1]``
    for a quantile-rank axis (default x-axis label: "Quantile Rank").

    If ``df`` carries a boolean ``mass_col`` (set by ``Analyzer._is_mass_qbin`` for a
    zero_inflated/edge_mass covariate's dedicated point-mass row(s), e.g. Gain==0),
    those rows are pulled OUT of the interior circle-marker/line and drawn as their
    own ✕ marker + CI whisker — visually flagging a hurdle/mass point as a
    qualitatively different kind of point, not an interior quantile.
    """
    apply_poster_style()
    df = df.sort_values(x_col).copy()
    is_mass = (df[mass_col].to_numpy(dtype=bool) if mass_col in df.columns
               else np.zeros(len(df), dtype=bool))

    if normalized:
        x_vals = np.arange(1, len(df) + 1) / len(df)
        if x_label is None:
            x_label = "Quantile Rank"
    else:
        x_vals = df[x_col].to_numpy()

    def _ci(mask):
        sub = df[mask]
        return 1.96 * sub[std_col].to_numpy() / np.sqrt(sub[n_col].to_numpy())

    interior, mass = ~is_mass, is_mass
    y_mean = df.loc[interior, y_col]
    ci_y = _ci(interior)
    # Fixed marker/line weight: every quantile-binned trend should read with the same
    # visual weight regardless of how many bins THIS covariate happens to produce
    # (a boolean collapses to 2 bins; a continuous covariate keeps ~10).
    ax.plot(x_vals[interior], y_mean, marker='o', color=color, lw=3, markersize=6, label=label)
    fb_kwargs = {"color": color, "alpha": 0.2}
    if ci_legend_label is not None:
        fb_kwargs["label"] = ci_legend_label
    ax.fill_between(x_vals[interior], y_mean - ci_y, y_mean + ci_y, **fb_kwargs)

    if mass.any():
        ax.errorbar(x_vals[mass], df.loc[mass, y_col], yerr=_ci(mass),
                    marker='x', ms=14, mew=3, linestyle='none',
                    color=color, ecolor=color, elinewidth=2.2, capsize=5, zorder=5)

    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)
    # Fewer, cleaner ticks — matters more now that dashboards render at a smaller
    # figsize (see board.py's bivariate_analysis/move_time_summary): a locator set
    # here is transparently replaced by the scale's own default (e.g. LogLocator) if
    # the caller applies set_xscale("log")/set_yscale("log") afterward, so this is
    # safe regardless of which axis ends up log-scaled. Mirrors the MaxNLocator
    # convention already used in plot_heatmap_with_alpha (this module) below.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), frameon=False)



def _annotate_n(fig, n: int) -> None:
    """``n = {count:,}`` at the bottom-left of the figure, below everything else —
    the only headline text a dashboard carries (house style omits titles/suptitles
    uniformly; see ``outputs/reports/reference.md`` "Plot standards"). Sized to
    read as part of the figure (same size as a legend), not a footnote. Position
    is computed from the actual rendered tight bounding box of every axes (ticks,
    axis labels, and any below-axes legend all included), so it clears whatever
    sits lowest in THIS figure — a two-line x-label, a legend, both — rather than
    a fixed guess that either collides with tall content or leaves excess
    whitespace for short content. ``bbox_inches="tight"`` on save still expands
    the canvas to include it, however far below y=0 it lands."""
    fig.canvas.draw()  # force a layout pass so tight bboxes are accurate
    renderer = fig.canvas.get_renderer()
    min_y = min(
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
        for ax in fig.axes
    )
    fig.text(0.01, min_y - 0.02, f"n = {n:,}", fontsize=LEGEND_FONTSIZE, ha="left", va="top")


def _draw_feature_histogram(
    conn, table, col, kind, clip, ax, *,
    name: str | None = None,
    zero_inflated: bool = False,
    zero_threshold: float = 0.0,
    extra_where: str | None = None,
    n_hist_bins: int = 50,
):
    """Marginal distribution of one covariate (SQL-aggregated), shared by
    board.py's ``bivariate_analysis``/``feature_histograms`` and engine.py's
    per-signal tree dashboards. ``kind`` ∈ {"cont", "disc", "bin"}:
      disc/bin -> one bar per distinct value (bin -> False/True tick labels).
      cont     -> ``n_hist_bins`` fixed-width bins over ``clip`` (auto-derived
                  from the 0.5th/99.5th percentile when ``clip`` is None — the
                  engine tree signals vary in scale by value unit, pwin vs cp).
    ``zero_inflated``: for a ``cont`` covariate with a point mass at/near
    ``zero_threshold`` (e.g. Gain==0, ~55% of rows), that mass is split OUT of
    the continuous binning and drawn as its own isolated bar (solid + black
    edge) rather than being smeared across a couple of bins next to it —
    the histogram-panel analogue of the ✕-marker split on the trend panels.
    """
    extra = f" AND ({extra_where})" if extra_where else ""
    where = (f"WHERE {col} BETWEEN {clip[0]} AND {clip[1]}{extra}" if clip
              else f"WHERE {col} IS NOT NULL{extra}")
    if kind in ("disc", "bin"):
        g = conn.execute(
            f"SELECT {col}::DOUBLE AS value, count(*) AS n FROM {table} {where} GROUP BY 1 ORDER BY 1"
        ).df()
        ax.bar(g.value, g.n / g.n.sum(), width=(0.4 if kind == "bin" else 0.9),
               color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
        if kind == "bin":
            ax.set_xticks([0, 1]); ax.set_xticklabels(["False", "True"])
    else:
        lo, hi = clip if clip else conn.execute(
            f"SELECT quantile_cont({col}, 0.005), quantile_cont({col}, 0.995) FROM {table} {where}"
        ).fetchone()
        width = (hi - lo) / n_hist_bins
        if zero_inflated:
            zero_pred = f"abs({col}) <= {zero_threshold}"
            n_total = conn.execute(f"SELECT count(*) FROM {table} {where}").fetchone()[0]
            n_zero = conn.execute(f"SELECT count(*) FROM {table} {where} AND {zero_pred}").fetchone()[0]
            g = conn.execute(
                f"SELECT least({n_hist_bins - 1}, greatest(0, floor(({col} - {lo}) / {width})))::INT AS b, "
                f"count(*) AS n FROM {table} {where} AND NOT ({zero_pred}) GROUP BY 1 ORDER BY 1"
            ).df()
            if n_total:
                ax.bar(lo + (g.b + 0.5) * width, g.n / n_total / width, width=width,
                       color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
                ax.bar([zero_threshold], [n_zero / n_total / width], width=width,
                       color=MAIN_COLOR, alpha=1.0, edgecolor="black", linewidth=2.0, zorder=3)
        else:
            g = conn.execute(
                f"SELECT least({n_hist_bins - 1}, floor(({col} - {lo}) / {width}))::INT AS b, count(*) AS n "
                f"FROM {table} {where} GROUP BY 1 ORDER BY 1"
            ).df()
            ax.bar(lo + (g.b + 0.5) * width, g.n / g.n.sum() / width, width=width,
                   color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
    ax.set(ylabel="Density")
    if name:
        ax.set_xlabel(_wrap_long_label(name))
    # Fewer ticks — same reasoning as plot_qbin_stats' MaxNLocator calls (this
    # histogram panel shares the same smaller-figsize dashboards). Skip the "bin"
    # (boolean False/True) case: its 2 explicit xticks above are already minimal.
    if kind != "bin":
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def plot_histogram_from_bins(
    ax, df_bins,
    left_col="bin_left", right_col="bin_right", count_col="n",
    x_label=None, y_label="Count", color=MAIN_COLOR,
    mean=None, median=None,
):
    """Aligned-edge bar histogram from SQL histogram tables.

    Draws optional dashed vertical lines at ``mean`` and ``median`` when provided.
    No-op if ``df_bins`` is empty.
    """
    apply_poster_style()
    if df_bins.empty:
        return
    widths = df_bins[right_col] - df_bins[left_col]
    ax.bar(df_bins[left_col], df_bins[count_col], width=widths, align="edge",
           color=color, alpha=0.5, edgecolor=color, linewidth=1.5)
    if mean is not None:
        ax.axvline(mean, color="black", linestyle="--", lw=2.5, label=f"Mean = {mean:.3f}")
    if median is not None:
        ax.axvline(median, color="dimgray", linestyle=":", lw=2.5, label=f"Median = {median:.3f}")
    if mean is not None or median is not None:
        ax.legend(fontsize=LEGEND_FONTSIZE)
    if x_label:
        ax.set_xlabel(x_label, fontsize=FONT_SIZE_LABEL)
    if y_label:
        ax.set_ylabel(y_label, fontsize=FONT_SIZE_LABEL)


def get_isoluminant_cmap(name="isoluminant_azure", h1=0.58, h2=None, s1=0.0, s2=0.85, lightness=0.6, saturation=None):
    """Matplotlib ``ListedColormap`` at fixed HSL lightness.

    Varies hue from ``h1`` to ``h2`` and saturation from ``s1`` to ``s2`` (``saturation`` pins both).
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
    """Heatmap: facecolor from ``pivot_values``, per-cell alpha from ``pivot_counts``.

    ``alpha_mode``: ``"log"`` uses ``log1p`` normalization vs max count; ``"linear"`` uses raw fractions.
    ``imshow_aspect`` is forwarded to ``imshow`` (use ``"equal"`` for square quantile×quantile cells).

    Returns the ``ScalarMappable`` used for the colorbar.
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


def highlight_corr_row(ax, n_cols, idx=0):
    """Bold row ``idx``'s cells and its tick labels to flag the response variable
    (log RT, kept first) as the row that matters — no outline/colour, just weight.
    Assumes cell text was added row-major (``for i: for j: ax.text(...)``), so
    ``ax.texts[idx*n_cols : idx*n_cols+n_cols]`` are that row's annotations."""
    for labs in (ax.get_yticklabels(), ax.get_xticklabels()):
        if len(labs) > idx:
            labs[idx].set_fontweight("bold")
    for j in range(n_cols):
        k = idx * n_cols + j
        if k < len(ax.texts):
            ax.texts[k].set_fontweight("bold")


def _typed_path(base_dir: str, kind: str, filename: str) -> str:
    """``base_dir/<kind>/<filename>``, creating the ``<kind>`` (pdf/png/csv) subdir."""
    d = os.path.join(base_dir, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def save_pdf_png(fig, base_dir: str, base: str, *, dpi: int = 300, **savefig_kwargs) -> str:
    """Write ``base.pdf`` -> ``base_dir/pdf/`` and ``base.png`` -> ``base_dir/png/``.

    PDF and PNG live in separate type subfolders (pdf/ png/); CSV tables go to
    csv/ via ``save_table``. Extra kwargs (``bbox_extra_artists``, ``pad_inches``,
    …) are forwarded to ``fig.savefig``. Returns the PDF path."""
    pdf = _typed_path(base_dir, "pdf", f"{base}.pdf")
    png = _typed_path(base_dir, "png", f"{base}.png")
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
    plt.close(fig)
    print(f"Saved figures: {pdf} and {png}")
    return pdf


def save_table(df, base_dir: str, filename: str, **to_csv_kwargs) -> str:
    """Write a CSV table to ``base_dir/csv/<filename>`` (sibling of pdf/ and png/)."""
    path = _typed_path(base_dir, "csv", filename)
    df.to_csv(path, **to_csv_kwargs)
    print(f"Saved table: {path}")
    return path


def save_figure(fig, category: str, filename: str) -> str:
    """Save a figure under figures/<category>/{pdf,png}/<name> — PDF and PNG in
    separate type subfolders. Creates the subdirectories if needed."""
    if category not in ("board", "engine"):
        raise ValueError(f"Invalid figure category: {category}. Must be 'board' or 'engine'.")
    base, _ = os.path.splitext(filename)
    base_dir = os.path.join(CONFIG["figures_dir"], category)
    return save_pdf_png(fig, base_dir, base, dpi=300, pad_inches=0.3)

