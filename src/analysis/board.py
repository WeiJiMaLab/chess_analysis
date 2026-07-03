"""
Unified board-level (no-model) response time analyses (DuckDB dashboards).
The log(RT) distribution + normal QQ plot, bivariate RT-vs-feature dashboards
for ply, legal moves, and clock time left, plus a Spearman correlation matrix
over those features and log(RT).

All analyses read a ply-windowed view (move_ply in [min_ply, max_ply] from the
active config) so the filter is applied on arrival and ply tertiles are
conditioned on the window. All plots are saved (PDF + PNG) under
<figures_dir>/board/ (figures_dir is namespaced by run_name in the config).
"""

import argparse
import os

# Resolve --config BEFORE importing analysis.utils so helpers loads the right
# config file (its CONFIG global is built at import from the CONFIG env var).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config")
_cfg, _ = _pre.parse_known_args()
if _cfg.config:
    os.environ["CONFIG"] = _cfg.config

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Align imports with the src/analysis package structure
from analysis.utils import Variable, Analyzer
from analysis.utils.helpers import (
    apply_poster_style,
    db_connection,
    create_ply_windowed_views,
    WIN_PROCESSED_MOVES_NONZERO,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    MAIN_COLOR,
    CONFIG,
)
from analysis.utils.plots import (
    highlight_corr_row,
    plot_histogram_from_bins,
    save_figure,
    save_table,
)
from analysis.utils.selected_db import (
    SELECTED_DB_DEFAULT,
)


def run_move_time_summary(conn):
    """Response time distribution: log(RT) histogram (left) + normal QQ plot (right)."""
    n_bins = CONFIG["response_time_histogram_bins"]
    n_qq = CONFIG["qq_plot_quantile_probes"]

    n_moves = conn.execute(f"SELECT count(*) FROM {WIN_PROCESSED_MOVES_NONZERO}").fetchone()[0]
    print(f"Running response time summary: n = {n_moves:,} moves")

    conn.execute(
        "CREATE OR REPLACE TEMPORARY VIEW _summary_view AS "
        f"SELECT ln(move_time) AS ln_move_time FROM {WIN_PROCESSED_MOVES_NONZERO}"
    )

    # SQL-side histogram binning
    conn.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE _lmt_bins AS
        WITH stats AS (SELECT min(ln_move_time) AS min_v, max(ln_move_time) AS max_v FROM _summary_view),
        bins AS (
            SELECT i AS bin_idx,
                   min_v + (max_v - min_v) * i / {n_bins} AS bin_left,
                   min_v + (max_v - min_v) * (i + 1) / {n_bins} AS bin_right
            FROM stats, range({n_bins}) AS t(i)
        )
        SELECT b.bin_idx, b.bin_left, b.bin_right, count(v.ln_move_time) AS n
        FROM bins b
        LEFT JOIN _summary_view v
            ON v.ln_move_time >= b.bin_left
           AND (v.ln_move_time < b.bin_right OR (b.bin_idx = {n_bins - 1} AND v.ln_move_time <= b.bin_right))
        GROUP BY b.bin_idx, b.bin_left, b.bin_right
        ORDER BY b.bin_idx
    """)
    df_lmt = conn.execute("SELECT * FROM _lmt_bins ORDER BY bin_idx").df()

    # Moments + empirical quantiles for the QQ plot (all SQL-side).
    mean, std = conn.execute("SELECT avg(ln_move_time), stddev(ln_move_time) FROM _summary_view").fetchone()
    probs = (np.arange(1, n_qq + 1)) / (n_qq + 1)
    emp_q = np.array(conn.execute(
        "SELECT quantile_cont(ln_move_time, ?) FROM _summary_view", [probs.tolist()]
    ).fetchone()[0], dtype=float)

    theo_q = stats.norm.ppf(probs)  # standard-normal quantiles
    ref = mean + std * theo_q       # reference line if log(RT) ~ Normal(mean, std)

    def _weighted_median(df):
        cumsum = df["n"].cumsum()
        idx = (cumsum >= df["n"].sum() / 2).idxmax()
        return float((df["bin_left"].iloc[idx] + df["bin_right"].iloc[idx]) / 2)

    med_log = _weighted_median(df_lmt)

    apply_poster_style()
    fig, (ax_h, ax_q, ax_p) = plt.subplots(1, 3, figsize=(34, 11), constrained_layout=True)

    # --- Panel 1: RT distribution in SECONDS on a log x-axis (bins are uniform in
    # ln(RT), so exponentiating the edges gives geometric bins that read evenly on
    # a log axis). Mean/median drawn in seconds. ---
    sec = df_lmt.assign(left_s=np.exp(df_lmt.bin_left), right_s=np.exp(df_lmt.bin_right))
    ax_h.bar(sec.left_s, sec.n, width=(sec.right_s - sec.left_s), align="edge",
             color=MAIN_COLOR, alpha=0.5, edgecolor=MAIN_COLOR, linewidth=1.2)
    ax_h.set_xscale("log")
    ax_h.axvline(np.exp(mean), color="black", ls="--", lw=2.5, label=f"mean = {np.exp(mean):.1f}s")
    ax_h.axvline(np.exp(med_log), color="dimgray", ls=":", lw=2.5, label=f"median = {np.exp(med_log):.1f}s")
    ax_h.set_xlabel("RT (s, log axis)", fontsize=FONT_SIZE_LABEL)
    ax_h.set_ylabel("count", fontsize=FONT_SIZE_LABEL)
    ax_h.set_title("RT distribution", fontsize=FONT_SIZE_TICKS)
    ax_h.legend(fontsize=FONT_SIZE_TICKS)

    # --- Panel 2: normal QQ (points only; no reference line, per request) ---
    ax_q.scatter(theo_q, emp_q, color=MAIN_COLOR, s=40, zorder=3)
    ax_q.set_xlabel("Theoretical normal quantile", fontsize=FONT_SIZE_LABEL)
    ax_q.set_ylabel("log(RT) quantile", fontsize=FONT_SIZE_LABEL)
    ax_q.set_title("Normal QQ", fontsize=FONT_SIZE_TICKS)

    # --- Panel 3: median RT (s) vs ply over the WHOLE game (unwindowed "total";
    # log y so the multiplicative arc is legible) ---
    ply = conn.execute(
        f"SELECT move_ply, median(move_time) AS med_s, count(*) AS n "
        f"FROM {CONFIG['table_processed_moves_nonzero']} "
        f"WHERE move_time > 0 AND move_ply BETWEEN 1 AND 150 GROUP BY 1 ORDER BY 1"
    ).df()
    ply = ply[ply.n >= 200]
    ax_p.plot(ply.move_ply, ply.med_s, color=MAIN_COLOR, lw=3)
    ax_p.axvspan(CONFIG["min_ply"], CONFIG["max_ply"], color="gray", alpha=0.12,
                 label=f"analysis window [{CONFIG['min_ply']},{CONFIG['max_ply']}]")
    ax_p.set_yscale("log")
    ax_p.set_xlabel("Ply (whole game)", fontsize=FONT_SIZE_LABEL)
    ax_p.set_ylabel("median RT (s, log axis)", fontsize=FONT_SIZE_LABEL)
    ax_p.set_title("RT vs ply (total)", fontsize=FONT_SIZE_TICKS)
    ax_p.legend(fontsize=FONT_SIZE_TICKS - 4)

    fig.suptitle(f"Response time — distribution, normal QQ, RT vs ply  (n = {n_moves:,} moves)",
                 fontsize=FONT_SIZE_LABEL + 4)
    save_figure(fig, "board", "rt_distribution.pdf")


def run_bivariate_analysis(conn, column: str, name: str, filename: str, filter_query: str | None = None,
                           n_bins: int = 10, tie_safe: bool = True, table: str = "pmnz_gf",
                           reverse_x: bool = False, analyzer_opts: dict | None = None):
    """RT-vs-covariate 1x2 dashboard: overall, colored by ply tertile. Reads
    ``pmnz_gf`` (windowed view).

    ``analyzer_opts`` are forwarded to ``Analyzer`` and override the defaults below
    (e.g. ``bin_mode="integer"``, ``min_bin_count=300``) — the battery's small-integer
    features bin per-integer per the trust protocol, not by ntile."""
    def _analyzer(segment_column, segment_label, **extra):
        kw = dict(
            db_conn=conn,
            table_name=table,
            x_var=Variable(column=column, is_log=False, name=name),
            y_var=Variable(column="move_time", is_log=True, name="RT"),
            filter_query=filter_query,
            title=name,
            n_bins=n_bins,
            tie_safe=tie_safe,
            min_bin_count=100 if tie_safe else 0,
            # Tertiles conditioned on the (windowed) analysis table, not the whole dataset.
            ply_tertile_source=table,
            segment_column=segment_column,
            segment_source=table,
            segment_label=segment_label,
        )
        kw.update(extra)
        kw.update(analyzer_opts or {})
        return Analyzer(**kw)
    a_ply = _analyzer("move_ply", "Ply")

    fig = plt.figure(figsize=(30, 13.72))
    ax1, ax2 = (fig.add_subplot(121), fig.add_subplot(122))
    a_ply.plot_quantile_bins(ax1)                     # overall
    a_ply.plot_quantile_bins_tertile_segmented(ax2)   # color: ply
    ax1.set_title("overall", fontsize=FONT_SIZE_TICKS)
    ax2.set_title("by ply", fontsize=FONT_SIZE_TICKS)
    if reverse_x:
        for ax in (ax1, ax2):
            ax.invert_xaxis()
    fig.suptitle(f"{a_ply.title}\nn = {a_ply.n_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    # wider + more horizontal gap so the two panels don't crowd.
    fig.subplots_adjust(top=0.80, wspace=0.28)
    save_figure(fig, "board", filename)


def run_board_corr(conn, n_sample=1_000_000, seed=42, resid_col=None,
                   filename="board_feature_corr.pdf", title="correlation — board features",
                   table="pmnz_gf", extra_cols: list[tuple[str, str, str]] | None = None):
    """Correlation matrices (Spearman AND Pearson, per the trust protocol) over
    board structural features + log(RT). The Pearson matrix saves with a
    ``_pearson`` filename suffix.

    If ``resid_col`` is given (e.g. ``"move_ply"``), RT is first **residualized**
    on it — ``log(RT) − mean(log(RT) | resid_col)`` — so the correlations reflect
    the part of think-time NOT explained by that covariate (e.g. game stage). The
    conditional mean is taken over the sampled rows via a partitioned window.

    ``extra_cols``: list of (sql_expr, alias, label) appended to the matrix — the
    P0 battery features join in via ``table="pmnz_bat"``.
    """
    if resid_col:
        rt_sql = f"ln(move_time) - avg(ln(move_time)) OVER (PARTITION BY {resid_col})"
        rt_label = f"log(RT) ⟂ {resid_col}"
    else:
        rt_sql = "ln(move_time)"
        rt_label = "log(RT)"
    extra_cols = extra_cols or []
    extra_sql = "".join(f", {expr} AS {alias}" for expr, alias, _ in extra_cols)
    df = conn.execute(f"""
        SELECT move_ply AS ply, n_possible_moves AS legal_moves,
               player_clock_time AS player_clock, {rt_sql} AS log_T
               {extra_sql}
        FROM {table}
        USING SAMPLE {n_sample} ROWS (reservoir, {seed})
    """).df()
    # Order: RT, Ply, Player clock, Legal moves, then battery features.
    labels = {
        "log_T": rt_label, "ply": "Ply",
        "player_clock": "Player clock", "legal_moves": "Legal moves",
    }
    labels.update({alias: label for _, alias, label in extra_cols})
    base, ext = os.path.splitext(filename)
    for method, mlabel, suffix in (("spearman", "Spearman ρ", ""), ("pearson", "Pearson r", "_pearson")):
        corr = df[list(labels)].corr(method=method).rename(columns=labels, index=labels)
        n = len(corr)
        apply_poster_style()
        side = max(9, 1.6 * n)  # grow with battery columns so cells stay legible
        fig, ax = plt.subplots(figsize=(side, side * 0.83))
        ax.grid(False)
        im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(corr.columns, fontsize=18, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.index, fontsize=18)
        for i in range(n):
            for j in range(n):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=15,
                        color="white" if abs(v) > 0.5 else "black",
                        fontweight="bold" if i == j else "normal")
        highlight_corr_row(ax, n)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(mlabel, fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        ax.set_title(f"{mlabel} {title} (n = {len(df):,})", fontsize=18, pad=12)
        plt.tight_layout()
        save_figure(fig, "board", f"{base}{suffix}{ext}")


# =============================================================================
# P0 board-correlate battery (engine-free; features from analysis.featurize_board;
# rules per outputs/reports/trust_protocol.md). Merged from board_battery.py.
# =============================================================================

# The new engine-free board features (subset that gets the partial-vs-legal-moves
# analysis — legal moves itself is the control, so it is not in this list).
BATTERY_FEATURES = {
    "n_captures_avail": "Captures available",
    "n_checks_avail": "Checks available",
    "self_material": "Self material (weighted)",
    "material_imbalance": "Material imbalance (weighted)",
    "in_check": "In check",
    "prev_move_was_capture": "Prev move was capture",
}

# The FULL battery for the descriptive figures (histograms / lowess / scatter):
# the classic board covariates PLUS the new features. Each entry is
# (label, kind, display_clip) — kind drives binning/jitter, clip trims display
# tails only (never drops rows from the stats).
BATTERY_ALL = {
    "move_ply":            ("Ply", "disc", (0, 150)),
    "n_possible_moves":    ("Legal moves", "disc", (0, 60)),
    "player_clock_time":   ("Player clock (s)", "cont", (0, 600)),
    "game_fraction":       ("Game fraction", "cont", (0.0, 1.0)),
    "n_captures_avail":    ("Captures available", "disc", (0, 15)),
    "n_checks_avail":      ("Checks available", "disc", (0, 8)),
    "self_material":       ("Self material (weighted)", "disc", (0, 40)),
    "material_imbalance":  ("Material imbalance (weighted)", "disc", (-15, 15)),
    "in_check":            ("In check", "bin", None),
    "prev_move_was_capture": ("Prev move was capture", "bin", None),
}
_BOOT_HEADLINE = 1000  # trust protocol: B=1000 headline CIs
_BOOT_CLUSTER = 200    # B=200 for the heavier per-instance runs


def default_features_dir() -> str:
    """Featurizer output lives beside the run's other scratch outputs."""
    return os.path.join(os.path.dirname(CONFIG["cache_default"]), "board_features")


def setup_battery_views(conn, features_dir: str) -> None:
    """feats (per-FEN parquet), prev_cap (lag over the FULL moves table so ply-15
    rows still see ply-14), and pmnz_bat = windowed instances x features."""
    glob_pat = os.path.join(features_dir, "features_*.parquet")
    conn.execute(f"CREATE OR REPLACE TEMP VIEW feats AS SELECT * FROM read_parquet('{glob_pat}')")
    conn.execute(
        "CREATE OR REPLACE TEMP VIEW prev_cap AS "
        "SELECT gid, move_ply, "
        "       (n_pieces < lag(n_pieces) OVER (PARTITION BY gid ORDER BY move_ply)) "
        "         AS prev_move_was_capture "
        f"FROM {CONFIG['table_moves']}"
    )
    conn.execute(
        "CREATE OR REPLACE TEMP VIEW pmnz_bat AS "
        "SELECT m.*, f.in_check, f.n_captures_avail, f.n_checks_avail, "
        "       f.self_material, f.opp_material, f.material_imbalance, "
        "       p.prev_move_was_capture "
        "FROM pmnz_gf m "
        "JOIN feats f ON f.fen = m.fen "
        "LEFT JOIN prev_cap p ON p.gid = m.gid AND p.move_ply = m.move_ply"
    )


def _binned_rt(x, y, kind, clip, *, stat="median", min_n=100):
    """(centres, stat(log RT)) per bin. Discrete/binary -> per value; continuous
    -> up to 20 quantile bins. Bins with < min_n rows are dropped."""
    import pandas as pd
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if clip:
        d = d[(d.x >= clip[0]) & (d.x <= clip[1])]
    if len(d) == 0:
        return np.array([]), np.array([])
    if kind in ("disc", "bin"):
        g = d.groupby("x").y.agg([stat, "size"])
        g = g[g["size"] >= min_n]
        return g.index.to_numpy(float), g[stat].to_numpy()
    nb = min(20, max(2, d.x.nunique()))
    d = d.assign(b=pd.qcut(d.x, nb, duplicates="drop"))
    g = d.groupby("b", observed=True).agg(cx=("x", "mean"), sy=("y", stat), n=("y", "size"))
    g = g[g.n >= min_n]
    return g.cx.to_numpy(), g.sy.to_numpy()


def _battery_grid(n_panels):
    """A constrained-layout grid (3 cols) sized to hold n_panels; extra axes off."""
    ncol = 3
    nrow = -(-n_panels // ncol)
    apply_poster_style()
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 6.2 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes


def run_battery_figures(conn, sample_rows=200_000, seed=7):
    """The full-battery descriptive figures over ALL features (classic covariates
    + new): battery_histograms (marginals), battery_lowess (log RT vs feature,
    LOWESS + binned median), battery_scatter (subsample points + binned median).
    constrained_layout keeps titles/labels from colliding with the panels."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    cols = list(BATTERY_ALL)
    df = conn.execute(
        f"SELECT {', '.join(cols)}, ln(move_time) AS log_rt "
        f"FROM pmnz_bat WHERE move_time > 0 USING SAMPLE {sample_rows} ROWS (reservoir, {seed})"
    ).df()
    df["in_check"] = df.in_check.astype(float)
    df["prev_move_was_capture"] = df.prev_move_was_capture.fillna(False).astype(float)
    n = len(df)
    rng = np.random.default_rng(seed)

    # --- 1. histograms (marginal distribution of each feature) ---
    fig, axes = _battery_grid(len(cols))
    for ax, col in zip(axes, cols):
        label, kind, clip = BATTERY_ALL[col]
        x = df[col].dropna()
        if clip:
            x = x[(x >= clip[0]) & (x <= clip[1])]
        if kind in ("disc", "bin"):
            vc = x.value_counts().sort_index()
            ax.bar(vc.index, vc.values / len(x), width=(0.4 if kind == "bin" else 0.9),
                   color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
        else:
            ax.hist(x, bins=50, density=True, color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
        ax.set_title(label, fontsize=FONT_SIZE_TICKS)
        ax.set_ylabel("density", fontsize=FONT_SIZE_LABEL - 8)
    fig.suptitle(f"Board battery — feature distributions (n = {n:,} move-instances)",
                 fontsize=FONT_SIZE_LABEL)
    save_figure(fig, "board", "battery_histograms.pdf")

    # --- 2. LOWESS: log RT vs feature (+ binned median for context) ---
    # lowess is O(n^2 * frac); keep the subsample small and use it=0 + a delta so
    # nearby x-values are interpolated rather than each getting a local fit — this
    # is the difference between seconds and many minutes at this scale.
    lw = df.sample(min(6_000, n), random_state=seed)
    fig, axes = _battery_grid(len(cols))
    for ax, col in zip(axes, cols):
        label, kind, clip = BATTERY_ALL[col]
        cx, my = _binned_rt(df[col], df.log_rt, kind, clip, stat="median")
        ax.plot(cx, my, "o", color="gray", alpha=0.55, ms=7, label="binned median")
        if kind == "cont" or (kind == "disc" and df[col].nunique() > 3):
            d = lw[[col, "log_rt"]].dropna()
            if clip:
                d = d[(d[col] >= clip[0]) & (d[col] <= clip[1])]
            xv = d[col].to_numpy()
            delta = 0.01 * (xv.max() - xv.min()) if len(xv) else 0.0
            lo = lowess(d.log_rt.to_numpy(), xv, frac=0.3, it=0, delta=delta,
                        return_sorted=True)
            ax.plot(lo[:, 0], lo[:, 1], color=MAIN_COLOR, lw=3.5, label="LOWESS")
        ax.set_title(label, fontsize=FONT_SIZE_TICKS)
        ax.set_ylabel("log RT", fontsize=FONT_SIZE_LABEL - 8)
        ax.legend(fontsize=FONT_SIZE_TICKS - 8)
    fig.suptitle(f"Board battery — log RT vs feature (LOWESS + binned median, n = {n:,})",
                 fontsize=FONT_SIZE_LABEL)
    save_figure(fig, "board", "battery_lowess.pdf")

    # --- 3. scatter (subsample) + binned median ---
    sc = df.sample(min(8_000, n), random_state=seed)
    fig, axes = _battery_grid(len(cols))
    for ax, col in zip(axes, cols):
        label, kind, clip = BATTERY_ALL[col]
        d = sc[[col, "log_rt"]].dropna()
        if clip:
            d = d[(d[col] >= clip[0]) & (d[col] <= clip[1])]
        jit = rng.uniform(-0.35, 0.35, len(d)) if kind in ("disc", "bin") else 0.0
        ax.scatter(d[col].to_numpy() + jit, d.log_rt, s=5, alpha=0.15,
                   color=MAIN_COLOR, edgecolors="none")
        cx, my = _binned_rt(df[col], df.log_rt, kind, clip, stat="median")
        ax.plot(cx, my, "-o", color="black", lw=2.5, ms=6, label="binned median")
        ax.set_title(label, fontsize=FONT_SIZE_TICKS)
        ax.set_ylabel("log RT", fontsize=FONT_SIZE_LABEL - 8)
        ax.legend(fontsize=FONT_SIZE_TICKS - 8)
    fig.suptitle("Board battery — log RT vs feature (scatter subsample + binned median)",
                 fontsize=FONT_SIZE_LABEL)
    save_figure(fig, "board", "battery_scatter.pdf")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    import pandas as pd
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _partial(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, rank: bool) -> float:
    """Partial corr of x,y | z (rank=True -> partial Spearman via ranked vars)."""
    import pandas as pd
    if rank:
        x = pd.Series(x).rank().to_numpy()
        y = pd.Series(y).rank().to_numpy()
        z = pd.Series(z).rank().to_numpy()
    zc = np.column_stack([np.ones_like(z), z])
    bx, *_ = np.linalg.lstsq(zc, x, rcond=None)
    by, *_ = np.linalg.lstsq(zc, y, rcond=None)
    return float(np.corrcoef(x - zc @ bx, y - zc @ by)[0, 1])


def _boot_ci(fn, df, b: int, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap resampling ROWS of df (rows = FENs on the per-FEN unit)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    stats_ = [fn(df.iloc[rng.integers(0, n, n)]) for _ in range(b)]
    return float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))


def run_battery_corr(conn, sample_games: int = 25_000):
    """P0.b — Spearman+Pearson vs ln(RT), partials vs legal moves, bootstrap CIs,
    on BOTH units (per-instance / per-FEN) from a game-level sample."""
    import pandas as pd
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE sample_gids AS "
        f"SELECT DISTINCT gid FROM pmnz_gf USING SAMPLE {sample_games}"
    )
    inst = conn.execute(
        "SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves, "
        "       m.in_check::INT AS in_check, m.n_captures_avail, m.n_checks_avail, "
        "       m.self_material, m.material_imbalance, "
        "       COALESCE(m.prev_move_was_capture, FALSE)::INT AS prev_move_was_capture "
        "FROM pmnz_bat m JOIN sample_gids sg ON sg.gid = m.gid"
    ).df()
    per_fen = inst.groupby("fen").agg(
        log_rt=("log_rt", "mean"), n_possible_moves=("n_possible_moves", "first"),
        **{c: (c, "first") for c in BATTERY_FEATURES},
    ).reset_index(drop=True)

    rows = []
    for col in BATTERY_FEATURES:
        for unit, df, b in (("instance", inst, _BOOT_CLUSTER), ("fen", per_fen, _BOOT_HEADLINE)):
            x = df[col].to_numpy(float)
            y = df.log_rt.to_numpy()
            z = df.n_possible_moves.to_numpy(float)
            rec = {
                "feature": col, "unit": unit, "n": len(df),
                "spearman": _spearman(x, y), "pearson": float(np.corrcoef(x, y)[0, 1]),
                "partial_spearman_legal": _partial(x, y, z, rank=True),
                "partial_pearson_legal": _partial(x, y, z, rank=False),
            }
            lo, hi = _boot_ci(
                lambda d, c=col: _partial(d[c].to_numpy(float), d.log_rt.to_numpy(),
                                          d.n_possible_moves.to_numpy(float), rank=True),
                df, b)
            rec["partial_spearman_ci_lo"], rec["partial_spearman_ci_hi"] = lo, hi
            rows.append(rec)
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    save_table(res, os.path.join(CONFIG["figures_dir"], "board"),
               "battery_correlations.csv", index=False)


def run_imbalance_shape(conn, sample_games: int = 25_000):
    """P0.b — pre-registered ∩-shape test for material_imbalance (weighted units).

    Weighted material (incl pawns) spans a wide range, so: peak = {−1, 0, +1}
    (within ~a pawn of balance), tails = |imbalance| >= 4 (at least a minor piece
    up/down), display clipped ±12. Test 1: the peak bins' CI lower bound sits above
    the CI upper bound of at least one bin in EACH tail. Test 2: quadratic beta < 0
    with bootstrap CI excluding 0 AND quadratic dR2 >= 0.001.
    """
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE sample_gids AS "
        f"SELECT DISTINCT gid FROM pmnz_gf USING SAMPLE {sample_games}"
    )
    df = conn.execute(
        "SELECT m.material_imbalance, ln(m.move_time) AS log_rt "
        "FROM pmnz_bat m JOIN sample_gids sg ON sg.gid = m.gid"
    ).df()
    rng = np.random.default_rng(7)

    imb = df.material_imbalance.clip(-12, 12)
    agg = df.assign(imb=imb).groupby("imb").log_rt.agg(["mean", "median", "count"])
    ci = {}
    for v, grp in df.assign(imb=imb).groupby("imb"):
        vals = grp.log_rt.to_numpy()
        boots = [np.median(vals[rng.integers(0, len(vals), len(vals))])
                 for _ in range(_BOOT_HEADLINE)]
        ci[v] = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    agg["ci_lo"] = [ci[v][0] for v in agg.index]
    agg["ci_hi"] = [ci[v][1] for v in agg.index]

    peak_lo = agg.loc[[i for i in (-1, 0, 1) if i in agg.index], "ci_lo"].min()
    left_tail = agg.loc[agg.index <= -4, "ci_hi"]
    right_tail = agg.loc[agg.index >= 4, "ci_hi"]
    test1 = bool(len(left_tail) and len(right_tail)
                 and (peak_lo > left_tail.min()) and (peak_lo > right_tail.min()))

    x = df.material_imbalance.to_numpy(float)
    y = df.log_rt.to_numpy()
    X1 = np.column_stack([np.ones_like(x), x])
    X2 = np.column_stack([np.ones_like(x), x, x ** 2])
    _, res1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    b2, res2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    sst = ((y - y.mean()) ** 2).sum()
    dr2 = float((res1[0] - res2[0]) / sst) if len(res1) and len(res2) else float("nan")
    boots_b2 = []
    n = len(df)
    for _ in range(_BOOT_CLUSTER):
        idx = rng.integers(0, n, n)
        bb, *_ = np.linalg.lstsq(X2[idx], y[idx], rcond=None)
        boots_b2.append(bb[2])
    b2_lo, b2_hi = np.percentile(boots_b2, [2.5, 97.5])
    test2 = bool(b2[2] < 0 and b2_hi < 0 and dr2 >= 0.001)

    verdict = {True: {True: "CONFIRMED (both tests)", False: "suggestive (binned only)"},
               False: {True: "suggestive (quadratic only)", False: "NOT confirmed"}}[test1][test2]
    print(f"∩-shape test: binned={test1}, quadratic={test2} (β2={b2[2]:+.5f} "
          f"[{b2_lo:+.5f},{b2_hi:+.5f}], ΔR²={dr2:.5f}) -> {verdict}")

    apply_poster_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.errorbar(agg.index, agg["median"],
                yerr=[agg["median"] - agg.ci_lo, agg.ci_hi - agg["median"]],
                marker="o", lw=3, capsize=4, color=MAIN_COLOR, label="median log RT (95% CI)")
    ax.plot(agg.index, agg["mean"], marker="s", lw=1.5, ls="--", color="gray", label="mean log RT")
    ax.set_xlabel("Material imbalance (mover POV, weighted incl pawns; clipped ±12)",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("log RT", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"∩-shape test — {verdict}\n(n = {len(df):,} instances)",
                 fontsize=FONT_SIZE_TICKS + 4)
    ax.legend(fontsize=FONT_SIZE_TICKS)
    save_table(agg, os.path.join(CONFIG["figures_dir"], "board"), "imbalance_shape_bins.csv")
    save_figure(fig, "board", "imbalance_shape.pdf")


def run_battery_binaries(conn):
    """Group means + SEM for the binary features (dashboards are meaningless)."""
    for col in ("in_check", "prev_move_was_capture"):
        stats_ = conn.execute(
            f"SELECT {col} AS v, count(*) AS n, avg(ln(move_time)) AS mean_lrt, "
            f"       median(ln(move_time)) AS median_lrt, "
            f"       stddev(ln(move_time)) / sqrt(count(*)) AS sem "
            f"FROM pmnz_bat WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY 1"
        ).df()
        print(f"\n{col}:")
        print(stats_.round(4).to_string(index=False))
        save_table(stats_, os.path.join(CONFIG["figures_dir"], "board"),
                   f"battery_{col}_groups.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified board-level response time analyses")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--config", help="Path to the run config (else $CONFIG or the default).")
    parser.add_argument("--all", action="store_true", default=True, help="Run all analyses (default/always)")
    parser.add_argument("--features-dir", default=None,
                        help="P0 battery features dir (default: <run scratch>/board_features)")
    args = parser.parse_args(argv)
    print(f"Board analysis: ply window [{CONFIG['min_ply']}, {CONFIG['max_ply']}], db={args.db}")

    # The legacy per-feature 1x3 dashboards (ply/legal_moves/clock) are subsumed by
    # the full-battery figures below (which include ply/legal-moves/clock + the new
    # features under one clean constrained-layout), so only the RT summary runs here.
    analyses = {
        "summary": run_move_time_summary,
    }

    with db_connection(args.db, read_only=True) as conn:
        # Apply the ply window ON ARRIVAL: every analysis reads these views.
        create_ply_windowed_views(conn)
        # game_fraction = move_ply / TRUE total plies in the game. The total must
        # come from the UNWINDOWED move table: taking max(move_ply) over the
        # ply-windowed view would cap the denominator at max_ply (=75), pinning
        # every game longer than the window to the same denominator and inflating
        # game_fraction toward 1.0 for the ~55% of moves whose game exceeds 75 plies.
        conn.execute(
            "CREATE OR REPLACE TEMP VIEW pmnz_gf AS "
            "SELECT w.*, w.move_ply * 1.0 / g.game_len AS game_fraction "
            f"FROM {WIN_PROCESSED_MOVES_NONZERO} w "
            "JOIN (SELECT gid, max(move_ply) AS game_len "
            f"      FROM {CONFIG['table_processed_moves']} GROUP BY gid) g USING (gid)"
        )
        for name, config in analyses.items():
            print(f"Executing board analysis: {name}...")
            if callable(config):
                config(conn)
            else:
                run_bivariate_analysis(
                    conn,
                    column=config["column"],
                    name=config["name"],
                    filename=config["filename"],
                    filter_query=config.get("filter_query"),
                    tie_safe=config.get("tie_safe", True),
                    reverse_x=config.get("reverse_x", False),
                )

        # ---- P0 battery (skipped with a warning if the featurizer hasn't run) ----
        features_dir = args.features_dir or default_features_dir()
        import glob as _glob
        if not _glob.glob(os.path.join(features_dir, "features_*.parquet")):
            print(f"WARNING: no battery features at {features_dir}; "
                  "run analysis.featurize_board first. Skipping P0 battery.")
            # Fallback: emit the base correlation matrix (no battery features) so
            # board_feature_corr.pdf still exists when the featurizer hasn't run.
            print("Executing board analysis: correlation matrix (base, no battery)...")
            run_board_corr(conn)
            return
        setup_battery_views(conn, features_dir)
        print("Executing board analysis: battery figures (histograms / lowess / scatter)...")
        run_battery_figures(conn)
        print("Executing board analysis: battery_binaries...")
        run_battery_binaries(conn)
        print("Executing board analysis: battery_corr (units + partials + CIs)...")
        run_battery_corr(conn)
        print("Executing board analysis: imbalance ∩-shape test...")
        run_imbalance_shape(conn)
        print("Executing board analysis: correlation matrix (board + battery)...")
        run_board_corr(
            conn, table="pmnz_bat", filename="board_feature_corr.pdf",
            extra_cols=[
                ("n_captures_avail", "captures", "Captures"),
                ("n_checks_avail", "checks", "Checks"),
                ("self_material", "self_mat", "Self material"),
                ("material_imbalance", "imbalance", "Imbalance"),
                ("in_check::INT", "in_check", "In check"),
                ("COALESCE(prev_move_was_capture, FALSE)::INT", "prev_cap", "Prev capture"),
            ],
        )


if __name__ == "__main__":
    main()
