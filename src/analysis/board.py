"""
Board-level (no-model) response time analyses (DuckDB + matplotlib).

Three ordered modes (each depends on the previous — mirror this in the pipeline):
  --featurize  precondition: featurize the legal-move covariates (captures/checks)
               over this task's hash-slice of the run's ``filtered_moves`` FENs →
               a shard. Runs single, or as a Slurm array (task per hash partition).
  --merge      merge the featurize shards → the ``board_features`` table (single).
  --plot       (default) the four analyses over filtered_moves ⋈ board_features:
                 move_time_summary  log(RT) distribution + normal QQ + RT-vs-ply arc
                 bivariate_analysis RT vs each covariate: histogram, overall trend, by-ply trend
                 correlation_matrix Spearman + Pearson over log(RT), ply, covariates
                 feature_histograms marginal distribution of each covariate

The canonical windowed table (``table_filtered``) is built by the filter stage; the
covariate list is defined in run_plot().
"""

import argparse
import math
import multiprocessing as mp
import os

import numpy as np
from scipy import stats
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analysis.utils import Variable, Analyzer
from analysis.utils.analysis import _seconds_from_log
from analysis.utils.helpers import (
    apply_poster_style, db_connection, sql_str,
    MAIN_COLOR, PHASE_COLORS, LEGEND_FONTSIZE, CONFIG,
)
from analysis.utils.plots import (
    highlight_corr_row, save_figure, _annotate_n, _draw_feature_histogram,
    get_isoluminant_cmap, plot_heatmap_with_alpha,
)
from analysis.utils.pgfvals import pgf_set, write_pgf_tex

FEATURE_COLS = ["n_captures_avail", "n_checks_avail"]

# Material-imbalance bands for the checks-available x material interaction views
# (checks_material_interaction_heatmap / checks_material_band_curves): signed
# material_imbalance (mover POV) bucketed into 5 bands, ordered heavily-behind ->
# heavily-ahead. Colors are the SAME indigo-blue hue family as MAIN_COLOR/PHASE_COLORS
# (light->dark = behind->ahead) at 5 steps instead of PHASE_COLORS' 3 (ply tertiles) —
# one consistent "ordered category" color language across the report, not a new palette.
MATERIAL_BAND_ORDER = ["≤-5", "-4..-1", "0", "+1..+4", "≥+5"]
MATERIAL_BAND_COLORS = ["#abb6f7", "#697df2", "#2845ec", "#112abb", "#0b1b7a"]
_MATERIAL_BAND_SQL = (
    "CASE WHEN material_imbalance <= -5 THEN '≤-5' "
    "WHEN material_imbalance BETWEEN -4 AND -1 THEN '-4..-1' "
    "WHEN material_imbalance = 0 THEN '0' "
    "WHEN material_imbalance BETWEEN 1 AND 4 THEN '+1..+4' "
    "ELSE '≥+5' END"
)


# =============================================================================
# --featurize / --merge : the board_features precondition (captures/checks)
# =============================================================================

def calc_captures_checks(fen: str) -> tuple[int, int]:
    """(#legal captures, #legal checks) for one FEN — needs move enumeration."""
    board = chess.Board(fen)
    caps = checks = 0
    for move in board.legal_moves:
        caps += board.is_capture(move)
        checks += board.gives_check(move)
    return caps, checks


def _shards_dir() -> str:
    scratch = os.path.dirname(os.path.abspath(CONFIG["selected_db_default"]))
    return os.path.join(scratch, "tmp", f"{CONFIG['table_board_features']}_shards")


def run_featurize(db: str) -> None:
    """Featurize this task's hash-slice of filtered_moves' distinct FENs → a shard.
    Single job (1 task) or a Slurm array (SLURM_ARRAY_TASK_ID / _COUNT, NWORKERS
    overrides the count for straggler re-runs)."""
    import pandas as pd
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_tasks = int(os.environ.get("NWORKERS") or os.environ.get("SLURM_ARRAY_TASK_COUNT") or 1)
    with db_connection(db, read_only=True) as conn:
        fens = [r[0] for r in conn.execute(
            f"SELECT DISTINCT fen FROM {CONFIG['table_filtered']} WHERE hash(fen) % {n_tasks} = {task}"
        ).fetchall()]
    # The DB connection is closed, so no live DuckDB threads — a fork pool is safe
    # here (and fast: workers inherit the imported module, no re-import).
    print(f"board featurize task {task}/{n_tasks}: {len(fens):,} FENs", flush=True)
    with mp.Pool() as pool:
        counts = pool.map(calc_captures_checks, fens, chunksize=4000)
    df = pd.DataFrame([(f, c, k) for f, (c, k) in zip(fens, counts)], columns=["fen", *FEATURE_COLS])
    os.makedirs(_shards_dir(), exist_ok=True)
    out = os.path.join(_shards_dir(), f"shard_{task:04d}.parquet")
    df.to_parquet(out, index=False)
    print(f"✅ task {task}: {len(df):,} FENs → {out}", flush=True)


def run_merge(db: str) -> None:
    """Merge the featurize shards → the board_features table (single job)."""
    shards = os.path.join(_shards_dir(), "shard_*.parquet")
    with db_connection(db, read_only=False) as conn:
        conn.execute(f"CREATE OR REPLACE TABLE {CONFIG['table_board_features']} AS "
                     f"SELECT * FROM read_parquet('{sql_str(shards)}')")
        n = conn.execute(f"SELECT count(*) FROM {CONFIG['table_board_features']}").fetchone()[0]
    print(f"✅ {CONFIG['table_board_features']}: {n:,} FENs", flush=True)


# =============================================================================
# --plot : the four analyses (read filtered_moves ⋈ board_features)
# =============================================================================

_LEGEND_KW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=LEGEND_FONTSIZE, frameon=False)


def move_time_summary(conn, table, *, ply_table: str | None = None, filename: str = "rt_distribution.pdf",
                      smoke: bool = False):
    """log(RT) distribution histogram + normal QQ + mean-RT-vs-ply arc (whole game).

    ``ply_table`` overrides the whole-game (unwindowed) table the ply-arc panel
    reads from — defaults to ``table_processed_moves_nonzero``, but ``--smoke``
    passes its own sampled view so this panel stays fast too. ``smoke=True`` skips
    the pgfvals registration (see ``analysis.utils.pgfvals``) — a small sample must
    never overwrite the canonical numbers a report cites."""
    n_bins = CONFIG["response_time_histogram_bins"]
    n_qq = CONFIG["qq_plot_quantile_probes"]

    n_moves = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"Running response time summary: n = {n_moves:,} moves")

    conn.execute(
        "CREATE OR REPLACE TEMPORARY VIEW _summary_view AS "
        f"SELECT ln(move_time) AS ln_move_time FROM {table}"
    )

    # SQL-side histogram binning (uniform bins in ln(RT)).
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
    lmt_bins = conn.execute("SELECT * FROM _lmt_bins ORDER BY bin_idx").df()

    # Moments + empirical quantiles for the QQ plot (all SQL-side).
    mean, std = conn.execute("SELECT avg(ln_move_time), stddev(ln_move_time) FROM _summary_view").fetchone()
    probs = (np.arange(1, n_qq + 1)) / (n_qq + 1)
    empirical = np.array(conn.execute(
        "SELECT quantile_cont(ln_move_time, ?) FROM _summary_view", [probs.tolist()]
    ).fetchone()[0], dtype=float)
    theoretical = stats.norm.ppf(probs, loc=mean, scale=std)  # QQ plot: theoretical Normal(mean, std) quantiles

    def _weighted_median(df):
        cumsum = df["n"].cumsum()
        idx = (cumsum >= df["n"].sum() / 2).idxmax()
        return float((df["bin_left"].iloc[idx] + df["bin_right"].iloc[idx]) / 2)

    med_log = _weighted_median(lmt_bins)

    apply_poster_style()
    fig, (ax_h, ax_q, ax_p) = plt.subplots(1, 3, figsize=(34, 12), constrained_layout=True)

    # Panel 1: RT distribution in SECONDS on a log x-axis (bins uniform in ln(RT),
    # so exponentiating the edges gives geometric bins that read evenly on a log axis).
    sec = lmt_bins.assign(left_s=np.exp(lmt_bins.bin_left), right_s=np.exp(lmt_bins.bin_right))
    ax_h.bar(sec.left_s, sec.n, width=(sec.right_s - sec.left_s), align="edge",
             color=MAIN_COLOR, alpha=0.5, edgecolor=MAIN_COLOR, linewidth=1.2)
    ax_h.set_xscale("log")
    mean_s = pgf_set("board/rt/mean_s", np.exp(mean), "{:.1f}") if not smoke else f"{np.exp(mean):.1f}"
    median_s = pgf_set("board/rt/median_s", np.exp(med_log), "{:.1f}") if not smoke else f"{np.exp(med_log):.1f}"
    ax_h.axvline(np.exp(mean), color="black", ls="--", lw=2.5, label=f"Mean = {mean_s}s")
    ax_h.axvline(np.exp(med_log), color="dimgray", ls=":", lw=2.5, label=f"Median = {median_s}s")
    ax_h.set(xlabel="RT (s, log axis)", ylabel="Count")
    ax_h.legend(**_LEGEND_KW)

    # Panel 2: QQ plot — theoretical Normal(mean, std) quantile vs. empirical quantile
    # of RT, both in SECONDS on a log-log scale (RT spans orders of magnitude, same
    # reasoning as panel 1's log x-axis). A straight y=x line means log(RT) is well
    # fit by a Normal; curvature away from it shows where/how the fit departs (e.g.
    # heavier tails than lognormal).
    theo_s, emp_s = np.exp(theoretical), np.exp(empirical)
    ax_q.scatter(theo_s, emp_s, color=MAIN_COLOR, s=40, zorder=3)
    lims = (float(min(theo_s.min(), emp_s.min())), float(max(theo_s.max(), emp_s.max())))
    ax_q.plot(lims, lims, "k--", lw=1.5, zorder=2, label="y = x")
    ax_q.set_xscale("log")
    ax_q.set_yscale("log")
    ax_q.set(xlabel="Theoretical Quantile (s)", ylabel="Empirical Quantile (s)", xlim=lims, ylim=lims)
    ax_q.legend(**_LEGEND_KW)

    # Panel 3: RT vs ply over the WHOLE game (unwindowed) — same quantile-binned
    # trend machinery as every other panel, so it reads with identical visual weight.
    ply_analyzer = Analyzer(
        db_conn=conn, table_name=ply_table or CONFIG["table_processed_moves_nonzero"],
        x_var=Variable(column="move_ply", is_log=False, name="Ply (whole game)"),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        n_bins=20, tie_safe=True, min_bin_count=200,
    )
    ply_analyzer.plot_quantile_bins(ax_p)
    ax_p.axvspan(int(CONFIG["min_ply"]), int(CONFIG["max_ply"]), color="gray", alpha=0.12,
                 label=f"Analysis window [{CONFIG['min_ply']},{CONFIG['max_ply']}]")
    ax_p.legend(**_LEGEND_KW)
    # `constrained_layout` + `axvspan` + a below-axes `legend()` on a multi-panel
    # figure corrupts this axis' tick FORMATTER into a 2-entry FixedFormatter keyed
    # off the axvspan's own (min_ply, max_ply) bounds (reproduced in isolation —
    # a matplotlib layout-engine interaction, not anything-specific to this data):
    # every tick beyond the first two renders blank, and the surviving two land on
    # top of each other since the locator's positions don't match the formatter's
    # assumptions. Force both back to sane, explicit state so this can't recur.
    ax_p.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax_p.xaxis.set_major_formatter(mticker.ScalarFormatter())

    _annotate_n(fig, n_moves)
    save_figure(fig, "board", filename)


def _binning_opts(col: str, kind: str, clip) -> dict:
    """Binning scheme by covariate kind, chosen so every bin carries roughly the
    same weight of evidence:
      cont -> plain rank-based ntile: EXACTLY equal count per bin by construction.
      disc -> integer (one bin per value), tail-merged at the display clip's upper
              bound, floor-excluded below its lower bound. These covariates are
              point-massed (e.g. n_checks_avail: 54% at value 0) — tie-safe VALUE-based
              quantile cuts collapse several nominal bins onto that one value and leave
              others empty, while slicing the thin tail into scraps that fall below
              min_bin_count and vanish silently. Integer bins don't promise equal
              counts, but every surviving bin is real and well-powered (the rare tail
              is one merged point, not fragments) — clip bounds are chosen (see
              run_plot's features dict) so that merged tail/floor still clears
              min_bin_count in every ply tertile, not just overall.

    A negative ``clip[0]`` (a genuinely SIGNED covariate, e.g. signed material
    imbalance) merges the sparse low tail into its own point too (``integer_floor_cut``),
    mirroring the upper-tail merge, instead of excluding rows below it — dropping real
    "mover is way behind" rows would bias a signed variable, unlike a naturally
    nonnegative one's sparse floor (self_material below 14 is display-excluded, not
    merged, since that's a display choice about the near-empty-board tail, not about
    preserving symmetry).
    """
    if kind == "cont":
        return dict(bin_mode="ntile", tie_safe=False)
    if kind == "bin":
        return dict(bin_mode="ntile", tie_safe=True)   # 2 values; tie-safe collapses cleanly to {0,1}
    opts = dict(bin_mode="integer", integer_bin_width=1, integer_tail_cut=clip[1])   # disc
    if clip[0] > 0:
        opts["filter_query"] = f"{col} >= {clip[0]}"   # exclude the sparse floor, same as the histogram
    elif clip[0] < 0:
        opts["integer_floor_cut"] = clip[0]            # merge the sparse low tail, don't drop it
    return opts


def _bool_bars(ax, df, *, width=0.6, offset=0.0, color=MAIN_COLOR, label=None):
    """One group of bars + 95% CI error bars, one per boolean value, at df's mean_x
    positions (0.0/1.0) — the categorical counterpart to the quantile trend line."""
    df = df.sort_values("mean_x")
    x = df["mean_x"].to_numpy() + offset
    ci = 1.96 * df["std_y"].to_numpy() / np.sqrt(df["n"].to_numpy())
    ax.bar(x, df["mean_y"].to_numpy(), yerr=ci, width=width, capsize=6,
           color=color, alpha=0.8, label=label)


def _plot_boolean_overall(analyzer, ax):
    _bool_bars(ax, analyzer.quantile_df)
    ax.set_xticks([0.0, 1.0]); ax.set_xticklabels(["False", "True"])
    _seconds_from_log(ax.yaxis)
    ax.set(ylabel="Response Time (s)")


def _plot_boolean_by_tertile(analyzer, ax):
    tertiles = sorted(analyzer.quantile_tertile_df["tertile_id"].unique())
    width = 0.8 / len(tertiles)
    for i, t in enumerate(tertiles):
        sub = analyzer.quantile_tertile_df[analyzer.quantile_tertile_df["tertile_id"] == t]
        offset = (i - (len(tertiles) - 1) / 2) * width
        _bool_bars(ax, sub, width=width, offset=offset, color=PHASE_COLORS.get(int(t), MAIN_COLOR),
                  label=analyzer._ply_tertile_legend_label(int(t)))
    ax.set_xticks([0.0, 1.0]); ax.set_xticklabels(["False", "True"])
    _seconds_from_log(ax.yaxis)
    ax.set(ylabel="Response Time (s)")
    ax.legend(**_LEGEND_KW)


def bivariate_analysis(conn, column: str, name: str, filename: str, table: str, *,
                       kind: str, clip, reverse_x: bool = False):
    """RT-vs-covariate 1x3 dashboard: marginal histogram (left), overall trend
    (middle), trend by ply tertile (right) — see ``_binning_opts`` for the binning
    scheme and ``_draw_feature_histogram`` for the left panel. Boolean covariates
    (``kind == "bin"``) render the trend panels as bar charts + 95% CI instead of a
    quantile trend line (a line over 2 categories reads as a spurious trend)."""
    apply_poster_style()
    kw = dict(
        db_conn=conn,
        table_name=table,
        x_var=Variable(column=column, is_log=False, name=name),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        title=name,
        n_bins=10,
        min_bin_count=300,             # trust_protocol.md's established house scheme
        ply_tertile_source=table,     # tertiles conditioned on the windowed table
        segment_column="move_ply",
        segment_source=table,
        segment_label="Ply",
        **_binning_opts(column, kind, clip),
    )
    analyzer = Analyzer(**kw)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(42, 13), constrained_layout=True)
    _draw_feature_histogram(conn, table, column, kind, clip, ax0, name=name)
    if kind == "bin":
        _plot_boolean_overall(analyzer, ax1)
        _plot_boolean_by_tertile(analyzer, ax2)
    else:
        analyzer.plot_quantile_bins(ax1)                     # overall
        analyzer.plot_quantile_bins_tertile_segmented(ax2)   # color: ply
    if reverse_x:
        for ax in (ax0, ax1, ax2):
            ax.invert_xaxis()
    _annotate_n(fig, analyzer.n_moves)
    save_figure(fig, "board", filename)


def checks_material_interaction_heatmap(conn, table, filename: str = "checks_material_interaction_heatmap.pdf"):
    """Checks-available (x) x material-imbalance band (y, signed, mover POV) heatmap:
    cell color = geometric-mean RT, cell alpha = frequency (see ``plot_heatmap_with_alpha``).
    The joint view backing ``checks_material_band_curves``' per-band lines — shows WHY
    ``bivariate_n_checks_avail``'s pooled inverted-U flips direction with material context
    (board.md "Why does material context flip the direction of the checks-avail effect?").
    """
    q = f"""
        SELECT LEAST(n_checks_avail, 10) AS checks_bin,
               {_MATERIAL_BAND_SQL} AS band,
               count(*) AS n,
               exp(avg(ln(move_time))) AS geo_rt
        FROM {table}
        WHERE n_checks_avail IS NOT NULL AND material_imbalance IS NOT NULL
        GROUP BY 1, 2
    """
    df = conn.execute(q).df()
    n_rows = int(df["n"].sum())
    band_rank = {b: i for i, b in enumerate(MATERIAL_BAND_ORDER)}
    df["band_rank"] = df["band"].map(band_rank)
    mean_piv = df.pivot(index="band_rank", columns="checks_bin", values="geo_rt")
    count_piv = df.pivot(index="band_rank", columns="checks_bin", values="n")

    apply_poster_style()
    fig, ax = plt.subplots(figsize=(20, 12), constrained_layout=True)
    plot_heatmap_with_alpha(ax, mean_piv, count_piv, get_isoluminant_cmap(),
                            alpha_mode="log", value_label="Mean RT (geo. mean, s)",
                            imshow_aspect="auto")
    ax.invert_xaxis()  # plot_heatmap_with_alpha always renders columns high->low; put 0 back on the left
    ax.set_yticks(list(band_rank.values()))
    ax.set_yticklabels(MATERIAL_BAND_ORDER)
    ax.set_xticks(list(range(11)))
    ax.set_xticklabels([*map(str, range(10)), "10+"])
    ax.set(xlabel="Checks Available (k; 10 = 10+)", ylabel="Material Imbalance (signed)")
    _annotate_n(fig, n_rows)
    save_figure(fig, "board", filename)


def checks_material_band_curves(conn, table, filename: str = "checks_material_band_curves.pdf"):
    """RT-vs-checks-available, one curve per material-imbalance band (signed, mover POV)
    + 95% CI band — the per-band breakdown of ``checks_material_interaction_heatmap``.
    Same quantile-trend visual language as every other board dashboard (mean +/- 1.96*SEM
    in log(RT) space, relabeled to seconds), just colored by band instead of ply tertile."""
    q = f"""
        SELECT LEAST(n_checks_avail, 10) AS checks_bin,
               {_MATERIAL_BAND_SQL} AS band,
               count(*) AS n,
               avg(ln(move_time)) AS mean_log_rt,
               stddev(ln(move_time)) AS std_log_rt
        FROM {table}
        WHERE n_checks_avail IS NOT NULL AND material_imbalance IS NOT NULL
        GROUP BY 1, 2
    """
    df = conn.execute(q).df()
    n_rows = int(df["n"].sum())

    apply_poster_style()
    fig, ax = plt.subplots(figsize=(20, 14), constrained_layout=True)
    for i, band in enumerate(MATERIAL_BAND_ORDER):
        sub = df[df["band"] == band].sort_values("checks_bin")
        if sub.empty:
            continue
        x = sub["checks_bin"].to_numpy()
        y = sub["mean_log_rt"].to_numpy()
        ci = 1.96 * sub["std_log_rt"].to_numpy() / np.sqrt(sub["n"].to_numpy())
        color = MATERIAL_BAND_COLORS[i]
        ax.plot(x, y, marker="o", lw=3, markersize=9, color=color, label=band)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.15)
    _seconds_from_log(ax.yaxis)
    ax.set_xticks(list(range(11)))
    ax.set_xticklabels([*map(str, range(10)), "10+"])
    ax.set(xlabel="Checks Available (k; 10 = 10+)", ylabel="Response Time (s)")
    ax.legend(title="Material Imbalance (signed)", fontsize=LEGEND_FONTSIZE,
              title_fontsize=LEGEND_FONTSIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    _annotate_n(fig, n_rows)
    save_figure(fig, "board", filename)


def correlation_matrix(conn, features, table, filename="board_feature_corr.pdf", smoke: bool = False):
    """Spearman AND Pearson correlation matrices over log(RT), ply, and every
    covariate, computed in SQL over the full windowed table (Spearman = Pearson on
    tie-corrected ranks). Pearson saves with a ``_pearson`` suffix.

    Also registers each feature's Pearson r vs. log(RT) (the row/column reports
    quote in prose) to the pgfvals registry — skipped when ``smoke=True``, so a
    ~50K-row iteration sample can never overwrite the canonical numbers."""
    labels = {"log_rt": "log(RT)", "move_ply": "Ply"}
    labels.update({col: lbl for col, (lbl, _, _) in features.items()})
    cols = list(labels)

    terms = ["ln(move_time) AS log_rt", "move_ply::DOUBLE AS move_ply"]
    for col, (_, kind, _) in features.items():
        terms.append(f"COALESCE({col}, FALSE)::INT::DOUBLE AS {col}" if kind == "bin" else f"{col}::DOUBLE AS {col}")
    # Correlation is stable at far less than the full ~89M rows; a 1M-row sample keeps
    # this fast and memory-light (the rank() window below runs over every column,
    # TWICE, for Spearman AND Pearson — full-table this needed 128G). Already-smoke-
    # sized input (~50K) is left as-is (no point sampling a sample).
    source = f"(SELECT * FROM {table} USING SAMPLE 1000000 ROWS)" if not smoke else table
    conn.execute(f"CREATE OR REPLACE TEMP TABLE _corr_base AS SELECT {', '.join(terms)} FROM {source}")
    n_rows = conn.execute("SELECT count(*) FROM _corr_base").fetchone()[0]
    # Average ranks (tie-corrected) so Pearson-on-ranks equals Spearman.
    ranks = [f"rank() OVER (ORDER BY {c}) + (count(*) OVER (PARTITION BY {c}) - 1) / 2.0 AS {c}" for c in cols]
    conn.execute(f"CREATE OR REPLACE TEMP TABLE _corr_rank AS SELECT {', '.join(ranks)} FROM _corr_base")

    def matrix(source):
        pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
        vals = conn.execute(
            f"SELECT {', '.join(f'corr({cols[i]}, {cols[j]})' for i, j in pairs)} FROM {source}"
        ).fetchone()
        m = np.eye(len(cols))
        for (i, j), v in zip(pairs, vals):
            m[i, j] = m[j, i] = v
        return m

    display = [labels[c] for c in cols]
    base, ext = os.path.splitext(filename)
    matrices = [("Spearman ρ", "", matrix("_corr_rank")), ("Pearson r", "_pearson", matrix("_corr_base"))]
    if not smoke:
        pearson_m = matrices[1][2]
        log_rt_row = cols.index("log_rt")
        for j, col in enumerate(cols):
            if col == "log_rt":
                continue
            pgf_set(f"board/corr/pearson/{col}", pearson_m[log_rt_row, j], "{:+.3f}")
    for mlabel, suffix, m in matrices:
        n = len(cols)
        apply_poster_style()
        side = max(9, 1.6 * n)  # grow with the feature count so cells stay legible
        fig, ax = plt.subplots(figsize=(side, side * 0.83))
        ax.grid(False)
        im = ax.imshow(m, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(display, fontsize=18, rotation=30, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(display, fontsize=18)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=15,
                        color="white" if abs(m[i, j]) > 0.5 else "black",
                        fontweight="bold" if i == j else "normal")
        highlight_corr_row(ax, n)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(mlabel, fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        plt.tight_layout()
        _annotate_n(fig, n_rows)
        save_figure(fig, "board", f"{base}{suffix}{ext}")


def feature_histograms(conn, features, table, filename: str = "feature_histograms.pdf"):
    """Marginal distribution of each covariate over the full windowed table, one
    panel per feature (same drawing code as bivariate_analysis' left panel)."""
    n_rows = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    cols = list(features)
    ncol = 3
    nrow = math.ceil(len(cols) / ncol)
    apply_poster_style()
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 6.2 * nrow), constrained_layout=True,
                              gridspec_kw={"wspace": 0.35})
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(cols):]:
        ax.axis("off")

    for ax, col in zip(axes, cols):
        label, kind, clip = features[col]
        _draw_feature_histogram(conn, table, col, kind, clip, ax, name=label)
    _annotate_n(fig, n_rows)
    save_figure(fig, "board", filename)


def run_plot(db: str, smoke: bool = False) -> None:
    """The four board analyses over filtered_moves ⋈ board_features.

    ``smoke=True`` runs the SAME analyses against a ~50,000-row DuckDB-native
    reservoir sample (``USING SAMPLE 50000 ROWS`` — a real random sample, not a
    ``LIMIT`` that would just take the first N rows in file order; DuckDB's
    Bernoulli method doesn't support a discrete row count, only a percentage,
    so reservoir is the right method for a "give me exactly N rows" sample) of
    both the windowed view and the whole-game table, for fast local iteration. Outputs get a
    ``smoke_`` filename prefix (matching the established convention); every n=
    annotation still reflects the actual observed post-sample row count (every
    count is a live ``count(*)``/analyzer row count — never the nominal 50,000)."""
    # Covariates analyzed against RT. (label, kind, display_clip); kind ∈ {cont, disc,
    # bin} drives histogram style (bivariate binning is quantile for every covariate;
    # "bin" kind additionally renders as bar+CI, not a trend line); clip trims DISPLAY
    # tails only (feature_histograms; never drops rows).
    features = {
        "n_possible_moves":      ("Legal Moves", "disc", (0, 60)),
        "player_clock_time":     ("Player Clock (s)", "cont", (0, 600)),
        "game_fraction":         ("Game Fraction", "cont", (0.0, 1.0)),
        "n_captures_avail":      ("Captures Available", "disc", (0, 10)),
        "n_checks_avail":        ("Checks Available", "disc", (0, 8)),
        "self_material":         ("Self Material", "disc", (14, 40)),
        "material_imbalance":    ("Material Imbalance", "disc", (-15, 15)),
        "abs_material_imbalance": ("Material Imbalance (Absolute)", "disc", (0, 15)),
    }
    # Supplementary confound-removed dashboards: the SAME bivariate_analysis, on a
    # filtered sub-population, so the plot itself demonstrates a wrinkle's cause
    # instead of needing a table in the report (e.g. legal-moves' 9-10 dip is an
    # in-check composition-shift artifact — supp_n_legal shows it vanish once
    # in-check rows are excluded). Add more entries here as new wrinkles are found.
    supplements = {
        "supp_n_legal": dict(column="n_possible_moves", name="Legal Moves (Excl. In Check)",
                             kind="disc", clip=(0, 60), filter_sql="NOT in_check"),
        # abs_material_imbalance's staggered dips at |3| and |9| (board.md "staggered
        # dips"): the mover-behind-side is checked FIRST since it's who dominates the
        # population there (7.2:1 / 23.2:1 behind:ahead at |3|/|9|) — restricting to
        # that side alone (still) shows the dip, so side-mixing alone doesn't resolve it.
        "supp_abs_material_imbalance_behind": dict(
            column="abs_material_imbalance", name="Material Imbalance (Absolute, Mover Behind)",
            kind="disc", clip=(0, 15), filter_sql="material_imbalance < 0"),
        # A second, narrower composition candidate: rows reachable via a PURE one-sided
        # material hang (the OTHER side's army is still fully intact — self_material=39
        # or, symmetrically, opponent_material=39, computed as self_material -
        # material_imbalance since opponent_material isn't its own column) rather than a
        # multi-piece trade sequence that nets to the same total. |3| and |9| are exactly
        # the totals reachable by losing ONE piece (a minor, a queen) outright, so this
        # subpopulation's share should be structurally elevated there vs. neighbors.
        "supp_abs_material_imbalance_excl_hangs": dict(
            column="abs_material_imbalance", name="Material Imbalance (Absolute, Excl. One-Sided Hangs)",
            kind="disc", clip=(0, 15),
            filter_sql="NOT (self_material = 39 OR (self_material - material_imbalance) = 39)"),
        # n_checks_avail's inverted-U (board.md "is the inverted-U real?"): re-test the
        # SAME in-check exclusion that resolved legal-moves' dip, in case a mover already
        # in check (restricted to block/capture/king-move) drags the checks-available
        # curve the same way.
        "supp_checks_avail_no_incheck": dict(
            column="n_checks_avail", name="Checks Available (Excl. In Check)",
            kind="disc", clip=(0, 8), filter_sql="NOT in_check"),
        # Second candidate: does the inverted-U survive once decisively-ahead/behind
        # positions (where checks-available's material-band story above shows a very
        # different RT profile) are excluded, leaving only roughly-balanced positions?
        "supp_checks_avail_balanced": dict(
            column="n_checks_avail", name="Checks Available (|Imbalance| < 5)",
            kind="disc", clip=(0, 8), filter_sql="abs_material_imbalance < 5"),
    }
    with db_connection(db, read_only=True) as conn:
        # The one canonical view every analysis reads: the windowed moves (with
        # game_fraction from the filter stage) joined to the featurized captures/checks.
        # material_imbalance keeps its natural SIGNED value (self − opponent, mover POV)
        # AND gets an abs_material_imbalance twin — both are tracked features below, so
        # the report can show "is the mover ahead or behind" (signed) alongside "how
        # decided is the position" (magnitude) as two separate curves.
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW board_view AS "
            f"SELECT m.*, abs(m.material_imbalance) AS abs_material_imbalance, "
            f"       b.n_captures_avail, b.n_checks_avail "
            f"FROM {CONFIG['table_filtered']} m LEFT JOIN {CONFIG['table_board_features']} b USING (fen)"
        )
        table, ply_table, prefix = "board_view", None, ""
        if smoke:
            # Materialize as a TABLE, not a VIEW: a view would re-run the (expensive,
            # full-table-scanning) reservoir sample on EVERY downstream query — 9
            # bivariate dashboards + correlation matrix + histograms all re-sampling
            # the ~89M-row join from scratch, which is both slow and, run concurrently
            # with other jobs, OOM-prone. A table takes the sample once.
            conn.execute(
                "CREATE OR REPLACE TEMP TABLE board_view_smoke AS "
                "SELECT * FROM board_view USING SAMPLE 50000 ROWS"
            )
            conn.execute(
                "CREATE OR REPLACE TEMP TABLE processed_moves_nonzero_smoke AS "
                f"SELECT * FROM {CONFIG['table_processed_moves_nonzero']} USING SAMPLE 50000 ROWS"
            )
            table, ply_table, prefix = "board_view_smoke", "processed_moves_nonzero_smoke", "smoke_"
            n_smoke = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"board analysis: --smoke sampled {n_smoke:,} rows (requested 50,000) from board_view")

        print("board analysis: RT distribution summary...")
        move_time_summary(conn, table, ply_table=ply_table, filename=f"{prefix}rt_distribution.pdf", smoke=smoke)

        print("board analysis: bivariate RT-vs-covariate dashboards...")
        for col, (label, kind, clip) in features.items():
            bivariate_analysis(conn, column=col, name=label, filename=f"{prefix}bivariate_{col}.pdf",
                               table=table, kind=kind, clip=clip)

        print("board analysis: supplementary confound-removed dashboards...")
        for key, cfg in supplements.items():
            filt_view = f"{table}_{key}_src"
            conn.execute(f"CREATE OR REPLACE TEMP VIEW {filt_view} AS SELECT * FROM {table} WHERE {cfg['filter_sql']}")
            bivariate_analysis(conn, column=cfg["column"], name=cfg["name"], filename=f"{prefix}{key}.pdf",
                               table=filt_view, kind=cfg["kind"], clip=cfg["clip"])

        print("board analysis: checks-available x material-imbalance interaction views...")
        checks_material_interaction_heatmap(conn, table, filename=f"{prefix}checks_material_interaction_heatmap.pdf")
        checks_material_band_curves(conn, table, filename=f"{prefix}checks_material_band_curves.pdf")

        print("board analysis: correlation matrix...")
        correlation_matrix(conn, features, table, filename=f"{prefix}board_feature_corr.pdf", smoke=smoke)

        print("board analysis: feature histograms...")
        feature_histograms(conn, features, table, filename=f"{prefix}feature_histograms.pdf")

    if not smoke:
        # outputs/reports/ is a sibling of outputs/figures/ (CONFIG["figures_dir"] is
        # .../outputs/figures/<run_name>) — one canonical file, not per-run-namespaced,
        # since the numbers a report cites should always resolve to the latest real run.
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(CONFIG["figures_dir"].rstrip("/"))), "reports")
        write_pgf_tex(os.path.join(reports_dir, "board_stats.tex"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Board-level response time analyses")
    parser.add_argument("--db", default=CONFIG["selected_db_default"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--featurize", action="store_true",
                      help="featurize a hash-slice of filtered_moves' FENs → shard (single or Slurm array)")
    mode.add_argument("--merge", action="store_true", help="merge featurize shards → board_features table")
    mode.add_argument("--plot", action="store_true", help="run the analyses/figures (default)")
    parser.add_argument("--smoke", action="store_true",
                        help="--plot against a ~50,000-row DuckDB sample (not the full table) "
                             "for fast local iteration; outputs get a smoke_ filename prefix")
    args = parser.parse_args(argv)
    print(f"Board {('featurize' if args.featurize else 'merge' if args.merge else 'plot')}"
          f"{' (smoke)' if args.smoke else ''}: "
          f"window [{CONFIG['min_ply']}, {CONFIG['max_ply']}], db={args.db}")

    if args.featurize:
        run_featurize(args.db)
    elif args.merge:
        run_merge(args.db)
    else:
        run_plot(args.db, smoke=args.smoke)


if __name__ == "__main__":
    main()
