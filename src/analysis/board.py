"""
Board-level (no-model) response time analyses (DuckDB + matplotlib).

Three ordered modes (each depends on the previous — mirror this in the pipeline):
  --featurize  precondition: featurize the legal-move covariates (captures/checks)
               over this task's hash-slice of the run's ``filtered_moves`` FENs →
               a shard. Runs single, or as a Slurm array (task per hash partition).
  --merge      merge the featurize shards → the ``board_features`` table (single).
  --plot       (default) the four analyses over filtered_moves ⋈ board_features:
                 move_time_summary  log(RT) distribution + normal QQ + RT-vs-ply arc
                 bivariate_analysis RT vs each covariate, overall + by ply tertile
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

from analysis.utils import Variable, Analyzer
from analysis.utils.analysis import _seconds_from_log
from analysis.utils.helpers import (
    apply_poster_style, db_connection, sql_str,
    FONT_SIZE_LABEL, MAIN_COLOR, CONFIG,
)
from analysis.utils.plots import highlight_corr_row, save_figure

FEATURE_COLS = ["n_captures_avail", "n_checks_avail"]


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

def move_time_summary(conn, table):
    """log(RT) distribution histogram + normal QQ + mean-RT-vs-ply arc (whole game)."""
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
    theoretical = stats.norm.ppf(probs)  # standard-normal quantiles

    def _weighted_median(df):
        cumsum = df["n"].cumsum()
        idx = (cumsum >= df["n"].sum() / 2).idxmax()
        return float((df["bin_left"].iloc[idx] + df["bin_right"].iloc[idx]) / 2)

    med_log = _weighted_median(lmt_bins)

    apply_poster_style()
    fig, (ax_h, ax_q, ax_p) = plt.subplots(1, 3, figsize=(34, 11), constrained_layout=True)

    # Panel 1: RT distribution in SECONDS on a log x-axis (bins uniform in ln(RT),
    # so exponentiating the edges gives geometric bins that read evenly on a log axis).
    sec = lmt_bins.assign(left_s=np.exp(lmt_bins.bin_left), right_s=np.exp(lmt_bins.bin_right))
    ax_h.bar(sec.left_s, sec.n, width=(sec.right_s - sec.left_s), align="edge",
             color=MAIN_COLOR, alpha=0.5, edgecolor=MAIN_COLOR, linewidth=1.2)
    ax_h.set_xscale("log")
    ax_h.axvline(np.exp(mean), color="black", ls="--", lw=2.5, label=f"mean = {np.exp(mean):.1f}s")
    ax_h.axvline(np.exp(med_log), color="dimgray", ls=":", lw=2.5, label=f"median = {np.exp(med_log):.1f}s")
    ax_h.set(xlabel="RT (s, log axis)", ylabel="count", title="RT distribution")
    ax_h.legend()

    # Panel 2: normal QQ (points only).
    ax_q.scatter(theoretical, empirical, color=MAIN_COLOR, s=40, zorder=3)
    ax_q.set(xlabel="Theoretical quantile", ylabel="Actual quantile", title="Normal QQ")

    # Panel 3: mean ln(RT) vs ply over the WHOLE game (unwindowed), on a linear axis
    # relabeled to seconds (log-spaced positions) so the arc reads evenly.
    ply = conn.execute(
        f"SELECT move_ply, avg(ln(move_time)) AS mean_log, count(*) AS n "
        f"FROM {CONFIG['table_processed_moves_nonzero']} "
        f"WHERE move_time > 0 AND move_ply BETWEEN 1 AND 150 GROUP BY 1 ORDER BY 1"
    ).df()
    ply = ply[ply.n >= 200]
    ax_p.plot(ply.move_ply, ply.mean_log, color=MAIN_COLOR, lw=3)
    ax_p.axvspan(CONFIG["min_ply"], CONFIG["max_ply"], color="gray", alpha=0.12,
                 label=f"analysis window [{CONFIG['min_ply']},{CONFIG['max_ply']}]")
    _seconds_from_log(ax_p.yaxis)
    ax_p.set(xlabel="Ply (whole game)", ylabel="mean RT (s, log axis)",
             title=f"RT vs ply (whole game, n = {int(ply.n.sum()):,})",
             xlim=(1, 150), xticks=[0, 50, 100, 150])
    ax_p.legend()

    fig.suptitle(f"Response time — distribution + normal QQ (windowed n = {n_moves:,}), RT vs ply (whole game)",
                 fontsize=FONT_SIZE_LABEL + 4)
    save_figure(fig, "board", "rt_distribution.pdf")


def bivariate_analysis(conn, column: str, name: str, filename: str, table: str,
                       analyzer_opts: dict | None = None, reverse_x: bool = False):
    """RT-vs-covariate 1x2 dashboard: overall (left), colored by ply tertile (right).
    Bins x by quantile (tie-safe, so a repeated value never straddles a bin edge) —
    the uniform scheme for every covariate, discrete or continuous.

    ``analyzer_opts`` are forwarded to ``Analyzer`` for a caller that needs to
    override the default binning for a specific covariate."""
    kw = dict(
        db_conn=conn,
        table_name=table,
        x_var=Variable(column=column, is_log=False, name=name),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        title=name,
        n_bins=10,
        tie_safe=True,
        min_bin_count=100,
        ply_tertile_source=table,     # tertiles conditioned on the windowed table
        segment_column="move_ply",
        segment_source=table,
        segment_label="Ply",
    )
    kw.update(analyzer_opts or {})
    analyzer = Analyzer(**kw)

    fig = plt.figure(figsize=(30, 13.72))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    analyzer.plot_quantile_bins(ax1)                     # overall
    analyzer.plot_quantile_bins_tertile_segmented(ax2)   # color: ply
    ax1.set(title="overall")
    ax2.set(title="by ply")
    if reverse_x:
        for ax in (ax1, ax2):
            ax.invert_xaxis()
    fig.suptitle(f"{analyzer.title}\nn = {analyzer.n_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    fig.subplots_adjust(top=0.80, wspace=0.28)
    save_figure(fig, "board", filename)


def correlation_matrix(conn, features, table, filename="board_feature_corr.pdf"):
    """Spearman AND Pearson correlation matrices over log(RT), ply, and every
    covariate, computed in SQL over the full windowed table (Spearman = Pearson on
    tie-corrected ranks). Pearson saves with a ``_pearson`` suffix."""
    labels = {"log_rt": "log(RT)", "move_ply": "Ply"}
    labels.update({col: lbl for col, (lbl, _, _) in features.items()})
    cols = list(labels)

    terms = ["ln(move_time) AS log_rt", "move_ply::DOUBLE AS move_ply"]
    for col, (_, kind, _) in features.items():
        terms.append(f"COALESCE({col}, FALSE)::INT::DOUBLE AS {col}" if kind == "bin" else f"{col}::DOUBLE AS {col}")
    conn.execute(f"CREATE OR REPLACE TEMP TABLE _corr_base AS SELECT {', '.join(terms)} FROM {table}")
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
    for mlabel, suffix, m in (("Spearman ρ", "", matrix("_corr_rank")),
                              ("Pearson r", "_pearson", matrix("_corr_base"))):
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
        ax.set_title(f"{mlabel} — board features (n = {n_rows:,})", fontsize=18, pad=12)
        plt.tight_layout()
        save_figure(fig, "board", f"{base}{suffix}{ext}")


def feature_histograms(conn, features, table):
    """Marginal distribution of each covariate over the full windowed table,
    aggregated in SQL. Discrete/binary → per-value bars; continuous → 50-bin density."""
    n_rows = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    cols = list(features)
    ncol = 3
    nrow = math.ceil(len(cols) / ncol)
    apply_poster_style()
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 6.2 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(cols):]:
        ax.axis("off")

    for ax, col in zip(axes, cols):
        label, kind, clip = features[col]
        where = f"WHERE {col} BETWEEN {clip[0]} AND {clip[1]}" if clip else f"WHERE {col} IS NOT NULL"
        if kind in ("disc", "bin"):
            g = conn.execute(
                f"SELECT {col}::DOUBLE AS value, count(*) AS n FROM {table} {where} GROUP BY 1 ORDER BY 1"
            ).df()
            ax.bar(g.value, g.n / g.n.sum(), width=(0.4 if kind == "bin" else 0.9),
                   color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
        else:
            lo, hi = clip
            width = (hi - lo) / 50
            g = conn.execute(
                f"SELECT least(49, floor(({col} - {lo}) / {width}))::INT AS b, count(*) AS n "
                f"FROM {table} {where} GROUP BY 1 ORDER BY 1"
            ).df()
            ax.bar(lo + (g.b + 0.5) * width, g.n / g.n.sum() / width, width=width,
                   color=MAIN_COLOR, alpha=0.6, edgecolor=MAIN_COLOR)
        ax.set(title=label, ylabel="density")
    fig.suptitle(f"Board features — distributions (n = {n_rows:,} move-instances)", fontsize=FONT_SIZE_LABEL)
    save_figure(fig, "board", "feature_histograms.pdf")


def run_plot(db: str) -> None:
    """The four board analyses over filtered_moves ⋈ board_features."""
    # Covariates analyzed against RT. (label, kind, display_clip); kind ∈ {cont, disc,
    # bin} drives histogram style only (bivariate binning is quantile for every
    # covariate); clip trims DISPLAY tails only (feature_histograms; never drops rows).
    features = {
        "n_possible_moves":      ("Legal moves", "disc", (0, 60)),
        "player_clock_time":     ("Player clock (s)", "cont", (0, 600)),
        "game_fraction":         ("Game fraction", "cont", (0.0, 1.0)),
        "n_captures_avail":      ("Captures available", "disc", (0, 15)),
        "n_checks_avail":        ("Checks available", "disc", (0, 8)),
        "self_material":         ("Self material (weighted)", "disc", (0, 40)),
        "material_imbalance":    ("Material imbalance (weighted)", "disc", (-15, 15)),
        "in_check":              ("In check", "bin", None),
        "prev_move_was_capture": ("Prev move was capture", "bin", None),
    }
    with db_connection(db, read_only=True) as conn:
        # The one canonical view every analysis reads: the windowed moves (with
        # game_fraction from the filter stage) joined to the featurized captures/checks.
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW board_view AS "
            f"SELECT m.*, b.n_captures_avail, b.n_checks_avail "
            f"FROM {CONFIG['table_filtered']} m LEFT JOIN {CONFIG['table_board_features']} b USING (fen)"
        )
        print("board analysis: RT distribution summary...")
        move_time_summary(conn, "board_view")

        print("board analysis: bivariate RT-vs-covariate dashboards...")
        for col, (label, _kind, _clip) in features.items():
            bivariate_analysis(conn, column=col, name=label, filename=f"bivariate_{col}.pdf",
                               table="board_view")

        print("board analysis: correlation matrix...")
        correlation_matrix(conn, features, "board_view")

        print("board analysis: feature histograms...")
        feature_histograms(conn, features, "board_view")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Board-level response time analyses")
    parser.add_argument("--db", default=CONFIG["selected_db_default"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--featurize", action="store_true",
                      help="featurize a hash-slice of filtered_moves' FENs → shard (single or Slurm array)")
    mode.add_argument("--merge", action="store_true", help="merge featurize shards → board_features table")
    mode.add_argument("--plot", action="store_true", help="run the analyses/figures (default)")
    args = parser.parse_args(argv)
    print(f"Board {('featurize' if args.featurize else 'merge' if args.merge else 'plot')}: "
          f"window [{CONFIG['min_ply']}, {CONFIG['max_ply']}], db={args.db}")

    if args.featurize:
        run_featurize(args.db)
    elif args.merge:
        run_merge(args.db)
    else:
        run_plot(args.db)


if __name__ == "__main__":
    main()
