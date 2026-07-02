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

    apply_poster_style()
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(30, 13.72))

    def _weighted_mean(df):
        mid = (df["bin_left"] + df["bin_right"]) / 2
        return (mid * df["n"]).sum() / df["n"].sum()

    def _weighted_median(df):
        cumsum = df["n"].cumsum()
        idx = (cumsum >= df["n"].sum() / 2).idxmax()
        return float((df["bin_left"].iloc[idx] + df["bin_right"].iloc[idx]) / 2)

    plot_histogram_from_bins(
        ax_h, df_lmt, x_label="log(RT)",
        mean=_weighted_mean(df_lmt), median=_weighted_median(df_lmt),
    )

    ax_q.scatter(theo_q, emp_q, color=MAIN_COLOR, s=40, zorder=3)
    ax_q.plot(theo_q, ref, color="#C0392B", lw=3, label="Normal reference")
    ax_q.set_xlabel("Theoretical normal quantile", fontsize=FONT_SIZE_LABEL)
    ax_q.set_ylabel("log(RT) quantile", fontsize=FONT_SIZE_LABEL)
    ax_q.legend(fontsize=FONT_SIZE_TICKS, loc="upper left", frameon=False)

    fig.suptitle(f"Response Time (log) — distribution & normal QQ\nn = {n_moves:,} moves",
                 fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    fig.subplots_adjust(top=0.80)

    save_figure(fig, "board", "rt_distribution.pdf")


def run_bivariate_analysis(conn, column: str, name: str, filename: str, filter_query: str | None = None, n_bins: int = 10, tie_safe: bool = True, table: str = WIN_PROCESSED_MOVES_NONZERO, ply_tertile_source: str = WIN_PROCESSED_MOVES_NONZERO):
    """A generic bivariate analyzer running quantile-bin dashboards for arbitrary fields."""
    analyzer = Analyzer(
        db_conn=conn,
        table_name=table,
        x_var=Variable(column=column, is_log=False, name=name),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        filter_query=filter_query,
        title=name,
        n_bins=n_bins,
        tie_safe=tie_safe,
        min_bin_count=100 if tie_safe else 0,
        # Ply tertiles conditioned on the windowed subset (not the whole dataset).
        ply_tertile_source=ply_tertile_source,
    )
    fig = plt.figure(figsize=(30, 13.72))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    analyzer.plot_quantile_bins(ax1)
    analyzer.plot_quantile_bins_tertile_segmented(ax2)
    fig.suptitle(f"{analyzer.title}\nn = {analyzer.n_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    fig.subplots_adjust(top=0.80)
    save_figure(fig, "board", filename)


def run_board_corr(conn, n_sample=1_000_000, seed=42):
    """Spearman correlation over board structural features + log(RT).
    Collapsed from movetime_analysis.py.
    """
    df = conn.execute(f"""
        SELECT move_ply AS ply, n_possible_moves AS legal_moves,
               player_clock_time AS player_clock, ln(move_time) AS log_T,
               move_ply * 1.0 / max(move_ply) OVER (PARTITION BY gid) AS game_fraction
        FROM {WIN_PROCESSED_MOVES_NONZERO}
        USING SAMPLE {n_sample} ROWS (reservoir, {seed})
    """).df()
    labels = {
        "log_T": "log(RT)", "ply": "Ply", "legal_moves": "Legal moves",
        "player_clock": "Player clock", "game_fraction": "Game fraction",
    }
    corr = df[list(labels)].corr(method="spearman").rename(columns=labels, index=labels)
    n = len(corr)
    apply_poster_style()
    fig, ax = plt.subplots(figsize=(9, 7.5))
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
    cbar.set_label("Spearman ρ", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(f"Spearman correlation — board features (n = {len(df):,})", fontsize=18, pad=12)
    plt.tight_layout()
    save_figure(fig, "board", "board_feature_corr.pdf")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified board-level response time analyses")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--config", help="Path to the run config (else $CONFIG or the default).")
    parser.add_argument("--all", action="store_true", default=True, help="Run all analyses (default/always)")
    args = parser.parse_args(argv)
    print(f"Board analysis: ply window [{CONFIG['min_ply']}, {CONFIG['max_ply']}], db={args.db}")

    analyses = {
        "summary": run_move_time_summary,
        "ply": {
            "column": "move_ply",
            "name": "Game Stage",
            "filename": "ply.pdf",
            "filter_query": "move_ply <= 150",
        },
        "legal_moves": {
            "column": "n_possible_moves",
            "name": "Legal Moves",
            "filename": "legal_moves.pdf",
            "filter_query": "n_possible_moves < 50",
        },
        "clock": {
            "column": "player_clock_time",
            "name": "Player Clock Pressure",
            "filename": "clock.pdf",
            "filter_query": "player_clock_time < 600",
            "tie_safe": False,
        },
        "game_fraction": {
            "column": "game_fraction",
            "name": "Game Fraction",
            "filename": "game_fraction.pdf",
            "tie_safe": True,
            # game_fraction lives on pmnz_gf (built on top of the ply window).
            "table": "pmnz_gf",
            "ply_tertile_source": "pmnz_gf",
        },
        "boardcorr": run_board_corr,
    }

    with db_connection(args.db, read_only=True) as conn:
        # Apply the ply window ON ARRIVAL: every analysis reads these views.
        create_ply_windowed_views(conn)
        # game_fraction is computed on top of the windowed view so the ply
        # window still applies; max(move_ply) per game gives the total plies.
        conn.execute(
            "CREATE OR REPLACE TEMP VIEW pmnz_gf AS "
            "SELECT *, move_ply * 1.0 / max(move_ply) OVER (PARTITION BY gid) AS game_fraction "
            f"FROM {WIN_PROCESSED_MOVES_NONZERO}"
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
                    table=config.get("table", WIN_PROCESSED_MOVES_NONZERO),
                    ply_tertile_source=config.get("ply_tertile_source", WIN_PROCESSED_MOVES_NONZERO),
                )


if __name__ == "__main__":
    main()
