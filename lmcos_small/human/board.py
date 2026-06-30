"""
Unified board-level response time analyses (DuckDB dashboards).
Consolidates the baseline distribution, bivariate, and premove analyses
under reports/board.md. Exposes a clean CLI that always runs all analyses.

All generated plots are saved in PDF format under figures/board/.
"""

import argparse
import duckdb
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Align imports with the lmcos_small/human package structure
from utils import Variable, Analyzer
from utils.helpers import (
    apply_poster_style,
    db_connection,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    MAIN_COLOR,
    CONFIG,
)
from utils.plots import (
    highlight_corr_row,
    plot_histogram_from_bins,
    save_figure,
)
from utils.selected_db import (
    SELECTED_DB_DEFAULT,
    TABLE_PROCESSED_MOVES,
    TABLE_PROCESSED_MOVES_NONZERO,
)


def run_move_time_summary(conn):
    """Response time distribution: log(RT) histogram (left) + normal QQ plot (right).
    Collapsed from move_time_summary.py.
    """
    n_bins = CONFIG["response_time_histogram_bins"]
    n_qq = CONFIG["qq_plot_quantile_probes"]

    n_moves = conn.execute(f"SELECT count(*) FROM {TABLE_PROCESSED_MOVES_NONZERO}").fetchone()[0]
    print(f"Running response time summary: n = {n_moves:,} moves")

    conn.execute(
        "CREATE OR REPLACE TEMPORARY VIEW _summary_view AS "
        f"SELECT ln(move_time) AS ln_move_time FROM {TABLE_PROCESSED_MOVES_NONZERO}"
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

    # Helper weighted mean/median calculation
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


def run_bivariate_analysis(conn, column: str, name: str, filename: str, filter_query: str | None = None, n_bins: int = 10, tie_safe: bool = True):
    """A generic bivariate analyzer running quantile-bin dashboards for arbitrary fields."""
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column=column, is_log=False, name=name),
        y_var=Variable(column="move_time", is_log=True, name="RT"),
        filter_query=filter_query,
        title=name,
        n_bins=n_bins,
        tie_safe=tie_safe,
        min_bin_count=100 if tie_safe else 0,
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
               n_self_pieces_exc_pawns AS own_material,
               player_clock_time AS player_clock, ln(move_time) AS log_T
        FROM {TABLE_PROCESSED_MOVES_NONZERO}
        USING SAMPLE {n_sample} ROWS (reservoir, {seed})
    """).df()
    labels = {
        "log_T": "log(RT)", "ply": "Ply", "legal_moves": "Legal moves",
        "own_material": "Own material", "player_clock": "Player clock",
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


def run_ply_premove(conn):
    """Arc of probability of instant moves vs ply.
    Collapsed from ply_premove.py.
    """
    x_var = Variable(column="move_ply", is_log=False, name="Move Ply")
    y_var = Variable(
        column="(CASE WHEN move_time = 0 THEN 1 ELSE 0 END)", 
        is_log=False, 
        name="Pr. Instant Move"
    )
    
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES,
        x_var=x_var, 
        y_var=y_var, 
        filter_query="move_ply <= 150",
        title="Instant Move Arc: Probability vs Game Stage"
    )
    fig = plt.figure(figsize=(30, 13.72))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    analyzer.plot_quantile_bins(ax1)
    analyzer.plot_quantile_bins_tertile_segmented(ax2)
    fig.suptitle(f"{analyzer.title}\nn = {analyzer.n_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10, y=1.0)
    fig.subplots_adjust(top=0.80)
    save_figure(fig, "board", "ply_vs_pinstant.pdf")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unified board-level response time analyses")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument("--all", action="store_true", default=True, help="Run all analyses (default/always)")
    args = parser.parse_args(argv)

    analyses = {
        "summary": run_move_time_summary,
        "clock": {
            "column": "player_clock_time",
            "name": "Player Clock Pressure",
            "filename": "clock.pdf",
            "filter_query": "player_clock_time < 600",
            "tie_safe": False,
        },
        "legal_moves": {
            "column": "n_possible_moves",
            "name": "Legal Moves",
            "filename": "legal_moves.pdf",
            "filter_query": "n_possible_moves < 50",
        },
        "own_material": {
            "column": "n_self_pieces_exc_pawns",
            "name": "Own Material",
            "filename": "own_material.pdf",
            "filter_query": "n_self_pieces_exc_pawns IS NOT NULL",
        },
        "ply": {
            "column": "move_ply",
            "name": "Game Stage",
            "filename": "ply.pdf",
            "filter_query": "move_ply <= 150",
        },
        "boardcorr": run_board_corr,
        "premove": run_ply_premove,
    }

    with db_connection(args.db, read_only=True) as conn:
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
                )


if __name__ == "__main__":
    main()
