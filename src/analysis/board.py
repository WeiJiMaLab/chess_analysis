"""
Unified board-level (no-model) response time analyses (DuckDB dashboards).
Bivariate RT-vs-feature dashboards for ply, legal moves, and clock time left,
plus a Spearman correlation matrix over those features and log(RT).

All generated plots are saved (PDF + PNG) under <figures_dir>/board/.
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Align imports with the src/analysis package structure
from analysis.utils import Variable, Analyzer
from analysis.utils.helpers import (
    apply_poster_style,
    db_connection,
    FONT_SIZE_LABEL,
)
from analysis.utils.plots import (
    highlight_corr_row,
    save_figure,
)
from analysis.utils.selected_db import (
    SELECTED_DB_DEFAULT,
    TABLE_PROCESSED_MOVES_NONZERO,
)


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
               player_clock_time AS player_clock, ln(move_time) AS log_T
        FROM {TABLE_PROCESSED_MOVES_NONZERO}
        USING SAMPLE {n_sample} ROWS (reservoir, {seed})
    """).df()
    labels = {
        "log_T": "log(RT)", "ply": "Ply", "legal_moves": "Legal moves",
        "player_clock": "Player clock",
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
    parser.add_argument("--all", action="store_true", default=True, help="Run all analyses (default/always)")
    args = parser.parse_args(argv)

    analyses = {
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
        "boardcorr": run_board_corr,
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
