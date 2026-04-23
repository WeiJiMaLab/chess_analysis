"""
Analysis of the relationship between number of legal moves and move time.
This mirrors the clock-vs-move-time 2x2 analysis, but uses raw move time
and only player-side features (no opponent branch).
"""

import os
import duckdb
import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils import (
    apply_poster_style,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    calculate_ols,
    calculate_plywise_betas,
    preprocess,
)
from utils.plots import plot_qbin_stats, plot_beta_vs_ply, plot_raw_trend, plot_subset_scatterplot

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
LIMIT_N = None  # Set to int for debugging on smaller samples
RAW_TREND_MIN_N = 5


def main():
    base_table = "_selected_moves"
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--nonzero_T", action="store_true", help="Use nonzero move time variant")
    args = parser.parse_args()

    if args.nonzero_T:
        base_table = f"{base_table}_nonzero_T"

    if not args.skip_preprocess:
        limit_clause = f"LIMIT {LIMIT_N}" if LIMIT_N is not None else ""
        print(f"Preprocessing moves (limit={LIMIT_N or 'FULL'})...")
        preprocess(conn, target_table="_selected_moves", limit_clause=limit_clause)
    else:
        print("Skipping preprocessing as requested.")
        cols = {
            row[0]
            for row in conn.execute("DESCRIBE _selected_moves").fetchall()
        }
        required_cols = {"n_possible_moves", "n_possible_moves_qbin"}
        missing_cols = required_cols - cols
        if missing_cols:
            conn.close()
            missing_list = ", ".join(sorted(missing_cols))
            raise RuntimeError(
                f"_selected_moves is missing required columns: {missing_list}. "
                "Run once without --skip_preprocess to rebuild the preprocessed table."
            )

    print("Calculating aggregate statistics...")
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE _npossible_stats AS
        SELECT
            n_possible_moves,
            AVG(move_time) AS mean_y,
            STDDEV(move_time) AS std_y,
            COUNT(*) AS n
        FROM {base_table}
        GROUP BY n_possible_moves;
        """
    )

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE _npossible_qstats AS
        SELECT
            n_possible_moves_qbin,
            AVG(n_possible_moves) AS mean_x,
            AVG(move_time) AS mean_y,
            STDDEV(move_time) AS std_y,
            COUNT(*) AS n
        FROM {base_table}
        GROUP BY n_possible_moves_qbin;
        """
    )

    print("Calculating global OLS parameters (SQL-native)...")
    slope_global, intercept_global = calculate_ols(conn, base_table, "n_possible_moves", "move_time")
    print(f"Global OLS: Slope={slope_global:.4f}, Intercept={intercept_global:.4f}")

    print("Calculating per-ply OLS slopes (SQL-native)...")
    df_betas = calculate_plywise_betas(conn, base_table, "n_possible_moves", "move_time")

    n_games = conn.execute(f"SELECT count(distinct gid) FROM {base_table}").fetchone()[0]
    n_moves = conn.execute(f"SELECT count(*) FROM {base_table}").fetchone()[0]
    print(f"Total Games: {n_games:,} | Total Moves: {n_moves:,}")

    print("Loading data for visualization...")
    df_raw = conn.execute(f"SELECT n_possible_moves, move_time FROM {base_table}").df()
    df_stats = conn.execute("SELECT * FROM _npossible_stats ORDER BY n_possible_moves").df()
    df_qstats = conn.execute("SELECT * FROM _npossible_qstats ORDER BY n_possible_moves_qbin").df()
    conn.close()

    apply_poster_style()
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes = axes.flatten()

    plot_raw_trend(
        axes[0],
        df_stats,
        x_col="n_possible_moves",
        y_col="mean_y",
        std_col="std_y",
        n_col="n",
        x_label="Number of Legal Moves",
        y_label="Move Time (s)",
        min_n=RAW_TREND_MIN_N,
        show_legend=False,
    )

    x_raw_min, x_raw_max = df_stats["n_possible_moves"].min(), df_stats["n_possible_moves"].max()
    x_raw_lin = np.linspace(x_raw_min, x_raw_max, 300)
    y_pred_raw = slope_global * x_raw_lin + intercept_global
    axes[0].plot(x_raw_lin, y_pred_raw, color="red", lw=4, label=f"OLS Fit (β={slope_global:.2f})", linestyle="--")

    plot_qbin_stats(
        axes[1],
        df_qstats,
        x_col="mean_x",
        x_label="Qrank Number of Legal Moves",
        y_label="Move Time (s)",
        normalized=False,
        show_legend=False,
    )

    plot_subset_scatterplot(
        axes[2],
        df_raw,
        "n_possible_moves",
        "move_time",
        x_label="Number of Legal Moves",
        y_label="Move Time (s)",
        n=10000,
    )
    axes[2].plot(x_raw_lin, y_pred_raw, color="red", lw=4, label=f"OLS Fit (β={slope_global:.2f})", linestyle="--")
    axes[2].legend(fontsize=FONT_SIZE_TICKS)

    plot_beta_vs_ply(axes[3], df_betas, title=None)

    title_text = f"{n_games:,} games | {n_moves:,} moves"
    fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 10, y=1.02)

    plt.tight_layout()
    figures_dir = os.path.join(src_dir, "figures", "npossiblemoves_movetime")
    os.makedirs(figures_dir, exist_ok=True)

    name = "combined"
    if args.nonzero_T:
        name += "_nonzero_T"
    output_plot = os.path.join(figures_dir, f"{name}.png")
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"\n✅ Combined n-possible-moves trend plot saved to {output_plot}")


if __name__ == "__main__":
    main()
