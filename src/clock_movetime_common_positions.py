"""
Clock-vs-move-time analysis for the most common positions.
Builds one 2x2 figure per position (top-k most common after a ply threshold):
1) Clock (s) vs log T(s) with log-log OLS curve overlaid
2) Quantile-binned clock trend
3) Log-log sampled scatter + OLS line
4) Chess board rendering of the position
"""

import os
import duckdb
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import chess

from utils import (
    apply_poster_style,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    calculate_ols,
    preprocess,
    EPSILON,
)
from utils.plots import plot_qbin_stats, plot_raw_trend, plot_subset_scatterplot

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
LIMIT_N = None  # Use None for full dataset
RAW_TREND_MIN_N = 5
DEFAULT_MIN_PLY = 50
DEFAULT_MAX_PLY = 70
DEFAULT_TOP_K = 5


def _render_board(ax, board_position: str, player_white_mode: bool) -> None:
    """
    Render board as unicode text on a matplotlib axis.
    """
    try:
        turn = "w" if player_white_mode else "b"
        fen = f"{board_position} {turn} - - 0 1"
        board = chess.Board(fen)
        board_txt = board.unicode(empty_square="·")
        ax.text(
            0.5,
            0.5,
            board_txt,
            ha="center",
            va="center",
            family="monospace",
            fontsize=20,
        )
        ax.set_title(f"Board (side to move: {'white' if player_white_mode else 'black'})", fontsize=FONT_SIZE_TICKS)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Board render failed:\n{exc}", ha="center", va="center", fontsize=FONT_SIZE_TICKS)
    ax.axis("off")


def main():
    # 1. Setup
    base_table = "_selected_moves"
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument(
        "--all_T",
        action="store_true",
        help="Use all move times (default uses nonzero move time variant)",
    )
    parser.add_argument("--opp", action="store_true", help="Use opponent clock instead of player clock")
    parser.add_argument("--min_ply", type=int, default=DEFAULT_MIN_PLY, help="Only positions from move_ply >= min_ply")
    parser.add_argument("--max_ply", type=int, default=DEFAULT_MAX_PLY, help="Only positions from move_ply <= max_ply")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of top common positions to analyze")
    args = parser.parse_args()

    player = "player" if not args.opp else "opponent"
    ln_clock_col = f"ln_{player}_clock_time"
    clock_col = f"{player}_clock_time"
    clock_qbin_col = f"{player}_clock_qbin"
    clock_label = f"{player.capitalize()} Clock"

    if not args.all_T:
        base_table = f"{base_table}_nonzero_T"

    # 2. SQL Preprocessing
    if not args.skip_preprocess:
        limit_clause = f"LIMIT {LIMIT_N}" if LIMIT_N is not None else ""
        print(f"Preprocessing moves (limit={LIMIT_N or 'FULL'})...")
        preprocess(conn, target_table=base_table, limit_clause=limit_clause)
    else:
        print("Skipping preprocessing as requested.")

    # 3. Find top positions
    print(f"Finding top {args.top_k} positions with {args.min_ply} <= move_ply <= {args.max_ply}...")
    df_top = conn.execute(
        f"""
        SELECT
            board_position,
            SUM(CASE WHEN player_white THEN 1 ELSE 0 END) AS n_white_to_move,
            SUM(CASE WHEN NOT player_white THEN 1 ELSE 0 END) AS n_black_to_move,
            COUNT(*) AS n
        FROM {base_table}
        WHERE move_ply BETWEEN {args.min_ply} AND {args.max_ply}
        GROUP BY board_position
        ORDER BY n DESC
        LIMIT {args.top_k}
        """
    ).df()

    figures_dir = os.path.join(src_dir, "figures", "clock_movetime_common_positions")
    os.makedirs(figures_dir, exist_ok=True)

    if df_top.empty:
        print("No positions found for the requested filters.")
        conn.close()
        return

    # 4. Per-position analysis and plotting
    for rank, row in df_top.reset_index(drop=True).iterrows():
        board_position = row["board_position"]
        board_position_sql = str(board_position).replace("'", "''")
        position_count = int(row["n"])
        player_white_mode = bool(row["n_white_to_move"] >= row["n_black_to_move"])
        print(f"Analyzing position {rank + 1}/{len(df_top)} (n={position_count})...")

        conn.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW _position_moves AS
            SELECT *
            FROM {base_table}
            WHERE move_ply BETWEEN {args.min_ply} AND {args.max_ply}
              AND board_position = '{board_position_sql}'
            """,
        )

        # Skip if not enough support in the selected subset
        n_rows = conn.execute("SELECT count(*) FROM _position_moves").fetchone()[0]
        if n_rows < 30:
            print(f"Skipping position rank {rank + 1}: too few rows ({n_rows})")
            continue

        conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE _clock_stats AS
            SELECT
                {ln_clock_col},
                AVG({clock_col}) AS mean_clock,
                AVG(ln_move_time) AS mean_y,
                STDDEV(ln_move_time) AS std_y,
                COUNT(*) AS n
            FROM _position_moves
            GROUP BY {ln_clock_col}
            """
        )

        conn.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE _clock_qstats AS
            SELECT
                {clock_qbin_col},
                AVG({clock_col}) AS mean_clock,
                AVG(ln_move_time) AS mean_y,
                STDDEV(ln_move_time) AS std_y,
                COUNT(*) AS n
            FROM _position_moves
            GROUP BY {clock_qbin_col}
            """
        )

        slope_global, intercept_global = calculate_ols(conn, "_position_moves", ln_clock_col, "ln_move_time")
        n_games = conn.execute("SELECT count(distinct gid) FROM _position_moves").fetchone()[0]
        n_moves = conn.execute("SELECT count(*) FROM _position_moves").fetchone()[0]

        df_raw = conn.execute(f"SELECT {ln_clock_col}, ln_move_time FROM _position_moves").df()
        df_stats = conn.execute("SELECT * FROM _clock_stats ORDER BY mean_clock").df()
        df_qstats = conn.execute(f"SELECT * FROM _clock_qstats ORDER BY {clock_qbin_col}").df()

        apply_poster_style()
        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        axes = axes.flatten()

        plot_raw_trend(
            axes[0],
            df_stats,
            x_col="mean_clock",
            y_col="mean_y",
            std_col="std_y",
            n_col="n",
            x_label=f"{clock_label} (s)",
            y_label="log T (s)",
            min_n=RAW_TREND_MIN_N,
            show_legend=False,
        )

        # Same model as log-log OLS, transformed onto raw clock axis.
        x_raw_min, x_raw_max = df_stats["mean_clock"].min(), df_stats["mean_clock"].max()
        x_raw_lin = np.linspace(x_raw_min, x_raw_max, 300)
        y_pred_raw = slope_global * np.log(x_raw_lin + EPSILON) + intercept_global
        axes[0].plot(x_raw_lin, y_pred_raw, color="red", lw=4, label=f"OLS Fit (β={slope_global:.2f})", linestyle="--")

        plot_qbin_stats(
            axes[1],
            df_qstats,
            x_col="mean_clock",
            x_label=f"Qrank {clock_label}",
            y_label="log T (s)",
            normalized=False,
            show_legend=False,
        )

        plot_subset_scatterplot(
            axes[2],
            df_raw,
            ln_clock_col,
            "ln_move_time",
            x_label=f"log {clock_label} (s)",
            y_label="log T (s)",
            n=10000,
        )
        x_log_min, x_log_max = df_raw[ln_clock_col].min(), df_raw[ln_clock_col].max()
        x_log_lin = np.linspace(x_log_min, x_log_max, 300)
        y_pred_log = slope_global * x_log_lin + intercept_global
        axes[2].plot(x_log_lin, y_pred_log, color="red", lw=4, label=f"OLS Fit (β={slope_global:.2f})", linestyle="--")
        axes[2].legend(fontsize=FONT_SIZE_TICKS)

        _render_board(axes[3], board_position, player_white_mode)

        title_text = f"rank {rank + 1} | occ={position_count:,} | {n_games:,} games | {n_moves:,} moves"
        fig.suptitle(title_text, fontsize=FONT_SIZE_LABEL + 8, y=1.02)

        plt.tight_layout()
        name = f"position_{rank + 1:02d}"
        if not args.all_T:
            name += "_nonzero_T"
        if args.opp:
            name += "_opp"
        out_path = os.path.join(figures_dir, f"{name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Saved {out_path}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
