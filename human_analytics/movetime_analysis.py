"""
Move-time bivariate analyses (DuckDB dashboards).

Each analysis saves one figure to figures/{x_var}_vs_{y_var}.png.
Run everything via main(), or import and call individually.
"""

from __future__ import annotations

import argparse
import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import Variable, Analyzer
from utils.helpers import apply_poster_style
from utils.plots import highlight_corr_row
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES_NONZERO

DEFAULT_ANALYSES = (
    "clock",
    "npossiblemoves",
    "self_pieces_exc",
    "ply",
    "boardcorr",
)


def _fig(src_dir: str, name: str) -> str:
    repo_root = os.path.dirname(src_dir)
    return os.path.join(repo_root, "figures", name)


def _src_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def npossiblemoves_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column="n_possible_moves", is_log=False, name="# Legal Moves"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query="n_possible_moves < 50",
        title="Branching Factor",
    )
    analyzer.save_dashboard(_fig(src_dir, "npossiblemoves_vs_movetime.png"))


def self_pieces_exc_pawns_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column="n_self_pieces_exc_pawns", is_log=False, name="Own Non-Pawn Pieces"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query="n_self_pieces_exc_pawns IS NOT NULL",
        title="Own Non-Pawn Material",
    )
    analyzer.save_dashboard(_fig(src_dir, "self_pieces_exc_pawns_vs_movetime.png"))


def ply_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column="move_ply", is_log=False, name="Move Ply"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query="move_ply <= 150",
        title="Game Stage",
    )
    analyzer.save_dashboard(_fig(src_dir, "ply_vs_movetime.png"))


def clock_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column="player_clock_time", is_log=False, name="Player Clock"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query="player_clock_time < 600",
        title="Player Clock Pressure",
    )
    analyzer.save_dashboard(_fig(src_dir, "clock_vs_movetime.png"))


def board_feature_corr(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None,
                       n_sample: int = 1_000_000, seed: int = 42) -> None:
    """Spearman ρ matrix over the board structural features + log(RT) — no engine.
    Rank correlation on a reservoir sample of ``processed_moves_nonzero`` (the full
    135M is unnecessary for a correlation estimate)."""
    if src_dir is None:
        src_dir = _src_dir()
    df = conn.execute(f"""
        SELECT move_ply AS ply, n_possible_moves AS branching,
               n_self_pieces_exc_pawns AS own_material,
               player_clock_time AS player_clock, ln(move_time) AS log_T
        FROM {TABLE_PROCESSED_MOVES_NONZERO}
        USING SAMPLE {n_sample} ROWS (reservoir, {seed})
    """).df()
    labels = {  # log(RT) first (the response variable), then structure → clock
        "log_T": "log(RT)", "ply": "Ply", "branching": "Branching",
        "own_material": "Own material", "player_clock": "Player clock",
    }
    corr = df[list(labels)].corr(method="spearman").rename(columns=labels, index=labels)
    n = len(corr)
    apply_poster_style()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.grid(False)
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, fontsize=18, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=18)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=15,
                    color="white" if abs(v) > 0.5 else "black",
                    fontweight="bold" if i == j else "normal")
    highlight_corr_row(ax, n)  # log(RT) row (index 0) — the response variable
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(f"Spearman correlation — board features (n = {len(df):,})", fontsize=18, pad=12)
    plt.tight_layout()
    out = _fig(src_dir, "board_feature_corr.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ {out}")


def _run_duckdb_analysis(name: str, conn: duckdb.DuckDBPyConnection, src_dir: str) -> None:
    if name == "npossiblemoves":
        npossiblemoves_movetime(conn, src_dir)
    elif name == "self_pieces_exc":
        self_pieces_exc_pawns_movetime(conn, src_dir)
    elif name == "ply":
        ply_movetime(conn, src_dir)
    elif name == "clock":
        clock_movetime(conn, src_dir)
    elif name == "boardcorr":
        board_feature_corr(conn, src_dir)
    else:
        raise ValueError(f"Unknown analysis: {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified move-time analysis dashboards")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument(
        "--only", nargs="+",
        choices=["npossiblemoves", "self_pieces_exc", "ply", "clock", "boardcorr", "all"],
        metavar="NAME",
    )
    args = parser.parse_args(argv)

    selected = list(DEFAULT_ANALYSES) if (args.only is None or args.only == ["all"]) else list(args.only)

    src_dir = _src_dir()
    conn = duckdb.connect(database=args.db, read_only=False)
    try:
        for name in selected:
            _run_duckdb_analysis(name, conn, src_dir)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
