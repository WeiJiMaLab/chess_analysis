"""
Move-time bivariate analyses (DuckDB dashboards).

Each analysis saves one figure to figures/{x_var}_vs_{y_var}.png.
Run everything via main(), or import and call individually.
"""

from __future__ import annotations

import argparse
import os

import duckdb

from utils import Variable, Analyzer
from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES_NONZERO

DEFAULT_ANALYSES = (
    "clock",
    "clock_opp",
    "npossiblemoves",
    "pieces_exc",
    "self_pieces_exc",
    "ply",
)


def _fig(src_dir: str, name: str) -> str:
    return os.path.join(src_dir, "figures", name)


def _src_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def npossiblemoves_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column="n_possible_moves", is_log=False, name="Number of Legal Moves"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query="n_possible_moves < 50",
        title="Branching Factor",
    )
    analyzer.save_dashboard(_fig(src_dir, "npossiblemoves_vs_movetime.png"))


def pieces_exc_pawns_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column="n_pieces_on_board_exc_pawns", is_log=False, name="Pieces (excl. pawns)"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query="n_pieces_on_board_exc_pawns IS NOT NULL",
        title="Non-Pawn Material",
    )
    analyzer.save_dashboard(_fig(src_dir, "pieces_exc_pawns_vs_movetime.png"))


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


def clock_movetime(
    conn: duckdb.DuckDBPyConnection,
    src_dir: str | None = None,
    *,
    opponent: bool = False,
) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    player = "opponent" if opponent else "player"
    clock_col = f"{player}_clock_time"
    analyzer = Analyzer(
        db_conn=conn,
        table_name=TABLE_PROCESSED_MOVES_NONZERO,
        x_var=Variable(column=clock_col, is_log=False, name=f"{player.capitalize()} Clock"),
        y_var=Variable(column="move_time", is_log=True, name="T"),
        filter_query=f"{clock_col} < 600",
        title=f"{'Opponent' if opponent else 'Player'} Clock Pressure",
    )
    fname = "clock_opp_vs_movetime.png" if opponent else "clock_vs_movetime.png"
    analyzer.save_dashboard(_fig(src_dir, fname))


def _run_duckdb_analysis(name: str, conn: duckdb.DuckDBPyConnection, src_dir: str) -> None:
    if name == "npossiblemoves":
        npossiblemoves_movetime(conn, src_dir)
    elif name == "pieces_exc":
        pieces_exc_pawns_movetime(conn, src_dir)
    elif name == "self_pieces_exc":
        self_pieces_exc_pawns_movetime(conn, src_dir)
    elif name == "ply":
        ply_movetime(conn, src_dir)
    elif name == "clock":
        clock_movetime(conn, src_dir, opponent=False)
    elif name == "clock_opp":
        clock_movetime(conn, src_dir, opponent=True)
    else:
        raise ValueError(f"Unknown analysis: {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified move-time analysis dashboards")
    parser.add_argument("--db", default=SELECTED_DB_DEFAULT)
    parser.add_argument(
        "--only", nargs="+",
        choices=["npossiblemoves", "pieces_exc", "self_pieces_exc", "ply", "clock", "clock_opp", "all"],
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
