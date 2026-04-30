"""
Move-time bivariate analyses (DuckDB dashboards).

Each analysis is a function; run everything via main(), or import and call individually.
"""

from __future__ import annotations

import argparse
import os

import duckdb

from utils import Variable, Analyzer

PERSONAL_DB_DEFAULT = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

# Default run order (matches slurm script_analysis.sh where applicable).
DEFAULT_ANALYSES = (
    "clock",
    "clock_opp",
    "npossiblemoves",
    "ply",
)


def _src_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def npossiblemoves_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    x_var = Variable(column="n_possible_moves", is_log=False, name="Number of Legal Moves")
    y_var = Variable(column="move_time", is_log=False, name="Move Time (s)")
    analyzer = Analyzer(
        db_conn=conn,
        table_name="_selected_moves_nonzero_T",
        x_var=x_var,
        y_var=y_var,
        filter_query="n_possible_moves < 50",
        title="Branching Factor Influence",
        quantile_heatmap_row="move_ply",
        quantile_heatmap_row_label="Move ply",
    )
    figures_dir = os.path.join(src_dir, "figures", "npossiblemoves_movetime")
    analyzer.save_dashboard(os.path.join(figures_dir, "combined.png"), include_quantile_heatmap=True)


def ply_movetime(conn: duckdb.DuckDBPyConnection, src_dir: str | None = None) -> None:
    if src_dir is None:
        src_dir = _src_dir()
    x_var = Variable(column="move_ply", is_log=False, name="Move Ply")
    y_var = Variable(column="move_time", is_log=True, name="T")
    analyzer = Analyzer(
        db_conn=conn,
        table_name="_selected_moves_nonzero_T",
        x_var=x_var,
        y_var=y_var,
        filter_query="move_ply <= 150",
        title="Thinking Arc: Time vs Game Stage",
    )
    figures_dir = os.path.join(src_dir, "figures", "ply_movetime")
    analyzer.save_dashboard(os.path.join(figures_dir, "combined.png"), layout="1x2")


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
    x_var = Variable(column=clock_col, is_log=False, name=f"{player.capitalize()} Clock")
    y_var = Variable(column="move_time", is_log=True, name="T")
    analyzer = Analyzer(
        db_conn=conn,
        table_name="_selected_moves_nonzero_T",
        x_var=x_var,
        y_var=y_var,
        filter_query=f"{clock_col} < 600",
        title=f"{player.capitalize()} Clock Pressure",
        quantile_heatmap_row="move_ply",
        quantile_heatmap_row_label="Move ply",
    )
    figures_dir = os.path.join(src_dir, "figures", "clock_movetime")
    filename = f"combined{'_opp' if opponent else ''}.png"
    analyzer.save_dashboard(os.path.join(figures_dir, filename), include_quantile_heatmap=True)


def _run_duckdb_analysis(
    name: str,
    conn: duckdb.DuckDBPyConnection,
    src_dir: str,
) -> None:
    if name == "npossiblemoves":
        npossiblemoves_movetime(conn, src_dir)
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
    parser.add_argument(
        "--db",
        default=PERSONAL_DB_DEFAULT,
        help="Path to personal.db (uses table _selected_moves_nonzero_T)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["npossiblemoves", "ply", "clock", "clock_opp", "all"],
        metavar="NAME",
        help="Run only these analyses (default: full DEFAULT_ANALYSES set). Use 'all' as a single token.",
    )
    args = parser.parse_args(argv)

    if args.only is None or (len(args.only) == 1 and args.only[0] == "all"):
        selected = list(DEFAULT_ANALYSES)
    else:
        selected = list(args.only)

    src_dir = _src_dir()
    conn = duckdb.connect(database=args.db, read_only=False)
    try:
        for name in selected:
            _run_duckdb_analysis(name, conn, src_dir)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
