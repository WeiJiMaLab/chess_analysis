"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

import os

import chess
import chess.svg
import chess.engine
from IPython.display import SVG, display
import datetime
import duckdb

# Default paths (override as needed)
STOCKFISH_PATH = "/home/hl4291/Stockfish/src/stockfish"
STOCKFISH_DIR = os.path.dirname(os.path.abspath(STOCKFISH_PATH))
# NNUE default filenames (must match Stockfish src/evaluate.h; keep .nnue files in STOCKFISH_DIR)
NNUE_BIG = "nn-5227780996d3.nnue"
NNUE_SMALL = "nn-37f18f62d772.nnue"

def display_fen(fen: str, size: int = 200) -> None:
    """Render a FEN position as an SVG and display it in the notebook."""
    board = chess.Board(fen)
    display(SVG(chess.svg.board(board=board, size=size)))


def get_db_connection(
    database: str = "personal.db",
    *,
    threads: int = 10,
    memory_limit: str = "20GB",
    temp_directory: str = ".",
    **kwargs,
):
    """Create a DuckDB connection with default config for analysis."""
    import duckdb

    config = {
        "threads": threads,
        "memory_limit": memory_limit,
        "temp_directory": temp_directory,
        **kwargs,
    }
    return duckdb.connect(database=database, config=config)


def get_stockfish_engine(
    path: str | None = None,
    *,
    cwd: str | None = None,
    threads: int = 1,
    hash_mb: int = 128,
):
    """
    Spawn a Stockfish UCI engine process. Caller must call engine.quit() when done.

    Uses explicit EvalFile paths and single-thread / modest hash to reduce
    EngineTerminatedError (segfault) when running many positions.
    """
    engine_path = path or STOCKFISH_PATH
    work_dir = cwd or STOCKFISH_DIR
    big_path = os.path.join(work_dir, NNUE_BIG)
    small_path = os.path.join(work_dir, NNUE_SMALL)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path, cwd=work_dir)
    options = {
        "Threads": threads,
        "Hash": hash_mb,
        "EvalFile": big_path,
        "EvalFileSmall": small_path,
    }
    engine.configure(options)
    return engine


def get_games_between_dates(conn: duckdb.Connection, start_date: datetime.date, end_date: datetime.date):
    """Get games between two dates."""
    return conn.sql(f"""
        SELECT * FROM games
        WHERE utc_datetime BETWEEN '{start_date}' AND '{end_date}'
    """).df()

def get_endgame_positions(conn: duckdb.Connection):
    """Get endgame positions."""
    return conn.sql(f"""
        SELECT * FROM core.moves m
        JOIN games g ON m.gid = g.gid
        WHERE length(regexp_replace(m.board_position, '[^a-zA-Z]', '', 'g')) BETWEEN 6 AND 8
        AND g.initial_clock >= 300
        AND g.white_elo >= 1500
        AND g.black_elo >= 1500;
    """).df()


def evaluate_position(board: chess.Board, engine, *, depth: int = 18, time_limit: float | None = None):
    """
    Run Stockfish on a position and return the analysis info dict.

    Args:
        board: Position to analyse.
        engine: Stockfish engine from get_stockfish_engine().
        depth: Search depth (plies). Ignored if time_limit is set.
        time_limit: If set, limit search by time in seconds instead of depth.

    Returns:
        Engine analysis dict with at least "score" and optionally "pv", "depth", etc.
    """
    limit = chess.engine.Limit(time=time_limit) if time_limit else chess.engine.Limit(depth=depth)
    return engine.analyse(board, limit)
