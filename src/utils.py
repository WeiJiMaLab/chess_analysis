"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

import os

import chess
import chess.svg
import chess.engine
from IPython.display import SVG, display
import duckdb

STOCKFISH_SF14_PATH = "/scratch/hl4291/Stockfish-sf_14/src/stockfish"
STOCKFISH_SF15_PATH = "/scratch/hl4291/Stockfish-sf_15/src/stockfish"
STOCKFISH_SF15_DIR  = "/scratch/hl4291/Stockfish-sf_15/src"
NNUE_SF15           = "nn-6877cd24400e.nnue"

# Keep old names pointing at SF15 for backward compat
STOCKFISH_PATH = STOCKFISH_SF15_PATH
STOCKFISH_DIR  = STOCKFISH_SF15_DIR

def get_stockfish_engine(
    path: str | None = None,
    *,
    cwd: str | None = None,
    threads: int = 1,
    hash_mb: int = 128,
    version: int = 14,   # default to SF14 for VOC work
):
    """
    Spawn a Stockfish UCI engine process. Caller must call engine.quit() when done.

    version=14: bundled NNUE, no EvalFile needed, better VOC variance at shallow depths
    version=15: external NNUE required, hybrid classical+NNUE eval
    """
    if path is not None:
        engine_path = path
        work_dir = cwd or os.path.dirname(path)
        nnue_path = None
    elif version == 14:
        engine_path = STOCKFISH_SF14_PATH
        work_dir = os.path.dirname(STOCKFISH_SF14_PATH)
        nnue_path = None  # bundled
    elif version == 15:
        engine_path = STOCKFISH_SF15_PATH
        work_dir = cwd or STOCKFISH_SF15_DIR
        nnue_path = os.path.join(work_dir, NNUE_SF15)
    else:
        raise ValueError(f"Unsupported version: {version}")

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Stockfish binary not found at {engine_path}")
    if nnue_path and not os.path.exists(nnue_path):
        raise FileNotFoundError(f"NNUE file not found at {nnue_path}")

    engine = chess.engine.SimpleEngine.popen_uci(engine_path, cwd=work_dir)
    options = {"Threads": threads, "Hash": hash_mb}
    if nnue_path:
        options["EvalFile"] = nnue_path
    engine.configure(options)
    return engine


def display_fen(fen: str, size: int = 200) -> None:
    """Render a FEN position as an SVG and display it in the notebook."""
    board = chess.Board(fen)
    display(SVG(chess.svg.board(board=board, size=size)))


def get_db_connection(
    database: str = ":memory:",  # default to in-memory to avoid personal.db/WAL clutter
    *,
    threads: int = 10,
    memory_limit: str = "20GB",
    temp_directory: str = ".",
    **kwargs,
):
    """Create a DuckDB connection with sensible defaults for analysis.
    
    Uses in-memory database by default — no persistent files created.
    Pass database='personal.db' explicitly if you need persistence.
    """
    config = {
        "threads": threads,
        "memory_limit": memory_limit,
        "temp_directory": temp_directory,
        **kwargs,
    }
    return duckdb.connect(database=database, config=config)
    """
    Spawn a Stockfish 15 UCI engine process. Caller must call engine.quit() when done.

    SF15 uses a single NNUE network (no small net) with hybrid classical+NNUE eval,
    which gives more VOC variance at shallow depths than SF17+.
    Single-thread / modest hash reduces EngineTerminatedError (segfault) risk.
    """
    engine_path = path or STOCKFISH_PATH
    work_dir = cwd or STOCKFISH_DIR
    nnue_path = os.path.join(work_dir, NNUE_BIG)

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Stockfish binary not found at {engine_path}")
    if not os.path.exists(nnue_path):
        raise FileNotFoundError(f"NNUE file not found at {nnue_path}")

    engine = chess.engine.SimpleEngine.popen_uci(engine_path, cwd=work_dir)
    engine.configure({
        "Threads": threads,
        "Hash": hash_mb,
        "EvalFile": nnue_path,
    })
    return engine