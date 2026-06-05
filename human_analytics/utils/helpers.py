"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

import os

import chess
import chess.engine
import chess.svg
import duckdb
import matplotlib.pyplot as plt
from IPython.display import SVG, display

# Constants
EPSILON = 1e-6

# Stockfish paths
_STOCKFISH_HOME = os.path.expanduser("~/stockfish")
_STOCKFISH_SF15_HOME = os.path.expanduser("~/stockfish-sf_15")

STOCKFISH_SF14_DIR = os.path.join(_STOCKFISH_HOME, "src")
STOCKFISH_SF14_PATH = os.path.join(STOCKFISH_SF14_DIR, "stockfish")
STOCKFISH_SF15_PATH = os.path.join(_STOCKFISH_SF15_HOME, "src", "stockfish")
STOCKFISH_SF15_DIR = os.path.join(_STOCKFISH_SF15_HOME, "src")
NNUE_SF15 = "nn-6877cd24400e.nnue"

# Default binary
STOCKFISH_PATH = STOCKFISH_SF14_PATH
STOCKFISH_DIR = STOCKFISH_SF14_DIR

# lc0 paths (aligned with lmcos/ysagiv version)
LC0_PATH = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS_PATH = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"

# --- Plotting Design System (Poster Style) ---
MAIN_COLOR = "#2E86C1"  # Consistent Steel Blue for all analysis
FONT_SIZE_LABEL = 45
FONT_SIZE_TICKS = 35

# Standard palette for ply tertiles 1–3 (segmented dashboards; matches preprocess ntile)
PHASE_COLORS = {
    1: "#16a085",  # Teal (Early)
    2: "#2980b9",  # Blue (Mid)
    3: "#8e44ad"   # Purple (End)
}

def apply_poster_style():
    """Apply global matplotlib settings for Poster Style."""
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICKS
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL


def analysis_style() -> None:
    """Apply standard matplotlib settings for lmcos analysis figures."""
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

def get_stockfish_engine(
    path: str | None = None,
    *,
    cwd: str | None = None,
    threads: int = 1,
    hash_mb: int = 128,
    version: int = 14,
):
    if path is not None:
        engine_path = path
        work_dir = cwd or os.path.dirname(path)
        nnue_path = None
    elif version == 14:
        engine_path = STOCKFISH_SF14_PATH
        work_dir = STOCKFISH_SF14_DIR
        nnue_path = None
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

def get_lc0_engine(
    path: str = LC0_PATH,
    weights_path: str = LC0_WEIGHTS_PATH,
    threads: int = 1,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"lc0 binary not found at {path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"lc0 weights not found at {weights_path}")

    engine = chess.engine.SimpleEngine.popen_uci(path)
    options = {
        "Threads": threads,
        "WeightsFile": weights_path,
        "UCI_ShowWDL": "true",
    }
    engine.configure(options)
    return engine

def display_fen(fen: str, size: int = 200) -> None:
    board = chess.Board(fen)
    display(SVG(chess.svg.board(board=board, size=size)))

def get_db_connection(
    database: str = ":memory:",
    *,
    threads: int = 10,
    memory_limit: str = "20GB",
    temp_directory: str = ".",
    **kwargs,
):
    config = {
        "threads": threads,
        "memory_limit": memory_limit,
        "temp_directory": temp_directory,
        **kwargs,
    }
    return duckdb.connect(database=database, config=config)

