"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

from collections import defaultdict
import os

import chess
import chess.engine
import chess.svg
import duckdb
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG, display

# Stockfish lives under home (not scratch). SF14: copied tree; SF15: built from official-stockfish sf_15.
_STOCKFISH_HOME = os.path.expanduser("~/stockfish")
_STOCKFISH_SF15_HOME = os.path.expanduser("~/stockfish-sf_15")

STOCKFISH_SF14_DIR = os.path.join(_STOCKFISH_HOME, "src")
STOCKFISH_SF14_PATH = os.path.join(STOCKFISH_SF14_DIR, "stockfish")
STOCKFISH_SF15_PATH = os.path.join(_STOCKFISH_SF15_HOME, "src", "stockfish")
STOCKFISH_SF15_DIR = os.path.join(_STOCKFISH_SF15_HOME, "src")
NNUE_SF15 = "nn-6877cd24400e.nnue"

# Default binary and working dir: Stockfish 14 (matches get_stockfish_engine() default).
STOCKFISH_PATH = STOCKFISH_SF14_PATH
STOCKFISH_DIR = STOCKFISH_SF14_DIR

# --- Plotting Design System ---
MAIN_COLOR = "#2E86C1"  # Consistent Steel Blue for all analysis
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 16
FONT_SIZE_TICKS = 14

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
        work_dir = STOCKFISH_SF14_DIR
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

# Compute bootstrapped confidence interval for the mean
def bootstrapped_ci(data, n_bootstraps=1000):
    if len(data) <= 1:
        return (np.nan, np.nan)
    bootstrap_means = []
    data = np.array(data)
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    return np.percentile(bootstrap_means, [2.5, 97.5])

def compute_metrics_by_bin(data):
    metrics = defaultdict(list)
    for x in sorted(data["bin"].unique()):
        bin_data = data[data["bin"] == x]["move_time"]
        n = len(bin_data)
        if n > 1:
            mean = bin_data.mean()
            ci = bootstrapped_ci(bin_data)
        elif n == 1:
            mean = bin_data.iloc[0]
            ci = (mean, mean)
        else:
            continue  # skip bins with no data
        metrics["x"].append(x)
        metrics["y"].append(mean)
        metrics["ci_lower"].append(ci[0])
        metrics["ci_upper"].append(ci[1])
    return metrics

def compute_metrics_by_qbin(data, qbin_edges):
    metrics = defaultdict(list)
    # Use range of number of bins, not the sorted unique qbin (which might have missing bins)
    n_bins = len(qbin_edges) - 1
    for q in range(n_bins):
        bin_data = data[data["qbins"] == q]["move_time"]
        n = len(bin_data)
        if n > 1:
            mean = bin_data.mean()
            ci = bootstrapped_ci(bin_data)
        elif n == 1:
            mean = bin_data.iloc[0]
            ci = (mean, mean)
        else:
            continue  # skip bins with no data
        left_edge = qbin_edges[q]
        right_edge = qbin_edges[q + 1]
        bin_mid = (left_edge + right_edge) / 2
        metrics["x"].append(bin_mid)
        metrics["y"].append(mean)
        metrics["ci_lower"].append(ci[0])
        metrics["ci_upper"].append(ci[1])
    return metrics

def plot_metrics(metrics, color="#4682B4", ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(metrics["x"], metrics["y"], label="mean", color=color)
    ax.fill_between(
        metrics["x"],
        metrics["ci_lower"],
        metrics["ci_upper"],
        color=color,
        alpha=0.2,
        label="95% CI",
    )