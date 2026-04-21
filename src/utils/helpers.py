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
import dask.dataframe as dd
from IPython.display import SVG, display

from .features import row_to_fen

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

# --- Plotting Design System (Poster Style) ---
MAIN_COLOR = "#2E86C1"  # Consistent Steel Blue for all analysis
FONT_SIZE_LABEL = 45
FONT_SIZE_TICKS = 35

def apply_poster_style():
    """Apply global matplotlib settings for Poster Style."""
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICKS
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL

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
            continue
        metrics["x"].append(x)
        metrics["y"].append(mean)
        metrics["ci_lower"].append(ci[0])
        metrics["ci_upper"].append(ci[1])
    return metrics

def compute_metrics_by_qbin(data, qbin_edges):
    metrics = defaultdict(list)
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
            continue
        left_edge = qbin_edges[q]
        right_edge = qbin_edges[q + 1]
        bin_mid = (left_edge + right_edge) / 2
        metrics["x"].append(bin_mid)
        metrics["y"].append(mean)
        metrics["ci_lower"].append(ci[0])
        metrics["ci_upper"].append(ci[1])
    return metrics

def plot_metrics(metrics, color=MAIN_COLOR, ax=None):
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

# Preprocess sample games to add time_left and time_left_after columns
def preprocess_data(df: dd.DataFrame) -> dd.DataFrame:
    df = df.sort_values(["gid", "move_ply"], kind="mergesort").reset_index(drop=True)
    df = df.set_index("gid") #most of our work is done on gids

    # clean up move time
    df["move_time"] = df["move_time"].fillna(0).clip(lower=0) 

    # group by gid and player number to get player-wise move time left
    grouped = df.groupby(["gid", "player_white"], sort=False)
    df["spent_prior"] = grouped["move_time"].cumsum() - df["move_time"]
    df["n_prior"] = grouped.cumcount()
    inc = df["clock_increment"]
    df["player_time_left"] = (df["initial_clock"] - df["spent_prior"] + df["n_prior"] * inc).clip(lower=0) 

    # append active player to fen
    df["fen"] = df.apply(row_to_fen, axis=1, meta=("object", "object"))
    df["move_number"] = (df["move_ply"] + 1) // 2
    df["total_time_left"] = df.groupby(["gid", "move_number"])["player_time_left"].transform("sum", meta=("total_time_left", "float64"))

    # Shifts within groups
    df["move_time(t-1)"] = df.groupby("gid")["move_time"].shift(1, meta=("move_time", "float64"))
    df["move_time(t-2)"] = df.groupby("gid")["move_time"].shift(2, meta=("move_time", "float64"))
    df["fen(t-1)"] = df.groupby("gid")["fen"].shift(1, meta=("fen", "object"))
    return df
