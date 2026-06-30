"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

import os
from pathlib import Path
import yaml

# Load the shared configuration.
# After the human_analytics + lmcos_tiny merge, the human-analysis keys live in
# the single unified config at lmcos_small/configs/core.yaml, under the
# top-level `human_analysis:` section. helpers.py sits at
# lmcos_small/src/human/utils/helpers.py, so the config is ../../configs/core.yaml.
def _load_shared_config() -> dict:
    config_path = (
        Path(__file__).resolve().parent.parent.parent.parent / "configs" / "core.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Shared config not found at {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    section = data.get("human_analysis")
    if not isinstance(section, dict):
        raise KeyError(
            f"'human_analysis' section missing from {config_path}; "
            "the human-RT analysis keys were folded there during the "
            "human_analytics->lmcos_small merge."
        )
    return section

CONFIG = _load_shared_config()

import chess
import chess.svg
import duckdb
import matplotlib.pyplot as plt
from IPython.display import SVG, display

# --- Plotting Design System (Poster Style) ---
MAIN_COLOR = "#2E86C1"  # Consistent Steel Blue for all analysis
FONT_SIZE_LABEL = 52
FONT_SIZE_TICKS = 42

# Standard palette for ply tertiles 1–3 (segmented dashboards; matches preprocess ntile)
PHASE_COLORS = {
    1: "#08519C",  # Dark blue (Early)
    2: "#3182BD",  # Medium blue (Mid)
    3: "#9ECAE1",  # Light blue (End)
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

def display_fen(fen: str, size: int = 200) -> None:
    board = chess.Board(fen)
    display(SVG(chess.svg.board(board=board, size=size)))

import contextlib

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


@contextlib.contextmanager
def db_connection(database: str = CONFIG["selected_db_default"], read_only: bool = True):
    """Context manager for acquiring and safely releasing a DuckDB connection."""
    conn = duckdb.connect(database=database, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def partial_spearman(df, x: str, y: str, controls: list[str]) -> float:
    """Spearman rank correlation ρ(x, y) controlling for covariates."""
    import numpy as np
    df = df[[x, y] + controls].dropna()
    R = df.rank()
    A = np.c_[np.ones(len(R)), R[controls].to_numpy()]
    def resid(col: str) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(A, R[col].to_numpy(), rcond=None)
        return R[col].to_numpy() - A @ beta
    return float(np.corrcoef(resid(x), resid(y))[0, 1])

