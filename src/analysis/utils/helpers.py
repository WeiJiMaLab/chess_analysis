"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import yaml


def _interpolate(value, variables):
    """Recursively substitute ${key} / ${globals.key} placeholders in value.

    Mirrors render_stage.py so the human_analysis section can reference the
    shared `globals` dirs (e.g. ${scratch_dir}, ${trees_dir})."""
    if isinstance(value, str):
        def repl(match):
            name = match.group(1).removeprefix("globals.")
            return str(variables[name]) if name in variables else match.group(0)
        return re.sub(r"\$\{([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, variables) for v in value]
    return value


# Load the shared configuration.
# The human-analysis keys live in the single unified config at the repo-root
# config.yaml, under the top-level `human_analysis:` section. helpers.py sits at
# src/analysis/utils/helpers.py, so the repo root is three dirs up. ${...}
# placeholders in human_analysis are resolved against `globals` (same contract
# as render_stage.py / cts._config), so analysis shares the pipeline's dirs.
def _load_shared_config() -> dict:
    config_path = (
        Path(__file__).resolve().parent.parent.parent.parent / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Shared config not found at {config_path}")
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    section = data.get("human_analysis")
    if not isinstance(section, dict):
        raise KeyError(f"'human_analysis' section missing from {config_path}.")

    # Resolve globals (which may reference each other), then the section.
    variables = dict(data.get("globals", {}))
    for _ in range(5):
        variables = {k: _interpolate(v, variables) for k, v in variables.items()}
    return _interpolate(section, variables)

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


import contextlib



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

