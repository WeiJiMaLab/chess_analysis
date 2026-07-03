"""
Shared utilities for chess_analysis: DB connection, FEN display, Stockfish engine.
"""

from __future__ import annotations

from analysis.config import load_config_section

# Shared analysis config. The active file is chosen by the ``CONFIG`` env var
# (which slurm/setup_env.sh exports and board.py / engine.py set from --config),
# falling back to the repo default; ${...} placeholders in ``human_analysis`` are
# resolved against ``globals`` (see analysis.config).
CONFIG = load_config_section("human_analysis")

import chess
import chess.svg
import duckdb
import matplotlib.pyplot as plt
from IPython.display import SVG, display

# --- Plotting Design System (Poster Style) ---
MAIN_COLOR = "#2E86C1"  # Consistent Steel Blue for all analysis
FONT_SIZE_LABEL = 52
FONT_SIZE_TICKS = 42

# Standard palette for progress tertiles 1–3 (segmented dashboards).
# Early = light, late = dark (so deeper into the game reads darker).
PHASE_COLORS = {
    1: "#9ECAE1",  # Light blue (Early)
    2: "#3182BD",  # Medium blue (Mid)
    3: "#08519C",  # Dark blue (Late)
}

# Game-fraction tertiles use FIXED thirds of the game (not empirical quantiles),
# so the split reads as "early / mid / late third" with clean labels.
GAME_FRAC_CUTS = (1.0 / 3.0, 2.0 / 3.0)
GAME_FRAC_LABELS = {1: "< 1/3", 2: "1/3–2/3", 3: "> 2/3"}

def apply_poster_style():
    """Apply global matplotlib settings for Poster Style."""
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICKS
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TICKS
    plt.rcParams['legend.fontsize'] = FONT_SIZE_TICKS


import contextlib



@contextlib.contextmanager
def db_connection(database: str = CONFIG["selected_db_default"], read_only: bool = True):
    """Context manager for acquiring and safely releasing a DuckDB connection."""
    conn = duckdb.connect(database=database, read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


# Ply window (Russek et al.-style): all board/engine analyses filter to
# move_ply in [min_ply, max_ply] "on arrival". These are the names of the
# windowed views create_ply_windowed_views() installs; downstream SQL references
# ONLY these so the filter (and the derived ply tertiles) are consistent
# everywhere, including on the backwards tree->move join.
WIN_PROCESSED_MOVES = "pm_win"
WIN_PROCESSED_MOVES_NONZERO = "pmnz_win"


def create_ply_windowed_views(conn) -> tuple[str, str]:
    """Install temp views of the processed-move tables filtered to the config ply
    window (move_ply BETWEEN min_ply AND max_ply). Returns (pm_view, pmnz_view).

    Every board/engine query reads these instead of the raw tables, so the ply
    filter is applied once, on arrival, and ply tertiles computed off these views
    are conditioned on the window (not the whole dataset)."""
    lo, hi = int(CONFIG["min_ply"]), int(CONFIG["max_ply"])
    for view, base in (
        (WIN_PROCESSED_MOVES, CONFIG["table_processed_moves"]),
        (WIN_PROCESSED_MOVES_NONZERO, CONFIG["table_processed_moves_nonzero"]),
    ):
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW {view} AS "
            f"SELECT * FROM {base} WHERE move_ply BETWEEN {lo} AND {hi}"
        )
    return WIN_PROCESSED_MOVES, WIN_PROCESSED_MOVES_NONZERO


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

