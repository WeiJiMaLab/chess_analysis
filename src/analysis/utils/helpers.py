"""
Shared analysis utilities: config loading, DuckDB connections/helpers, ply-window
views, plotting style, and small stats helpers.
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config_allply.yaml"


def config_path() -> Path:
    """Active config file: $CONFIG if set, else the repo default."""
    env = os.environ.get("CONFIG")
    return Path(env) if env else _DEFAULT_CONFIG


def _interpolate(value, variables):
    """Recursively substitute ${key} / ${globals.key} placeholders."""
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


def load_config_section(section: str, path=None) -> dict:
    """Return one top-level section of the run config with ${...} placeholders
    resolved against ``globals``."""
    path = Path(path) if path else config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    body = data.get(section)
    if not isinstance(body, dict):
        raise KeyError(f"'{section}' section missing from {path}")
    variables = dict(data.get("globals", {}))
    for _ in range(5):
        variables = {k: _interpolate(v, variables) for k, v in variables.items()}
    return _interpolate(body, variables)


# Shared analysis config: active file from $CONFIG (slurm/setup_env.sh exports it),
# else the repo default; ${...} in ``human_analysis`` resolved against ``globals``.
CONFIG = load_config_section("human_analysis")

# --- Plotting Design System (Poster Style) ---
MAIN_COLOR = "#6378f1"  # Blue-leaning indigo — the presentations' indigo accent (VOC in
                        # make_rt_figures.py, hue 239°) with its hue nudged ~35% toward
                        # Tailwind blue-500 (217°) to read a bit more blue; same lightness/
                        # saturation as the original indigo.
FONT_SIZE_LABEL = 52
FONT_SIZE_TICKS = 42
# Legend text size == the n= annotation size (house style: the n= label and every
# legend should read as the same visual weight — see reference.md "Plot standards").
LEGEND_FONTSIZE = 36

# Standard palette for progress tertiles 1–3 (segmented dashboards). Same hue as
# MAIN_COLOR (blue-leaning indigo) at 3 lightness steps, so every plot — base
# series AND ply-tertile segmentation — reads as one consistent color family,
# rather than MAIN_COLOR's indigo-blue next to a separate plain-blue ColorBrewer
# ramp. Early = light, late = dark (so deeper into the game reads darker).
PHASE_COLORS = {
    1: "#a7b2f1",  # Light indigo-blue (Early)
    2: "#3e57ea",  # Medium indigo-blue (Mid)
    3: "#0c2197",  # Dark indigo-blue (Late)
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
    plt.rcParams['axes.titlesize'] = FONT_SIZE_TICKS
    plt.rcParams['legend.fontsize'] = FONT_SIZE_TICKS


def sql_str(s: str) -> str:
    """Single-quoted SQL literal fragment."""
    return s.replace("'", "''")


def duckdb_connect_config(work_dir: str, threads: int, memory_limit: str) -> dict:
    """DuckDB ``connect`` config: spill/sort temp files live in ``work_dir``."""
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    return {"threads": int(threads), "memory_limit": str(memory_limit), "temp_directory": work_dir}


def connect(database: str, work_dir: str, threads: int, memory_limit: str,
            read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open ``database`` with DuckDB spill/sort temp files routed to ``work_dir``."""
    return duckdb.connect(database=database, read_only=read_only,
                          config=duckdb_connect_config(work_dir, threads, memory_limit))


@contextlib.contextmanager
def db_connection(database: str = CONFIG["selected_db_default"], read_only: bool = True):
    """Acquire/release a DuckDB connection with spill routed to scratch (beside the
    DB) and memory capped, so the full-table sorts in the analyses don't spill to
    CWD or hit the default ceiling. Cap overridable via $DUCKDB_MEMORY_LIMIT."""
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(database)), "duckdb_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    conn = duckdb.connect(
        database=database, read_only=read_only,
        config={"temp_directory": temp_dir,
                "memory_limit": os.environ.get("DUCKDB_MEMORY_LIMIT", "64GB")},
    )
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

