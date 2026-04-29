"""Shared utilities for paper figure scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from budgeted_controller_oracle import (
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
    continue_cost,
    return_for_stop_step,
)


# ---------------------------------------------------------------------------
# Diagnostics I/O
# ---------------------------------------------------------------------------


def load_diagnostics(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL diagnostics file as a list of episode dicts."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_multiple_diagnostics(
    label_path_pairs: Sequence[Tuple[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Load diagnostics for multiple models. Returns {label: records}."""
    return {label: load_diagnostics(path) for label, path in label_path_pairs}


# ---------------------------------------------------------------------------
# Oracle config recovery from training log
# ---------------------------------------------------------------------------


def recover_oracle_config_from_log(log_path: str) -> BudgetedOracleConfig:
    """Extract BudgetedOracleConfig parameters from a training stdout log.

    Looks for lines like ``time_lambda=18.537`` printed at job start.
    Falls back to defaults for any parameter not found.
    """
    text = Path(log_path).read_text()
    defaults = BudgetedOracleConfig()

    def _float(name: str, default: float) -> float:
        m = re.search(rf"{name}=([\d.eE+-]+)", text)
        return float(m.group(1)) if m else default

    def _int(name: str, default: int) -> int:
        m = re.search(rf"{name}=(\d+)", text)
        return int(m.group(1)) if m else default

    return BudgetedOracleConfig(
        maintenance_scale=_float("maintenance_scale", defaults.maintenance_scale),
        maintenance_ref_nodes=_float("maintenance_ref_nodes", defaults.maintenance_ref_nodes),
        maintenance_exponent=_float("maintenance_exponent", defaults.maintenance_exponent),
        time_lambda=_float("time_lambda", defaults.time_lambda),
        time_p=_float("time_p", defaults.time_p),
        time_tau=_float("time_tau", defaults.time_tau),
        time_delta=_int("time_delta", defaults.time_delta),
        timeout_value=_float("timeout_value", defaults.timeout_value),
        samples_per_bucket=_int("samples_per_bucket", defaults.samples_per_bucket),
    )


# ---------------------------------------------------------------------------
# Return computation helpers
# ---------------------------------------------------------------------------


def episode_returns_at_all_steps(
    episode: Dict[str, Any],
    config: BudgetedOracleConfig,
) -> np.ndarray:
    """Compute the return for every possible stop step in an episode.

    Returns a 1-D float64 array of length ``episode_length``.
    Uses cumulative cost for O(n) instead of O(n^2).
    """
    hr = np.asarray(episode["halt_rewards"], dtype=np.float64)
    ts = episode["tree_sizes"]
    tb = episode["time_budgets"]
    n = len(hr)
    costs = np.empty(n, dtype=np.float64)
    for i in range(n):
        costs[i] = continue_cost(int(ts[i]), int(tb[i]), config)
    cumcost = np.empty(n, dtype=np.float64)
    cumcost[0] = 0.0
    if n > 1:
        np.cumsum(costs[:-1], out=cumcost[1:])
    return hr - cumcost


def padded_return_matrix(
    diagnostics: List[Dict[str, Any]],
    config: BudgetedOracleConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a padded [num_episodes, max_steps] return matrix.

    Returns ``(returns_matrix, lengths)`` where ``lengths[i]`` is the
    episode length of episode *i* and invalid entries are filled with
    ``returns_matrix[i, lengths[i] - 1]`` (the last valid return).
    """
    lengths = np.array([len(ep["halt_rewards"]) for ep in diagnostics], dtype=np.int64)
    max_steps = int(lengths.max())
    mat = np.zeros((len(diagnostics), max_steps), dtype=np.float64)
    for i, ep in enumerate(diagnostics):
        r = episode_returns_at_all_steps(ep, config)
        mat[i, : len(r)] = r
        mat[i, len(r) :] = r[-1]  # pad with last valid return
    return mat, lengths


# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "slw01": "#1f77b4",
    "reweight_w4": "#ff7f0e",
    "inv_freq": "#2ca02c",
    "affine": "#d62728",
    "affine+rw": "#9467bd",
}

MODEL_MARKERS = {
    "slw01": "*",
    "reweight_w4": "D",
    "inv_freq": "^",
    "affine": "s",
    "affine+rw": "P",
}

BUDGET_BUCKET_ORDER = [
    "scramble",
    "medium-small",
    "medium-large",
    "large",
    "very-large",
]
