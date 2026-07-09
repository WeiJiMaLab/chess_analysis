"""TDD-lite sanity tests for ``analysis.quiescence_pruning_sweep`` (plan.md T3, 2026-07-08).

A few well-chosen checks per pure function, not an exhaustive suite -- mirrors the
convention in ``test_tree_diagnostics.py``. Only ``trajectory_metrics`` and
``aggregate_grid_metrics`` are pure/synthetic-testable; ``load_tree_metrics``/
``scan_grid_point`` are thin I/O wrappers exercised against real generated trees
manually as part of the T3 sweep run itself.
"""
from __future__ import annotations

import numpy as np

from analysis.quiescence_pruning_sweep import (
    ORACLE_CONFIG,
    aggregate_grid_metrics,
    trajectory_metrics,
)


def _toy_trajectory(halt_rewards: list[float], depth: list[int], node_cutoffs: list[int]) -> dict:
    """Minimal ``build_compact_trajectory``-shaped dict: one node per depth step, node_cutoffs
    grows by however many nodes each step's depth vector's prefix should cover."""
    return {
        "halt_rewards": np.asarray(halt_rewards, dtype=np.float32),
        "depth": depth,
        "step_node_cutoffs": node_cutoffs,
        "tree_sizes": node_cutoffs,
    }


def test_trajectory_metrics_noise_proxy_is_mean_abs_consecutive_delta():
    # halt_rewards steps by +0.1 each time -> |delta| is constant 0.1.
    traj = _toy_trajectory(
        halt_rewards=[0.0, 0.1, 0.2, 0.3],
        depth=[0, 1, 1, 2, 2, 2],
        node_cutoffs=[1, 3, 4, 6],
    )
    result = trajectory_metrics(traj, oracle_config=ORACLE_CONFIG, starting_budget=4)
    assert abs(result["mean_abs_delta_halt_reward"] - 0.1) < 1e-6  # float32 storage tolerance
    assert result["n_steps"] == 4


def test_trajectory_metrics_single_step_has_zero_noise_and_no_crash():
    traj = _toy_trajectory(halt_rewards=[0.5], depth=[0], node_cutoffs=[1])
    result = trajectory_metrics(traj, oracle_config=ORACLE_CONFIG, starting_budget=1)
    assert result["mean_abs_delta_halt_reward"] == 0.0
    assert result["argmax_step"] == 0
    assert result["thinking_helps"] is False  # argmax_step=0 <= threshold=2


def test_trajectory_metrics_final_depth_width_match_last_step_prefix():
    # step 0: nodes[0:1] depth=[0] -> height 0, width 1.
    # step 1: nodes[0:4] depth=[0,1,1,1] -> height 1, width 3 (three depth-1 nodes).
    traj = _toy_trajectory(
        halt_rewards=[0.0, 1.0],
        depth=[0, 1, 1, 1],
        node_cutoffs=[1, 4],
    )
    result = trajectory_metrics(traj, oracle_config=ORACLE_CONFIG, starting_budget=2)
    assert result["final_depth"] == 1
    assert result["final_width"] == 3


def test_trajectory_metrics_thinking_helps_flag_uses_threshold():
    # A monotonically improving halt_reward all the way through -> oracle keeps
    # continuing (argmax at the last step), well above threshold=2.
    halt_rewards = [0.0, 0.05, 0.1, 0.5, 0.9]
    depth = [0] * 15  # depth values don't matter for the argmax check
    node_cutoffs = [1, 2, 3, 4, 5]
    traj = _toy_trajectory(halt_rewards, depth, node_cutoffs)
    result = trajectory_metrics(traj, argmax_threshold=2, oracle_config=ORACLE_CONFIG, starting_budget=5)
    assert result["argmax_step"] == 4
    assert result["thinking_helps"] is True


def test_aggregate_grid_metrics_empty_returns_nan_not_crash():
    agg = aggregate_grid_metrics([])
    assert agg["n_trees"] == 0
    assert np.isnan(agg["mean_abs_delta_halt_reward"])
    assert np.isnan(agg["yield_pct"])


def test_aggregate_grid_metrics_computes_yield_pct_and_medians():
    per_tree = [
        {"mean_abs_delta_halt_reward": 0.1, "thinking_helps": True, "final_depth": 4, "final_width": 10},
        {"mean_abs_delta_halt_reward": 0.3, "thinking_helps": False, "final_depth": 2, "final_width": 20},
        {"mean_abs_delta_halt_reward": 0.2, "thinking_helps": True, "final_depth": 6, "final_width": 30},
    ]
    agg = aggregate_grid_metrics(per_tree)
    assert agg["n_trees"] == 3
    assert abs(agg["mean_abs_delta_halt_reward"] - 0.2) < 1e-9
    assert abs(agg["yield_pct"] - (200.0 / 3.0)) < 1e-6
    assert agg["median_final_depth"] == 4
    assert agg["median_final_width"] == 20
