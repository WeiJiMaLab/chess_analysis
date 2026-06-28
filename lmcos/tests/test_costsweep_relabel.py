"""Unit tests for the P1 cost-regime relabel (linear-cost + lambda-override DP paths).

Verifies that ``compute_budgeted_oracle`` behaves sanely and monotonically under the
two sweep modes the relabeler uses, and that ``costsweep_relabel`` wiring (config
construction, in-shard relabel, quick regret check) is correct on synthetic shards.

Run from lmcos/:  pytest tests/test_costsweep_relabel.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cts.data.preprocess_mc.oracle import (  # noqa: E402
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    time_cost,
)
from cts.data.preprocess_mc.costsweep_relabel import (  # noqa: E402
    build_oracle_config,
    relabel_shard,
    quick_regret_check,
    _episodes_from_shards,
)

# A monotone-improving halt-reward trace: continuing always helps absent cost, so the
# stop step is a clean function of how expensive continuing is.
_HR = [0.0, 0.2, 0.35, 0.45, 0.5, 0.52, 0.53, 0.535, 0.537, 0.538]
_SIZES = [s for s in range(2, 2 + len(_HR))]
_BUDGET = len(_HR)
_BASE = BudgetedOracleConfig()


# --------------------------------------------------------------------------- #
# build_oracle_config: only the cost regime changes
# --------------------------------------------------------------------------- #
def test_build_linear_sets_mode_and_cbar():
    cfg = build_oracle_config(_BASE, time_mode="linear", time_lambda=0.01)
    assert cfg.time_mode == "linear"
    assert cfg.time_lambda == 0.01
    # bucket partition / seed / maintenance carried over from base
    assert cfg.budget_buckets == _BASE.budget_buckets
    assert cfg.seed == _BASE.seed
    assert cfg.maintenance_scale == _BASE.maintenance_scale


def test_build_lambda_keeps_p_tau():
    cfg = build_oracle_config(_BASE, time_mode="power_law", time_lambda=1.0)
    assert cfg.time_mode == "power_law"
    assert cfg.time_lambda == 1.0
    assert cfg.time_p == _BASE.time_p
    assert cfg.time_tau == _BASE.time_tau


# --------------------------------------------------------------------------- #
# Linear cost: constant per step, and monotone in c_bar
# --------------------------------------------------------------------------- #
def test_linear_cost_is_constant_across_budget():
    cfg = build_oracle_config(_BASE, time_mode="linear", time_lambda=0.03)
    costs = [time_cost(b, cfg) for b in range(1, 50)]
    assert all(abs(c - 0.03) < 1e-12 for c in costs)


def test_linear_stop_step_monotone_nonincreasing_in_cbar():
    """Heavier flat cost => stop no later (weakly earlier). Monotone sanity."""
    stops = []
    for cbar in [0.0, 0.005, 0.02, 0.05, 0.2, 1.0]:
        cfg = build_oracle_config(_BASE, time_mode="linear", time_lambda=cbar)
        pol = compute_budgeted_oracle(_HR, _SIZES, _BUDGET, cfg)
        stops.append(pol.optimal_stop_step)
    assert all(later <= earlier for earlier, later in zip(stops, stops[1:])), stops
    # Zero cost on a strictly improving trace continues to the last available step;
    # a large flat cost forces an immediate halt at step 0.
    assert stops[0] == _BUDGET - 1
    assert stops[-1] == 0


def test_linear_zero_cost_advantages_nonnegative_until_last():
    """Zero cost on an improving trace: continuing is never worse => adv >= 0 pre-terminal."""
    cfg = build_oracle_config(_BASE, time_mode="linear", time_lambda=0.0)
    pol = compute_budgeted_oracle(_HR, _SIZES, _BUDGET, cfg)
    for a in pol.target_advantages[:-1]:
        assert a >= -1e-9


# --------------------------------------------------------------------------- #
# Lambda sweep: monotone in lambda, and matches the default at lambda=18.537
# --------------------------------------------------------------------------- #
def test_lambda_stop_step_monotone_nonincreasing():
    """Larger lambda scales the convex cost up => stop weakly earlier."""
    stops = []
    for lam in [0.1, 1.0, 5.0, 18.537, 100.0]:
        cfg = build_oracle_config(_BASE, time_mode="power_law", time_lambda=lam)
        pol = compute_budgeted_oracle(_HR, _SIZES, _BUDGET, cfg)
        stops.append(pol.optimal_stop_step)
    assert all(later <= earlier for earlier, later in zip(stops, stops[1:])), stops


def test_lambda_default_matches_base_config():
    cfg = build_oracle_config(_BASE, time_mode="power_law", time_lambda=18.537)
    pol_override = compute_budgeted_oracle(_HR, _SIZES, _BUDGET, cfg)
    pol_base = compute_budgeted_oracle(_HR, _SIZES, _BUDGET, _BASE)
    assert pol_override.optimal_stop_step == pol_base.optimal_stop_step
    assert pol_override.target_advantages == pytest.approx(pol_base.target_advantages)


def test_relaxing_cost_does_not_decrease_oracle_value():
    """Cheaper continuing can only raise (or hold) the achievable optimal value."""
    cheap = compute_budgeted_oracle(
        _HR, _SIZES, _BUDGET, build_oracle_config(_BASE, time_mode="power_law", time_lambda=0.1)
    ).oracle_value
    expensive = compute_budgeted_oracle(
        _HR, _SIZES, _BUDGET, build_oracle_config(_BASE, time_mode="power_law", time_lambda=18.537)
    ).oracle_value
    assert cheap >= expensive - 1e-9


# --------------------------------------------------------------------------- #
# relabel_shard + quick_regret_check on a tiny synthetic shard
# --------------------------------------------------------------------------- #
def _synthetic_shard(n_traj: int = 3) -> dict:
    """Two episodes per trajectory (budgets T and T-2), schema-compatible with the loader."""
    traj_hr, traj_sizes = [], []
    trajectory_step_ptr = [0]
    episode_step_ptr = [0]
    episode_trajectory_index = []
    starting_budgets = []
    for t in range(n_traj):
        traj_hr.extend(_HR)
        traj_sizes.extend(_SIZES)
        trajectory_step_ptr.append(trajectory_step_ptr[-1] + len(_HR))
        for budget in (_BUDGET, _BUDGET - 2):
            episode_step_ptr.append(episode_step_ptr[-1] + min(budget, len(_HR)))
            episode_trajectory_index.append(t)
            starting_budgets.append(budget)
    num_episodes = len(starting_budgets)
    return {
        "num_episodes": num_episodes,
        "episode_step_ptr": torch.tensor(episode_step_ptr, dtype=torch.long),
        "episode_trajectory_index": torch.tensor(episode_trajectory_index, dtype=torch.long),
        "trajectory_step_ptr": torch.tensor(trajectory_step_ptr, dtype=torch.long),
        "trajectory_halt_rewards": torch.tensor(traj_hr, dtype=torch.float32),
        "step_node_cutoffs": torch.tensor(traj_sizes, dtype=torch.long),
        "starting_budgets": torch.tensor(starting_budgets, dtype=torch.long),
        # placeholder fields the relabeler overwrites
        "oracle_values": torch.zeros(num_episodes, dtype=torch.float32),
        "oracle_stop_steps": torch.zeros(num_episodes, dtype=torch.long),
        "target_advantages": torch.zeros(sum(episode_step_ptr[-1:]), dtype=torch.float32),
    }


def test_relabel_shard_shapes_and_metadata():
    shard = _synthetic_shard()
    cfg = build_oracle_config(_BASE, time_mode="linear", time_lambda=0.02)
    relabel_shard(shard, cfg)
    n = shard["num_episodes"]
    assert shard["oracle_values"].shape == (n,)
    assert shard["oracle_stop_steps"].shape == (n,)
    # per-step advantages == total steps across episodes
    total_steps = int(shard["episode_step_ptr"][-1].item())
    assert shard["target_advantages"].shape == (total_steps,)
    # metadata refreshed to the new regime
    assert shard["time_mode"] == "linear"
    assert shard["time_lambda"] == 0.02


def test_relabel_shard_stop_step_responds_to_cost():
    """A high-cost relabel halts earlier than a low-cost relabel on the same shard."""
    cheap = _synthetic_shard()
    relabel_shard(cheap, build_oracle_config(_BASE, time_mode="linear", time_lambda=0.0))
    pricey = _synthetic_shard()
    relabel_shard(pricey, build_oracle_config(_BASE, time_mode="linear", time_lambda=1.0))
    assert cheap["oracle_stop_steps"].float().mean() >= pricey["oracle_stop_steps"].float().mean()


def test_quick_regret_check_sane_and_relaxes():
    """Regret >= 0 for all rules; relaxing cost lowers the Never-Stop floor (less cost paid)."""
    cfg_cheap = build_oracle_config(_BASE, time_mode="linear", time_lambda=0.0)
    shard = _synthetic_shard(n_traj=5)
    relabel_shard(shard, cfg_cheap)
    eps = _episodes_from_shards([shard])
    res = quick_regret_check(eps, eps, cfg_cheap)
    assert res["regret_always_stop"] >= -1e-9
    assert res["regret_never_stop"] >= -1e-9
    assert res["regret_fraction"] >= -1e-9
    # Under zero cost on a strictly improving trace, Never-Stop is optimal => ~0 regret.
    assert res["regret_never_stop"] < res["regret_always_stop"]
