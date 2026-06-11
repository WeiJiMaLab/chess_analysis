"""
Tests for the hand-written stop-rule baselines (Analysis U3 — model comparison).

These verify the baseline harness in ``analysis/_budgeted/baselines.py`` purely
on synthetic episodes, with no dependence on generated trees. They pin the
behavioural anchors the comparison report relies on:

  - always-halt-immediately stops at step 0,
  - always-search-to-the-end stops at the last step,
  - threshold rules fire at the first satisfying step,
  - out-of-range stop steps are clamped into the episode, and
  - every replayed baseline has non-negative regret against the DP oracle
    (the oracle return is optimal by construction).

Run from lmcos/: pytest tests/test_budgeted_baselines.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis._budgeted.baselines import (
    _baseline_sweep,
    _evaluate_baseline,
    _fixed_fraction_stop_step,
    _gain_depth_stop_step,
    _value_plateau_stop_step,
)
from src.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    return_for_stop_step,
)

_CONFIG = BudgetedOracleConfig()


def _episode_from_oracle(halt_rewards, tree_sizes, starting_budget):
    """Build a diagnostics-shaped episode whose oracle fields are the true DP optimum.

    Routing the labels through ``compute_budgeted_oracle`` (rather than hand-set
    values) means ``oracle_value`` is the maximal achievable return, so every
    replayed baseline must have non-negative regret — the key invariant below.
    """
    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, starting_budget, _CONFIG)
    return {
        "halt_rewards": list(policy.halt_rewards),
        "tree_sizes": list(policy.tree_sizes),
        "time_budgets": list(policy.time_budgets),
        "oracle_stop_step": int(policy.optimal_stop_step),
        "oracle_value": float(policy.oracle_value),
    }


@pytest.fixture
def episodes():
    """Two small, structurally different episodes.

    A: rewards keep improving with depth (continuing tends to pay off).
    B: rewards plateau immediately (halting early tends to pay off).
    """
    a = _episode_from_oracle(
        halt_rewards=[0.00, 0.10, 0.20, 0.30, 0.40],
        tree_sizes=[1, 3, 7, 15, 31],
        starting_budget=96,
    )
    b = _episode_from_oracle(
        halt_rewards=[0.50, 0.50, 0.50, 0.50, 0.50],
        tree_sizes=[1, 3, 7, 15, 31],
        starting_budget=96,
    )
    return [a, b]


def test_always_halt_zero_stops_at_step_zero(episodes):
    metrics = _evaluate_baseline(episodes, _CONFIG, lambda episode: 0)
    assert metrics["average_expansions"] == 0.0


def test_always_continue_stops_at_last_step(episodes):
    metrics = _evaluate_baseline(episodes, _CONFIG, lambda episode: len(episode["halt_rewards"]) - 1)
    expected = sum(len(e["halt_rewards"]) - 1 for e in episodes) / len(episodes)
    assert metrics["average_expansions"] == pytest.approx(expected)


def test_out_of_range_stop_steps_are_clamped(episodes):
    """A rule that never triggers (huge index) or underflows is clamped in-episode."""
    last = _evaluate_baseline(episodes, _CONFIG, lambda episode: 10_000)
    first = _evaluate_baseline(episodes, _CONFIG, lambda episode: -10_000)
    n_minus_1 = sum(len(e["halt_rewards"]) - 1 for e in episodes) / len(episodes)
    assert last["average_expansions"] == pytest.approx(n_minus_1)
    assert first["average_expansions"] == 0.0


def test_time_threshold_rule_fires_at_first_satisfying_step():
    """halt-when-T-low must stop at the first step whose remaining budget <= threshold."""
    # starting_budget 5 -> time_budgets [5, 4, 3, 2, 1]; threshold 3 -> first hit at idx 2.
    episode = _episode_from_oracle(
        halt_rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
        tree_sizes=[1, 2, 3, 4, 5],
        starting_budget=5,
    )
    rule = lambda e: next(
        (i for i, t in enumerate(e["time_budgets"]) if int(t) <= 3),
        len(e["halt_rewards"]) - 1,
    )
    metrics = _evaluate_baseline([episode], _CONFIG, rule)
    assert metrics["average_expansions"] == 2.0


def test_size_threshold_rule_fires_at_first_satisfying_step():
    """halt-when-N-high must stop at the first step whose tree size >= threshold."""
    episode = _episode_from_oracle(
        halt_rewards=[0.0, 0.0, 0.0, 0.0, 0.0],
        tree_sizes=[1, 2, 3, 4, 5],
        starting_budget=96,
    )
    rule = lambda e: next(
        (i for i, s in enumerate(e["tree_sizes"]) if int(s) >= 3),
        len(e["halt_rewards"]) - 1,
    )
    metrics = _evaluate_baseline([episode], _CONFIG, rule)
    assert metrics["average_expansions"] == 2.0


def test_exact_and_first_action_accuracy_track_the_oracle():
    """On a plateau episode the oracle halts at step 0, so always-halt-0 scores 1.0."""
    episode = _episode_from_oracle(
        halt_rewards=[0.9, 0.9, 0.9, 0.9],
        tree_sizes=[1, 2, 3, 4],
        starting_budget=96,
    )
    assert episode["oracle_stop_step"] == 0  # plateau -> halting immediately is optimal

    halt0 = _evaluate_baseline([episode], _CONFIG, lambda e: 0)
    assert halt0["exact_stop_step_accuracy"] == 1.0
    assert halt0["first_action_accuracy"] == 1.0

    to_end = _evaluate_baseline([episode], _CONFIG, lambda e: len(e["halt_rewards"]) - 1)
    assert to_end["exact_stop_step_accuracy"] == 0.0
    assert to_end["first_action_accuracy"] == 0.0


def test_gain_depth_stop_step_fires_when_marginal_gain_decays():
    """Gain-depth rule halts at the first expansion whose marginal gain <= threshold."""
    # gains: [_, 0.10, 0.05, 0.00, 0.00]
    episode = {"halt_rewards": [0.0, 0.10, 0.15, 0.15, 0.15]}
    assert _gain_depth_stop_step(episode, threshold=0.0) == 3   # first gain <= 0.0
    assert _gain_depth_stop_step(episode, threshold=0.05) == 2  # first gain <= 0.05
    assert _gain_depth_stop_step(episode, threshold=0.20) == 1  # first gain <= 0.20


def test_gain_depth_rule_falls_through_to_last_step_when_gains_stay_high():
    """If every expansion keeps paying off, the rule searches to the end."""
    episode = {"halt_rewards": [0.0, 1.0, 2.0, 3.0]}  # gains all 1.0
    assert _gain_depth_stop_step(episode, threshold=0.5) == 3  # last step


def test_gain_depth_baseline_is_present_in_sweep(episodes):
    result = _baseline_sweep(episodes, _CONFIG)
    assert any(name.startswith("halt_when_gain_le_") for name in result["all_baselines"])


def test_value_plateau_fires_on_windowed_flattening():
    """Halts at the first step whose improvement over the window is <= threshold."""
    # window=2 improvements: i2: 0.5-0.0=0.5, i3: 0.5-0.5=0.0
    episode = {"halt_rewards": [0.0, 0.5, 0.5, 0.5, 0.5]}
    assert _value_plateau_stop_step(episode, threshold=0.1, window=2) == 3
    # threshold high enough that even the first window improvement qualifies
    assert _value_plateau_stop_step(episode, threshold=0.6, window=2) == 2


def test_value_plateau_is_noise_robust_vs_gain_depth():
    """A single-step dip trips gain-depth early but not the windowed plateau."""
    # gains: [_, +0.5, -0.01, +0.11, 0.0] -> gain-depth(<=0) stops at the i=2 dip
    episode = {"halt_rewards": [0.0, 0.5, 0.49, 0.60, 0.60]}
    assert _gain_depth_stop_step(episode, threshold=0.0) == 2
    # window-2 improvements stay positive until the real flattening at the end
    assert _value_plateau_stop_step(episode, threshold=0.0, window=2) == 4


def test_fixed_fraction_stop_step_uses_starting_budget():
    """Stops at round(rho * starting_budget); starting budget = time_budgets[0]."""
    episode = {"halt_rewards": [0.0] * 12, "time_budgets": list(range(10, -2, -1))}  # starts at 10
    assert _fixed_fraction_stop_step(episode, 0.5) == 5
    assert _fixed_fraction_stop_step(episode, 0.25) == 2   # round(2.5) -> 2 (banker's rounding)
    assert _fixed_fraction_stop_step(episode, 0.1) == 1


def test_sweep_includes_plateau_and_fraction_baselines(episodes):
    names = _baseline_sweep(episodes, _CONFIG)["all_baselines"]
    assert any(n.startswith("value_plateau_w") for n in names)
    assert any(n.startswith("halt_at_frac_") for n in names)


def test_sweep_exposes_anchor_baselines_and_best_picks(episodes):
    result = _baseline_sweep(episodes, _CONFIG)
    assert "always_halt_0" in result["all_baselines"]
    assert "always_continue_to_end" in result["all_baselines"]
    assert result["all_baselines"]["always_halt_0"]["average_expansions"] == 0.0
    for key in ("best_by_regret", "best_by_return"):
        assert "name" in result[key]
        assert "average_regret" in result[key]
        assert "average_return" in result[key]


def test_every_baseline_has_non_negative_regret(episodes):
    """The DP oracle return is optimal, so no hand-written rule can beat it."""
    result = _baseline_sweep(episodes, _CONFIG)
    for name, metrics in result["all_baselines"].items():
        assert metrics["average_regret"] >= -1e-9, f"{name} beat the oracle (regret < 0)"
    # The best baseline by regret is the closest approach to the oracle from below.
    assert result["best_by_regret"]["average_regret"] >= -1e-9


def test_return_for_stop_step_matches_evaluate_baseline_single_episode():
    """Sanity bridge: the harness's reported return equals a direct oracle call."""
    episode = _episode_from_oracle(
        halt_rewards=[0.1, 0.25, 0.2, 0.35, 0.3],
        tree_sizes=[1, 3, 7, 15, 31],
        starting_budget=96,
    )
    stop_step = 3
    direct = return_for_stop_step(
        episode["halt_rewards"],
        episode["tree_sizes"],
        episode["time_budgets"],
        stop_step,
        _CONFIG,
    )
    metrics = _evaluate_baseline([episode], _CONFIG, lambda e: stop_step)
    assert metrics["average_return"] == pytest.approx(direct)
