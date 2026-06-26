"""Unit tests for the cost-free value-gain target + sweep-many cost decision rule.

Covers:
  * value_gain >= 0 and equals max(halt_reward[t:]) - halt_reward[t] (greedy/GSS
    reachability at zero cost), budget-truncated;
  * value_gain is non-flat on a non-trivial trajectory (NOT collapsed to ~0);
  * the analytic-cost-subtraction eval reproduces the relabeled oracle's optimal
    stop at the matching lambda when the head is the PERFECT (oracle) value gain.
"""
from __future__ import annotations

import torch

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
)
from cts.data.preprocess_mc.value_gain import (
    value_gain_trace,
    analytic_cost_trace,
    swept_advantage_trace,
    predicted_halt_curve,
    dp_stop_over_curve,
)
from cts.models.readout import stop_step_from_advantages


def test_value_gain_non_negative_and_matches_future_max():
    halt = [0.1, 0.3, 0.2, 0.5, 0.4]
    sizes = [1, 2, 3, 4, 5]
    budget = 10  # > len, so no truncation
    vg = value_gain_trace(halt, sizes, budget)
    assert len(vg) == len(halt)
    for v in vg:
        assert v >= -1e-9, "value_gain must be non-negative"
    # zero-cost V*(t) == max(halt[t:]); gain == that minus halt[t]
    for t in range(len(halt)):
        expected = max(halt[t:]) - halt[t]
        assert abs(vg[t] - expected) < 1e-6, (t, vg[t], expected)


def test_value_gain_budget_truncation():
    halt = [0.1, 0.2, 0.9, 0.3]
    sizes = [1, 2, 3, 4]
    budget = 2  # truncates to first 2 steps -> 0.9 unreachable
    vg = value_gain_trace(halt, sizes, budget)
    assert len(vg) == 2
    # within truncated [0.1, 0.2]: future max at t=0 is 0.2
    assert abs(vg[0] - (0.2 - 0.1)) < 1e-6
    assert abs(vg[1] - 0.0) < 1e-6


def test_value_gain_non_flat():
    # A rising-then-flat trajectory: gains should span a real range, not ~0.
    halt = [0.0, 0.2, 0.5, 0.6, 0.6, 0.6]
    sizes = list(range(1, 7))
    vg = value_gain_trace(halt, sizes, 20)
    assert max(vg) - min(vg) > 0.1, "value_gain landscape must be non-flat"
    assert vg[0] > 0.5  # large gain available from the start


def test_analytic_cost_matches_oracle_continue_cost():
    sizes = [3, 5, 7]
    budget = 10
    cfg = BudgetedOracleConfig(time_lambda=5.0, time_mode="power_law")
    costs = analytic_cost_trace(sizes, budget, cfg, len(sizes))
    from cts.data.preprocess_mc.oracle import continue_cost
    for t in range(len(sizes)):
        assert abs(costs[t] - continue_cost(sizes[t], budget - t, cfg)) < 1e-9


def test_swept_perfect_head_brackets_oracle_and_is_monotone():
    """Sanity on the analytic-cost-subtraction eval with the PERFECT (oracle)
    value-gain head.

    The swept rule charges the MYOPIC single-step cost against the cost-free
    TOTAL reachable gain, so it need not equal the full backward-induction
    oracle stop exactly; but two genuine guarantees must hold:
      (1) it never stops BEFORE the relabeled oracle's optimal stop (a positive
          total gain minus a single step's cost is an optimistic continue
          signal, so it stops no earlier than the cost-aware optimum), and
      (2) the swept stop is MONOTONE non-increasing in the cost level (more
          expensive thinking => stop no later).
    """
    halt = [0.0, 0.15, 0.30, 0.42, 0.50, 0.55, 0.57, 0.58]
    sizes = list(range(1, len(halt) + 1))
    budget = 12
    n = min(len(halt), budget)
    perfect_gain = value_gain_trace(halt, sizes, budget)[:n]

    # (1) never earlier than the cost-aware optimum, every regime.
    for mode, lam in [("linear", 0.0), ("power_law", 1.0), ("power_law", 5.0),
                      ("power_law", 18.537), ("linear", 0.02)]:
        cfg = BudgetedOracleConfig(time_lambda=lam, time_mode=mode, maintenance_scale=0.0)
        adv = swept_advantage_trace(perfect_gain, sizes[:n], budget, cfg)
        swept_stop = stop_step_from_advantages(torch.tensor(adv, dtype=torch.float32))
        oracle = compute_budgeted_oracle(halt, sizes, budget, cfg)
        assert swept_stop >= oracle.optimal_stop_step, (mode, lam, swept_stop, oracle.optimal_stop_step)

    # (2) monotone non-increasing in lambda within one cost mode (more expensive
    #     thinking => stop no later).
    prev_stop = None
    for lam in [0.0, 0.1, 1.0, 5.0, 18.537, 50.0]:
        cfg = BudgetedOracleConfig(time_lambda=lam, time_mode="power_law", maintenance_scale=0.0)
        adv = swept_advantage_trace(perfect_gain, sizes[:n], budget, cfg)
        swept_stop = stop_step_from_advantages(torch.tensor(adv, dtype=torch.float32))
        if prev_stop is not None:
            assert swept_stop <= prev_stop, (lam, swept_stop, prev_stop)
        prev_stop = swept_stop


def test_dp_over_perfect_curve_reproduces_relabeled_oracle_exactly():
    """The KEY acceptance test for the trajectory + real-DP design.

    Feeding the PERFECT predicted curve (== the true halt rewards) into
    ``dp_stop_over_curve`` must reproduce the relabeled oracle's optimal stop
    EXACTLY for every lambda -- including the transition regime where the greedy
    rule fails -- because it is the same backward-induction recursion.
    """
    # A lumpy landscape: flat, then a late jump (exactly where greedy mis-stops).
    halt = [0.10, 0.10, 0.10, 0.10, 0.60, 0.62, 0.63]
    sizes = list(range(1, len(halt) + 1))
    budget = 12
    curve = predicted_halt_curve(halt, sizes, budget)

    for mode, lam in [("linear", 0.0), ("power_law", 0.1), ("power_law", 1.0),
                      ("power_law", 5.0), ("power_law", 18.537), ("linear", 0.003),
                      ("linear", 0.02)]:
        cfg = BudgetedOracleConfig(time_lambda=lam, time_mode=mode, maintenance_scale=0.0)
        dp_stop = dp_stop_over_curve(curve, sizes, budget, cfg)
        oracle = compute_budgeted_oracle(halt, sizes, budget, cfg)
        assert dp_stop == oracle.optimal_stop_step, (mode, lam, dp_stop, oracle.optimal_stop_step)


def test_dp_zero_cost_searches_to_global_max():
    """At zero cost the DP stops at the global-max halt step (search-forever)."""
    halt = [0.1, 0.3, 0.2, 0.9, 0.4]
    sizes = list(range(1, 6))
    cfg = BudgetedOracleConfig(time_lambda=0.0, time_mode="linear", maintenance_scale=0.0,
                               timeout_value=0.0)
    curve = predicted_halt_curve(halt, sizes, 20)
    assert dp_stop_over_curve(curve, sizes, 20, cfg) == 3  # index of 0.9


def test_zero_cost_limit_searches_forever():
    """At lambda->0 the swept rule should never stop early (search-forever),
    since cost ~ 0 < any positive gain except at the plateau."""
    halt = [0.0, 0.2, 0.4, 0.6, 0.8]  # strictly increasing -> gain>0 until last
    sizes = list(range(1, 6))
    budget = 10
    gain = value_gain_trace(halt, sizes, budget)
    cfg = BudgetedOracleConfig(time_lambda=0.0, time_mode="linear", maintenance_scale=0.0)
    adv = swept_advantage_trace(gain, sizes, budget, cfg)
    stop = stop_step_from_advantages(torch.tensor(adv, dtype=torch.float32))
    assert stop == len(halt) - 1  # runs to the end
