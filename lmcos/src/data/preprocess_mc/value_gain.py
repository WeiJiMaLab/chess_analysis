"""Cost-free value-gain target + train-once / sweep-many cost decision rule.

THE IDEA (mc_minimal_plan.md cost/value discussion).  The production MC head
regresses a *cost-baked* advantage (``continue_value - halt_value`` with the
time-cost already subtracted under one fixed ``time_lambda``), so a trained head
is welded to its lambda and cannot be swept.  The fix is to train the head on a
*cost-free* quantity -- the expected move-quality GAIN, in pure reward units,
from continuing the search past step ``t`` -- and bolt the cost on only at
DECISION time:

    advantage(t ; lambda) = pred_value_gain(t)  -  analytic_cost(t ; lambda, T)
    stop at the first t where advantage(t ; lambda) <= 0   (continue iff > 0)

Train ONCE on the cost-free target, then sweep ANY cost regime at eval by
re-subtracting the analytic cost -- no retrain.

THE COST-FREE TARGET.  ``value_gain(t)`` is the best halt-reward reachable by
continuing from ``t`` onward at ZERO cost, minus the current halt reward:

    value_gain(t) = V*_zerocost(t) - halt_reward(t)
                  = max_{t <= j < budget-truncated} halt_reward[j] - halt_reward[t]

This is exactly the zero-cost / lambda->0 DP value at step ``t``: with cost
disabled the optimal stopper just runs to the global future maximum, so
``V*_zerocost(t) == max(halt_reward[t:])`` over the budget-truncated episode.
We REUSE the oracle (``compute_budgeted_oracle`` with ``time_lambda=0`` in linear
mode -> constant zero per-step cost, ``maintenance_scale=0``) rather than
reimplementing the landscape.  ``value_gain(t) >= 0`` always, and -- unlike the
cost-baked advantage that collapsed to ~0 at the optimal boundary -- it is a
rich non-flat landscape (the whole point).

THE ANALYTIC COST.  At decision time we subtract the EXACT per-step continue
cost from the same oracle: for an episode with starting budget ``T``, the cost
charged for the expansion taken AT step ``t`` (which consumes remaining budget
``T - t``) is ``continue_cost(n_nodes[t], T - t, config)`` -- maintenance + the
power-law/linear ``time_cost``.  Reusing ``continue_cost`` guarantees the swept
advantage uses byte-identical cost arithmetic to the relabeled oracle.
"""
from __future__ import annotations

from typing import List, Sequence

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    continue_cost,
)


# Zero-cost oracle: linear mode with lambda=0 => constant 0 per-step cost,
# maintenance disabled. Its V*(t) is exactly max(halt_reward[t:]) on the
# budget-truncated episode, so value_gain reduces to greedy/GSS reachability.
_ZERO_COST_CONFIG = BudgetedOracleConfig(
    maintenance_scale=0.0,
    time_lambda=0.0,
    time_mode="linear",
    timeout_value=0.0,  # no forced-timeout penalty when search is free
)


def value_gain_trace(
    halt_rewards: Sequence[float],
    tree_sizes: Sequence[int],
    starting_budget: int,
) -> List[float]:
    """Per-step cost-free value gain ``[num_steps]`` for one episode.

    ``value_gain[t] = V*_zerocost(t) - halt_reward[t] >= 0`` where the zero-cost
    optimal value is computed by the shared oracle (cost disabled), so this is
    the genuine reachable future best, budget-truncated, NOT a re-implementation.

    Returns a list aligned with the *budget-truncated* episode (length
    ``min(len(halt_rewards), starting_budget)``).
    """
    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, starting_budget, _ZERO_COST_CONFIG)
    return [v - h for v, h in zip(policy.values, policy.halt_rewards)]


def analytic_cost_trace(
    tree_sizes: Sequence[int],
    starting_budget: int,
    config: BudgetedOracleConfig,
    num_steps: int,
) -> List[float]:
    """Per-step analytic continue cost ``[num_steps]`` under ``config``.

    ``cost[t] = continue_cost(n_nodes[t], T - t, config)`` -- the maintenance +
    time cost charged for the expansion taken at step ``t`` (remaining budget
    ``T - t``).  Reuses the oracle's own cost function so the swept advantage is
    arithmetically identical to the relabeled oracle at the matching lambda.
    The final step has no "continue", but we still emit a cost (never consumed
    by the stop rule, which can't continue past the last index).
    """
    costs: List[float] = []
    for t in range(num_steps):
        remaining = starting_budget - t
        if remaining <= 0:
            costs.append(float("inf"))  # exhausted budget => continuing impossible
        else:
            costs.append(continue_cost(int(tree_sizes[t]), remaining, config))
    return costs


def swept_advantage_trace(
    pred_value_gain: Sequence[float],
    tree_sizes: Sequence[int],
    starting_budget: int,
    config: BudgetedOracleConfig,
) -> List[float]:
    """[GREEDY BASELINE] advantage(t ; lambda) = pred_value_gain(t) - analytic_cost(t ; lambda, T).

    The MYOPIC quantity a one-step greedy rule consumes: a cost-free predicted
    *total* gain minus the regime's *per-step* analytic cost.  This is exact only
    at the lambda extremes (lambda->0 search-forever; lambda huge stop-instantly)
    and is horizon-/unit-mismatched in the transition regime, because the total
    realizable gain depends on lambda (a higher cost stops sooner and never
    reaches the zero-cost optimum the gain was measured against).  Kept ONLY as a
    labelled baseline column so the misalignment is visible; the production rule
    is the DP over the predicted halt-reward curve (``dp_stop_over_curve``).
    """
    costs = analytic_cost_trace(tree_sizes, starting_budget, config, len(pred_value_gain))
    return [float(g) - c for g, c in zip(pred_value_gain, costs)]


def predicted_halt_curve(
    halt_rewards: Sequence[float],
    tree_sizes: Sequence[int],
    starting_budget: int,
) -> List[float]:
    """Cost-free per-step halt-reward curve for one episode (the trajectory target).

    The trajectory-prediction target is simply the per-step achievable halt
    reward, budget-truncated -- a rich PER-STEP curve, not a single collapsed
    gain scalar.  Returned at the same length the oracle uses
    (``min(len(halt_rewards), starting_budget)``) so a predicted curve can be fed
    straight back into the DP.  (This is identity on the cached trace; it exists
    so the head has a per-step regression target and the eval has a canonical
    truncation.)
    """
    max_steps = min(len(halt_rewards), starting_budget)
    return [float(halt_rewards[t]) for t in range(max_steps)]


def dp_stop_over_curve(
    pred_halt_curve: Sequence[float],
    tree_sizes: Sequence[int],
    starting_budget: int,
    config: BudgetedOracleConfig,
) -> int:
    """[PRODUCTION] True optimal stop for regime ``config`` from a predicted halt curve.

    Runs the EXACT oracle backward-induction (``compute_budgeted_oracle``) over
    the *predicted* per-step halt-reward curve with the regime's analytic
    per-step cost (maintenance + power-law/linear time cost).  Because this is
    the same DP the relabeled oracle uses, a single trajectory-prediction head
    reconstructs the true OSS for ANY lambda -- not greedy, not horizon-
    mismatched: train once, sweep many, and match the per-lambda oracle across
    the whole sweep including the transition rung.

    Returns the DP-optimal stop step (``policy.optimal_stop_step``).
    """
    policy = compute_budgeted_oracle(pred_halt_curve, tree_sizes, starting_budget, config)
    return policy.optimal_stop_step
