"""Difficulty and landscape metrics for halt/continue stopping episodes.

Each metric takes a halt_rewards sequence and (where needed) a continue_cost,
and returns a single float characterising how "hard" or "interesting" the
episode is for the meta-controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from controller_oracle import compute_oracle_policy
from planning_cost import PlanningCostConfig, planning_cost_config, return_for_stop_step


@dataclass(frozen=True)
class EpisodeDifficultyMetrics:
    halt_reward_range: float
    optimal_vs_second_best_gap: float
    return_variance: float
    softmax_entropy: float
    regret_of_always_halt: float
    reward_curvature: int


def halt_reward_range(halt_rewards: Sequence[float]) -> float:
    """Max minus min halt reward across the episode."""
    if len(halt_rewards) < 2:
        return 0.0
    return float(max(halt_rewards) - min(halt_rewards))


def _returns_at_all_stops(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> List[float]:
    resolved_cost = cost_config or planning_cost_config(
        continue_cost,
        kind=planning_cost_kind,
        exponent=planning_cost_exponent,
    )
    return [
        return_for_stop_step(list(halt_rewards), t, resolved_cost)
        for t in range(len(halt_rewards))
    ]


def optimal_vs_second_best_gap(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> float:
    """Gap between best and second-best return across all possible stop times."""
    returns = _returns_at_all_stops(
        halt_rewards,
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )
    if len(returns) < 2:
        return 0.0
    sorted_returns = sorted(returns, reverse=True)
    return sorted_returns[0] - sorted_returns[1]


def return_variance(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> float:
    """Variance of returns across all possible stopping times."""
    returns = _returns_at_all_stops(
        halt_rewards,
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    return sum((r - mean) ** 2 for r in returns) / len(returns)


def softmax_entropy(
    halt_rewards: Sequence[float],
    continue_cost: float,
    temperature: float = 1.0,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> float:
    """Entropy of the softmax distribution over returns at each stop time.

    Low entropy = one stop time dominates (hard, precise stopping matters).
    High entropy = many stop times are similarly good (easy, flat landscape).
    """
    returns = _returns_at_all_stops(
        halt_rewards,
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )
    if len(returns) < 2:
        return 0.0
    scaled = [r / temperature for r in returns]
    max_val = max(scaled)
    exp_vals = [math.exp(s - max_val) for s in scaled]
    total = sum(exp_vals)
    entropy = 0.0
    for e in exp_vals:
        p = e / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def regret_of_always_halt(
    halt_rewards: Sequence[float],
    continue_cost: float,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> float:
    """How much value is lost by always halting at step 0."""
    if not halt_rewards:
        return 0.0
    policy = compute_oracle_policy(
        halt_rewards,
        continue_cost,
        cost_config=cost_config,
        planning_cost_kind=planning_cost_kind,
        planning_cost_exponent=planning_cost_exponent,
    )
    return float(policy.values[0]) - float(halt_rewards[0])


def reward_curvature(halt_rewards: Sequence[float]) -> int:
    """Number of sign changes in consecutive halt reward differences.

    0 = monotone (trivially easy landscape).
    High = wiggly / non-monotone (complex landscape where the controller
    must learn to "wait out" dips).
    """
    if len(halt_rewards) < 3:
        return 0
    diffs = [halt_rewards[t + 1] - halt_rewards[t] for t in range(len(halt_rewards) - 1)]
    sign_changes = 0
    prev_sign = 0
    for d in diffs:
        if d > 0:
            current_sign = 1
        elif d < 0:
            current_sign = -1
        else:
            continue
        if prev_sign != 0 and current_sign != prev_sign:
            sign_changes += 1
        prev_sign = current_sign
    return sign_changes


def compute_difficulty_metrics(
    halt_rewards: Sequence[float],
    continue_cost: float,
    softmax_temperature: float = 1.0,
    *,
    cost_config: PlanningCostConfig | None = None,
    planning_cost_kind: str = "linear",
    planning_cost_exponent: float = 1.0,
) -> EpisodeDifficultyMetrics:
    """Compute all difficulty metrics for a single episode."""
    resolved_cost = cost_config or planning_cost_config(
        continue_cost,
        kind=planning_cost_kind,
        exponent=planning_cost_exponent,
    )
    return EpisodeDifficultyMetrics(
        halt_reward_range=halt_reward_range(halt_rewards),
        optimal_vs_second_best_gap=optimal_vs_second_best_gap(
            halt_rewards,
            continue_cost,
            cost_config=resolved_cost,
        ),
        return_variance=return_variance(
            halt_rewards,
            continue_cost,
            cost_config=resolved_cost,
        ),
        softmax_entropy=softmax_entropy(
            halt_rewards,
            continue_cost,
            softmax_temperature,
            cost_config=resolved_cost,
        ),
        regret_of_always_halt=regret_of_always_halt(
            halt_rewards,
            continue_cost,
            cost_config=resolved_cost,
        ),
        reward_curvature=reward_curvature(halt_rewards),
    )
