"""Validation policy metrics for Stage 2b proxy training.

Greedy regret stops at the first step with predicted advantage ≤ 0. Expected regret
uses a probabilistic stop rule: at each step t < T−1, stop with probability
``sigmoid(−advantage_t / temperature)``; any remaining probability mass stops at
the final step. Expected return is computed exactly (no sampling).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Sequence

from cts.data.preprocess_mc.oracle import BudgetedOracleConfig, return_for_stop_step
from cts.train.controller_train import EpisodeMetadata


def stop_probability(advantage: float, temperature: float) -> float:
    """Probability of stopping at this step given predicted advantage (logistic link)."""
    tau = max(float(temperature), 1e-8)
    x = -float(advantage) / tau
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def expected_return_and_stop_step(
    meta: EpisodeMetadata,
    advantages: Sequence[float],
    oracle_config: BudgetedOracleConfig,
    *,
    temperature: float,
) -> tuple[float, float]:
    """Expected halt return and expected stop step under independent per-step stopping."""
    if len(advantages) != meta.num_steps:
        raise ValueError(
            f"Episode {meta.path}: {len(advantages)} advantages vs {meta.num_steps} steps"
        )
    n = meta.num_steps
    if n == 0:
        raise ValueError(f"Episode {meta.path} has zero steps")

    survival = 1.0
    expected_return = 0.0
    expected_stop = 0.0

    for step in range(n):
        if step == n - 1:
            stop_mass = survival
        else:
            p_stop = stop_probability(advantages[step], temperature)
            stop_mass = survival * p_stop
            survival *= 1.0 - p_stop

        step_return = return_for_stop_step(
            meta.halt_rewards,
            meta.tree_sizes,
            meta.time_budgets,
            step,
            oracle_config,
        )
        expected_return += stop_mass * step_return
        expected_stop += stop_mass * float(step)

    return expected_return, expected_stop


@dataclass(frozen=True)
class ExpectedPolicyMetrics:
    """Aggregated validation metrics under probabilistic stopping."""

    average_return: float
    average_oracle_value: float
    average_regret: float
    average_expansions: float
    evaluated_episodes: int


def aggregate_expected_regret_metrics(
    episode_metadata: List[EpisodeMetadata],
    all_advantages: Sequence[float],
    oracle_config: BudgetedOracleConfig,
    *,
    temperature: float,
) -> ExpectedPolicyMetrics:
    """Mean oracle − expected return across episodes (contiguous advantage slices)."""
    offset = 0
    total_return = 0.0
    total_oracle = 0.0
    total_expansions = 0.0

    for meta in episode_metadata:
        episode_advantages = all_advantages[offset : offset + meta.num_steps]
        offset += meta.num_steps
        exp_return, exp_stop = expected_return_and_stop_step(
            meta,
            episode_advantages,
            oracle_config,
            temperature=temperature,
        )
        total_return += exp_return
        total_oracle += meta.oracle_value
        total_expansions += exp_stop

    if offset != len(all_advantages):
        raise ValueError(
            f"Advantage tensor length {len(all_advantages)} != episode steps {offset}"
        )

    evaluated = len(episode_metadata)
    if evaluated == 0:
        raise ValueError("Expected-regret evaluation produced no episodes.")

    return ExpectedPolicyMetrics(
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle / evaluated,
        average_regret=(total_oracle - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
    )


def evaluate_expected_regret_batched(
    episode_metadata: List[EpisodeMetadata],
    all_advantages: Sequence[float],
    oracle_config: BudgetedOracleConfig,
    *,
    temperature: float,
) -> ExpectedPolicyMetrics:
    """Entry point mirroring ``evaluate_batched_greedy_policy`` for analysis runs."""
    _ = time.perf_counter()
    return aggregate_expected_regret_metrics(
        episode_metadata,
        all_advantages,
        oracle_config,
        temperature=temperature,
    )
