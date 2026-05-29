"""Hand-written stop-rule baselines so the controller's lift is contextualized.

Sweeps over halt-when-T-low / halt-when-N-high / OR-combination / always-
halt-immediately / always-search-to-the-end stop rules, replays each rule
on every episode, and reports the per-baseline greedy-style metrics plus
the best-by-regret and best-by-return picks for the report.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    return_for_stop_step,
)


# Quantiles of the tree-size distribution at which to plant the halt-when-N-high
# threshold sweep. Covers the low / lower-quartile / median / upper-quartile /
# high regions so the baseline grid spans the realized range of tree sizes
# without requiring the dataset's exact extremes.
_BASELINE_SWEEP_SIZE_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
# Hand-picked time-budget thresholds for the halt-when-T-low sweep. Denser at
# the low end (1, 2, 3, 4, 5) where the decision matters most, with a few
# larger anchors (7, 10, 15, 20) so the OR combinations still cover the
# slack-budget regime.
_BASELINE_SWEEP_TIME_THRESHOLDS = [1, 2, 3, 4, 5, 7, 10, 15, 20]


def _evaluate_baseline(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
    stop_rule: Callable[[dict[str, Any]], int],
) -> dict[str, Any]:
    """Replay a hand-written stop rule on every episode and report greedy-style metrics.

    ``stop_rule`` receives an episode dict and returns the step at which the
    baseline would halt. The clamp guards against rules that emit indices
    outside the episode (e.g. never-trigger thresholds).
    """
    returns = []
    regrets = []
    exact = 0
    first = 0
    expansions = 0
    for episode in diagnostics:
        stop_step = stop_rule(episode)
        stop_step = max(0, min(stop_step, len(episode["halt_rewards"]) - 1))
        predicted_return = return_for_stop_step(
            episode["halt_rewards"],
            episode["tree_sizes"],
            episode["time_budgets"],
            stop_step,
            oracle_config,
        )
        returns.append(predicted_return)
        regrets.append(float(episode["oracle_value"]) - predicted_return)
        exact += int(stop_step == int(episode["oracle_stop_step"]))
        first += int((stop_step == 0) == (int(episode["oracle_stop_step"]) == 0))
        expansions += stop_step
    n = len(diagnostics)
    return {
        "average_return": statistics.mean(returns),
        "average_regret": statistics.mean(regrets),
        "exact_stop_step_accuracy": exact / n,
        "first_action_accuracy": first / n,
        "average_expansions": expansions / n,
    }


def _baseline_sweep(diagnostics: list[dict[str, Any]], oracle_config: BudgetedOracleConfig) -> dict[str, Any]:
    """Evaluate a suite of trivial baselines so the controller's lift is contextualized.

    Sweeps over halt-when-T-low, halt-when-N-high, and the OR combination of
    both, plus always-halt-immediately and always-search-to-the-end. Returns
    every baseline's metrics plus the best-by-regret and best-by-return.
    """
    tree_sizes = sorted({int(size) for episode in diagnostics for size in episode["tree_sizes"]})
    size_thresholds = sorted(
        {tree_sizes[int(round(idx * (len(tree_sizes) - 1)))] for idx in _BASELINE_SWEEP_SIZE_QUANTILES}
    )
    time_thresholds = sorted(set(_BASELINE_SWEEP_TIME_THRESHOLDS))

    baselines: dict[str, dict[str, Any]] = {}
    baselines["always_halt_0"] = _evaluate_baseline(diagnostics, oracle_config, lambda episode: 0)
    baselines["always_continue_to_end"] = _evaluate_baseline(
        diagnostics,
        oracle_config,
        lambda episode: len(episode["halt_rewards"]) - 1,
    )

    # Halt as soon as remaining time budget drops below threshold; fall back
    # to the final step if the threshold never triggers.
    for threshold in time_thresholds:
        baselines[f"halt_when_T_le_{threshold}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, threshold=threshold: next(
                (idx for idx, time_budget in enumerate(episode["time_budgets"]) if int(time_budget) <= threshold),
                len(episode["halt_rewards"]) - 1,
            ),
        )
    # Halt as soon as tree size grows past threshold.
    for threshold in size_thresholds:
        baselines[f"halt_when_N_ge_{threshold}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, threshold=threshold: next(
                (idx for idx, size in enumerate(episode["tree_sizes"]) if int(size) >= threshold),
                len(episode["halt_rewards"]) - 1,
            ),
        )
    # OR-combine: halt the moment either threshold is satisfied.
    for t_threshold in time_thresholds:
        for n_threshold in size_thresholds:
            baselines[f"halt_when_T_le_{t_threshold}_or_N_ge_{n_threshold}"] = _evaluate_baseline(
                diagnostics,
                oracle_config,
                lambda episode, t=t_threshold, n=n_threshold: next(
                    (
                        idx
                        for idx, (time_budget, size) in enumerate(zip(episode["time_budgets"], episode["tree_sizes"]))
                        if int(time_budget) <= t or int(size) >= n
                    ),
                    len(episode["halt_rewards"]) - 1,
                ),
            )

    best_regret_name, best_regret_metrics = min(baselines.items(), key=lambda item: item[1]["average_regret"])
    best_return_name, best_return_metrics = max(baselines.items(), key=lambda item: item[1]["average_return"])
    return {
        "all_baselines": baselines,
        "best_by_regret": {"name": best_regret_name, **best_regret_metrics},
        "best_by_return": {"name": best_return_name, **best_return_metrics},
    }
