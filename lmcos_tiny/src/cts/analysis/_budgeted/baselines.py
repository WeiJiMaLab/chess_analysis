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
# Marginal-gain thresholds for the gain-depth-only sweep. The "gain" of the
# i-th expansion is the change in the haltable reward, halt_rewards[i] -
# halt_rewards[i-1] — a myopic value-of-computation signal. Halting when that
# gain falls to/below the threshold is the dynamic stop rule that uses gain
# alone (Russek-style VOC), independent of tree size or remaining budget.
_BASELINE_SWEEP_GAIN_THRESHOLDS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
# Value-plateau: a noise-robust gain-depth. Instead of one step's gain, look at
# the improvement over a window of ``w`` steps and halt once it falls to/below
# the threshold (the value curve has flattened over the window, not just dipped
# for a single noisy step). ``w == 1`` reduces to gain-depth.
_BASELINE_SWEEP_PLATEAU_WINDOW = 3
_BASELINE_SWEEP_PLATEAU_THRESHOLDS = [0.0, 0.01, 0.05]
# Fixed-fraction-of-budget: a constant "spend rho of the starting budget then
# stop" anchor (rho in (0, 1)). A non-trivial control that ignores the value
# landscape entirely and just allocates a fixed share of the budget.
_BASELINE_SWEEP_FRACTION_RHOS = [0.1, 0.25, 0.5, 0.75]


def _gain_depth_stop_step(episode: dict[str, Any], threshold: float) -> int:
    """First step (>=1) whose marginal halt-reward gain is <= threshold, else the last step."""
    halt_rewards = episode["halt_rewards"]
    return next(
        (
            idx
            for idx in range(1, len(halt_rewards))
            if (halt_rewards[idx] - halt_rewards[idx - 1]) <= threshold
        ),
        len(halt_rewards) - 1,
    )


def _value_plateau_stop_step(episode: dict[str, Any], threshold: float, window: int) -> int:
    """First step (>=window) whose improvement over the last ``window`` steps is <= threshold.

    Noise-robust gain-depth: ``halt_rewards[i] - halt_rewards[i-window] <= threshold``
    means the value has not climbed meaningfully across the window — a plateau —
    rather than reacting to a single noisy step. Falls through to the last step.
    """
    halt_rewards = episode["halt_rewards"]
    return next(
        (
            idx
            for idx in range(window, len(halt_rewards))
            if (halt_rewards[idx] - halt_rewards[idx - window]) <= threshold
        ),
        len(halt_rewards) - 1,
    )


def _fixed_fraction_stop_step(episode: dict[str, Any], rho: float) -> int:
    """Halt after spending ``rho`` of the starting budget (clamped in-episode by the caller).

    ``time_budgets[0]`` is the starting budget by construction, so this stops at
    ``round(rho * starting_budget)`` regardless of the value landscape.
    """
    starting_budget = int(episode["time_budgets"][0])
    return int(round(rho * starting_budget))


def stop_fit_metrics(
    stops: list[int],
    oracle_stops: list[int],
    regrets: list[float],
) -> dict[str, float]:
    """Tier-1 goodness-of-fit of stop decisions vs the oracle's OSS.

    All derived from per-episode ``(predicted stop, oracle stop, regret)``:

    - ``stop_bias`` = E[stop - OSS]; **positive => systematic over-search** (stops late),
      negative => under-search. The single most diagnostic stat for the
      Fraction-vs-MCHalt gap (MCHalt stops much later).
    - ``stop_mae`` = E[|stop - OSS|]: absolute stop-step fit (vs the brittle
      exact-match P(stop==OSS), which scores a 1-step miss as fully wrong even
      though the return curve is ~flat near OSS).
    - ``tol_acc_1`` / ``tol_acc_2`` = P(|stop - OSS| <= 1 / 2): near-miss-tolerant
      stop accuracy (k=0 recovers exact-match).
    - ``frac_early`` / ``frac_late`` = P(stop < OSS) / P(stop > OSS).
    - ``regret_from_early`` / ``regret_from_late`` = mean over ALL episodes of the
      regret restricted to stop<OSS / stop>OSS. Regret at stop==OSS is 0 by
      construction, so these two **sum to average_regret** — an exact decomposition
      of the value loss into *under-searching* (missed value) vs *over-searching*
      (wasted cost).
    - ``regret_median`` / ``regret_p90`` / ``regret_p99`` = the regret distribution's
      median and upper tail (nearest-rank percentiles), so a low *mean* that hides a
      heavy catastrophic-stop tail is visible.
    """
    n = len(stops)
    if n == 0:
        keys = ("stop_bias", "stop_mae", "tol_acc_1", "tol_acc_2",
                "frac_early", "frac_late", "regret_from_early", "regret_from_late",
                "regret_median", "regret_p90", "regret_p99")
        return {k: 0.0 for k in keys}
    diffs = [int(s) - int(o) for s, o in zip(stops, oracle_stops)]
    srt = sorted(regrets)

    def _pct(p: float) -> float:  # nearest-rank percentile of the regret distribution
        return srt[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "stop_bias": sum(diffs) / n,
        "stop_mae": sum(abs(d) for d in diffs) / n,
        "tol_acc_1": sum(abs(d) <= 1 for d in diffs) / n,
        "tol_acc_2": sum(abs(d) <= 2 for d in diffs) / n,
        "frac_early": sum(d < 0 for d in diffs) / n,
        "frac_late": sum(d > 0 for d in diffs) / n,
        "regret_from_early": sum(r for d, r in zip(diffs, regrets) if d < 0) / n,
        "regret_from_late": sum(r for d, r in zip(diffs, regrets) if d > 0) / n,
        # regret DISTRIBUTION, not just its mean: median + upper tail. The mean can
        # be dragged by a few catastrophic over/under-stops; p90/p99 expose whether
        # a policy's edge is broad or hides a heavy tail.
        "regret_median": _pct(0.50),
        "regret_p90": _pct(0.90),
        "regret_p99": _pct(0.99),
    }


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
    stops: list[int] = []
    oracle_stops: list[int] = []
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
        oracle_stop = int(episode["oracle_stop_step"])
        returns.append(predicted_return)
        regrets.append(float(episode["oracle_value"]) - predicted_return)
        stops.append(stop_step)
        oracle_stops.append(oracle_stop)
        exact += int(stop_step == oracle_stop)
        first += int((stop_step == 0) == (oracle_stop == 0))
        expansions += stop_step
    n = len(diagnostics)
    return {
        "average_return": statistics.mean(returns),
        "average_regret": statistics.mean(regrets),
        "exact_stop_step_accuracy": exact / n,
        "first_action_accuracy": first / n,
        "average_expansions": expansions / n,
        "per_episode_regrets": list(regrets),  # for bootstrap CIs
        **stop_fit_metrics(stops, oracle_stops, regrets),
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
    # Gain-depth-only: halt as soon as the marginal value of one more expansion
    # decays to/below threshold. Uses the halt-reward trajectory alone.
    for threshold in _BASELINE_SWEEP_GAIN_THRESHOLDS:
        baselines[f"halt_when_gain_le_{threshold}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, threshold=threshold: _gain_depth_stop_step(episode, threshold),
        )
    # Value-plateau: noise-robust gain-depth over a window of expansions.
    plateau_window = _BASELINE_SWEEP_PLATEAU_WINDOW
    for threshold in _BASELINE_SWEEP_PLATEAU_THRESHOLDS:
        baselines[f"value_plateau_w{plateau_window}_le_{threshold}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, threshold=threshold, window=plateau_window: _value_plateau_stop_step(
                episode, threshold, window
            ),
        )
    # Fixed-fraction-of-budget: spend rho of the starting budget, then stop.
    for rho in _BASELINE_SWEEP_FRACTION_RHOS:
        baselines[f"halt_at_frac_{rho}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, rho=rho: _fixed_fraction_stop_step(episode, rho),
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
