"""Oracle-halt drivers, factor magnitudes, and the oversearch-tail decomposition.

Computes the per-step factor rows that flatten each episode into "what
drove the oracle's halt decision?" tuples (both the V*-based local view
and the clean future-best-stop view), then renders the bucket-stacked
driver bar plots, factor-magnitude grouped bars, and false-continue
rates per driver. Also covers the error-episode oracle-stop
decomposition and the per-extra-step oversearch-tail cost profile.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cts.analysis._budgeted._shared import (
    ANALYSIS_FIGURE_DPI,
    FIGSIZE_SINGLE_WIDE,
)
from cts.analysis._common import _episode_regret_decomposition
from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    maintenance_cost,
    time_cost,
)


def _oracle_stop_driver(
    future_value_gain: float,
    maintenance_term: float,
    time_term: float,
) -> str:
    """Classify the local oracle-halt reason from V*(s+1) - V_halt and the per-step costs."""
    if future_value_gain <= 0.0:
        return "future_already_worse"
    if maintenance_term >= time_term:
        return "maintenance_dominated"
    return "time_dominated"


def _oracle_stop_driver_clean(
    cost_free_future_reward_gain: float,
    future_net_gain_before_current_cost: float,
) -> str:
    """Classify the oracle-halt reason using the "clean" decomposition fields."""
    if cost_free_future_reward_gain <= 0.0:
        return "no_cost_free_gain"
    if future_net_gain_before_current_cost <= 0.0:
        return "downstream_cost_dominated"
    return "current_step_cost_dominated"


def _best_future_stop_decomposition(
    halt_rewards: list[float],
    tree_sizes: list[int],
    time_budgets: list[int],
    step_index: int,
    oracle_config: BudgetedOracleConfig,
) -> dict[str, float | int]:
    """"Clean" decomposition of the oracle's advantage at one step.

    Splits the target advantage into three intelligible pieces:
    ``cost_free_future_reward_gain`` (the best later halt reward minus the
    current halt reward, ignoring all costs), ``downstream_future_cost``
    (the costs paid to reach that best future stop), and ``current_step_cost``
    (the immediate cost of taking one more expansion now). Sums back to the
    target advantage modulo the timeout / data-truncation edge cases.
    """
    current_halt_reward = float(halt_rewards[step_index])
    current_step_cost = maintenance_cost(int(tree_sizes[step_index]), oracle_config) + time_cost(int(time_budgets[step_index]), oracle_config)
    current_time_budget = int(time_budgets[step_index])

    # Budget exhausted: the only "future" is the timeout penalty.
    if current_time_budget <= oracle_config.time_delta:
        future_net_gain_before_current_cost = float(oracle_config.timeout_value) - current_halt_reward
        return {
            "best_future_stop_step": step_index,
            "cost_free_future_reward_gain": float(oracle_config.timeout_value) - current_halt_reward,
            "downstream_future_cost": 0.0,
            "current_step_cost": float(current_step_cost),
            "future_net_gain_before_current_cost": float(future_net_gain_before_current_cost),
            "reconstructed_target_advantage": float(future_net_gain_before_current_cost - current_step_cost),
        }

    # Data-truncation edge: no future steps available, only the current cost.
    if step_index + 1 >= len(halt_rewards):
        return {
            "best_future_stop_step": step_index,
            "cost_free_future_reward_gain": 0.0,
            "downstream_future_cost": 0.0,
            "current_step_cost": float(current_step_cost),
            "future_net_gain_before_current_cost": 0.0,
            "reconstructed_target_advantage": -float(current_step_cost),
        }

    # General case: search every later candidate stop for the one with the
    # highest cost-discounted value, accumulating its downstream cost as
    # we go. O(T^2) but T is small (<= ~120 steps).
    best_stop = step_index + 1
    best_future_value = float("-inf")
    for future_stop in range(step_index + 1, len(halt_rewards)):
        downstream_cost = 0.0
        for idx in range(step_index + 1, future_stop):
            downstream_cost += maintenance_cost(int(tree_sizes[idx]), oracle_config)
            downstream_cost += time_cost(int(time_budgets[idx]), oracle_config)
        future_value = float(halt_rewards[future_stop]) - downstream_cost
        if future_value > best_future_value:
            best_future_value = future_value
            best_stop = future_stop

    # Re-accumulate the downstream cost up to the chosen best stop so we
    # can report it cleanly.
    downstream_future_cost = 0.0
    for idx in range(step_index + 1, best_stop):
        downstream_future_cost += maintenance_cost(int(tree_sizes[idx]), oracle_config)
        downstream_future_cost += time_cost(int(time_budgets[idx]), oracle_config)
    cost_free_future_reward_gain = float(halt_rewards[best_stop]) - current_halt_reward
    future_net_gain_before_current_cost = cost_free_future_reward_gain - downstream_future_cost
    reconstructed_target_advantage = future_net_gain_before_current_cost - current_step_cost
    return {
        "best_future_stop_step": best_stop,
        "cost_free_future_reward_gain": float(cost_free_future_reward_gain),
        "downstream_future_cost": float(downstream_future_cost),
        "current_step_cost": float(current_step_cost),
        "future_net_gain_before_current_cost": float(future_net_gain_before_current_cost),
        "reconstructed_target_advantage": float(reconstructed_target_advantage),
    }


def _oracle_step_factor_rows(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> list[dict[str, Any]]:
    """Flatten episodes into per-step rows annotated with the oracle's local decision factors.

    Each row contains the per-step ``future_value_gain``/``maintenance_cost``/
    ``time_cost`` along with the controller's predicted vs target advantage,
    and at oracle-stop steps records which factor "drove" the halt. Used as
    the input to several factor/driver summary plots.
    """
    rows: list[dict[str, Any]] = []
    for episode in diagnostics:
        policy = compute_budgeted_oracle(
            episode["halt_rewards"],
            episode["tree_sizes"],
            int(episode["starting_budget"]),
            oracle_config,
        )
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        for step_index, (halt_reward, tree_size, time_budget, predicted_advantage, target_advantage) in enumerate(
            zip(
                episode["halt_rewards"],
                episode["tree_sizes"],
                episode["time_budgets"],
                episode["predicted_advantages"],
                episode["target_advantages"],
            )
        ):
            maintenance_term = maintenance_cost(int(tree_size), oracle_config)
            time_term = time_cost(int(time_budget), oracle_config)
            # Local view of the value of "next" state: timeout penalty when
            # the budget is about to run out, otherwise V*(s+1) from the DP.
            if int(time_budget) <= oracle_config.time_delta:
                next_value = float(oracle_config.timeout_value)
            elif step_index + 1 < len(policy.values):
                next_value = float(policy.values[step_index + 1])
            else:
                next_value = float(halt_reward)
            future_value_gain = next_value - float(halt_reward)
            reconstructed = future_value_gain - maintenance_term - time_term
            rows.append(
                {
                    "path": episode["path"],
                    "source_path": episode["source_path"],
                    "budget_bucket_name": episode["budget_bucket_name"],
                    "starting_budget": int(episode["starting_budget"]),
                    "step_index": step_index,
                    "oracle_stop_step": oracle_stop,
                    "predicted_stop_step": predicted_stop,
                    "at_oracle_stop": step_index == oracle_stop,
                    "at_predicted_stop": step_index == predicted_stop,
                    "halt_reward": float(halt_reward),
                    "future_value_gain": float(future_value_gain),
                    "maintenance_cost": float(maintenance_term),
                    "time_cost": float(time_term),
                    "total_cost": float(maintenance_term + time_term),
                    "target_advantage": float(target_advantage),
                    "predicted_advantage": float(predicted_advantage),
                    "reconstructed_target_advantage": float(reconstructed),
                    "target_residual": float(reconstructed - float(target_advantage)),
                    "sign_correct": (float(predicted_advantage) > 0.0) == (float(target_advantage) > 0.0),
                    "false_continue": float(predicted_advantage) > 0.0 and float(target_advantage) <= 0.0,
                    "false_halt": float(predicted_advantage) <= 0.0 and float(target_advantage) > 0.0,
                    "oracle_stop_driver": _oracle_stop_driver(future_value_gain, maintenance_term, time_term)
                    if step_index == oracle_stop
                    else None,
                }
            )
    return rows


def _oracle_stop_clean_factor_rows(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> list[dict[str, Any]]:
    """Per-step rows annotated with the "clean" future-best-stop decomposition.

    Parallel to ``_oracle_step_factor_rows`` but uses ``_best_future_stop_decomposition``
    instead of the V*-based local decomposition. Both views are produced
    because they emphasize different aspects of the halt decision.
    """
    rows: list[dict[str, Any]] = []
    for episode in diagnostics:
        halt_rewards = [float(value) for value in episode["halt_rewards"]]
        tree_sizes = [int(value) for value in episode["tree_sizes"]]
        time_budgets = [int(value) for value in episode["time_budgets"]]
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        predicted_advantages = [float(value) for value in episode["predicted_advantages"]]
        target_advantages = [float(value) for value in episode["target_advantages"]]

        for step_index in range(len(halt_rewards)):
            decomp = _best_future_stop_decomposition(
                halt_rewards,
                tree_sizes,
                time_budgets,
                step_index,
                oracle_config,
            )
            rows.append(
                {
                    "path": str(episode["path"]),
                    "source_path": str(episode["source_path"]),
                    "budget_bucket_name": str(episode["budget_bucket_name"]),
                    "starting_budget": int(episode["starting_budget"]),
                    "step_index": step_index,
                    "oracle_stop_step": oracle_stop,
                    "predicted_stop_step": predicted_stop,
                    "at_oracle_stop": step_index == oracle_stop,
                    "at_predicted_stop": step_index == predicted_stop,
                    "cost_free_future_reward_gain": float(decomp["cost_free_future_reward_gain"]),
                    "downstream_future_cost": float(decomp["downstream_future_cost"]),
                    "current_step_cost": float(decomp["current_step_cost"]),
                    "future_net_gain_before_current_cost": float(decomp["future_net_gain_before_current_cost"]),
                    "best_future_stop_step": int(decomp["best_future_stop_step"]),
                    "target_advantage": target_advantages[step_index],
                    "predicted_advantage": predicted_advantages[step_index],
                    "reconstructed_target_advantage": float(decomp["reconstructed_target_advantage"]),
                    "target_residual": float(decomp["reconstructed_target_advantage"] - target_advantages[step_index]),
                    "sign_correct": (predicted_advantages[step_index] > 0.0) == (target_advantages[step_index] > 0.0),
                    "false_continue": predicted_advantages[step_index] > 0.0 and target_advantages[step_index] <= 0.0,
                    "false_halt": predicted_advantages[step_index] <= 0.0 and target_advantages[step_index] > 0.0,
                    "oracle_stop_driver_clean": _oracle_stop_driver_clean(
                        float(decomp["cost_free_future_reward_gain"]),
                        float(decomp["future_net_gain_before_current_cost"]),
                    )
                    if step_index == oracle_stop
                    else None,
                }
            )
    return rows


def _oracle_stop_factor_summary(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the V*-based driver/factor stats at the oracle stop step (overall, per driver, per bucket)."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_driver: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket_driver: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in stop_rows:
        bucket = str(row["budget_bucket_name"])
        driver = str(row["oracle_stop_driver"])
        by_bucket[bucket].append(row)
        by_driver[driver].append(row)
        by_bucket_driver[bucket][driver].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "mean_future_value_gain": statistics.mean(float(row["future_value_gain"]) for row in rows),
            "mean_maintenance_cost": statistics.mean(float(row["maintenance_cost"]) for row in rows),
            "mean_time_cost": statistics.mean(float(row["time_cost"]) for row in rows),
            "mean_total_cost": statistics.mean(float(row["total_cost"]) for row in rows),
            "mean_target_advantage": statistics.mean(float(row["target_advantage"]) for row in rows),
            "mean_predicted_advantage": statistics.mean(float(row["predicted_advantage"]) for row in rows),
            "false_continue_rate": statistics.mean(1.0 if row["false_continue"] else 0.0 for row in rows),
            "sign_accuracy": statistics.mean(1.0 if row["sign_correct"] else 0.0 for row in rows),
        }

    driver_counts = Counter(str(row["oracle_stop_driver"]) for row in stop_rows)
    return {
        "overall": summarize(stop_rows),
        "driver_counts": dict(driver_counts),
        "by_driver": {driver: summarize(rows) for driver, rows in sorted(by_driver.items())},
        "by_bucket": {bucket: summarize(rows) for bucket, rows in sorted(by_bucket.items())},
        "by_bucket_driver": {
            bucket: {driver: summarize(rows) for driver, rows in sorted(driver_map.items())}
            for bucket, driver_map in sorted(by_bucket_driver.items())
        },
        "max_abs_target_residual": max(abs(float(row["target_residual"])) for row in stop_rows) if stop_rows else 0.0,
    }


def _plot_oracle_stop_driver_by_bucket(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Stacked-bar plot of the V*-based driver mix at the oracle stop step, per bucket."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    buckets = sorted({str(row["budget_bucket_name"]) for row in stop_rows})
    drivers = ["future_already_worse", "maintenance_dominated", "time_dominated"]
    fractions: dict[str, list[float]] = {driver: [] for driver in drivers}
    summary: dict[str, Any] = {}

    for bucket in buckets:
        bucket_rows = [row for row in stop_rows if str(row["budget_bucket_name"]) == bucket]
        counts = Counter(str(row["oracle_stop_driver"]) for row in bucket_rows)
        total = len(bucket_rows)
        summary[bucket] = {driver: counts.get(driver, 0) / total for driver in drivers}
        for driver in drivers:
            fractions[driver].append(summary[bucket][driver])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_WIDE, constrained_layout=True)
    x = np.arange(len(buckets))
    bottom = np.zeros(len(buckets))
    colors = {
        "future_already_worse": "tab:blue",
        "maintenance_dominated": "tab:red",
        "time_dominated": "tab:orange",
    }
    for driver in drivers:
        values = np.array(fractions[driver], dtype=np.float32)
        ax.bar(x, values, bottom=bottom, label=driver, color=colors[driver], alpha=0.85)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of oracle-stop episodes")
    ax.set_title("What Makes the Oracle Halt?")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_oracle_stop_factor_magnitudes(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Grouped-bar plot of the three V*-based factor magnitudes at the oracle stop step, per bucket."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    buckets = sorted({str(row["budget_bucket_name"]) for row in stop_rows})
    future_gain = []
    maintenance_vals = []
    time_vals = []
    summary: dict[str, Any] = {}
    for bucket in buckets:
        rows = [row for row in stop_rows if str(row["budget_bucket_name"]) == bucket]
        future_gain.append(statistics.mean(float(row["future_value_gain"]) for row in rows))
        maintenance_vals.append(statistics.mean(float(row["maintenance_cost"]) for row in rows))
        time_vals.append(statistics.mean(float(row["time_cost"]) for row in rows))
        summary[bucket] = {
            "mean_future_value_gain": future_gain[-1],
            "mean_maintenance_cost": maintenance_vals[-1],
            "mean_time_cost": time_vals[-1],
            "episodes": len(rows),
        }

    x = np.arange(len(buckets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width, future_gain, width=width, label="future value gain", color="tab:blue")
    ax.bar(x, maintenance_vals, width=width, label="maintenance cost", color="tab:red")
    ax.bar(x + width, time_vals, width=width, label="time cost", color="tab:orange")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylabel("Mean magnitude at oracle stop")
    ax.set_title("Oracle Stop Factors by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_false_continue_by_oracle_driver(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """False-continue rate at the oracle stop step, sliced by V*-based driver category."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    drivers = ["future_already_worse", "maintenance_dominated", "time_dominated"]
    rates = []
    counts = []
    summary: dict[str, Any] = {}
    for driver in drivers:
        rows = [row for row in stop_rows if str(row["oracle_stop_driver"]) == driver]
        rate = statistics.mean(1.0 if row["false_continue"] else 0.0 for row in rows) if rows else 0.0
        rates.append(rate)
        counts.append(len(rows))
        summary[driver] = {"false_continue_rate": rate, "episodes": len(rows)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(drivers, rates, color=["tab:blue", "tab:red", "tab:orange"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("False-continue rate")
    axes[0].set_title("Miss Rate at Oracle Stop by Driver")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(drivers, counts, color="tab:gray")
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Oracle-Stop Episode Count by Driver")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _oracle_stop_clean_factor_summary(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the clean-decomposition factor magnitudes at the oracle stop step.

    Companion to ``_oracle_stop_factor_summary`` but for the clean decomposition.
    Reports per-driver and per-bucket means plus the max absolute residual
    so callers can spot reconstruction drift.
    """
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_driver: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stop_rows:
        by_bucket[str(row["budget_bucket_name"])].append(row)
        by_driver[str(row["oracle_stop_driver_clean"])].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "mean_cost_free_future_reward_gain": statistics.mean(float(row["cost_free_future_reward_gain"]) for row in rows),
            "mean_downstream_future_cost": statistics.mean(float(row["downstream_future_cost"]) for row in rows),
            "mean_current_step_cost": statistics.mean(float(row["current_step_cost"]) for row in rows),
            "mean_future_net_gain_before_current_cost": statistics.mean(float(row["future_net_gain_before_current_cost"]) for row in rows),
            "mean_target_advantage": statistics.mean(float(row["target_advantage"]) for row in rows),
            "mean_predicted_advantage": statistics.mean(float(row["predicted_advantage"]) for row in rows),
            "false_continue_rate": statistics.mean(1.0 if row["false_continue"] else 0.0 for row in rows),
            "sign_accuracy": statistics.mean(1.0 if row["sign_correct"] else 0.0 for row in rows),
        }

    driver_counts = Counter(str(row["oracle_stop_driver_clean"]) for row in stop_rows)
    return {
        "overall": summarize(stop_rows),
        "driver_counts": dict(driver_counts),
        "by_driver": {driver: summarize(rows) for driver, rows in sorted(by_driver.items())},
        "by_bucket": {bucket: summarize(rows) for bucket, rows in sorted(by_bucket.items())},
        "max_abs_target_residual": max(abs(float(row["target_residual"])) for row in stop_rows) if stop_rows else 0.0,
    }


def _plot_oracle_stop_clean_driver_by_bucket(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Stacked-bar plot of the clean-driver mix at the oracle stop step, per bucket."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    buckets = sorted({str(row["budget_bucket_name"]) for row in stop_rows})
    drivers = ["no_cost_free_gain", "downstream_cost_dominated", "current_step_cost_dominated"]
    fractions: dict[str, list[float]] = {driver: [] for driver in drivers}
    summary: dict[str, Any] = {}
    for bucket in buckets:
        bucket_rows = [row for row in stop_rows if str(row["budget_bucket_name"]) == bucket]
        counts = Counter(str(row["oracle_stop_driver_clean"]) for row in bucket_rows)
        total = len(bucket_rows)
        summary[bucket] = {driver: counts.get(driver, 0) / total for driver in drivers}
        for driver in drivers:
            fractions[driver].append(summary[bucket][driver])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_WIDE, constrained_layout=True)
    x = np.arange(len(buckets))
    bottom = np.zeros(len(buckets))
    colors = {
        "no_cost_free_gain": "tab:blue",
        "downstream_cost_dominated": "tab:red",
        "current_step_cost_dominated": "tab:orange",
    }
    for driver in drivers:
        values = np.array(fractions[driver], dtype=np.float32)
        ax.bar(x, values, bottom=bottom, label=driver, color=colors[driver], alpha=0.85)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of oracle-stop episodes")
    ax.set_title("Clean Oracle Halt Drivers by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_oracle_stop_clean_factor_magnitudes(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """Grouped-bar plot of the three clean-factor magnitudes (future gain, downstream cost, current cost) per bucket."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    buckets = sorted({str(row["budget_bucket_name"]) for row in stop_rows})
    future_reward = []
    downstream_cost = []
    current_cost = []
    summary: dict[str, Any] = {}
    for bucket in buckets:
        rows = [row for row in stop_rows if str(row["budget_bucket_name"]) == bucket]
        future_reward.append(statistics.mean(float(row["cost_free_future_reward_gain"]) for row in rows))
        downstream_cost.append(statistics.mean(float(row["downstream_future_cost"]) for row in rows))
        current_cost.append(statistics.mean(float(row["current_step_cost"]) for row in rows))
        summary[bucket] = {
            "mean_cost_free_future_reward_gain": future_reward[-1],
            "mean_downstream_future_cost": downstream_cost[-1],
            "mean_current_step_cost": current_cost[-1],
            "episodes": len(rows),
        }

    x = np.arange(len(buckets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width, future_reward, width=width, label="cost-free future reward gain", color="tab:blue")
    ax.bar(x, downstream_cost, width=width, label="downstream future cost", color="tab:red")
    ax.bar(x + width, current_cost, width=width, label="current-step cost", color="tab:orange")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylabel("Mean magnitude at oracle stop")
    ax.set_title("Clean Oracle Stop Factors by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_false_continue_by_clean_oracle_driver(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    """False-continue rate at the oracle stop step, sliced by clean driver category."""
    stop_rows = [row for row in step_rows if bool(row["at_oracle_stop"])]
    drivers = ["no_cost_free_gain", "downstream_cost_dominated", "current_step_cost_dominated"]
    rates = []
    counts = []
    summary: dict[str, Any] = {}
    for driver in drivers:
        rows = [row for row in stop_rows if str(row["oracle_stop_driver_clean"]) == driver]
        rate = statistics.mean(1.0 if row["false_continue"] else 0.0 for row in rows) if rows else 0.0
        rates.append(rate)
        counts.append(len(rows))
        summary[driver] = {"false_continue_rate": rate, "episodes": len(rows)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(drivers, rates, color=["tab:blue", "tab:red", "tab:orange"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("False-continue rate")
    axes[0].set_title("Miss Rate at Oracle Stop by Clean Driver")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(drivers, counts, color="tab:gray")
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Oracle-Stop Episode Count by Clean Driver")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _oversearch_regret_component_summary(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, Any]:
    """Mean regret-decomposition components for oversearch episodes only, overall and per bucket."""
    oversearch = [
        _episode_regret_decomposition(episode, oracle_config)
        for episode in diagnostics
        if int(episode["predicted_stop_step"]) > int(episode["oracle_stop_step"])
    ]
    by_bucket: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in oversearch:
        by_bucket[str(row["budget_bucket_name"])].append(row)

    def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "mean_regret": statistics.mean(float(row["saved_regret"]) for row in rows),
            "mean_halt_reward_term": statistics.mean(float(row["halt_reward_term"]) for row in rows),
            "mean_maintenance_term": statistics.mean(float(row["maintenance_term"]) for row in rows),
            "mean_time_term": statistics.mean(float(row["time_term"]) for row in rows),
        }

    return {
        "overall": summarize(oversearch),
        "by_bucket": {bucket: summarize(rows) for bucket, rows in sorted(by_bucket.items())},
    }


def _plot_oversearch_regret_components_by_bucket(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
    out_path: Path,
) -> dict[str, Any]:
    """Stacked-bar plot of where oversearch regret comes from (halt vs maintenance vs time) per bucket."""
    summary = _oversearch_regret_component_summary(diagnostics, oracle_config)
    bucket_summary = summary["by_bucket"]
    buckets = list(bucket_summary)
    halt_terms = [bucket_summary[bucket]["mean_halt_reward_term"] for bucket in buckets]
    maintenance_terms = [bucket_summary[bucket]["mean_maintenance_term"] for bucket in buckets]
    time_terms = [bucket_summary[bucket]["mean_time_term"] for bucket in buckets]

    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_WIDE, constrained_layout=True)
    ax.bar(x, halt_terms, label="halt-reward term", color="tab:blue")
    ax.bar(x, maintenance_terms, bottom=halt_terms, label="maintenance term", color="tab:red")
    stacked_bottom = np.array(halt_terms) + np.array(maintenance_terms)
    ax.bar(x, time_terms, bottom=stacked_bottom, label="time term", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylabel("Mean regret contribution")
    ax.set_title("Oversearch Regret Components by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _error_episode_decomposition_records(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> list[dict[str, Any]]:
    """One row per error episode (predicted != oracle stop), with clean-decomposition + model-advantage at oracle stop.

    Written to ``error_episode_decomposition.jsonl`` so downstream notebook
    analyses can slice/dice without re-running the analyzer.
    """
    records: list[dict[str, Any]] = []
    for episode in diagnostics:
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        if predicted_stop == oracle_stop:
            continue
        halt_rewards = [float(value) for value in episode["halt_rewards"]]
        tree_sizes = [int(value) for value in episode["tree_sizes"]]
        time_budgets = [int(value) for value in episode["time_budgets"]]
        predicted_advantages = [float(value) for value in episode["predicted_advantages"]]
        target_advantages = [float(value) for value in episode["target_advantages"]]
        clean = _best_future_stop_decomposition(
            halt_rewards,
            tree_sizes,
            time_budgets,
            oracle_stop,
            oracle_config,
        )
        current_maintenance_cost = maintenance_cost(tree_sizes[oracle_stop], oracle_config)
        current_time_cost = time_cost(time_budgets[oracle_stop], oracle_config)
        total_decomp = _episode_regret_decomposition(episode, oracle_config)
        error_type = "oversearch" if predicted_stop > oracle_stop else "undersearch"
        records.append(
            {
                "path": str(episode["path"]),
                "source_path": str(episode["source_path"]),
                "budget_bucket_name": str(episode["budget_bucket_name"]),
                "starting_budget": int(episode["starting_budget"]),
                "error_type": error_type,
                "oracle_stop_step": oracle_stop,
                "predicted_stop_step": predicted_stop,
                "stop_delta": predicted_stop - oracle_stop,
                "regret": float(episode["regret"]),
                "oracle_halt_reward": halt_rewards[oracle_stop],
                "best_future_stop_step": int(clean["best_future_stop_step"]),
                "cost_free_future_reward_gain": float(clean["cost_free_future_reward_gain"]),
                "downstream_future_cost": float(clean["downstream_future_cost"]),
                "current_step_maintenance_cost": float(current_maintenance_cost),
                "current_step_time_cost": float(current_time_cost),
                "current_step_cost": float(clean["current_step_cost"]),
                "target_advantage_at_oracle_stop": target_advantages[oracle_stop],
                "predicted_advantage_at_oracle_stop": predicted_advantages[oracle_stop],
                "prediction_gap_at_oracle_stop": predicted_advantages[oracle_stop] - target_advantages[oracle_stop],
                "would_model_continue_at_oracle_stop": predicted_advantages[oracle_stop] > 0.0,
                "oracle_stop_driver_clean": _oracle_stop_driver_clean(
                    float(clean["cost_free_future_reward_gain"]),
                    float(clean["future_net_gain_before_current_cost"]),
                ),
                # Carry the full-episode decomposition terms so notebooks can
                # reconstruct oversearch regret without re-running the analyzer.
                "oversearch_total_halt_reward_term": float(total_decomp["halt_reward_term"]) if error_type == "oversearch" else 0.0,
                "oversearch_total_maintenance_term": float(total_decomp["maintenance_term"]) if error_type == "oversearch" else 0.0,
                "oversearch_total_time_term": float(total_decomp["time_term"]) if error_type == "oversearch" else 0.0,
            }
        )
    return records


def _oversearch_tail_step_records(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> list[dict[str, Any]]:
    """One row per extra step taken in an oversearch episode, with cumulative cost and halt-reward erosion.

    Captures "what did the controller pay for each unnecessary expansion?".
    Used to plot the per-extra-step cost profile across oversearch episodes.
    """
    records: list[dict[str, Any]] = []
    for episode in diagnostics:
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        if predicted_stop <= oracle_stop:
            continue
        halt_rewards = [float(value) for value in episode["halt_rewards"]]
        tree_sizes = [int(value) for value in episode["tree_sizes"]]
        time_budgets = [int(value) for value in episode["time_budgets"]]
        cumulative_maintenance = 0.0
        cumulative_time = 0.0
        cumulative_halt_reward_deterioration = 0.0
        for step_index in range(oracle_stop, predicted_stop):
            maintenance_term = maintenance_cost(tree_sizes[step_index], oracle_config)
            time_term = time_cost(time_budgets[step_index], oracle_config)
            halt_reward_delta = halt_rewards[step_index + 1] - halt_rewards[step_index]
            halt_reward_deterioration = halt_rewards[step_index] - halt_rewards[step_index + 1]
            cumulative_maintenance += maintenance_term
            cumulative_time += time_term
            cumulative_halt_reward_deterioration += halt_reward_deterioration
            records.append(
                {
                    "path": str(episode["path"]),
                    "source_path": str(episode["source_path"]),
                    "budget_bucket_name": str(episode["budget_bucket_name"]),
                    "starting_budget": int(episode["starting_budget"]),
                    "oracle_stop_step": oracle_stop,
                    "predicted_stop_step": predicted_stop,
                    "stop_delta": predicted_stop - oracle_stop,
                    "relative_extra_step": step_index - oracle_stop + 1,
                    "absolute_step_index": step_index,
                    "maintenance_cost": float(maintenance_term),
                    "time_cost": float(time_term),
                    "halt_reward_before": halt_rewards[step_index],
                    "halt_reward_after": halt_rewards[step_index + 1],
                    "halt_reward_delta": float(halt_reward_delta),
                    "halt_reward_deterioration": float(halt_reward_deterioration),
                    "cumulative_maintenance_cost": float(cumulative_maintenance),
                    "cumulative_time_cost": float(cumulative_time),
                    "cumulative_halt_reward_deterioration": float(cumulative_halt_reward_deterioration),
                }
            )
    return records


def _summarize_error_episode_decomposition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate error-episode records by error type and by (bucket, error type)."""
    by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket_error: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        error_type = str(row["error_type"])
        bucket = str(row["budget_bucket_name"])
        by_error[error_type].append(row)
        by_bucket_error[bucket][error_type].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"episodes": 0}
        return {
            "episodes": len(rows),
            "mean_stop_delta": statistics.mean(float(row["stop_delta"]) for row in rows),
            "mean_regret": statistics.mean(float(row["regret"]) for row in rows),
            "mean_cost_free_future_reward_gain": statistics.mean(float(row["cost_free_future_reward_gain"]) for row in rows),
            "mean_downstream_future_cost": statistics.mean(float(row["downstream_future_cost"]) for row in rows),
            "mean_current_step_maintenance_cost": statistics.mean(float(row["current_step_maintenance_cost"]) for row in rows),
            "mean_current_step_time_cost": statistics.mean(float(row["current_step_time_cost"]) for row in rows),
            "mean_target_advantage_at_oracle_stop": statistics.mean(float(row["target_advantage_at_oracle_stop"]) for row in rows),
            "mean_predicted_advantage_at_oracle_stop": statistics.mean(float(row["predicted_advantage_at_oracle_stop"]) for row in rows),
            "would_model_continue_rate_at_oracle_stop": statistics.mean(1.0 if row["would_model_continue_at_oracle_stop"] else 0.0 for row in rows),
        }

    return {
        "by_error_type": {error_type: summarize(rows) for error_type, rows in sorted(by_error.items())},
        "by_bucket_error_type": {
            bucket: {error_type: summarize(rows) for error_type, rows in sorted(error_map.items())}
            for bucket, error_map in sorted(by_bucket_error.items())
        },
    }


def _summarize_oversearch_tail(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate oversearch-tail records by relative extra step (and per bucket)."""
    by_relative_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_bucket_relative_step: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        rel = int(row["relative_extra_step"])
        bucket = str(row["budget_bucket_name"])
        by_relative_step[rel].append(row)
        by_bucket_relative_step[bucket][rel].append(row)

    def summarize(groups: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for rel, rows in sorted(groups.items()):
            summary[str(rel)] = {
                "records": len(rows),
                "mean_maintenance_cost": statistics.mean(float(row["maintenance_cost"]) for row in rows),
                "mean_time_cost": statistics.mean(float(row["time_cost"]) for row in rows),
                "mean_halt_reward_delta": statistics.mean(float(row["halt_reward_delta"]) for row in rows),
                "mean_halt_reward_deterioration": statistics.mean(float(row["halt_reward_deterioration"]) for row in rows),
                "mean_cumulative_maintenance_cost": statistics.mean(float(row["cumulative_maintenance_cost"]) for row in rows),
                "mean_cumulative_time_cost": statistics.mean(float(row["cumulative_time_cost"]) for row in rows),
                "mean_cumulative_halt_reward_deterioration": statistics.mean(float(row["cumulative_halt_reward_deterioration"]) for row in rows),
            }
        return summary

    return {
        "overall": summarize(by_relative_step),
        "by_bucket": {bucket: summarize(groups) for bucket, groups in sorted(by_bucket_relative_step.items())},
    }


def _plot_error_episode_oracle_stop_factors(
    records: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """For oversearch episodes, plot clean-factor magnitudes at oracle stop alongside the model's predicted advantage."""
    oversearch_rows = [row for row in records if str(row["error_type"]) == "oversearch"]
    if not oversearch_rows:
        # Render a placeholder figure so downstream code that expects the
        # file to exist doesn't need an existence check.
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No oversearch error episodes", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
        plt.close(fig)
        return {}
    buckets = sorted({str(row["budget_bucket_name"]) for row in oversearch_rows})
    reward_gain = []
    downstream_cost = []
    maintenance_costs = []
    time_costs = []
    predicted_adv = []
    summary: dict[str, Any] = {}
    for bucket in buckets:
        rows = [row for row in oversearch_rows if str(row["budget_bucket_name"]) == bucket]
        reward_gain.append(statistics.mean(float(row["cost_free_future_reward_gain"]) for row in rows))
        downstream_cost.append(statistics.mean(float(row["downstream_future_cost"]) for row in rows))
        maintenance_costs.append(statistics.mean(float(row["current_step_maintenance_cost"]) for row in rows))
        time_costs.append(statistics.mean(float(row["current_step_time_cost"]) for row in rows))
        predicted_adv.append(statistics.mean(float(row["predicted_advantage_at_oracle_stop"]) for row in rows))
        summary[bucket] = {
            "mean_cost_free_future_reward_gain": reward_gain[-1],
            "mean_downstream_future_cost": downstream_cost[-1],
            "mean_current_step_maintenance_cost": maintenance_costs[-1],
            "mean_current_step_time_cost": time_costs[-1],
            "mean_predicted_advantage_at_oracle_stop": predicted_adv[-1],
            "episodes": len(rows),
        }

    x = np.arange(len(buckets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(x - 1.5 * width, reward_gain, width=width, label="cost-free future reward gain", color="tab:blue")
    ax.bar(x - 0.5 * width, downstream_cost, width=width, label="downstream future cost", color="tab:red")
    ax.bar(x + 0.5 * width, maintenance_costs, width=width, label="current maint cost", color="tab:orange")
    ax.bar(x + 1.5 * width, time_costs, width=width, label="current time cost", color="tab:green")
    ax.plot(x, predicted_adv, marker="o", color="black", linewidth=2, label="predicted advantage @ oracle stop")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylabel("Mean magnitude")
    ax.set_title("Oversearch Episodes: Oracle-Stop Factors and Predicted Advantage")
    ax.legend()
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary


def _plot_oversearch_tail_components_by_extra_step(
    records: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """Per-step maintenance, time, and halt-reward-deterioration curves indexed by extra step taken past the oracle."""
    if not records:
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "No oversearch tail steps", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
        plt.close(fig)
        return {}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["relative_extra_step"])].append(row)
    xs = sorted(grouped)
    maintenance_vals = [statistics.mean(float(row["maintenance_cost"]) for row in grouped[x]) for x in xs]
    time_vals = [statistics.mean(float(row["time_cost"]) for row in grouped[x]) for x in xs]
    churn_vals = [statistics.mean(float(row["halt_reward_deterioration"]) for row in grouped[x]) for x in xs]
    counts = [len(grouped[x]) for x in xs]
    summary = {
        str(x): {
            "records": counts[i],
            "mean_maintenance_cost": maintenance_vals[i],
            "mean_time_cost": time_vals[i],
            "mean_halt_reward_deterioration": churn_vals[i],
        }
        for i, x in enumerate(xs)
    }

    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax1.plot(xs, maintenance_vals, marker="o", label="maintenance cost")
    ax1.plot(xs, time_vals, marker="o", label="time cost")
    ax1.plot(xs, churn_vals, marker="o", label="halt-reward deterioration")
    ax1.set_xlabel("Relative extra oversearch step")
    ax1.set_ylabel("Mean per-step contribution")
    ax1.set_title("Oversearch Tail: Per-Step Cost and Halt-Reward Deterioration")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Twin axis with a faded count bar so the reader can read off support
    # at each relative step.
    ax2 = ax1.twinx()
    ax2.bar(xs, counts, color="tab:gray", alpha=0.15)
    ax2.set_ylabel("Record count")
    fig.savefig(out_path, dpi=ANALYSIS_FIGURE_DPI)
    plt.close(fig)
    return summary
