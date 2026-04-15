from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from budgeted_controller_oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    budgeted_oracle_config_from_metadata,
    maintenance_cost,
    return_for_stop_step,
    time_cost,
)
try:
    import torch
    from cts_episode_envs import build_trimmed_decision_episode
    from cts_pretrain import PretrainExample, TeacherSearchConfig
except ModuleNotFoundError:
    torch = None
    build_trimmed_decision_episode = None
    PretrainExample = None
    TeacherSearchConfig = None


TRAIN_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"train_advantage_mse=(?P<train_advantage_mse>-?\d+(?:\.\d+)?) "
    r"train_mean_abs_advantage_error=(?P<train_mean_abs_advantage_error>-?\d+(?:\.\d+)?) "
    r"train_sign_accuracy=(?P<train_sign_accuracy>-?\d+(?:\.\d+)?) "
    r"train_snapshots=(?P<train_snapshots>\d+)$"
)

VALIDATION_RE = re.compile(
    r"^validation_epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"validation_advantage_mse=(?P<validation_advantage_mse>-?\d+(?:\.\d+)?) "
    r"validation_mean_abs_advantage_error=(?P<validation_mean_abs_advantage_error>-?\d+(?:\.\d+)?) "
    r"validation_sign_accuracy=(?P<validation_sign_accuracy>-?\d+(?:\.\d+)?) "
    r"validation_snapshots=(?P<validation_snapshots>\d+)$"
)

GREEDY_RE = re.compile(
    r"^greedy_epoch=(?P<epoch>\d+)/(?P<total_epochs>\d+) "
    r"exact_stop_step_accuracy=(?P<exact_stop_step_accuracy>-?\d+(?:\.\d+)?) "
    r"first_action_accuracy=(?P<first_action_accuracy>-?\d+(?:\.\d+)?) "
    r"average_return=(?P<average_return>-?\d+(?:\.\d+)?) "
    r"average_oracle_value=(?P<average_oracle_value>-?\d+(?:\.\d+)?) "
    r"average_regret=(?P<average_regret>-?\d+(?:\.\d+)?) "
    r"average_expansions=(?P<average_expansions>-?\d+(?:\.\d+)?) "
    r"evaluated_episodes=(?P<evaluated_episodes>\d+)"
)


def _parse_number(value: str) -> int | float:
    return float(value) if "." in value or "e" in value.lower() else int(value)


def _analysis_quality_config(metadata: dict[str, Any]) -> TeacherSearchConfig:
    if TeacherSearchConfig is None:
        raise RuntimeError("TeacherSearchConfig is unavailable; install the CTS environment to reconstruct moves.")
    return TeacherSearchConfig(
        max_depth=int(metadata.get("max_depth", 10)),
        search_budget=int(metadata.get("search_budget", 64)),
        c_puct=float(metadata.get("c_puct", 1.0)),
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def parse_log(log_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata: dict[str, Any] = {}
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    greedy_rows: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("oracle_type") == "budgeted_controller_v1":
                metadata.update(payload)
                continue

        match = TRAIN_RE.match(line)
        if match is not None:
            train_rows.append({key: _parse_number(value) for key, value in match.groupdict().items()})
            continue
        match = VALIDATION_RE.match(line)
        if match is not None:
            validation_rows.append({key: _parse_number(value) for key, value in match.groupdict().items()})
            continue
        match = GREEDY_RE.match(line)
        if match is not None:
            greedy_rows.append({key: _parse_number(value) for key, value in match.groupdict().items()})
            continue
        if "=" in line and " " not in line:
            key, value = line.split("=", 1)
            if key and value:
                try:
                    metadata[key] = _parse_number(value)
                except ValueError:
                    metadata[key] = value
    return metadata, train_rows, validation_rows, greedy_rows


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_state_rows(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in diagnostics:
        for step_index, (tree_size, time_budget, pred_adv, target_adv) in enumerate(
            zip(
                episode["tree_sizes"],
                episode["time_budgets"],
                episode["predicted_advantages"],
                episode["target_advantages"],
            )
        ):
            pred_continue = float(pred_adv) > 0.0
            target_continue = float(target_adv) > 0.0
            rows.append(
                {
                    "source_path": episode["source_path"],
                    "path": episode["path"],
                    "budget_bucket_name": episode["budget_bucket_name"],
                    "starting_budget": int(episode["starting_budget"]),
                    "oracle_stop_step": int(episode["oracle_stop_step"]),
                    "predicted_stop_step": int(episode["predicted_stop_step"]),
                    "episode_regret": float(episode["regret"]),
                    "step_index": step_index,
                    "tree_size": int(tree_size),
                    "time_budget": int(time_budget),
                    "predicted_advantage": float(pred_adv),
                    "target_advantage": float(target_adv),
                    "pred_continue": pred_continue,
                    "target_continue": target_continue,
                    "sign_correct": pred_continue == target_continue,
                    "state_error_type": (
                        "false_continue"
                        if pred_continue and not target_continue
                        else "false_halt"
                        if (not pred_continue) and target_continue
                        else "correct_continue"
                        if pred_continue
                        else "correct_halt"
                    ),
                }
            )
    return rows


def _episode_error_type(episode: dict[str, Any]) -> str:
    pred = int(episode["predicted_stop_step"])
    oracle = int(episode["oracle_stop_step"])
    if pred == oracle:
        return "exact"
    if pred > oracle:
        return "oversearch"
    return "undersearch"


def _plot_loss_curves(train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(
        [row["epoch"] for row in train_rows],
        [row["train_advantage_mse"] for row in train_rows],
        marker="o",
        linewidth=2,
        label="train",
    )
    ax.plot(
        [row["epoch"] for row in validation_rows],
        [row["validation_advantage_mse"] for row in validation_rows],
        marker="o",
        linewidth=2,
        label="validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Advantage MSE")
    ax.set_title("Advantage Loss Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_greedy_metrics(greedy_rows: list[dict[str, Any]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = [row["epoch"] for row in greedy_rows]
    axes[0].plot(epochs, [row["average_return"] for row in greedy_rows], marker="o", label="return")
    axes[0].plot(epochs, [row["average_oracle_value"] for row in greedy_rows], marker="o", label="oracle")
    axes[0].plot(epochs, [row["average_regret"] for row in greedy_rows], marker="o", label="regret")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Greedy Returns and Regret")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["exact_stop_step_accuracy"] for row in greedy_rows], marker="o", label="exact stop")
    axes[1].plot(epochs, [row["first_action_accuracy"] for row in greedy_rows], marker="o", label="first action")
    axes[1].plot(epochs, [row["average_expansions"] for row in greedy_rows], marker="o", label="avg expansions")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Greedy Boundary Metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_episode_regret_by_bucket(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    buckets = sorted({episode["budget_bucket_name"] for episode in diagnostics})
    means = []
    stds = []
    counts = []
    for bucket in buckets:
        regrets = [float(episode["regret"]) for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        means.append(statistics.mean(regrets))
        stds.append(statistics.pstdev(regrets))
        counts.append(len(regrets))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(buckets, means, yerr=stds, color="tab:red", alpha=0.8)
    ax.set_ylabel("Episode regret")
    ax.set_title("Regret by Starting-Budget Bucket")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {bucket: {"mean_regret": mean, "std_regret": std, "episodes": count} for bucket, mean, std, count in zip(buckets, means, stds, counts)}


def _plot_stop_error_by_bucket(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    buckets = sorted({episode["budget_bucket_name"] for episode in diagnostics})
    error_types = ["exact", "undersearch", "oversearch"]
    fractions: dict[str, list[float]] = {error_type: [] for error_type in error_types}
    summary: dict[str, Any] = {}
    for bucket in buckets:
        bucket_eps = [episode for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        counts = Counter(_episode_error_type(episode) for episode in bucket_eps)
        summary[bucket] = {error_type: counts.get(error_type, 0) / len(bucket_eps) for error_type in error_types}
        for error_type in error_types:
            fractions[error_type].append(summary[bucket][error_type])

    x = np.arange(len(buckets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = {"exact": "tab:green", "undersearch": "tab:orange", "oversearch": "tab:red"}
    for idx, error_type in enumerate(error_types):
        ax.bar(x + (idx - 1) * width, fractions[error_type], width=width, label=error_type, color=colors[error_type])
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of episodes")
    ax.set_title("Predicted Stop vs Oracle Stop by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_regret_by_oracle_stop(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for episode in diagnostics:
        grouped[int(episode["oracle_stop_step"])].append(float(episode["regret"]))
    xs = sorted(grouped)
    means = [statistics.mean(grouped[x]) for x in xs]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(xs, means, marker="o")
    ax.set_xlabel("Oracle stop step")
    ax.set_ylabel("Mean regret")
    ax.set_title("Regret by Oracle Stop Step")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {str(x): {"mean_regret": means[i], "episodes": len(grouped[x])} for i, x in enumerate(xs)}


def _stop_step_delta_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[int, list[float]] = defaultdict(list)
    by_bucket: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    direction_by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for episode in diagnostics:
        delta = int(episode["predicted_stop_step"]) - int(episode["oracle_stop_step"])
        regret = float(episode["regret"])
        bucket = str(episode["budget_bucket_name"])
        overall[delta].append(regret)
        by_bucket[bucket][delta].append(regret)
        if delta == 0:
            direction = "exact"
        elif delta > 0:
            direction = "oversearch"
        else:
            direction = "undersearch"
        direction_by_bucket[bucket][direction].append(regret)

    overall_summary = {
        str(delta): {
            "episodes": len(regrets),
            "mean_regret": statistics.mean(regrets),
        }
        for delta, regrets in sorted(overall.items())
    }
    bucket_summary = {
        bucket: {
            str(delta): {
                "episodes": len(regrets),
                "mean_regret": statistics.mean(regrets),
            }
            for delta, regrets in sorted(delta_map.items())
        }
        for bucket, delta_map in by_bucket.items()
    }
    direction_summary = {
        bucket: {
            direction: {
                "episodes": len(regrets),
                "mean_regret": statistics.mean(regrets) if regrets else 0.0,
            }
            for direction, regrets in sorted(direction_map.items())
        }
        for bucket, direction_map in direction_by_bucket.items()
    }
    return {
        "overall": overall_summary,
        "by_bucket": bucket_summary,
        "direction_by_bucket": direction_summary,
    }


def _plot_regret_by_stop_step_delta(diagnostics: list[dict[str, Any]], out_path: Path) -> None:
    buckets = ["overall", "scramble", "medium-small", "medium-large", "large", "very-large"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes_list = list(axes.flat)

    for ax, bucket in zip(axes_list, buckets):
        selected = diagnostics if bucket == "overall" else [episode for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        grouped: dict[int, list[float]] = defaultdict(list)
        for episode in selected:
            delta = int(episode["predicted_stop_step"]) - int(episode["oracle_stop_step"])
            grouped[delta].append(float(episode["regret"]))
        deltas = sorted(grouped)
        means = [statistics.mean(grouped[delta]) for delta in deltas]
        counts = [len(grouped[delta]) for delta in deltas]
        ax.bar(deltas, means, color="tab:red", alpha=0.75)
        ax.set_title(bucket)
        ax.set_xlabel("Predicted stop - Oracle stop")
        ax.set_ylabel("Mean regret")
        ax.grid(True, axis="y", alpha=0.3)
        for x, y, count in zip(deltas, means, counts):
            if count >= 50:
                ax.text(x, y, str(count), ha="center", va="bottom", fontsize=7, rotation=90)

    fig.suptitle("Regret by Stop-Step Error")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _episode_regret_decomposition(
    episode: dict[str, Any],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, float | int | str]:
    oracle_stop = int(episode["oracle_stop_step"])
    predicted_stop = int(episode["predicted_stop_step"])
    halt_rewards = [float(value) for value in episode["halt_rewards"]]
    tree_sizes = [int(value) for value in episode["tree_sizes"]]
    time_budgets = [int(value) for value in episode["time_budgets"]]

    oracle_maintenance = sum(maintenance_cost(tree_sizes[idx], oracle_config) for idx in range(oracle_stop))
    predicted_maintenance = sum(maintenance_cost(tree_sizes[idx], oracle_config) for idx in range(predicted_stop))
    oracle_time = sum(time_cost(time_budgets[idx], oracle_config) for idx in range(oracle_stop))
    predicted_time = sum(time_cost(time_budgets[idx], oracle_config) for idx in range(predicted_stop))

    halt_reward_term = halt_rewards[oracle_stop] - halt_rewards[predicted_stop]
    maintenance_term = predicted_maintenance - oracle_maintenance
    time_term = predicted_time - oracle_time
    total = halt_reward_term + maintenance_term + time_term

    return {
        "delta": predicted_stop - oracle_stop,
        "budget_bucket_name": str(episode["budget_bucket_name"]),
        "halt_reward_term": float(halt_reward_term),
        "maintenance_term": float(maintenance_term),
        "time_term": float(time_term),
        "decomposed_regret": float(total),
        "saved_regret": float(episode["regret"]),
        "residual": float(total - float(episode["regret"])),
    }


def _regret_decomposition_summary(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, Any]:
    decomposed = [_episode_regret_decomposition(episode, oracle_config) for episode in diagnostics]

    overall: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    by_bucket: dict[str, dict[int, list[dict[str, float | int | str]]]] = defaultdict(lambda: defaultdict(list))
    for row in decomposed:
        delta = int(row["delta"])
        bucket = str(row["budget_bucket_name"])
        overall[delta].append(row)
        by_bucket[bucket][delta].append(row)

    def summarize(groups: dict[int, list[dict[str, float | int | str]]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for delta, rows in sorted(groups.items()):
            summary[str(delta)] = {
                "episodes": len(rows),
                "mean_regret": statistics.mean(float(row["saved_regret"]) for row in rows),
                "mean_halt_reward_term": statistics.mean(float(row["halt_reward_term"]) for row in rows),
                "mean_maintenance_term": statistics.mean(float(row["maintenance_term"]) for row in rows),
                "mean_time_term": statistics.mean(float(row["time_term"]) for row in rows),
                "mean_decomposed_regret": statistics.mean(float(row["decomposed_regret"]) for row in rows),
                "max_abs_residual": max(abs(float(row["residual"])) for row in rows),
            }
        return summary

    residuals = [abs(float(row["residual"])) for row in decomposed]
    return {
        "overall": summarize(overall),
        "by_bucket": {bucket: summarize(groups) for bucket, groups in sorted(by_bucket.items())},
        "max_abs_residual": max(residuals) if residuals else 0.0,
        "mean_abs_residual": statistics.mean(residuals) if residuals else 0.0,
    }


def _plot_regret_decomposition_by_delta(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
    out_path: Path,
) -> None:
    selected_buckets = ["overall", "medium-large", "large", "very-large"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes_list = list(axes.flat)

    for ax, bucket in zip(axes_list, selected_buckets):
        selected = diagnostics if bucket == "overall" else [episode for episode in diagnostics if episode["budget_bucket_name"] == bucket]
        grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
        for episode in selected:
            row = _episode_regret_decomposition(episode, oracle_config)
            grouped[int(row["delta"])].append(row)

        deltas = sorted(grouped)
        halt_terms = [statistics.mean(float(row["halt_reward_term"]) for row in grouped[delta]) for delta in deltas]
        maintenance_terms = [statistics.mean(float(row["maintenance_term"]) for row in grouped[delta]) for delta in deltas]
        time_terms = [statistics.mean(float(row["time_term"]) for row in grouped[delta]) for delta in deltas]
        regrets = [statistics.mean(float(row["saved_regret"]) for row in grouped[delta]) for delta in deltas]
        counts = [len(grouped[delta]) for delta in deltas]

        ax.plot(deltas, regrets, marker="o", linewidth=2, color="black", label="total regret")
        ax.plot(deltas, halt_terms, marker="o", linewidth=1.5, label="halt-reward term")
        ax.plot(deltas, maintenance_terms, marker="o", linewidth=1.5, label="maintenance term")
        ax.plot(deltas, time_terms, marker="o", linewidth=1.5, label="time term")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(bucket)
        ax.set_xlabel("Predicted stop - Oracle stop")
        ax.set_ylabel("Mean contribution")
        ax.grid(True, alpha=0.3)
        for x, y, count in zip(deltas, regrets, counts):
            if count >= 50:
                ax.text(x, y, str(count), ha="center", va="bottom", fontsize=7, rotation=90)

    axes_list[0].legend(loc="upper left")
    fig.suptitle("Regret Decomposition by Stop-Step Error")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _oracle_stop_driver(
    future_value_gain: float,
    maintenance_term: float,
    time_term: float,
) -> str:
    if future_value_gain <= 0.0:
        return "future_already_worse"
    if maintenance_term >= time_term:
        return "maintenance_dominated"
    return "time_dominated"


def _oracle_step_factor_rows(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> list[dict[str, Any]]:
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


def _reconstruct_episode_best_moves(
    source_path: str,
    quality_config: TeacherSearchConfig,
) -> list[str]:
    if torch is None or build_trimmed_decision_episode is None or PretrainExample is None:
        raise RuntimeError("Torch/CTS episode helpers are unavailable; cannot reconstruct best moves.")
    example = torch.load(source_path, weights_only=False)
    if not isinstance(example, PretrainExample):
        raise ValueError(f"Expected PretrainExample at {source_path}, got {type(example).__name__}.")
    episode = build_trimmed_decision_episode(example, quality_config)
    return list(episode.best_moves)


def _best_move_sequences(
    diagnostics: list[dict[str, Any]],
    quality_config: TeacherSearchConfig,
) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for source_path in sorted({str(episode["source_path"]) for episode in diagnostics}):
        cache[source_path] = _reconstruct_episode_best_moves(source_path, quality_config)
    return cache


def _future_worse_move_switch_summary(
    diagnostics: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    best_move_cache: dict[str, list[str]],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, Any]:
    stop_rows = {
        str(row["path"]): row
        for row in step_rows
        if bool(row["at_oracle_stop"]) and str(row["oracle_stop_driver"]) == "future_already_worse"
    }
    cases: list[dict[str, Any]] = []
    for episode in diagnostics:
        path = str(episode["path"])
        stop_row = stop_rows.get(path)
        if stop_row is None:
            continue
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        source_path = str(episode["source_path"])
        best_moves = best_move_cache[source_path]
        oracle_move = best_moves[oracle_stop]
        predicted_move = best_moves[predicted_stop]
        move_relation = "move_switch" if oracle_move != predicted_move else "same_move"
        decomposition = _episode_regret_decomposition(episode, oracle_config)
        cases.append(
            {
                "budget_bucket_name": str(episode["budget_bucket_name"]),
                "delta": predicted_stop - oracle_stop,
                "move_relation": move_relation,
                "oracle_move": oracle_move,
                "predicted_move": predicted_move,
                "halt_reward_term": float(decomposition["halt_reward_term"]),
                "maintenance_term": float(decomposition["maintenance_term"]),
                "time_term": float(decomposition["time_term"]),
                "regret": float(decomposition["saved_regret"]),
                "future_value_gain": float(stop_row["future_value_gain"]),
                "maintenance_cost": float(stop_row["maintenance_cost"]),
                "time_cost": float(stop_row["time_cost"]),
                "false_continue": bool(stop_row["false_continue"]),
            }
        )

    relation_counts = Counter(case["move_relation"] for case in cases)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_bucket[str(case["budget_bucket_name"])].append(case)
        by_relation[str(case["move_relation"])].append(case)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "mean_regret": statistics.mean(float(row["regret"]) for row in rows),
            "mean_halt_reward_term": statistics.mean(float(row["halt_reward_term"]) for row in rows),
            "mean_maintenance_term": statistics.mean(float(row["maintenance_term"]) for row in rows),
            "mean_time_term": statistics.mean(float(row["time_term"]) for row in rows),
            "mean_delta": statistics.mean(float(row["delta"]) for row in rows),
            "false_continue_rate": statistics.mean(1.0 if row["false_continue"] else 0.0 for row in rows),
        }

    by_bucket_relation: dict[str, dict[str, Any]] = {}
    for bucket, rows in sorted(by_bucket.items()):
        relation_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            relation_map[str(row["move_relation"])].append(row)
        by_bucket_relation[bucket] = {relation: summarize(rel_rows) for relation, rel_rows in sorted(relation_map.items())}

    return {
        "overall_relation_counts": dict(relation_counts),
        "overall": summarize(cases) if cases else {},
        "by_relation": {relation: summarize(rows) for relation, rows in sorted(by_relation.items())},
        "by_bucket_relation": by_bucket_relation,
    }


def _plot_future_worse_move_switch(
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    bucket_summary = summary["by_bucket_relation"]
    buckets = list(bucket_summary)
    same_move = [bucket_summary[bucket].get("same_move", {}).get("episodes", 0) for bucket in buckets]
    move_switch = [bucket_summary[bucket].get("move_switch", {}).get("episodes", 0) for bucket in buckets]
    totals = np.maximum(np.array(same_move, dtype=np.float32) + np.array(move_switch, dtype=np.float32), 1.0)
    same_frac = np.array(same_move, dtype=np.float32) / totals
    switch_frac = np.array(move_switch, dtype=np.float32) / totals

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(buckets))
    axes[0].bar(x, same_frac, label="same_move", color="tab:blue")
    axes[0].bar(x, switch_frac, bottom=same_frac, label="move_switch", color="tab:red")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(buckets, rotation=25)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Future-Worse Episodes: Same Move vs Move Switch")
    axes[0].legend()

    same_regret = [bucket_summary[bucket].get("same_move", {}).get("mean_regret", 0.0) for bucket in buckets]
    switch_regret = [bucket_summary[bucket].get("move_switch", {}).get("mean_regret", 0.0) for bucket in buckets]
    width = 0.35
    axes[1].bar(x - width / 2, same_regret, width=width, label="same_move", color="tab:blue")
    axes[1].bar(x + width / 2, switch_regret, width=width, label="move_switch", color="tab:red")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(buckets, rotation=25)
    axes[1].set_ylabel("Mean regret")
    axes[1].set_title("Future-Worse Regret by Move Relation")
    axes[1].legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _oracle_stop_factor_summary(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
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

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_oracle_stop_factor_magnitudes(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_false_continue_by_oracle_driver(step_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
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
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _oversearch_regret_component_summary(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
) -> dict[str, Any]:
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
    summary = _oversearch_regret_component_summary(diagnostics, oracle_config)
    bucket_summary = summary["by_bucket"]
    buckets = list(bucket_summary)
    halt_terms = [bucket_summary[bucket]["mean_halt_reward_term"] for bucket in buckets]
    maintenance_terms = [bucket_summary[bucket]["mean_maintenance_term"] for bucket in buckets]
    time_terms = [bucket_summary[bucket]["mean_time_term"] for bucket in buckets]

    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(x, halt_terms, label="halt-reward term", color="tab:blue")
    ax.bar(x, maintenance_terms, bottom=halt_terms, label="maintenance term", color="tab:red")
    stacked_bottom = np.array(halt_terms) + np.array(maintenance_terms)
    ax.bar(x, time_terms, bottom=stacked_bottom, label="time term", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, rotation=25)
    ax.set_ylabel("Mean regret contribution")
    ax.set_title("Oversearch Regret Components by Budget Bucket")
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_sign_accuracy_by_time(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in state_rows:
        grouped[int(row["time_budget"])].append(bool(row["sign_correct"]))
    xs = sorted(grouped)
    accs = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    counts = [len(grouped[x]) for x in xs]
    fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax1.plot(xs, accs, marker="o", color="tab:blue")
    ax1.set_xlabel("Current T_t")
    ax1.set_ylabel("Sign accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(xs, counts, color="tab:gray", alpha=0.2)
    ax2.set_ylabel("State count", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")
    ax1.set_title("Sign Accuracy by Current T_t")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {str(x): {"sign_accuracy": accs[i], "states": counts[i]} for i, x in enumerate(xs)}


def _plot_target_distribution_by_time(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in state_rows:
        grouped[int(row["time_budget"])].append(float(row["target_advantage"]))
    xs = sorted(grouped)
    box_values = [grouped[x] for x in xs]
    means = [statistics.mean(values) for values in box_values]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.boxplot(box_values, positions=xs, widths=0.6, showfliers=False)
    ax.plot(xs, means, marker="o", color="tab:red", label="mean")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Current T_t")
    ax.set_ylabel("Target advantage")
    ax.set_title("Target Advantage Distribution by Current T_t")
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        str(x): {
            "mean_target_advantage": means[i],
            "median_target_advantage": statistics.median(box_values[i]),
            "states": len(box_values[i]),
        }
        for i, x in enumerate(xs)
    }


def _plot_predicted_vs_target(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    buckets = ["overall"] + sorted({row["budget_bucket_name"] for row in state_rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes_list = list(axes.flat)
    summary: dict[str, Any] = {}
    for ax, bucket in zip(axes_list, buckets):
        selected = state_rows if bucket == "overall" else [row for row in state_rows if row["budget_bucket_name"] == bucket]
        xs = np.array([row["target_advantage"] for row in selected], dtype=np.float32)
        ys = np.array([row["predicted_advantage"] for row in selected], dtype=np.float32)
        if xs.size == 0:
            ax.set_visible(False)
            continue
        limit = float(np.percentile(np.abs(np.concatenate([xs, ys])), 99))
        limit = max(limit, 0.5)
        sample_size = min(4000, xs.size)
        if xs.size > sample_size:
            idx = np.linspace(0, xs.size - 1, sample_size, dtype=int)
            xs = xs[idx]
            ys = ys[idx]
        ax.scatter(xs, ys, s=5, alpha=0.2)
        ax.plot([-limit, limit], [-limit, limit], color="tab:red", linewidth=1)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(bucket)
        corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 1 else float("nan")
        summary[bucket] = {"sampled_points": int(xs.size), "correlation": corr}
    for ax in axes_list[len(buckets):]:
        ax.set_visible(False)
    fig.suptitle("Predicted vs Target Advantage")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_false_action_rates_by_time(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for row in state_rows:
        grouped[int(row["time_budget"])][str(row["state_error_type"])] += 1
    xs = sorted(grouped)
    false_continue = []
    false_halt = []
    summary: dict[str, Any] = {}
    for x in xs:
        total = sum(grouped[x].values())
        fc = grouped[x]["false_continue"] / total
        fh = grouped[x]["false_halt"] / total
        false_continue.append(fc)
        false_halt.append(fh)
        summary[str(x)] = {
            "false_continue_rate": fc,
            "false_halt_rate": fh,
            "states": total,
        }
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(xs, false_continue, marker="o", label="false continue")
    ax.plot(xs, false_halt, marker="o", label="false halt")
    ax.set_xlabel("Current T_t")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("False Continue / False Halt Rates by T_t")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_regret_by_tree_size(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    episode_rows = []
    for episode in diagnostics:
        first_tree_size = int(episode["tree_sizes"][0])
        episode_rows.append((first_tree_size, float(episode["regret"])))
    sizes = np.array([row[0] for row in episode_rows], dtype=np.int64)
    regrets = np.array([row[1] for row in episode_rows], dtype=np.float32)
    quantiles = np.unique(np.percentile(sizes, [0, 20, 40, 60, 80, 100]).astype(int))
    if len(quantiles) < 2:
        quantiles = np.array([sizes.min(), sizes.max()])
    labels = []
    means = []
    summary: dict[str, Any] = {}
    for lower, upper in zip(quantiles[:-1], quantiles[1:]):
        if upper <= lower:
            continue
        mask = (sizes >= lower) & (sizes <= upper if upper == quantiles[-1] else sizes < upper)
        bucket_regrets = regrets[mask]
        if bucket_regrets.size == 0:
            continue
        label = f"[{int(lower)}, {int(upper)}{'+' if upper == quantiles[-1] else ')'}"
        labels.append(label)
        means.append(float(bucket_regrets.mean()))
        summary[label] = {"mean_regret": float(bucket_regrets.mean()), "episodes": int(bucket_regrets.size)}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(labels, means, color="tab:purple")
    ax.set_ylabel("Mean regret")
    ax.set_title("Episode Regret by Initial Tree Size")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _plot_partial_dependence_heatmaps(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    time_bins = [1, 2, 3, 5, 10, 20, 40, 80, math.inf]
    tree_sizes = np.array([row["tree_size"] for row in state_rows], dtype=np.float32)
    tree_edges = np.unique(np.percentile(tree_sizes, [0, 25, 50, 75, 100]).astype(int))
    if len(tree_edges) < 2:
        tree_edges = np.array([int(tree_sizes.min()), int(tree_sizes.max())])
    pred_grid = np.full((len(tree_edges) - 1, len(time_bins) - 1), np.nan)
    acc_grid = np.full((len(tree_edges) - 1, len(time_bins) - 1), np.nan)
    count_grid = np.zeros((len(tree_edges) - 1, len(time_bins) - 1), dtype=int)

    for i, (n_lo, n_hi) in enumerate(zip(tree_edges[:-1], tree_edges[1:])):
        for j, (t_lo, t_hi) in enumerate(zip(time_bins[:-1], time_bins[1:])):
            selected = [
                row
                for row in state_rows
                if n_lo <= row["tree_size"] <= n_hi
                and t_lo <= row["time_budget"] < t_hi
            ]
            if not selected:
                continue
            pred_grid[i, j] = statistics.mean(row["predicted_advantage"] for row in selected)
            acc_grid[i, j] = statistics.mean(1.0 if row["sign_correct"] else 0.0 for row in selected)
            count_grid[i, j] = len(selected)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(pred_grid, aspect="auto", cmap="coolwarm")
    axes[0].set_title("Mean Predicted Advantage")
    im1 = axes[1].imshow(acc_grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Sign Accuracy")
    xlabels = [f"[{time_bins[j]}, {time_bins[j + 1]})" if math.isfinite(time_bins[j + 1]) else f"[{time_bins[j]}, inf)" for j in range(len(time_bins) - 1)]
    ylabels = [f"[{int(tree_edges[i])}, {int(tree_edges[i + 1])}]" for i in range(len(tree_edges) - 1)]
    for ax in axes:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("T_t bin")
        ax.set_ylabel("N_t bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "time_bins": xlabels,
        "tree_bins": ylabels,
        "mean_predicted_advantage": pred_grid.tolist(),
        "sign_accuracy": acc_grid.tolist(),
        "counts": count_grid.tolist(),
    }


def _same_tree_budget_consistency(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in diagnostics:
        groups[episode["source_path"]].append(episode)
    total_groups = 0
    oracle_monotone = 0
    predicted_monotone = 0
    same_tree_examples: list[dict[str, Any]] = []
    for source_path, episodes in groups.items():
        if len(episodes) < 2:
            continue
        total_groups += 1
        ordered = sorted(episodes, key=lambda episode: (int(episode["starting_budget"]), episode["budget_bucket_name"]))
        oracle_stops = [int(episode["oracle_stop_step"]) for episode in ordered]
        predicted_stops = [int(episode["predicted_stop_step"]) for episode in ordered]
        if all(b >= a for a, b in zip(oracle_stops, oracle_stops[1:])):
            oracle_monotone += 1
        if all(b >= a for a, b in zip(predicted_stops, predicted_stops[1:])):
            predicted_monotone += 1
        if len(same_tree_examples) < 10:
            same_tree_examples.append(
                {
                    "source_path": source_path,
                    "starting_budgets": [int(episode["starting_budget"]) for episode in ordered],
                    "oracle_stops": oracle_stops,
                    "predicted_stops": predicted_stops,
                }
            )
    return {
        "num_multibudget_source_paths": total_groups,
        "oracle_stop_monotone_fraction": oracle_monotone / max(total_groups, 1),
        "predicted_stop_monotone_fraction": predicted_monotone / max(total_groups, 1),
        "examples": same_tree_examples,
    }


def _oracle_stop_distribution(diagnostics: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(int(episode["oracle_stop_step"]) for episode in diagnostics)
    return {str(step): counts[step] for step in sorted(counts)}


def _calibration_by_margin(state_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    edges = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, math.inf]
    labels = []
    accs = []
    counts = []
    mean_targets = []
    summary: dict[str, Any] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [row for row in state_rows if lo <= abs(row["predicted_advantage"]) < hi]
        label = f"[{lo:.2f}, {hi:.2f})" if math.isfinite(hi) else f"[{lo:.2f}, inf)"
        labels.append(label)
        if not selected:
            accs.append(float("nan"))
            counts.append(0)
            mean_targets.append(float("nan"))
            continue
        acc = statistics.mean(1.0 if row["sign_correct"] else 0.0 for row in selected)
        mean_target = statistics.mean(abs(row["target_advantage"]) for row in selected)
        accs.append(acc)
        counts.append(len(selected))
        mean_targets.append(mean_target)
        summary[label] = {"sign_accuracy": acc, "mean_abs_target_advantage": mean_target, "states": len(selected)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(labels, accs, color="tab:blue")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Sign accuracy")
    axes[0].set_title("Calibration by |Predicted Advantage|")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(labels, counts, color="tab:gray")
    axes[1].set_ylabel("State count")
    axes[1].set_title("States per Calibration Bin")
    axes[1].tick_params(axis="x", rotation=35)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _best_epoch_summary(validation_rows: list[dict[str, Any]], greedy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_val = min(validation_rows, key=lambda row: row["validation_advantage_mse"])
    best_return = max(greedy_rows, key=lambda row: row["average_return"])
    best_regret = min(greedy_rows, key=lambda row: row["average_regret"])
    return {
        "best_validation_mse_epoch": best_val["epoch"],
        "best_validation_mse": best_val["validation_advantage_mse"],
        "best_greedy_return_epoch": best_return["epoch"],
        "best_greedy_return": best_return["average_return"],
        "best_greedy_regret_epoch": best_regret["epoch"],
        "best_greedy_regret": best_regret["average_regret"],
    }


def _evaluate_baseline(
    diagnostics: list[dict[str, Any]],
    oracle_config: BudgetedOracleConfig,
    stop_rule: Callable[[dict[str, Any]], int],
) -> dict[str, Any]:
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
    tree_sizes = sorted({int(size) for episode in diagnostics for size in episode["tree_sizes"]})
    size_thresholds = sorted({tree_sizes[int(round(idx * (len(tree_sizes) - 1)))] for idx in [0.1, 0.25, 0.5, 0.75, 0.9]})
    time_thresholds = sorted({1, 2, 3, 4, 5, 7, 10, 15, 20})

    baselines: dict[str, dict[str, Any]] = {}
    baselines["always_halt_0"] = _evaluate_baseline(diagnostics, oracle_config, lambda episode: 0)
    baselines["always_continue_to_end"] = _evaluate_baseline(
        diagnostics,
        oracle_config,
        lambda episode: len(episode["halt_rewards"]) - 1,
    )

    for threshold in time_thresholds:
        baselines[f"halt_when_T_le_{threshold}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, threshold=threshold: next(
                (idx for idx, time_budget in enumerate(episode["time_budgets"]) if int(time_budget) <= threshold),
                len(episode["halt_rewards"]) - 1,
            ),
        )
    for threshold in size_thresholds:
        baselines[f"halt_when_N_ge_{threshold}"] = _evaluate_baseline(
            diagnostics,
            oracle_config,
            lambda episode, threshold=threshold: next(
                (idx for idx, size in enumerate(episode["tree_sizes"]) if int(size) >= threshold),
                len(episode["halt_rewards"]) - 1,
            ),
        )
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


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    large_plus_six = summary["regret_decomposition"]["by_bucket"].get("large", {}).get("6")
    large_plus_six_line = (
        f"- Large-bucket `+6` episodes: regret={large_plus_six['mean_regret']:.4f}, "
        f"halt={large_plus_six['mean_halt_reward_term']:.4f}, "
        f"maint={large_plus_six['mean_maintenance_term']:.4f}, "
        f"time={large_plus_six['mean_time_term']:.4f}"
        if large_plus_six is not None
        else "- Large-bucket `+6` episodes: unavailable"
    )
    future_worse_counts = summary["future_worse_move_switch"]["overall_relation_counts"]
    future_worse_same = future_worse_counts.get("same_move", 0)
    future_worse_switch = future_worse_counts.get("move_switch", 0)
    lines = [
        "# Budgeted Controller Analysis",
        "",
        "## Run Summary",
        f"- Episodes: {summary['episodes']}",
        f"- States: {summary['states']}",
        f"- Final greedy regret: {summary['final_greedy']['average_regret']:.4f}",
        f"- Final greedy return: {summary['final_greedy']['average_return']:.4f}",
        f"- Final greedy expansions: {summary['final_greedy']['average_expansions']:.3f}",
        "",
        "## Best Epochs",
        f"- Best validation MSE epoch: {summary['best_epochs']['best_validation_mse_epoch']}",
        f"- Best greedy return epoch: {summary['best_epochs']['best_greedy_return_epoch']}",
        f"- Best greedy regret epoch: {summary['best_epochs']['best_greedy_regret_epoch']}",
        "",
        "## Best Baseline",
        f"- Best baseline by regret: `{summary['baselines']['best_by_regret']['name']}`",
        f"- Baseline regret: {summary['baselines']['best_by_regret']['average_regret']:.4f}",
        f"- Baseline return: {summary['baselines']['best_by_regret']['average_return']:.4f}",
        "",
        "## Oracle Decision Factors",
        "- At each step, oracle advantage is decomposed as:",
        "  `target_advantage = future_value_gain - maintenance_cost - time_cost`",
        "  where `future_value_gain = oracle_next_value - halt_reward_now`.",
        f"- Oracle-stop residual check: {summary['oracle_stop_factors']['max_abs_target_residual']:.6g}",
        f"- Oracle-stop overall means: future_gain={summary['oracle_stop_factors']['overall']['mean_future_value_gain']:.4f}, "
        f"maint={summary['oracle_stop_factors']['overall']['mean_maintenance_cost']:.4f}, "
        f"time={summary['oracle_stop_factors']['overall']['mean_time_cost']:.4f}, "
        f"target_adv={summary['oracle_stop_factors']['overall']['mean_target_advantage']:.4f}",
        f"- `future_already_worse` split: same_move={future_worse_same}, move_switch={future_worse_switch}",
        "",
        "## Regret Decomposition",
        "- Regret is decomposed as:",
        "  `oracle_value - predicted_value = (halt_reward@oracle - halt_reward@predicted) + (predicted maintenance paid - oracle maintenance paid) + (predicted time cost paid - oracle time cost paid)`",
        f"- Max absolute decomposition residual: {summary['regret_decomposition']['max_abs_residual']:.6g}",
        large_plus_six_line,
        "",
        "## Same-Tree Different-Budget Consistency",
        f"- Source paths with multiple budgets: {summary['consistency']['num_multibudget_source_paths']}",
        f"- Oracle monotone fraction: {summary['consistency']['oracle_stop_monotone_fraction']:.3f}",
        f"- Predicted monotone fraction: {summary['consistency']['predicted_stop_monotone_fraction']:.3f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a budgeted controller diagnostics JSONL and training .out log.")
    parser.add_argument("--diagnostics-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    diagnostics_path = Path(args.diagnostics_path)
    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, train_rows, validation_rows, greedy_rows = parse_log(log_path)
    diagnostics = load_diagnostics(diagnostics_path)
    state_rows = build_state_rows(diagnostics)
    oracle_config = budgeted_oracle_config_from_metadata(metadata)
    if oracle_config is None:
        raise ValueError("Could not recover budgeted oracle metadata from the .out file.")
    oracle_step_rows = _oracle_step_factor_rows(diagnostics, oracle_config)
    quality_config = _analysis_quality_config(metadata)
    best_move_cache = _best_move_sequences(diagnostics, quality_config)

    _plot_loss_curves(train_rows, validation_rows, output_dir / "advantage_loss.png")
    _plot_greedy_metrics(greedy_rows, output_dir / "greedy_metrics.png")
    regret_by_bucket = _plot_episode_regret_by_bucket(diagnostics, output_dir / "regret_by_budget_bucket.png")
    stop_by_bucket = _plot_stop_error_by_bucket(diagnostics, output_dir / "stop_error_by_budget_bucket.png")
    regret_by_oracle_stop = _plot_regret_by_oracle_stop(diagnostics, output_dir / "regret_by_oracle_stop.png")
    stop_step_delta = _stop_step_delta_summary(diagnostics)
    _plot_regret_by_stop_step_delta(diagnostics, output_dir / "regret_by_stop_step_delta.png")
    regret_decomposition = _regret_decomposition_summary(diagnostics, oracle_config)
    _plot_regret_decomposition_by_delta(diagnostics, oracle_config, output_dir / "regret_decomposition_by_stop_step_delta.png")
    oracle_stop_factors = _oracle_stop_factor_summary(oracle_step_rows)
    oracle_stop_driver_by_bucket = _plot_oracle_stop_driver_by_bucket(
        oracle_step_rows,
        output_dir / "oracle_stop_driver_by_bucket.png",
    )
    oracle_stop_factor_magnitudes = _plot_oracle_stop_factor_magnitudes(
        oracle_step_rows,
        output_dir / "oracle_stop_factor_magnitudes.png",
    )
    false_continue_by_oracle_driver = _plot_false_continue_by_oracle_driver(
        oracle_step_rows,
        output_dir / "false_continue_by_oracle_driver.png",
    )
    oversearch_regret_components = _plot_oversearch_regret_components_by_bucket(
        diagnostics,
        oracle_config,
        output_dir / "oversearch_regret_components_by_bucket.png",
    )
    future_worse_move_switch = _future_worse_move_switch_summary(
        diagnostics,
        oracle_step_rows,
        best_move_cache,
        oracle_config,
    )
    _plot_future_worse_move_switch(
        future_worse_move_switch,
        output_dir / "future_worse_move_switch.png",
    )
    sign_by_time = _plot_sign_accuracy_by_time(state_rows, output_dir / "sign_accuracy_by_time_budget.png")
    target_by_time = _plot_target_distribution_by_time(state_rows, output_dir / "target_advantage_by_time_budget.png")
    pred_vs_target = _plot_predicted_vs_target(state_rows, output_dir / "predicted_vs_target_advantage.png")
    false_action_by_time = _plot_false_action_rates_by_time(state_rows, output_dir / "false_action_rates_by_time_budget.png")
    regret_by_tree_size = _plot_regret_by_tree_size(diagnostics, output_dir / "regret_by_initial_tree_size.png")
    partial_dependence = _plot_partial_dependence_heatmaps(state_rows, output_dir / "partial_dependence_heatmaps.png")
    consistency = _same_tree_budget_consistency(diagnostics)
    oracle_stop_dist = _oracle_stop_distribution(diagnostics)
    calibration = _calibration_by_margin(state_rows, output_dir / "calibration_by_margin.png")
    best_epochs = _best_epoch_summary(validation_rows, greedy_rows)
    baselines = _baseline_sweep(diagnostics, oracle_config)

    episode_error_counts = Counter(_episode_error_type(episode) for episode in diagnostics)
    episode_error_regret = {
        error_type: statistics.mean(float(episode["regret"]) for episode in diagnostics if _episode_error_type(episode) == error_type)
        for error_type in sorted(episode_error_counts)
    }
    bucket_counts = Counter(episode["budget_bucket_name"] for episode in diagnostics)

    summary = {
        "episodes": len(diagnostics),
        "states": len(state_rows),
        "bucket_counts": dict(bucket_counts),
        "final_greedy": greedy_rows[-1] if greedy_rows else None,
        "best_epochs": best_epochs,
        "oracle_metadata": metadata,
        "oracle_stop_distribution": oracle_stop_dist,
        "regret_by_budget_bucket": regret_by_bucket,
        "stop_error_by_budget_bucket": stop_by_bucket,
        "regret_by_oracle_stop_step": regret_by_oracle_stop,
        "regret_by_stop_step_delta": stop_step_delta,
        "regret_decomposition": regret_decomposition,
        "oracle_stop_factors": oracle_stop_factors,
        "oracle_stop_driver_by_bucket": oracle_stop_driver_by_bucket,
        "oracle_stop_factor_magnitudes": oracle_stop_factor_magnitudes,
        "false_continue_by_oracle_driver": false_continue_by_oracle_driver,
        "oversearch_regret_components": oversearch_regret_components,
        "future_worse_move_switch": future_worse_move_switch,
        "sign_accuracy_by_time_budget": sign_by_time,
        "target_advantage_by_time_budget": target_by_time,
        "predicted_vs_target": pred_vs_target,
        "false_action_rates_by_time_budget": false_action_by_time,
        "regret_by_initial_tree_size": regret_by_tree_size,
        "partial_dependence": partial_dependence,
        "consistency": consistency,
        "calibration_by_margin": calibration,
        "episode_error_counts": dict(episode_error_counts),
        "episode_error_regret": episode_error_regret,
        "baselines": baselines,
    }
    _write_summary(output_dir / "summary.json", summary)
    _write_markdown_report(output_dir / "report.md", summary)
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
