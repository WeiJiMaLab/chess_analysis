from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from budgeted_controller_oracle import (
    budgeted_oracle_config_from_metadata,
    compute_budgeted_oracle,
    maintenance_cost,
    time_cost,
)
from cts_episode_envs import build_trimmed_decision_episode
from cts_pretrain import PretrainExample, TeacherSearchConfig, load_pretrain_example


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


def parse_log(log_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
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
        if GREEDY_RE.match(line):
            continue
        if "=" in line and " " not in line:
            key, value = line.split("=", 1)
            if key and value:
                try:
                    metadata[key] = _parse_number(value)
                except ValueError:
                    metadata[key] = value
    return metadata


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analysis_quality_config(metadata: dict[str, Any]) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=int(metadata.get("max_depth", 10)),
        search_budget=int(metadata.get("search_budget", 64)),
        c_puct=float(metadata.get("c_puct", 1.0)),
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def reconstruct_best_moves(source_path: str, quality_config: TeacherSearchConfig) -> list[str]:
    example = load_pretrain_example(source_path)
    if not isinstance(example, PretrainExample):
        raise ValueError(f"Expected PretrainExample at {source_path}, got {type(example).__name__}.")
    episode = build_trimmed_decision_episode(example, quality_config)
    return list(episode.best_moves)


def best_move_cache(diagnostics: list[dict[str, Any]], quality_config: TeacherSearchConfig) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for source_path in sorted({str(episode["source_path"]) for episode in diagnostics}):
        cache[source_path] = reconstruct_best_moves(source_path, quality_config)
    return cache


def episode_regret_decomposition(episode: dict[str, Any], oracle_config: Any) -> dict[str, float]:
    oracle_stop = int(episode["oracle_stop_step"])
    predicted_stop = int(episode["predicted_stop_step"])
    halt_rewards = [float(value) for value in episode["halt_rewards"]]
    tree_sizes = [int(value) for value in episode["tree_sizes"]]
    time_budgets = [int(value) for value in episode["time_budgets"]]

    oracle_maintenance = sum(maintenance_cost(tree_sizes[idx], oracle_config) for idx in range(oracle_stop))
    predicted_maintenance = sum(maintenance_cost(tree_sizes[idx], oracle_config) for idx in range(predicted_stop))
    oracle_time = sum(time_cost(time_budgets[idx], oracle_config) for idx in range(oracle_stop))
    predicted_time = sum(time_cost(time_budgets[idx], oracle_config) for idx in range(predicted_stop))

    return {
        "halt_reward_term": halt_rewards[oracle_stop] - halt_rewards[predicted_stop],
        "maintenance_term": predicted_maintenance - oracle_maintenance,
        "time_term": predicted_time - oracle_time,
        "regret": float(episode["regret"]),
    }


def future_worse_move_switch_summary(
    diagnostics: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    oracle_config = budgeted_oracle_config_from_metadata(metadata)
    if oracle_config is None:
        raise ValueError("Could not recover budgeted oracle metadata from log.")
    quality_config = analysis_quality_config(metadata)
    move_cache = best_move_cache(diagnostics, quality_config)

    cases: list[dict[str, Any]] = []
    for episode in diagnostics:
        policy = compute_budgeted_oracle(
            episode["halt_rewards"],
            episode["tree_sizes"],
            int(episode["starting_budget"]),
            oracle_config,
        )
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        if oracle_stop >= len(policy.values):
            continue
        halt_reward = float(episode["halt_rewards"][oracle_stop])
        time_budget = int(episode["time_budgets"][oracle_stop])
        tree_size = int(episode["tree_sizes"][oracle_stop])
        if time_budget <= oracle_config.time_delta:
            next_value = float(oracle_config.timeout_value)
        elif oracle_stop + 1 < len(policy.values):
            next_value = float(policy.values[oracle_stop + 1])
        else:
            next_value = halt_reward
        future_value_gain = next_value - halt_reward
        if future_value_gain > 0.0:
            continue

        moves = move_cache[str(episode["source_path"])]
        oracle_move = moves[oracle_stop]
        predicted_move = moves[predicted_stop]
        move_relation = "move_switch" if oracle_move != predicted_move else "same_move"
        decomp = episode_regret_decomposition(episode, oracle_config)
        cases.append(
            {
                "budget_bucket_name": str(episode["budget_bucket_name"]),
                "delta": predicted_stop - oracle_stop,
                "move_relation": move_relation,
                "oracle_move": oracle_move,
                "predicted_move": predicted_move,
                "future_value_gain": future_value_gain,
                "maintenance_cost": maintenance_cost(tree_size, oracle_config),
                "time_cost": time_cost(time_budget, oracle_config),
                **decomp,
            }
        )

    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket_relation: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        relation = str(case["move_relation"])
        bucket = str(case["budget_bucket_name"])
        by_relation[relation].append(case)
        by_bucket_relation[bucket][relation].append(case)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(rows),
            "mean_delta": statistics.mean(float(row["delta"]) for row in rows),
            "mean_regret": statistics.mean(float(row["regret"]) for row in rows),
            "mean_halt_reward_term": statistics.mean(float(row["halt_reward_term"]) for row in rows),
            "mean_maintenance_term": statistics.mean(float(row["maintenance_term"]) for row in rows),
            "mean_time_term": statistics.mean(float(row["time_term"]) for row in rows),
            "mean_future_value_gain": statistics.mean(float(row["future_value_gain"]) for row in rows),
        }

    return {
        "overall_relation_counts": dict(Counter(case["move_relation"] for case in cases)),
        "by_relation": {relation: summarize(rows) for relation, rows in sorted(by_relation.items())},
        "by_bucket_relation": {
            bucket: {relation: summarize(rows) for relation, rows in sorted(relation_map.items())}
            for bucket, relation_map in sorted(by_bucket_relation.items())
        },
        "examples": cases[:20],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary["overall_relation_counts"]
    lines = [
        "# Future-Worse Move-Switch Analysis",
        "",
        "This analysis only considers episodes where the oracle stop step has `future_value_gain <= 0`,",
        "i.e. the continuation path is already worse in value terms before maintenance/time costs are applied.",
        "",
        f"- same_move episodes: {counts.get('same_move', 0)}",
        f"- move_switch episodes: {counts.get('move_switch', 0)}",
        "",
        "## By Relation",
    ]
    for relation, metrics in summary["by_relation"].items():
        lines.extend(
            [
                f"### {relation}",
                f"- episodes: {metrics['episodes']}",
                f"- mean delta: {metrics['mean_delta']:.3f}",
                f"- mean regret: {metrics['mean_regret']:.4f}",
                f"- mean halt-reward term: {metrics['mean_halt_reward_term']:.4f}",
                f"- mean maintenance term: {metrics['mean_maintenance_term']:.4f}",
                f"- mean time term: {metrics['mean_time_term']:.4f}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether future-worse oracle stops correspond to same-move or move-switch failures.")
    parser.add_argument("--diagnostics-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    diagnostics = load_diagnostics(Path(args.diagnostics_path))
    metadata = parse_log(Path(args.log_path))
    summary = future_worse_move_switch_summary(diagnostics, metadata)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "future_worse_move_switch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(output_dir / "future_worse_move_switch_report.md", summary)
    print(f"Wrote move-switch analysis to {output_dir}")


if __name__ == "__main__":
    main()
