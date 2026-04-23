from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from budgeted_controller_oracle import budgeted_oracle_config_from_metadata
from scripts.pack_controller_episodes import _tree_budget_sensitivity_score


def _summarize_tree(
    source_path: str,
    episodes: list[dict[str, Any]],
    tree_sizes: list[int],
    oracle_config: Any,
) -> dict[str, Any]:
    oracle_stop_steps = [int(episode["oracle_stop_step"]) for episode in episodes]
    oracle_values = [float(episode["oracle_value"]) for episode in episodes]
    low_margin_states = 0
    total_states = 0
    for episode in episodes:
        for value in episode["target_advantages"]:
            total_states += 1
            if abs(float(value)) < 0.05:
                low_margin_states += 1
    return {
        "source_path": source_path,
        "score": _tree_budget_sensitivity_score(episodes, tree_sizes, oracle_config),
        "p_stop_le_1": sum(1 for step in oracle_stop_steps if step <= 1) / len(oracle_stop_steps),
        "mean_oracle_stop": statistics.mean(oracle_stop_steps),
        "mean_oracle_value": statistics.mean(oracle_values),
        "low_margin_frac": (low_margin_states / total_states) if total_states > 0 else 0.0,
        "episodes": len(episodes),
    }


def _load_tree_summaries(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest.get("oracle_metadata", manifest)
    oracle_config = budgeted_oracle_config_from_metadata(metadata)
    if oracle_config is None:
        raise ValueError("Manifest oracle metadata does not describe a budgeted controller oracle.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tree_sizes_by_source: dict[str, list[int]] = {}
    for entry in manifest["entries"]:
        payload = torch.load(entry["path"], weights_only=False)
        step_ptr = payload["episode_step_ptr"]
        target_advantages = payload["target_advantages"]
        oracle_stop_steps = payload["oracle_stop_steps"]
        oracle_values = payload["oracle_values"]
        episode_trajectory_index = payload["episode_trajectory_index"]
        trajectory_source_paths = payload["trajectory_source_paths"]
        trajectory_step_ptr = payload["trajectory_step_ptr"]
        trajectory_halt_rewards = payload["trajectory_halt_rewards"]
        step_node_cutoffs = payload["step_node_cutoffs"]

        num_episodes = int(payload["num_episodes"])
        for episode_index in range(num_episodes):
            trajectory_index = int(episode_trajectory_index[episode_index].item())
            source_path = trajectory_source_paths[trajectory_index]
            start = int(step_ptr[episode_index].item())
            end = int(step_ptr[episode_index + 1].item())
            num_steps = end - start
            step_begin = int(trajectory_step_ptr[trajectory_index].item())
            step_end = int(trajectory_step_ptr[trajectory_index + 1].item())
            tree_sizes_by_source.setdefault(source_path, step_node_cutoffs[step_begin:step_end].tolist())
            grouped[source_path].append(
                {
                    "oracle_stop_step": int(oracle_stop_steps[episode_index].item()),
                    "oracle_value": float(oracle_values[episode_index].item()),
                    "starting_budget": int(payload["starting_budgets"][episode_index].item()),
                    "target_advantages": target_advantages[start:end].tolist(),
                    "halt_rewards": trajectory_halt_rewards[step_begin:step_begin + num_steps].tolist(),
                }
            )
    return [
        _summarize_tree(source_path, episodes, tree_sizes_by_source[source_path], oracle_config)
        for source_path, episodes in grouped.items()
    ]


def _decile_rows(tree_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(tree_summaries, key=lambda row: row["score"])
    total = len(ordered)
    rows = []
    for decile in range(10):
        start = (decile * total) // 10
        end = ((decile + 1) * total) // 10
        bucket = ordered[start:end]
        if not bucket:
            continue
        rows.append(
            {
                "decile": decile,
                "count": len(bucket),
                "score_min": bucket[0]["score"],
                "score_max": bucket[-1]["score"],
                "mean_p_stop_le_1": statistics.mean(row["p_stop_le_1"] for row in bucket),
                "mean_oracle_stop": statistics.mean(row["mean_oracle_stop"] for row in bucket),
                "mean_oracle_value": statistics.mean(row["mean_oracle_value"] for row in bucket),
                "mean_low_margin_frac": statistics.mean(row["low_margin_frac"] for row in bucket),
                "mean_episodes_per_tree": statistics.mean(row["episodes"] for row in bucket),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze budget-score deciles for a packed controller manifest.")
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    tree_summaries = _load_tree_summaries(manifest_path)
    deciles = _decile_rows(tree_summaries)
    print(json.dumps({"manifest_path": str(manifest_path), "num_trees": len(tree_summaries), "deciles": deciles}, indent=2))


if __name__ == "__main__":
    main()
