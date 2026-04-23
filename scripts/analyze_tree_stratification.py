from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pack_controller_episodes import (
    _build_packed_tree_result,
    _compute_tree_stratified_keep_probabilities_from_rows,
    _deterministic_unit_interval,
    _feature_schema,
    _oracle_config,
    _read_manifest,
)


def _stats_one_task(
    task: tuple[str, float, float, float, bool, tuple[str, ...], Any],
) -> dict[str, Any] | None:
    path_str, reward_scale, min_halt_reward_range, min_decision_margin, exclude_xaba, feature_names, oracle_config = task
    result = _build_packed_tree_result(
        path_str,
        reward_scale,
        min_halt_reward_range,
        min_decision_margin,
        exclude_xaba,
        feature_names,
        oracle_config,
    )
    if result is None:
        return None
    return {
        "source_path": path_str,
        "stop_depth_excess": float(result["stop_depth_excess"]),
        "budget_action_variance": float(result["budget_action_variance"]),
        "num_episodes": len(result["episodes"]),
    }


def _compute_rows(
    manifest_path: Path,
    reward_scale: float,
    min_halt_reward_range: float,
    min_decision_margin: float,
    exclude_xaba: bool,
    oracle_config: Any,
    num_workers: int,
) -> tuple[int, list[dict[str, Any]]]:
    example_paths = _read_manifest(manifest_path)
    feature_names = tuple(_feature_schema().feature_names)
    tasks = [
        (
            str(path),
            reward_scale,
            min_halt_reward_range,
            min_decision_margin,
            exclude_xaba,
            feature_names,
            oracle_config,
        )
        for path in example_paths
    ]
    rows: list[dict[str, Any]] = []
    if num_workers <= 0:
        for task in tasks:
            result = _stats_one_task(task)
            if result is not None:
                rows.append(result)
        return len(example_paths), rows

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_stats_one_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)
    return len(example_paths), rows


def _summarize_mode(rows: list[dict[str, Any]], mode: str, seed: int) -> dict[str, Any]:
    keep_probabilities, metadata = _compute_tree_stratified_keep_probabilities_from_rows(rows, mode)
    total_trees = len(rows)
    if not rows:
        return {
            "mode": mode,
            "num_accepted_trees": 0,
            "expected_kept_trees": 0.0,
            "actual_kept_trees": 0,
            "expected_kept_episodes": 0.0,
            "actual_kept_episodes": 0,
            "stop_depth_thresholds": [],
            "budget_action_variance_thresholds": [],
            "strata": [],
        }

    expected_kept_trees = 0.0
    actual_kept_trees = 0
    expected_kept_episodes = 0.0
    actual_kept_episodes = 0
    strata_rows = []
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        path_str = str(row["source_path"])
        d_bin = 0
        for threshold in metadata.get("tree_stop_depth_thresholds", []):
            if float(row["stop_depth_excess"]) > float(threshold):
                d_bin += 1
        if mode == "d":
            stratum = f"d{d_bin}"
        elif mode == "j":
            j_bin = 0
            for threshold in metadata.get("tree_budget_action_variance_thresholds", []):
                if float(row["budget_action_variance"]) > float(threshold):
                    j_bin += 1
            stratum = f"j{j_bin}"
        else:
            thresholds = metadata.get("tree_budget_action_variance_thresholds_by_d_bin", {}).get(f"d{d_bin}", [])
            j_bin = 0
            for threshold in thresholds:
                if float(row["budget_action_variance"]) > float(threshold):
                    j_bin += 1
            stratum = f"d{d_bin}_j{j_bin}"
        grouped_rows[stratum].append(row)

    for stratum in sorted(grouped_rows):
        bucket_rows = grouped_rows[stratum]
        count = len(bucket_rows)
        episodes = sum(int(row["num_episodes"]) for row in bucket_rows)
        expected_kept_trees_in_stratum = sum(keep_probabilities[str(row["source_path"])] for row in bucket_rows)
        expected_kept_episodes_in_stratum = sum(
            keep_probabilities[str(row["source_path"])] * int(row["num_episodes"])
            for row in bucket_rows
        )
        actual_kept_in_stratum = 0
        actual_kept_episodes_in_stratum = 0
        for row in bucket_rows:
            keep_probability = keep_probabilities[str(row["source_path"])]
            if _deterministic_unit_interval(str(row["source_path"]), seed) < keep_probability:
                actual_kept_in_stratum += 1
                actual_kept_episodes_in_stratum += int(row["num_episodes"])
        expected_kept_trees += expected_kept_trees_in_stratum
        expected_kept_episodes += expected_kept_episodes_in_stratum
        actual_kept_trees += actual_kept_in_stratum
        actual_kept_episodes += actual_kept_episodes_in_stratum
        strata_rows.append(
            {
                "stratum": stratum,
                "tree_count": count,
                "episode_count": episodes,
                "mean_keep_probability": expected_kept_trees_in_stratum / float(count),
                "expected_kept_trees": expected_kept_trees_in_stratum,
                "actual_kept_trees": actual_kept_in_stratum,
                "expected_kept_episodes": expected_kept_episodes_in_stratum,
                "actual_kept_episodes": actual_kept_episodes_in_stratum,
            }
        )

    return {
        "mode": mode,
        "num_accepted_trees": total_trees,
        "expected_kept_trees": expected_kept_trees,
        "actual_kept_trees": actual_kept_trees,
        "expected_kept_episodes": expected_kept_episodes,
        "actual_kept_episodes": actual_kept_episodes,
        "stop_depth_thresholds": metadata.get("tree_stop_depth_thresholds", []),
        "budget_action_variance_thresholds": metadata.get("tree_budget_action_variance_thresholds", []),
        "budget_action_variance_thresholds_by_d_bin": metadata.get("tree_budget_action_variance_thresholds_by_d_bin", {}),
        "strata": strata_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run source-tree stratification before controller packing.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--min-halt-reward-range", type=float, default=0.0)
    parser.add_argument("--min-decision-margin", type=float, default=0.0)
    parser.add_argument("--exclude-xaba", action="store_true")
    parser.add_argument("--mode", choices=("d", "j", "dj", "all"), default="all")
    parser.add_argument("--maintenance-scale", type=float, default=0.0)
    parser.add_argument("--maintenance-ref-nodes", type=float, default=30.0)
    parser.add_argument("--maintenance-exponent", type=float, default=1.1)
    parser.add_argument("--time-lambda", type=float, default=18.537)
    parser.add_argument("--time-p", type=float, default=2.8)
    parser.add_argument("--time-tau", type=float, default=2.5)
    parser.add_argument("--time-delta", type=int, default=1)
    parser.add_argument("--timeout-value", type=float, default=-1.0)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scramble-min-time", type=int, default=1)
    parser.add_argument("--scramble-max-time", type=int, default=3)
    parser.add_argument("--medium-small-min-time", type=int, default=4)
    parser.add_argument("--medium-small-max-time", type=int, default=10)
    parser.add_argument("--medium-large-min-time", type=int, default=11)
    parser.add_argument("--medium-large-max-time", type=int, default=25)
    parser.add_argument("--large-min-time", type=int, default=26)
    parser.add_argument("--large-max-time", type=int, default=60)
    parser.add_argument("--very-large-min-time", type=int, default=61)
    parser.add_argument("--very-large-max-time", type=int, default=120)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    oracle_config = _oracle_config(args)
    num_manifest_paths, rows = _compute_rows(
        manifest_path,
        args.reward_scale,
        args.min_halt_reward_range,
        args.min_decision_margin,
        args.exclude_xaba,
        oracle_config,
        args.num_workers,
    )
    modes = ("d", "j", "dj") if args.mode == "all" else (args.mode,)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "num_manifest_paths": num_manifest_paths,
                "num_accepted_trees": len(rows),
                "modes": [_summarize_mode(rows, mode, args.seed) for mode in modes],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
