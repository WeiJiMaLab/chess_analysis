"""Dry-run analysis of source-tree stratification before controller packing.

Reuses the packing pipeline's tree-level statistics (stop-depth excess,
budget-action variance) and stratified keep-probability logic to report,
without writing any episodes, how trees would be distributed across strata
under each stratification ``mode`` ("d", "j", or "dj"). Used to answer
lab-notebook questions about how controller behavior partitions across
different kinds of trees (tactical vs quiet, depth profile, etc.).
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from cts.data.preprocess_mc.pack import (
    _build_packed_tree_result,
    _compute_tree_stratified_keep_probabilities_from_rows,
    _deterministic_unit_interval,
    _feature_schema,
    _oracle_config,
    _read_manifest,
)


class AnalyzeTreeStratificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    num_workers: int = 0
    reward_scale: float = 1.0
    min_halt_reward_range: float = 0.0
    min_decision_margin: float = 0.0
    exclude_xaba: bool = False
    mode: Literal["d", "j", "dj", "all"] = "all"
    maintenance_scale: float = 0.0
    maintenance_ref_nodes: float = 30.0
    maintenance_exponent: float = 1.1
    time_lambda: float = 18.537
    time_p: float = 2.8
    time_tau: float = 2.5
    time_delta: int = 1
    timeout_value: float = -1.0
    samples_per_bucket: int = 2
    seed: int = 0
    scramble_min_time: int = 1
    scramble_max_time: int = 3
    medium_small_min_time: int = 4
    medium_small_max_time: int = 10
    medium_large_min_time: int = 11
    medium_large_max_time: int = 25
    large_min_time: int = 26
    large_max_time: int = 60
    very_large_min_time: int = 61
    very_large_max_time: int = 120


def _stats_one_task(
    task: tuple[str, float, float, float, bool, tuple[str, ...], Any],
) -> dict[str, Any] | None:
    """Compute the per-tree stratification stats for a single source tree.

    Runs the same packing pipeline used at training-data generation time but
    only keeps the two scalars used by the stratifier plus the episode count.
    Returns None when the underlying packer rejects the tree (e.g. filtered
    out by ``min_halt_reward_range`` / ``min_decision_margin`` / xABA).

    Args:
        task: tuple form so this function can be ``executor.submit``-ed
            without per-arg pickling overhead.
    """
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
    """Pack every tree in ``manifest_path`` and return one stats row per accepted tree.

    Args:
        manifest_path: path to the controller-packing manifest (the same one
            ``pack_controller_episodes`` consumes).
        num_workers: 0 to run inline, >0 to fan out across processes. Matches
            the ``--num-workers`` semantics of the packer.

    Returns:
        ``(num_manifest_paths, rows)`` where ``num_manifest_paths`` is the
        total trees listed in the manifest (including ones the packer dropped)
        and ``rows`` is the surviving subset with per-tree stats.
    """
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
    # Serial path: avoids ProcessPoolExecutor startup cost and lets tracebacks
    # surface directly during debugging.
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
    """Summarize tree distribution and keep-counts for one stratification mode.

    For each row, assign it to a stratum by bucketing its ``stop_depth_excess``
    (the "d" axis) and/or ``budget_action_variance`` (the "j" axis) against the
    thresholds the packer would use. Then report per-stratum tree counts,
    episode counts, and both expected (probability-weighted) and actual
    (deterministically sampled) keep counts.

    Args:
        mode: "d" (depth bin only), "j" (variance bin only), or "dj"
            (variance bins computed within each depth bin).
        seed: PRNG seed for the deterministic per-tree keep decision, matching
            the one ``pack_controller_episodes`` would use.
    """
    keep_probabilities, metadata = _compute_tree_stratified_keep_probabilities_from_rows(rows, mode)
    total_trees = len(rows)
    if not rows:
        # Empty manifest / everything rejected: return a well-shaped empty
        # summary so downstream JSON consumers don't have to special-case it.
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
    # Bucket every row into its stratum label. The d-axis uses a single global
    # threshold list; the j-axis either uses a single global list ("j") or a
    # per-d-bin list ("dj") so j-bins are conditional on depth.
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
            # "dj": j thresholds are conditional on the row's d_bin.
            thresholds = metadata.get("tree_budget_action_variance_thresholds_by_d_bin", {}).get(f"d{d_bin}", [])
            j_bin = 0
            for threshold in thresholds:
                if float(row["budget_action_variance"]) > float(threshold):
                    j_bin += 1
            stratum = f"d{d_bin}_j{j_bin}"
        grouped_rows[stratum].append(row)

    # Per-stratum rollup. "expected" weights by keep probability; "actual"
    # replays the same deterministic per-path coin-flip the packer uses, so
    # the numbers here predict the real packing run rather than just its mean.
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


def main(config: AnalyzeTreeStratificationConfig) -> None:
    """CLI entry point: pack every manifest tree (stats only) and print stratum summaries as JSON."""
    manifest_path = Path(config.manifest_path)
    oracle_config = _oracle_config(config)
    num_manifest_paths, rows = _compute_rows(
        manifest_path,
        config.reward_scale,
        config.min_halt_reward_range,
        config.min_decision_margin,
        config.exclude_xaba,
        oracle_config,
        config.num_workers,
    )
    # "all" expands into a separate summary per mode so a single invocation
    # can compare d, j, and dj stratifications side by side.
    modes = ("d", "j", "dj") if config.mode == "all" else (config.mode,)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "num_manifest_paths": num_manifest_paths,
                "num_accepted_trees": len(rows),
                "modes": [_summarize_mode(rows, mode, config.seed) for mode in modes],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(AnalyzeTreeStratificationConfig, main)
