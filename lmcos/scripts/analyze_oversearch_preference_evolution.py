from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from cts_episode_envs import (
        _root_q_values_from_teacher_result,
        build_trimmed_decision_episode,
        build_snapshot_episode,
        oracle_root_q_trace_for_example,
        trim_episode_to_first_root_decision,
    )
    from cts_pretrain import PretrainExample, TeacherSearchConfig, compute_teacher_targets, load_pretrain_example
except ModuleNotFoundError:
    torch = None
    _root_q_values_from_teacher_result = None
    build_trimmed_decision_episode = None
    build_snapshot_episode = None
    oracle_root_q_trace_for_example = None
    trim_episode_to_first_root_decision = None
    PretrainExample = None
    TeacherSearchConfig = Any  # type: ignore[assignment]
    compute_teacher_targets = None
    load_pretrain_example = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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


def _analysis_quality_config(metadata: dict[str, Any]) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=int(metadata.get("max_depth", 10)),
        search_budget=int(metadata.get("search_budget", 64)),
        c_puct=float(metadata.get("c_puct", 1.0)),
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def _parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected path rewrite OLD=NEW, got {value!r}.")
        old_prefix, new_prefix = value.split("=", 1)
        if not old_prefix:
            raise ValueError(f"Rewrite prefix cannot be empty: {value!r}.")
        rewrites.append((old_prefix, new_prefix))
    return rewrites


def _apply_path_rewrites(source_path: str, rewrites: list[tuple[str, str]]) -> str:
    rewritten = source_path
    for old_prefix, new_prefix in rewrites:
        if rewritten.startswith(old_prefix):
            rewritten = new_prefix + rewritten[len(old_prefix):]
    return rewritten


def _load_trimmed_snapshot_q_maps(
    source_path: str,
    quality_config: TeacherSearchConfig,
) -> tuple[list[dict[str, float]], list[str], dict[str, float]]:
    if (
        torch is None
        or build_trimmed_decision_episode is None
        or oracle_root_q_trace_for_example is None
        or PretrainExample is None
    ):
        raise RuntimeError("Torch/CTS reconstruction stack is unavailable in this environment.")
    example = load_pretrain_example(source_path)
    if not isinstance(example, PretrainExample):
        raise ValueError(f"Expected PretrainExample at {source_path}, got {type(example).__name__}.")
    decision_episode = build_trimmed_decision_episode(example, quality_config)
    stored_trace = oracle_root_q_trace_for_example(example)
    if stored_trace is not None:
        _, q_maps, best_moves, final_root_q_values = stored_trace
        return q_maps, best_moves, dict(final_root_q_values)

    if (
        build_snapshot_episode is None
        or trim_episode_to_first_root_decision is None
        or _root_q_values_from_teacher_result is None
        or compute_teacher_targets is None
    ):
        raise RuntimeError("Stored oracle traces are unavailable and teacher-search reconstruction is unavailable.")
    snapshots, qualities = build_snapshot_episode(example, quality_config)
    trimmed_snapshots, _ = trim_episode_to_first_root_decision(snapshots, qualities)

    q_maps: list[dict[str, float]] = []
    best_moves: list[str] = []
    for snapshot in trimmed_snapshots:
        teacher_result = compute_teacher_targets(snapshot, quality_config, validate=False)
        root_q_values = _root_q_values_from_teacher_result(snapshot, teacher_result)
        if not root_q_values:
            raise ValueError(f"No root q-values reconstructed for {source_path}.")
        q_maps.append(root_q_values)
        best_moves.append(max(root_q_values.items(), key=lambda item: item[1])[0])
    return q_maps, best_moves, dict(decision_episode.final_root_q_values)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _plot_halt_reward_delta_trajectory(diagnostics: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable in this environment.")
    oversearch = [episode for episode in diagnostics if int(episode["predicted_stop_step"]) > int(episode["oracle_stop_step"])]
    if not oversearch:
        raise ValueError("No oversearch episodes found in diagnostics.")

    max_len = max(max(len(episode["halt_rewards"]) - 1, 0) for episode in oversearch)
    means = []
    medians = []
    p10s = []
    p90s = []
    counts = []

    for step_index in range(max_len):
        values = [
            float(episode["halt_rewards"][step_index + 1]) - float(episode["halt_rewards"][step_index])
            for episode in oversearch
            if step_index + 1 < len(episode["halt_rewards"])
        ]
        if not values:
            continue
        arr = np.array(values, dtype=np.float32)
        means.append(float(arr.mean()))
        medians.append(float(np.median(arr)))
        p10s.append(float(np.percentile(arr, 10)))
        p90s.append(float(np.percentile(arr, 90)))
        counts.append(int(arr.size))

    xs = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.fill_between(xs, p10s, p90s, alpha=0.2, color="tab:green", label="10-90 pct")
    ax.plot(xs, medians, linewidth=2, color="tab:green", label="Median delta")
    ax.plot(xs, means, linewidth=1.5, linestyle="--", color="tab:orange", label="Mean delta")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Search step t")
    ax.set_ylabel("halt_reward[t+1] - halt_reward[t]")
    ax.set_title("Oversearch Episodes: Halt-Reward Delta Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "mean": means,
        "median": medians,
        "p10": p10s,
        "p90": p90s,
        "counts": counts,
    }


def _collect_oversearch_preference_rows(
    diagnostics: list[dict[str, Any]],
    quality_config: TeacherSearchConfig,
    path_rewrites: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    oversearch = [episode for episode in diagnostics if int(episode["predicted_stop_step"]) > int(episode["oracle_stop_step"])]
    cache: dict[str, tuple[list[dict[str, float]], list[str], dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    missing_paths: list[str] = []

    for episode in oversearch:
        source_path = str(episode["source_path"])
        resolved_path = _apply_path_rewrites(source_path, path_rewrites)
        if resolved_path not in cache:
            if not Path(resolved_path).exists():
                missing_paths.append(resolved_path)
                continue
            cache[resolved_path] = _load_trimmed_snapshot_q_maps(resolved_path, quality_config)

        q_maps, best_moves, final_root_q_values = cache[resolved_path]
        episode_length = int(episode["episode_length"])
        oracle_stop = int(episode["oracle_stop_step"])
        predicted_stop = int(episode["predicted_stop_step"])
        if oracle_stop >= len(best_moves) or predicted_stop >= len(best_moves):
            continue

        oracle_move = best_moves[oracle_stop]
        metacontroller_move = best_moves[predicted_stop]
        oracle_move_final_value = float(final_root_q_values[oracle_move])
        metacontroller_move_final_value = float(final_root_q_values[metacontroller_move])
        oracle_q_trajectory: list[float | None] = []
        metacontroller_q_trajectory: list[float | None] = []
        q_gap_trajectory: list[float | None] = []
        oracle_available: list[bool] = []
        metacontroller_available: list[bool] = []

        for step_index in range(min(episode_length, len(q_maps))):
            q_map = q_maps[step_index]
            oracle_q = q_map.get(oracle_move)
            meta_q = q_map.get(metacontroller_move)
            oracle_available.append(oracle_q is not None)
            metacontroller_available.append(meta_q is not None)
            oracle_q_trajectory.append(None if oracle_q is None else float(oracle_q))
            metacontroller_q_trajectory.append(None if meta_q is None else float(meta_q))
            q_gap_trajectory.append(None if oracle_q is None or meta_q is None else float(meta_q - oracle_q))
            rows.append(
                {
                    "path": str(episode["path"]),
                    "source_path": source_path,
                    "resolved_source_path": resolved_path,
                    "budget_bucket_name": str(episode["budget_bucket_name"]),
                    "starting_budget": int(episode["starting_budget"]),
                    "oracle_stop_step": oracle_stop,
                    "predicted_stop_step": predicted_stop,
                    "stop_delta": predicted_stop - oracle_stop,
                    "step_index": step_index,
                    "relative_step": step_index - oracle_stop,
                    "oracle_move": oracle_move,
                    "metacontroller_move": metacontroller_move,
                    "oracle_move_final_value": oracle_move_final_value,
                    "metacontroller_move_final_value": metacontroller_move_final_value,
                    "oracle_move_q": None if oracle_q is None else float(oracle_q),
                    "metacontroller_move_q": None if meta_q is None else float(meta_q),
                    "q_gap_meta_minus_oracle": None if oracle_q is None or meta_q is None else float(meta_q - oracle_q),
                    "oracle_move_available": oracle_q is not None,
                    "metacontroller_move_available": meta_q is not None,
                }
            )

        episode_records.append(
            {
                "path": str(episode["path"]),
                "source_path": source_path,
                "resolved_source_path": resolved_path,
                "budget_bucket_name": str(episode["budget_bucket_name"]),
                "starting_budget": int(episode["starting_budget"]),
                "oracle_stop_step": oracle_stop,
                "predicted_stop_step": predicted_stop,
                "stop_delta": predicted_stop - oracle_stop,
                "oracle_move": oracle_move,
                "metacontroller_move": metacontroller_move,
                "same_move": oracle_move == metacontroller_move,
                "oracle_move_final_value": oracle_move_final_value,
                "metacontroller_move_final_value": metacontroller_move_final_value,
                "final_value_gap_meta_minus_oracle": metacontroller_move_final_value - oracle_move_final_value,
                "regret": float(episode["regret"]),
                "oracle_move_q_trajectory": oracle_q_trajectory,
                "metacontroller_move_q_trajectory": metacontroller_q_trajectory,
                "q_gap_meta_minus_oracle_trajectory": q_gap_trajectory,
                "oracle_move_available_trajectory": oracle_available,
                "metacontroller_move_available_trajectory": metacontroller_available,
            }
        )

    status = {
        "requested_oversearch_episodes": len(oversearch),
        "reconstructed_oversearch_episodes": len(episode_records),
        "missing_source_paths": sorted(set(missing_paths)),
    }
    return rows, episode_records, status


def _summarize_relative_q(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["relative_step"])].append(row)

    summary: dict[str, Any] = {}
    for relative_step in sorted(grouped):
        step_rows = grouped[relative_step]
        oracle_values = [float(row["oracle_move_q"]) for row in step_rows if row["oracle_move_q"] is not None]
        meta_values = [float(row["metacontroller_move_q"]) for row in step_rows if row["metacontroller_move_q"] is not None]
        gap_values = [float(row["q_gap_meta_minus_oracle"]) for row in step_rows if row["q_gap_meta_minus_oracle"] is not None]
        oracle_final_values = [float(row["oracle_move_final_value"]) for row in step_rows]
        meta_final_values = [float(row["metacontroller_move_final_value"]) for row in step_rows]
        summary[str(relative_step)] = {
            "episodes": len(step_rows),
            "oracle_move_available_fraction": statistics.mean(float(bool(row["oracle_move_available"])) for row in step_rows),
            "metacontroller_move_available_fraction": statistics.mean(float(bool(row["metacontroller_move_available"])) for row in step_rows),
            "oracle_move_q_mean": None if not oracle_values else float(statistics.mean(oracle_values)),
            "oracle_move_q_median": None if not oracle_values else float(statistics.median(oracle_values)),
            "metacontroller_move_q_mean": None if not meta_values else float(statistics.mean(meta_values)),
            "metacontroller_move_q_median": None if not meta_values else float(statistics.median(meta_values)),
            "q_gap_mean": None if not gap_values else float(statistics.mean(gap_values)),
            "q_gap_median": None if not gap_values else float(statistics.median(gap_values)),
            "oracle_move_final_value_mean": float(statistics.mean(oracle_final_values)),
            "metacontroller_move_final_value_mean": float(statistics.mean(meta_final_values)),
            "final_value_gap_mean": float(statistics.mean(meta - oracle for meta, oracle in zip(meta_final_values, oracle_final_values))),
        }
    return summary


def _same_move_regret_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    same = [record for record in records if bool(record["same_move"])]
    switch = [record for record in records if not bool(record["same_move"])]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "episodes": 0,
                "fraction": 0.0,
                "mean_regret": 0.0,
                "total_regret": 0.0,
                "mean_final_value_gap": 0.0,
            }
        return {
            "episodes": len(rows),
            "fraction": len(rows) / len(records),
            "mean_regret": float(statistics.mean(float(row["regret"]) for row in rows)),
            "total_regret": float(sum(float(row["regret"]) for row in rows)),
            "mean_final_value_gap": float(statistics.mean(float(row["final_value_gap_meta_minus_oracle"]) for row in rows)),
        }

    switch_better = [row for row in switch if float(row["final_value_gap_meta_minus_oracle"]) > 1e-9]
    switch_equal = [row for row in switch if abs(float(row["final_value_gap_meta_minus_oracle"])) <= 1e-9]
    switch_worse = [row for row in switch if float(row["final_value_gap_meta_minus_oracle"]) < -1e-9]
    return {
        "same_move": summarize(same),
        "move_switch": summarize(switch),
        "move_switch_final_value_relation": {
            "better": summarize(switch_better),
            "equal": summarize(switch_equal),
            "worse": summarize(switch_worse),
        },
    }


def _plot_relative_q_trajectories(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable in this environment.")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["relative_step"])].append(row)

    xs = sorted(grouped)
    xs_arr = np.array(xs, dtype=np.int32)
    oracle_means = []
    oracle_p10 = []
    oracle_p90 = []
    meta_means = []
    meta_p10 = []
    meta_p90 = []
    oracle_final_means = []
    meta_final_means = []
    oracle_availability = []
    meta_availability = []

    for x in xs:
        step_rows = grouped[x]
        oracle_values = np.array([float(row["oracle_move_q"]) for row in step_rows if row["oracle_move_q"] is not None], dtype=np.float32)
        meta_values = np.array([float(row["metacontroller_move_q"]) for row in step_rows if row["metacontroller_move_q"] is not None], dtype=np.float32)
        oracle_means.append(float(oracle_values.mean()) if oracle_values.size else np.nan)
        oracle_p10.append(float(np.percentile(oracle_values, 10)) if oracle_values.size else np.nan)
        oracle_p90.append(float(np.percentile(oracle_values, 90)) if oracle_values.size else np.nan)
        meta_means.append(float(meta_values.mean()) if meta_values.size else np.nan)
        meta_p10.append(float(np.percentile(meta_values, 10)) if meta_values.size else np.nan)
        meta_p90.append(float(np.percentile(meta_values, 90)) if meta_values.size else np.nan)
        oracle_final_means.append(float(np.mean([float(row["oracle_move_final_value"]) for row in step_rows])))
        meta_final_means.append(float(np.mean([float(row["metacontroller_move_final_value"]) for row in step_rows])))
        oracle_availability.append(float(np.mean([float(bool(row["oracle_move_available"])) for row in step_rows])))
        meta_availability.append(float(np.mean([float(bool(row["metacontroller_move_available"])) for row in step_rows])))

    fig, (ax_q, ax_avail) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    ax_q.fill_between(xs_arr, oracle_p10, oracle_p90, alpha=0.2, color="tab:blue")
    ax_q.plot(xs_arr, oracle_means, linewidth=2, color="tab:blue", label="Oracle move Q")
    ax_q.fill_between(xs_arr, meta_p10, meta_p90, alpha=0.2, color="tab:orange")
    ax_q.plot(xs_arr, meta_means, linewidth=2, color="tab:orange", label="Metacontroller move Q")
    ax_q.plot(xs_arr, oracle_final_means, linewidth=1.5, linestyle="--", color="tab:blue", label="Oracle move final value")
    ax_q.plot(xs_arr, meta_final_means, linewidth=1.5, linestyle="--", color="tab:orange", label="Metacontroller move final value")
    ax_q.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_q.set_ylabel("Snapshot root Q-value")
    ax_q.set_title("Oversearch Episodes: Oracle vs Metacontroller Choice Q-Trajectories")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best")

    ax_avail.plot(xs_arr, oracle_availability, linewidth=2, color="tab:blue", label="Oracle move available")
    ax_avail.plot(xs_arr, meta_availability, linewidth=2, color="tab:orange", label="Metacontroller move available")
    ax_avail.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_avail.set_xlabel("Step relative to oracle stop")
    ax_avail.set_ylabel("Availability fraction")
    ax_avail.set_ylim(-0.02, 1.02)
    ax_avail.grid(True, alpha=0.3)
    ax_avail.legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "relative_steps": xs,
        "oracle_move_q_mean": oracle_means,
        "metacontroller_move_q_mean": meta_means,
        "oracle_move_final_value_mean": oracle_final_means,
        "metacontroller_move_final_value_mean": meta_final_means,
        "oracle_move_available_fraction": oracle_availability,
        "metacontroller_move_available_fraction": meta_availability,
    }


def _plot_relative_q_gap(rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    if plt is None:
        raise RuntimeError("matplotlib is unavailable in this environment.")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["q_gap_meta_minus_oracle"] is None:
            continue
        grouped[int(row["relative_step"])].append(row)

    xs = sorted(grouped)
    xs_arr = np.array(xs, dtype=np.int32)
    means = []
    medians = []
    p10s = []
    p90s = []
    counts = []

    for x in xs:
        values = np.array([float(row["q_gap_meta_minus_oracle"]) for row in grouped[x]], dtype=np.float32)
        means.append(float(values.mean()))
        medians.append(float(np.median(values)))
        p10s.append(float(np.percentile(values, 10)))
        p90s.append(float(np.percentile(values, 90)))
        counts.append(int(values.size))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.fill_between(xs_arr, p10s, p90s, alpha=0.2, color="tab:purple", label="10-90 pct")
    ax.plot(xs_arr, medians, linewidth=2, color="tab:purple", label="Median Q gap")
    ax.plot(xs_arr, means, linewidth=1.5, linestyle="--", color="tab:red", label="Mean Q gap")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Step relative to oracle stop")
    ax.set_ylabel("Q(meta choice) - Q(oracle choice)")
    ax.set_title("Oversearch Episodes: Relative Root-Q Preference Gap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "relative_steps": xs,
        "mean_q_gap": means,
        "median_q_gap": medians,
        "p10_q_gap": p10s,
        "p90_q_gap": p90s,
        "counts": counts,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Oversearch Preference Evolution",
        "",
        f"- Oversearch episodes requested: {summary['source_status']['requested_oversearch_episodes']}",
        f"- Oversearch episodes reconstructed: {summary['source_status']['reconstructed_oversearch_episodes']}",
        f"- Missing source paths: {len(summary['source_status']['missing_source_paths'])}",
        "",
        "## Definitions",
        "- `oracle_move` is the root move chosen by the snapshot at the oracle stop step.",
        "- `metacontroller_move` is the root move chosen by the snapshot at the learned controller's later stop step.",
        "- Each plotted Q-value is the snapshot-local root child `q_value` for that fixed move at each search step.",
    ]
    split = summary.get("same_move_regret")
    if split is not None:
        lines.extend(
            [
                "",
                "## Same Move vs Move Switch",
                f"- Same-move oversearch: {split['same_move']['episodes']} episodes, mean regret {split['same_move']['mean_regret']:.4f}, total regret {split['same_move']['total_regret']:.4f}",
                f"- Move-switch oversearch: {split['move_switch']['episodes']} episodes, mean regret {split['move_switch']['mean_regret']:.4f}, total regret {split['move_switch']['total_regret']:.4f}",
                f"- Among move-switch episodes, final-value-worse cases: {split['move_switch_final_value_relation']['worse']['episodes']} episodes",
                f"- Among move-switch episodes, final-value-better cases: {split['move_switch_final_value_relation']['better']['episodes']} episodes",
            ]
        )
    if summary["relative_q_summary"]:
        zero = summary["relative_q_summary"].get("0")
        lines.extend(
            [
                "",
                "## Relative-Step Summary",
                f"- Same-move fraction among reconstructed oversearch episodes: {summary['same_move_fraction']:.3f}",
            ]
        )
        if zero is not None:
            if zero["oracle_move_q_mean"] is not None:
                lines.append(f"- At oracle stop: oracle move mean Q = {zero['oracle_move_q_mean']:.4f}")
            if zero["metacontroller_move_q_mean"] is not None:
                lines.append(f"- At oracle stop: metacontroller move mean Q = {zero['metacontroller_move_q_mean']:.4f}")
            if zero["q_gap_mean"] is not None:
                lines.append(f"- At oracle stop: mean Q gap (meta - oracle) = {zero['q_gap_mean']:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze oracle-vs-metacontroller preference evolution on oversearch episodes.")
    parser.add_argument("--diagnostics-path", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        help="Optional OLD=NEW prefix rewrite for source_path resolution. May be supplied multiple times.",
    )
    parser.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help="Still emit the halt-reward delta plot if source trees needed for Q reconstruction are unavailable.",
    )
    args = parser.parse_args()

    diagnostics_path = Path(args.diagnostics_path)
    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, _, _, _ = parse_log(log_path)
    diagnostics = load_diagnostics(diagnostics_path)
    path_rewrites = _parse_path_rewrites(list(args.path_rewrite))

    halt_reward_delta_summary: dict[str, Any] | None = None
    if plt is not None:
        halt_reward_delta_summary = _plot_halt_reward_delta_trajectory(
            diagnostics,
            output_dir / "oversearch_halt_reward_delta_trajectory.png",
        )
    oversearch = [episode for episode in diagnostics if int(episode["predicted_stop_step"]) > int(episode["oracle_stop_step"])]
    resolved_paths = {
        _apply_path_rewrites(str(episode["source_path"]), path_rewrites)
        for episode in oversearch
    }
    existing_resolved_paths = sorted(path for path in resolved_paths if Path(path).exists())
    missing_resolved_paths = sorted(path for path in resolved_paths if not Path(path).exists())

    rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    source_status = {
        "requested_oversearch_episodes": len(oversearch),
        "reconstructed_oversearch_episodes": 0,
        "missing_source_paths": missing_resolved_paths,
    }

    if existing_resolved_paths:
        quality_config = _analysis_quality_config(metadata)
        rows, episode_records, source_status = _collect_oversearch_preference_rows(
            diagnostics,
            quality_config,
            path_rewrites,
        )

    if source_status["missing_source_paths"] and not args.allow_missing_sources:
        preview = "\n".join(source_status["missing_source_paths"][:5])
        raise FileNotFoundError(
            "Source trees are unavailable locally. Provide --path-rewrite to a local mirror, or rerun with "
            f"--allow-missing-sources to at least emit the halt-reward delta plot.\nMissing examples:\n{preview}"
        )

    summary: dict[str, Any] = {
        "source_status": source_status,
        "halt_reward_delta_trajectory": halt_reward_delta_summary,
        "relative_q_summary": {},
        "same_move_fraction": float("nan"),
        "matplotlib_available": plt is not None,
    }
    if episode_records:
        _write_jsonl(output_dir / "oversearch_preference_q_trajectories.jsonl", episode_records)
        summary["relative_q_summary"] = _summarize_relative_q(rows)
        summary["same_move_regret"] = _same_move_regret_summary(episode_records)
        if plt is not None:
            summary["relative_q_plot"] = _plot_relative_q_trajectories(
                rows,
                output_dir / "oversearch_choice_q_trajectories.png",
            )
            summary["relative_q_gap_plot"] = _plot_relative_q_gap(
                rows,
                output_dir / "oversearch_choice_q_gap.png",
            )
        summary["same_move_fraction"] = float(statistics.mean(float(record["same_move"]) for record in episode_records))

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(output_dir / "report.md", summary)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
