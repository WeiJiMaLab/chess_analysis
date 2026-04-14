"""Compute episode-difficulty metrics over packed controller shards.

Reads one or more packed-episode manifests (train, validation, or both),
computes all six difficulty metrics per episode, and writes:

  1. A per-episode JSONL file with all metrics and episode metadata.
  2. A printed summary table (mean / std / min / max per metric per split).

This script is data-only -- it does not load any model.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from episode_difficulty import EpisodeDifficultyMetrics, compute_difficulty_metrics
from planning_cost import PlanningCostConfig, planning_cost_config, planning_cost_config_from_metadata

_FIELD_NAMES = [f.name for f in EpisodeDifficultyMetrics.__dataclass_fields__.values()]


def _load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "cts_controller_episode_manifest_v1":
        raise ValueError(f"Unexpected manifest format in {manifest_path}")
    return manifest


def _process_shard(
    shard_path: str,
    cost_config: PlanningCostConfig,
    softmax_temperature: float,
) -> List[Dict[str, Any]]:
    payload = torch.load(shard_path, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "cts_controller_episode_shard_v1":
        raise ValueError(f"Unexpected shard format: {shard_path}")

    episode_step_ptr = payload["episode_step_ptr"]
    halt_rewards_tensor = payload["halt_rewards"]
    oracle_stop_steps = payload["oracle_stop_steps"]
    oracle_values = payload["oracle_values"]
    source_paths = payload.get("source_paths", [])
    num_episodes = int(payload["num_episodes"])

    records: List[Dict[str, Any]] = []
    for ep_idx in range(num_episodes):
        step_begin = int(episode_step_ptr[ep_idx].item())
        step_end = int(episode_step_ptr[ep_idx + 1].item())
        halt_rewards = halt_rewards_tensor[step_begin:step_end].tolist()
        oracle_stop = int(oracle_stop_steps[ep_idx].item())
        oracle_value = float(oracle_values[ep_idx].item())
        path = source_paths[ep_idx] if ep_idx < len(source_paths) else ""

        metrics = compute_difficulty_metrics(
            halt_rewards,
            cost_config.base_cost,
            softmax_temperature,
            cost_config=cost_config,
        )

        records.append({
            "path": path,
            "episode_length": len(halt_rewards),
            "halt_rewards": halt_rewards,
            "oracle_stop_step": oracle_stop,
            "oracle_value": oracle_value,
            "halt_reward_range": metrics.halt_reward_range,
            "optimal_vs_second_best_gap": metrics.optimal_vs_second_best_gap,
            "return_variance": metrics.return_variance,
            "softmax_entropy": metrics.softmax_entropy,
            "regret_of_always_halt": metrics.regret_of_always_halt,
            "reward_curvature": metrics.reward_curvature,
        })

    return records


def _analyze_split(
    manifest_path: str,
    cost_config: PlanningCostConfig,
    softmax_temperature: float,
    log_interval: int,
) -> List[Dict[str, Any]]:
    manifest = _load_manifest(manifest_path)
    entries = manifest["entries"]
    split_name = manifest.get("split", "unknown")
    manifest_cost = planning_cost_config_from_metadata(manifest)
    if manifest_cost is not None and (
        manifest_cost.kind != cost_config.kind
        or abs(manifest_cost.base_cost - cost_config.base_cost) > 1e-9
        or abs(manifest_cost.exponent - cost_config.exponent) > 1e-9
    ):
        print(
            f"[warning] manifest cost=({manifest_cost.kind}, {manifest_cost.base_cost}, {manifest_cost.exponent}) "
            f"differs from requested cost=({cost_config.kind}, {cost_config.base_cost}, {cost_config.exponent}); "
            "using requested cost",
            flush=True,
        )

    all_records: List[Dict[str, Any]] = []
    started = time.time()
    for shard_idx, entry in enumerate(entries):
        shard_records = _process_shard(entry["path"], cost_config, softmax_temperature)
        for r in shard_records:
            r["split"] = split_name
        all_records.extend(shard_records)

        if log_interval > 0 and (shard_idx + 1) % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"[{split_name}] shards={shard_idx + 1}/{len(entries)} "
                f"episodes={len(all_records)} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"[{split_name}] done shards={len(entries)} episodes={len(all_records)} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return all_records


def _print_summary(records: List[Dict[str, Any]], label: str) -> None:
    if not records:
        print(f"[{label}] no episodes", flush=True)
        return

    print(f"\n{'=' * 70}", flush=True)
    print(f"  {label}  ({len(records)} episodes)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'metric':<32s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}", flush=True)
    print(f"  {'-' * 32} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}", flush=True)
    for field in _FIELD_NAMES:
        values = [r[field] for r in records]
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        mn = min(values)
        mx = max(values)
        print(f"  {field:<32s} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {mx:>10.4f}", flush=True)

    lengths = [r["episode_length"] for r in records]
    oracle_stops = [r["oracle_stop_step"] for r in records]
    halt_at_zero = sum(1 for s in oracle_stops if s == 0)
    print(f"\n  episode_length:  mean={statistics.mean(lengths):.1f}  min={min(lengths)}  max={max(lengths)}", flush=True)
    print(f"  oracle_stop=0 (halt immediately):  {halt_at_zero}/{len(records)} ({100 * halt_at_zero / len(records):.1f}%)", flush=True)
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze episode difficulty from packed controller shards.")
    parser.add_argument("--train-manifest", default=None, help="Path to packed train manifest JSON.")
    parser.add_argument("--validation-manifest", default=None, help="Path to packed validation manifest JSON.")
    parser.add_argument("--continue-cost", type=float, required=True)
    parser.add_argument("--planning-cost-kind", choices=["linear", "power"], default="linear")
    parser.add_argument("--planning-cost-exponent", type=float, default=1.0)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()

    if args.train_manifest is None and args.validation_manifest is None:
        parser.error("At least one of --train-manifest or --validation-manifest is required.")

    cost_config = planning_cost_config(
        args.continue_cost,
        kind=args.planning_cost_kind,
        exponent=args.planning_cost_exponent,
    )
    all_records: List[Dict[str, Any]] = []

    if args.train_manifest is not None:
        train_records = _analyze_split(
            args.train_manifest, cost_config, args.softmax_temperature, args.log_interval,
        )
        all_records.extend(train_records)
        _print_summary(train_records, "train")

    if args.validation_manifest is not None:
        val_records = _analyze_split(
            args.validation_manifest, cost_config, args.softmax_temperature, args.log_interval,
        )
        all_records.extend(val_records)
        _print_summary(val_records, "validation")

    if args.train_manifest is not None and args.validation_manifest is not None:
        _print_summary(all_records, "combined (train + validation)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(all_records)} records to {output_path}", flush=True)


if __name__ == "__main__":
    main()
