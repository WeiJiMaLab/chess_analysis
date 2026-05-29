"""Post-filter packed controller-episode shards by halt-reward dynamic range.

Runs between ``pack_controller_episodes.py`` and
``materialize_controller_cache.py``. Drops episodes whose halt-reward range
is below ``--min-halt-reward-range`` (i.e. trivially-decided trees where
every stop step yields essentially the same reward, contributing no signal
to fitted-Q training). Optionally stratifies by ``oracle_stop_step`` to
prevent any single stop bucket from dominating the kept set. Re-packs the
survivors into fresh shards and emits a new manifest.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from pydantic import BaseModel, ConfigDict


class FilterPackedEpisodesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_manifest: str
    validation_manifest: str
    output_dir: str
    min_halt_reward_range: float = 0.05
    max_per_stop_step: int = 0
    shard_size: int = 512
    seed: int = 42


def _extract_episodes(payload: dict) -> List[Dict[str, Any]]:
    """Unpack one shard's flat tensors back into a list of per-episode dicts.

    Inverse of the packing performed by ``pack_controller_episodes.py``:
    every per-step tensor is reconstructed by slicing the shard-level
    arrays with the pointer tables (``episode_step_ptr``,
    ``trajectory_step_ptr``, ``step_node_ptr``, ``step_edge_ptr``). The
    returned list is the natural unit we filter and re-pack against.
    """
    episode_step_ptr = payload["episode_step_ptr"]
    episode_trajectory_index = payload["episode_trajectory_index"]
    trajectory_step_ptr = payload["trajectory_step_ptr"]
    step_node_ptr = payload["step_node_ptr"]
    step_edge_ptr = payload["step_edge_ptr"]
    num_episodes = int(payload["num_episodes"])

    episodes = []
    for i in range(num_episodes):
        # Episode-level slice into the shard's step arrays.
        episode_s_begin = int(episode_step_ptr[i].item())
        episode_s_end = int(episode_step_ptr[i + 1].item())
        num_steps = episode_s_end - episode_s_begin
        # Trajectory-level slice: episodes share a trajectory's per-step
        # tree tensors, indexed by ``episode_trajectory_index``.
        trajectory_index = int(episode_trajectory_index[i].item())
        trajectory_s_begin = int(trajectory_step_ptr[trajectory_index].item())
        trajectory_s_end = trajectory_s_begin + num_steps

        step_nf, step_pi, step_ep, step_ec, step_es, step_d = [], [], [], [], [], []
        for s in range(trajectory_s_begin, trajectory_s_end):
            # Per-step node and edge slices into the shard-level flat tensors.
            n0 = int(step_node_ptr[s].item())
            n1 = int(step_node_ptr[s + 1].item())
            e0 = int(step_edge_ptr[s].item())
            e1 = int(step_edge_ptr[s + 1].item())
            step_nf.append(payload["node_features"][n0:n1])
            step_pi.append(payload["parent_index"][n0:n1])
            step_d.append(payload["depth"][n0:n1])
            step_ep.append(payload["edge_parent"][e0:e1])
            step_ec.append(payload["edge_child"][e0:e1])
            step_es.append(payload["edge_slot"][e0:e1])

        episodes.append(
            {
                "step_nf": step_nf,
                "step_pi": step_pi,
                "step_d": step_d,
                "step_ep": step_ep,
                "step_ec": step_ec,
                "step_es": step_es,
                "halt_rewards": payload["halt_rewards"][episode_s_begin:episode_s_end],
                "target_advantages": payload["target_advantages"][episode_s_begin:episode_s_end],
                "tree_sizes": payload["tree_sizes"][episode_s_begin:episode_s_end],
                "time_budgets": payload["time_budgets"][episode_s_begin:episode_s_end],
                "oracle_stop_step": int(payload["oracle_stop_steps"][i].item()),
                "oracle_value": float(payload["oracle_values"][i].item()),
                "starting_budget": int(payload["starting_budgets"][i].item()),
                "budget_bucket_index": int(payload["budget_bucket_indices"][i].item()),
                "budget_bucket_name": payload["budget_bucket_names"][i],
                "episode_key": payload["episode_keys"][i],
                "source_path": payload["trajectory_source_paths"][trajectory_index],
            }
        )
    return episodes


def _write_shard(episodes: List[Dict[str, Any]], shard_path: Path, feature_names: list, metadata: dict) -> None:
    """Re-pack a list of episode dicts into a single shard file on disk.

    Reverses ``_extract_episodes``: deduplicates trajectories by source
    path (so multiple episodes from one trajectory share its per-step
    tree tensors), rebuilds all pointer tables, and writes the v3 shard
    format that ``materialize_controller_cache.py`` expects downstream.
    """
    trajectory_step_ptr = [0]
    episode_step_ptr = [0]
    step_node_ptr = [0]
    step_edge_ptr = [0]
    all_nf, all_pi, all_d = [], [], []
    all_ep, all_ec, all_es = [], [], []
    all_hr, all_ta, all_ns, all_tb = [], [], [], []
    trajectory_source_paths = []
    episode_trajectory_index = []
    oracle_stops, oracle_vals = [], []
    starting_budgets, bucket_indices = [], []
    bucket_names, episode_keys = [], []
    trajectory_lookup: Dict[str, int] = {}  # source_path → trajectory_index, for dedup

    for ep in episodes:
        # Append the trajectory's per-step tensors only the first time we
        # see its source path; subsequent episodes from the same trajectory
        # reuse the existing trajectory_index.
        trajectory_index = trajectory_lookup.get(ep["source_path"])
        if trajectory_index is None:
            trajectory_index = len(trajectory_source_paths)
            trajectory_lookup[ep["source_path"]] = trajectory_index
            trajectory_source_paths.append(ep["source_path"])
            num_steps = len(ep["step_nf"])
            trajectory_step_ptr.append(trajectory_step_ptr[-1] + num_steps)
            for s in range(num_steps):
                all_nf.append(ep["step_nf"][s])
                all_pi.append(ep["step_pi"][s])
                all_d.append(ep["step_d"][s])
                all_ep.append(ep["step_ep"][s])
                all_ec.append(ep["step_ec"][s])
                all_es.append(ep["step_es"][s])
                step_node_ptr.append(step_node_ptr[-1] + ep["step_nf"][s].shape[0])
                step_edge_ptr.append(step_edge_ptr[-1] + ep["step_ep"][s].shape[0])

        num_steps = len(ep["step_nf"])
        episode_step_ptr.append(episode_step_ptr[-1] + num_steps)
        episode_trajectory_index.append(trajectory_index)
        all_hr.append(ep["halt_rewards"])
        all_ta.append(ep["target_advantages"])
        all_ns.append(ep["tree_sizes"])
        all_tb.append(ep["time_budgets"])
        oracle_stops.append(ep["oracle_stop_step"])
        oracle_vals.append(ep["oracle_value"])
        starting_budgets.append(ep["starting_budget"])
        bucket_indices.append(ep["budget_bucket_index"])
        bucket_names.append(ep["budget_bucket_name"])
        episode_keys.append(ep["episode_key"])

    torch.save(
        {
            "format": "cts_budgeted_controller_episode_shard_v3",
            "num_trajectories": len(trajectory_source_paths),
            "num_episodes": len(episodes),
            "feature_names": feature_names,
            **metadata,
            "trajectory_step_ptr": torch.tensor(trajectory_step_ptr, dtype=torch.long),
            "episode_step_ptr": torch.tensor(episode_step_ptr, dtype=torch.long),
            "episode_trajectory_index": torch.tensor(episode_trajectory_index, dtype=torch.long),
            "step_node_ptr": torch.tensor(step_node_ptr, dtype=torch.long),
            "step_edge_ptr": torch.tensor(step_edge_ptr, dtype=torch.long),
            "node_features": torch.cat(all_nf, dim=0),
            "parent_index": torch.cat(all_pi, dim=0),
            # Empty-edge guards: a shard of root-only trees would otherwise
            # crash ``torch.cat`` with an empty list. Long dtype matches the
            # producing packer's convention.
            "edge_parent": torch.cat(all_ep, dim=0) if all_ep else torch.empty(0, dtype=torch.long),
            "edge_child": torch.cat(all_ec, dim=0) if all_ec else torch.empty(0, dtype=torch.long),
            "edge_slot": torch.cat(all_es, dim=0) if all_es else torch.empty(0, dtype=torch.long),
            "depth": torch.cat(all_d, dim=0),
            "halt_rewards": torch.cat(all_hr, dim=0),
            "target_advantages": torch.cat(all_ta, dim=0),
            "tree_sizes": torch.cat(all_ns, dim=0),
            "time_budgets": torch.cat(all_tb, dim=0),
            "oracle_stop_steps": torch.tensor(oracle_stops, dtype=torch.long),
            "oracle_values": torch.tensor(oracle_vals, dtype=torch.float32),
            "starting_budgets": torch.tensor(starting_budgets, dtype=torch.long),
            "budget_bucket_indices": torch.tensor(bucket_indices, dtype=torch.long),
            "budget_bucket_names": bucket_names,
            "episode_keys": episode_keys,
            "trajectory_source_paths": trajectory_source_paths,
        },
        shard_path,
    )


def _filter_split(
    manifest_path: str,
    output_dir: Path,
    min_halt_reward_range: float,
    max_per_stop_step: int,
    shard_size: int,
    seed: int,
) -> tuple[Path, int, int, int]:
    """Filter one split's manifest end-to-end and write the re-packed output.

    Performs two filtering passes: first drops episodes whose halt-reward
    range falls below ``min_halt_reward_range``, then (if requested)
    stratifies by oracle stop step so no bucket exceeds
    ``max_per_stop_step``. Survivors are re-packed into fixed-size shards
    and a fresh manifest is written next to them.

    Returns:
        Tuple of ``(manifest_path, total_before, after_range, final_count)``
        for the caller's summary print.
    """
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    split_name = manifest.get("split", "unknown")
    entries = manifest["entries"]
    feature_names = None
    shard_metadata: Dict[str, Any] = {}
    manifest_metadata: Dict[str, Any] = {}

    kept: List[Dict[str, Any]] = []
    total_before = 0
    started = time.time()

    # Metadata-passthrough keys (below): the budget/maintenance/oracle
    # config baked into the input shards/manifest must survive into the
    # filtered output so downstream materialization and training see the
    # same hyperparameters. Snapshot from the first shard only; all shards
    # in a manifest share the same schema/config.
    for shard_idx, entry in enumerate(entries):
        payload = torch.load(entry["path"], weights_only=False)
        if feature_names is None:
            feature_names = list(payload["feature_names"])
            for key, value in payload.items():
                if key.startswith("maintenance_") or key.startswith("time_") or key in {
                    "oracle_type",
                    "timeout_value",
                    "samples_per_bucket",
                    "budget_seed",
                    "budget_buckets",
                    "reward_scale",
                }:
                    shard_metadata[key] = value
        if not manifest_metadata:
            for key, value in manifest.items():
                if key.startswith("maintenance_") or key.startswith("time_") or key in {
                    "oracle_type",
                    "timeout_value",
                    "samples_per_bucket",
                    "budget_seed",
                    "budget_buckets",
                    "reward_scale",
                }:
                    manifest_metadata[key] = value

        episodes = _extract_episodes(payload)
        total_before += len(episodes)
        # Pass 1: drop trivially-decided episodes (halt rewards nearly
        # constant across the stop steps, so no learning signal).
        for ep in episodes:
            hr = ep["halt_rewards"]
            reward_range = float(hr.max().item() - hr.min().item())
            if reward_range >= min_halt_reward_range:
                ep["_oracle_stop"] = ep["oracle_stop_step"]
                kept.append(ep)

        elapsed = time.time() - started
        print(
            f"[{split_name}] read shard {shard_idx + 1}/{len(entries)} "
            f"total_read={total_before} kept={len(kept)} elapsed_s={elapsed:.1f}",
            flush=True,
        )

    filtered_count = len(kept)
    # Pass 2 (optional): stratify by oracle stop step so no single bucket
    # dominates training. Seeded RNG keeps the subsample reproducible.
    if max_per_stop_step > 0:
        rng = random.Random(seed)
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ep in kept:
            buckets[ep["_oracle_stop"]].append(ep)
        sampled = []
        for stop_step in sorted(buckets):
            bucket = buckets[stop_step]
            if len(bucket) > max_per_stop_step:
                rng.shuffle(bucket)
                bucket = bucket[:max_per_stop_step]
            sampled.extend(bucket)
        # Final shuffle so consecutive shards aren't all the same stop step.
        rng.shuffle(sampled)
        kept = sampled

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    out_entries = []
    for shard_idx in range(0, len(kept), shard_size):
        chunk = kept[shard_idx:shard_idx + shard_size]
        shard_path = split_dir / f"shard_{shard_idx // shard_size:05d}.pt"
        _write_shard(chunk, shard_path, feature_names, shard_metadata)
        out_entries.append({"path": str(shard_path), "num_episodes": len(chunk), "shard_index": shard_idx // shard_size})

    out_manifest_path = output_dir / f"{split_name}_manifest.json"
    with out_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "cts_budgeted_controller_episode_manifest_v3",
                "split": split_name,
                "total_episodes": len(kept),
                "total_before_filter": total_before,
                "total_after_range_filter": filtered_count,
                "min_halt_reward_range": min_halt_reward_range,
                "max_per_stop_step": max_per_stop_step,
                **manifest_metadata,
                "entries": out_entries,
            },
            handle,
            indent=2,
        )

    return out_manifest_path, total_before, filtered_count, len(kept)


def main(config: FilterPackedEpisodesConfig) -> None:
    """CLI entry point: filter the train and validation manifests in one pass."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_manifest in (config.train_manifest, config.validation_manifest):
        path, before, after_range, final = _filter_split(
            split_manifest,
            output_dir,
            config.min_halt_reward_range,
            config.max_per_stop_step,
            config.shard_size,
            config.seed,
        )
        print(f"wrote {path}: {before} -> {after_range} (range filter) -> {final} (stratify)", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(FilterPackedEpisodesConfig, main)
