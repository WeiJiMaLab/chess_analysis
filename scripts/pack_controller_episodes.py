"""Pack controller episode data (snapshot episodes + Bellman Q targets) into shards.

Each raw PretrainExample is expanded into a decision episode (sequence of tree
snapshots), halt rewards and Bellman Q-targets are computed, and the tensorized
snapshot trees are written to shard files alongside the targets.  A JSON manifest
records the shard paths and counts.

Shard format:  cts_controller_episode_shard_v1
  - Two levels of pointer arrays let us slice by episode and by step.
  - Per-step tree data uses LOCAL (0-based) node indices so that the
    collator at train time can reassemble TreeBatch objects by adding
    offsets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from controller_oracle import compute_oracle_policy, has_strong_optimal_margins
from cts_episode_envs import build_trimmed_decision_episode_with_halt_rewards
from cts_pretrain import PretrainExample, TeacherSearchConfig
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from tensorizer import TreeTensorizer


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


def _quality_config(args: argparse.Namespace) -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=args.max_depth,
        search_budget=args.search_budget,
        c_puct=args.c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="compute_advantage_controller_v1",
        search_config_id="compute_advantage_controller",
    )


def _bellman_q_targets(
    halt_rewards: List[float],
    continue_cost: float,
) -> tuple[torch.Tensor, int, float]:
    policy = compute_oracle_policy(halt_rewards, continue_cost)
    targets: List[List[float]] = []
    for index, halt_reward in enumerate(halt_rewards):
        if index + 1 < len(halt_rewards):
            continue_value = -continue_cost + policy.values[index + 1]
        else:
            continue_value = -continue_cost + float(halt_reward)
        targets.append([continue_value, float(halt_reward)])
    return (
        torch.tensor(targets, dtype=torch.float32),
        policy.optimal_stop_step,
        float(policy.values[0]),
    )


def _read_manifest(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        examples = [Path(line.strip()) for line in handle if line.strip()]
    if not examples:
        raise ValueError(f"No example paths found in manifest: {path}")
    return examples


def _pack_split(
    manifest_path: Path,
    output_root: Path,
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    reward_scale: float,
    min_decision_margin: float,
    shard_size: int,
    num_workers: int,
    log_interval: int,
) -> tuple[Path, int, int]:
    split_name = manifest_path.stem.replace("_manifest", "")
    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    example_paths = _read_manifest(manifest_path)
    schema = _feature_schema()
    feature_names = tuple(schema.feature_names)
    start_time = time.time()
    entries: List[dict] = []
    total_episodes = 0
    total_skipped = 0
    total_shards = (len(example_paths) + shard_size - 1) // shard_size

    for shard_index, shard_start in enumerate(range(0, len(example_paths), shard_size)):
        shard_paths = example_paths[shard_start: shard_start + shard_size]

        tasks = [
            (
                str(path),
                quality_config.max_depth,
                quality_config.search_budget,
                quality_config.c_puct,
                quality_config.target_normalization_version,
                quality_config.search_config_id,
                continue_cost,
                reward_scale,
                min_decision_margin,
                feature_names,
            )
            for path in shard_paths
        ]

        results: List[Optional[dict]]
        if num_workers <= 0:
            results = []
            for completed_in_shard, task in enumerate(tasks, start=1):
                results.append(_process_one_task(task))
                if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                    elapsed = time.time() - start_time
                    accepted_so_far = total_episodes + sum(1 for r in results if r is not None)
                    skipped_so_far = total_skipped + sum(1 for r in results if r is None)
                    print(
                        f"split={split_name} shard={shard_index + 1}/{total_shards} "
                        f"shard_progress={completed_in_shard}/{len(tasks)} "
                        f"accepted={accepted_so_far} skipped={skipped_so_far} "
                        f"elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
        else:
            results = [None] * len(tasks)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_process_one_task, task): idx
                    for idx, task in enumerate(tasks)
                }
                for completed_in_shard, future in enumerate(as_completed(futures), start=1):
                    results[futures[future]] = future.result()
                    if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                        elapsed = time.time() - start_time
                        accepted_so_far = total_episodes + sum(1 for r in results if r is not None)
                        skipped_so_far = total_skipped + sum(1 for r in results if r is None)
                        print(
                            f"split={split_name} shard={shard_index + 1}/{total_shards} "
                            f"shard_progress={completed_in_shard}/{len(tasks)} "
                            f"accepted={accepted_so_far} skipped={skipped_so_far} "
                            f"elapsed_s={elapsed:.1f}",
                            flush=True,
                        )

        episode_step_ptr = [0]
        step_node_ptr = [0]
        step_edge_ptr = [0]
        all_node_features: List[torch.Tensor] = []
        all_parent_index: List[torch.Tensor] = []
        all_edge_parent: List[torch.Tensor] = []
        all_edge_child: List[torch.Tensor] = []
        all_edge_slot: List[torch.Tensor] = []
        all_depth: List[torch.Tensor] = []
        all_halt_rewards: List[torch.Tensor] = []
        all_q_targets: List[torch.Tensor] = []
        all_target_advantages: List[torch.Tensor] = []
        all_oracle_stop_steps: List[int] = []
        all_oracle_values: List[float] = []
        source_paths: List[str] = []
        shard_episodes = 0

        for episode_data in results:
            if episode_data is None:
                total_skipped += 1
            else:
                num_steps = episode_data["num_steps"]
                episode_step_ptr.append(episode_step_ptr[-1] + num_steps)

                for step_nf, step_pi, step_ep, step_ec, step_es, step_d in zip(
                    episode_data["step_node_features"],
                    episode_data["step_parent_index"],
                    episode_data["step_edge_parent"],
                    episode_data["step_edge_child"],
                    episode_data["step_edge_slot"],
                    episode_data["step_depth"],
                ):
                    all_node_features.append(step_nf)
                    all_parent_index.append(step_pi)
                    all_edge_parent.append(step_ep)
                    all_edge_child.append(step_ec)
                    all_edge_slot.append(step_es)
                    all_depth.append(step_d)
                    step_node_ptr.append(step_node_ptr[-1] + step_nf.shape[0])
                    step_edge_ptr.append(step_edge_ptr[-1] + step_ep.shape[0])

                all_halt_rewards.append(episode_data["halt_rewards"])
                all_q_targets.append(episode_data["q_targets"])
                all_target_advantages.append(episode_data["target_advantages"])
                all_oracle_stop_steps.append(episode_data["oracle_stop_step"])
                all_oracle_values.append(episode_data["oracle_value"])
                source_paths.append(episode_data["source_path"])
                shard_episodes += 1

        if shard_episodes == 0:
            continue

        shard_path = split_output_dir / f"shard_{shard_index:05d}.pt"
        payload = {
            "format": "cts_controller_episode_shard_v1",
            "num_episodes": shard_episodes,
            "feature_names": list(schema.feature_names),
            "continue_cost": continue_cost,
            "reward_scale": reward_scale,
            "episode_step_ptr": torch.tensor(episode_step_ptr, dtype=torch.long),
            "step_node_ptr": torch.tensor(step_node_ptr, dtype=torch.long),
            "step_edge_ptr": torch.tensor(step_edge_ptr, dtype=torch.long),
            "node_features": torch.cat(all_node_features, dim=0),
            "parent_index": torch.cat(all_parent_index, dim=0),
            "edge_parent": torch.cat(all_edge_parent, dim=0) if all_edge_parent else torch.empty(0, dtype=torch.long),
            "edge_child": torch.cat(all_edge_child, dim=0) if all_edge_child else torch.empty(0, dtype=torch.long),
            "edge_slot": torch.cat(all_edge_slot, dim=0) if all_edge_slot else torch.empty(0, dtype=torch.long),
            "depth": torch.cat(all_depth, dim=0),
            "halt_rewards": torch.cat(all_halt_rewards, dim=0),
            "q_targets": torch.cat(all_q_targets, dim=0),
            "target_advantages": torch.cat(all_target_advantages, dim=0),
            "oracle_stop_steps": torch.tensor(all_oracle_stop_steps, dtype=torch.long),
            "oracle_values": torch.tensor(all_oracle_values, dtype=torch.float32),
            "source_paths": source_paths,
        }
        torch.save(payload, shard_path)
        entries.append({
            "path": str(shard_path),
            "num_episodes": shard_episodes,
            "shard_index": shard_index,
        })
        total_episodes += shard_episodes

    packed_manifest_path = output_root / f"{split_name}_manifest.json"
    with packed_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "cts_controller_episode_manifest_v1",
                "split": split_name,
                "total_episodes": total_episodes,
                "total_skipped": total_skipped,
                "continue_cost": continue_cost,
                "reward_scale": reward_scale,
                "min_decision_margin": min_decision_margin,
                "entries": entries,
            },
            handle,
            indent=2,
        )

    return packed_manifest_path, total_episodes, total_skipped


def _process_one_task(
    task: Tuple[str, int, int, float, str, str, float, float, float, Tuple[str, ...]],
) -> Optional[dict]:
    """Top-level picklable worker: unpack task tuple, load example, build episode, tensorize."""
    (
        path_str, max_depth, search_budget, c_puct,
        target_normalization_version, search_config_id,
        continue_cost, reward_scale, min_decision_margin,
        feature_names,
    ) = task

    quality_config = TeacherSearchConfig(
        max_depth=max_depth,
        search_budget=search_budget,
        c_puct=c_puct,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version=target_normalization_version,
        search_config_id=search_config_id,
    )
    schema = NodeFeatureSchema(feature_names)
    tensorizer = TreeTensorizer(schema=schema, device="cpu")

    try:
        example = torch.load(path_str, weights_only=False)
        if not isinstance(example, PretrainExample):
            return None
        episode, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(
            example, quality_config,
        )
    except (ValueError, Exception):
        return None

    scaled_rewards = [reward_scale * r for r in halt_rewards]
    if not has_strong_optimal_margins(
        scaled_rewards,
        continue_cost=continue_cost,
        min_decision_margin=min_decision_margin,
    ):
        return None

    q_targets, oracle_stop_step, oracle_value = _bellman_q_targets(
        scaled_rewards, continue_cost,
    )
    target_advantages = q_targets[:, 0] - q_targets[:, 1]

    step_node_features = []
    step_parent_index = []
    step_edge_parent = []
    step_edge_child = []
    step_edge_slot = []
    step_depth = []

    for snapshot in episode.snapshots:
        batch = tensorizer.tensorize_tree(snapshot, validate=False)
        step_node_features.append(batch.node_features)
        step_parent_index.append(batch.parent_index)
        step_edge_parent.append(batch.edge_parent)
        step_edge_child.append(batch.edge_child)
        step_edge_slot.append(batch.edge_slot)
        step_depth.append(batch.depth)

    return {
        "num_steps": len(episode.snapshots),
        "step_node_features": step_node_features,
        "step_parent_index": step_parent_index,
        "step_edge_parent": step_edge_parent,
        "step_edge_child": step_edge_child,
        "step_edge_slot": step_edge_slot,
        "step_depth": step_depth,
        "halt_rewards": torch.tensor(scaled_rewards, dtype=torch.float32),
        "q_targets": q_targets,
        "target_advantages": target_advantages,
        "oracle_stop_step": oracle_stop_step,
        "oracle_value": oracle_value,
        "source_path": path_str,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack controller episode data (snapshot episodes + Bellman targets) into shards.",
    )
    parser.add_argument(
        "--split-root",
        default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split",
    )
    parser.add_argument(
        "--output-root",
        default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed",
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--min-decision-margin", type=float, default=0.0)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"

    if args.clear and output_root.exists():
        for child in output_root.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(output_root.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()

    output_root.mkdir(parents=True, exist_ok=True)
    quality_config = _quality_config(args)

    print(f"split_root={split_root}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"num_workers={args.num_workers}", flush=True)
    print(f"continue_cost={args.continue_cost}", flush=True)
    print(f"reward_scale={args.reward_scale}", flush=True)
    print(f"min_decision_margin={args.min_decision_margin}", flush=True)

    train_manifest_out, train_count, train_skipped = _pack_split(
        train_manifest, output_root, quality_config,
        args.continue_cost, args.reward_scale, args.min_decision_margin,
        args.shard_size, args.num_workers, args.log_interval,
    )
    validation_manifest_out, val_count, val_skipped = _pack_split(
        validation_manifest, output_root, quality_config,
        args.continue_cost, args.reward_scale, args.min_decision_margin,
        args.shard_size, args.num_workers, args.log_interval,
    )

    print(f"train_manifest={train_manifest_out}")
    print(f"validation_manifest={validation_manifest_out}")
    print(f"train_episodes={train_count} train_skipped={train_skipped}")
    print(f"validation_episodes={val_count} validation_skipped={val_skipped}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
