"""Pack budget-augmented controller episodes into tensorized shards."""
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

import numpy as np
import torch

from budgeted_controller_oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    budgeted_oracle_metadata,
    compute_budgeted_oracle,
    deterministic_starting_budgets,
    has_strong_budgeted_margins,
)
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


def _oracle_config(args: argparse.Namespace) -> BudgetedOracleConfig:
    return BudgetedOracleConfig(
        maintenance_scale=args.maintenance_scale,
        maintenance_ref_nodes=args.maintenance_ref_nodes,
        maintenance_exponent=args.maintenance_exponent,
        time_lambda=args.time_lambda,
        time_p=args.time_p,
        time_tau=args.time_tau,
        time_delta=args.time_delta,
        timeout_value=args.timeout_value,
        budget_buckets=(
            BudgetBucket("scramble", args.scramble_min_time, args.scramble_max_time),
            BudgetBucket("medium-small", args.medium_small_min_time, args.medium_small_max_time),
            BudgetBucket("medium-large", args.medium_large_min_time, args.medium_large_max_time),
            BudgetBucket("large", args.large_min_time, args.large_max_time),
            BudgetBucket("very-large", args.very_large_min_time, args.very_large_max_time),
        ),
        samples_per_bucket=args.samples_per_bucket,
        seed=args.seed,
    )


def _read_manifest(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        examples = [Path(line.strip()) for line in handle if line.strip()]
    if not examples:
        raise ValueError(f"No example paths found in manifest: {path}")
    return examples


def _halt_reward_range(halt_rewards: List[float]) -> float:
    if not halt_rewards:
        raise ValueError("halt_rewards must be non-empty.")
    return float(max(halt_rewards) - min(halt_rewards))


def _pack_split(
    manifest_path: Path,
    output_root: Path,
    quality_config: TeacherSearchConfig,
    oracle_config: BudgetedOracleConfig,
    reward_scale: float,
    min_halt_reward_range: float,
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
                reward_scale,
                min_halt_reward_range,
                min_decision_margin,
                feature_names,
                oracle_config,
            )
            for path in shard_paths
        ]

        results: List[Optional[List[dict]]]
        if num_workers <= 0:
            results = []
            for completed_in_shard, task in enumerate(tasks, start=1):
                results.append(_process_one_task(task))
                if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                    elapsed = time.time() - start_time
                    accepted_so_far = total_episodes + sum(len(r) for r in results if r is not None)
                    skipped_so_far = total_skipped + sum(1 for r in results if not r)
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
                futures = {executor.submit(_process_one_task, task): idx for idx, task in enumerate(tasks)}
                for completed_in_shard, future in enumerate(as_completed(futures), start=1):
                    results[futures[future]] = future.result()
                    if completed_in_shard % log_interval == 0 or completed_in_shard == len(tasks):
                        elapsed = time.time() - start_time
                        accepted_so_far = total_episodes + sum(len(r) for r in results if r is not None)
                        skipped_so_far = total_skipped + sum(1 for r in results if not r)
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
        all_target_advantages: List[torch.Tensor] = []
        all_tree_sizes: List[torch.Tensor] = []
        all_time_budgets: List[torch.Tensor] = []
        all_oracle_stop_steps: List[int] = []
        all_oracle_values: List[float] = []
        all_starting_budgets: List[int] = []
        all_bucket_indices: List[int] = []
        all_bucket_names: List[str] = []
        all_episode_keys: List[str] = []
        all_source_paths: List[str] = []
        shard_episodes = 0

        for raw_result in results:
            if not raw_result:
                total_skipped += 1
                continue

            for augmented_result in raw_result:
                episode_data = _numpy_to_torch(augmented_result)
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
                all_target_advantages.append(episode_data["target_advantages"])
                all_tree_sizes.append(episode_data["tree_sizes"])
                all_time_budgets.append(episode_data["time_budgets"])
                all_oracle_stop_steps.append(episode_data["oracle_stop_step"])
                all_oracle_values.append(episode_data["oracle_value"])
                all_starting_budgets.append(episode_data["starting_budget"])
                all_bucket_indices.append(episode_data["bucket_index"])
                all_bucket_names.append(episode_data["bucket_name"])
                all_episode_keys.append(episode_data["episode_key"])
                all_source_paths.append(episode_data["source_path"])
                shard_episodes += 1

        if shard_episodes == 0:
            continue

        shard_path = split_output_dir / f"shard_{shard_index:05d}.pt"
        payload = {
            "format": "cts_budgeted_controller_episode_shard_v2",
            "num_episodes": shard_episodes,
            "feature_names": list(schema.feature_names),
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
            "target_advantages": torch.cat(all_target_advantages, dim=0),
            "tree_sizes": torch.cat(all_tree_sizes, dim=0),
            "time_budgets": torch.cat(all_time_budgets, dim=0),
            "oracle_stop_steps": torch.tensor(all_oracle_stop_steps, dtype=torch.long),
            "oracle_values": torch.tensor(all_oracle_values, dtype=torch.float32),
            "starting_budgets": torch.tensor(all_starting_budgets, dtype=torch.long),
            "budget_bucket_indices": torch.tensor(all_bucket_indices, dtype=torch.long),
            "budget_bucket_names": all_bucket_names,
            "episode_keys": all_episode_keys,
            "source_paths": all_source_paths,
            **budgeted_oracle_metadata(oracle_config),
        }
        torch.save(payload, shard_path)
        entries.append({"path": str(shard_path), "num_episodes": shard_episodes, "shard_index": shard_index})
        total_episodes += shard_episodes

    packed_manifest_path = output_root / f"{split_name}_manifest.json"
    with packed_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format": "cts_budgeted_controller_episode_manifest_v2",
                "split": split_name,
                "total_episodes": total_episodes,
                "total_skipped": total_skipped,
                "reward_scale": reward_scale,
                "min_halt_reward_range": min_halt_reward_range,
                "min_decision_margin": min_decision_margin,
                **budgeted_oracle_metadata(oracle_config),
                "entries": entries,
            },
            handle,
            indent=2,
        )

    return packed_manifest_path, total_episodes, total_skipped


def _process_one_task(
    task: Tuple[str, int, int, float, str, str, float, float, float, Tuple[str, ...], BudgetedOracleConfig],
) -> Optional[List[dict]]:
    (
        path_str,
        max_depth,
        search_budget,
        c_puct,
        target_normalization_version,
        search_config_id,
        reward_scale,
        min_halt_reward_range,
        min_decision_margin,
        feature_names,
        oracle_config,
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
    tensorizer = TreeTensorizer(schema=NodeFeatureSchema(feature_names), device="cpu")

    try:
        example = torch.load(path_str, weights_only=False)
        if not isinstance(example, PretrainExample):
            return None
        episode, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(example, quality_config)
    except (ValueError, Exception):
        return None

    if not episode.snapshots:
        return None

    step_node_features = []
    step_parent_index = []
    step_edge_parent = []
    step_edge_child = []
    step_edge_slot = []
    step_depth = []
    tree_sizes = []

    for snapshot in episode.snapshots:
        batch = tensorizer.tensorize_tree(snapshot, validate=False)
        step_node_features.append(batch.node_features.numpy())
        step_parent_index.append(batch.parent_index.numpy())
        step_edge_parent.append(batch.edge_parent.numpy())
        step_edge_child.append(batch.edge_child.numpy())
        step_edge_slot.append(batch.edge_slot.numpy())
        step_depth.append(batch.depth.numpy())
        tree_sizes.append(snapshot.num_nodes())

    scaled_rewards = [float(reward_scale * reward) for reward in halt_rewards]
    if _halt_reward_range(scaled_rewards) < min_halt_reward_range:
        return None

    packed_episodes: List[dict] = []
    for sampled_budget in deterministic_starting_budgets(path_str, oracle_config):
        if not has_strong_budgeted_margins(
            scaled_rewards,
            tree_sizes,
            sampled_budget.starting_budget,
            oracle_config,
            min_decision_margin,
        ):
            continue

        policy = compute_budgeted_oracle(
            scaled_rewards,
            tree_sizes,
            sampled_budget.starting_budget,
            oracle_config,
        )
        max_steps = len(policy.halt_rewards)
        episode_key = f"{path_str}#bucket={sampled_budget.bucket_name}#budget={sampled_budget.starting_budget}"
        packed_episodes.append(
            {
                "num_steps": max_steps,
                "step_node_features": step_node_features[:max_steps],
                "step_parent_index": step_parent_index[:max_steps],
                "step_edge_parent": step_edge_parent[:max_steps],
                "step_edge_child": step_edge_child[:max_steps],
                "step_edge_slot": step_edge_slot[:max_steps],
                "step_depth": step_depth[:max_steps],
                "halt_rewards": np.asarray(policy.halt_rewards, dtype=np.float32),
                "target_advantages": np.asarray(policy.target_advantages, dtype=np.float32),
                "tree_sizes": np.asarray(policy.tree_sizes, dtype=np.int64),
                "time_budgets": np.asarray(policy.time_budgets, dtype=np.int64),
                "oracle_stop_step": policy.optimal_stop_step,
                "oracle_value": policy.oracle_value,
                "starting_budget": policy.starting_budget,
                "bucket_index": sampled_budget.bucket_index,
                "bucket_name": sampled_budget.bucket_name,
                "episode_key": episode_key,
                "source_path": path_str,
            }
        )

    return packed_episodes or None


def _numpy_to_torch(result: dict) -> dict:
    return {
        "num_steps": result["num_steps"],
        "step_node_features": [torch.from_numpy(a) for a in result["step_node_features"]],
        "step_parent_index": [torch.from_numpy(a) for a in result["step_parent_index"]],
        "step_edge_parent": [torch.from_numpy(a) for a in result["step_edge_parent"]],
        "step_edge_child": [torch.from_numpy(a) for a in result["step_edge_child"]],
        "step_edge_slot": [torch.from_numpy(a) for a in result["step_edge_slot"]],
        "step_depth": [torch.from_numpy(a) for a in result["step_depth"]],
        "halt_rewards": torch.from_numpy(result["halt_rewards"]),
        "target_advantages": torch.from_numpy(result["target_advantages"]),
        "tree_sizes": torch.from_numpy(result["tree_sizes"]),
        "time_budgets": torch.from_numpy(result["time_budgets"]),
        "oracle_stop_step": result["oracle_stop_step"],
        "oracle_value": result["oracle_value"],
        "starting_budget": result["starting_budget"],
        "bucket_index": result["bucket_index"],
        "bucket_name": result["bucket_name"],
        "episode_key": result["episode_key"],
        "source_path": result["source_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack budget-aware controller episode data into shards.")
    parser.add_argument("--split-root", default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split")
    parser.add_argument("--output-root", default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--min-halt-reward-range", type=float, default=0.0)
    parser.add_argument("--min-decision-margin", type=float, default=0.0)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--maintenance-scale", type=float, default=0.0025)
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
    oracle_config = _oracle_config(args)

    print(f"split_root={split_root}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"num_workers={args.num_workers}", flush=True)
    print(f"reward_scale={args.reward_scale}", flush=True)
    print(f"min_halt_reward_range={args.min_halt_reward_range}", flush=True)
    print(f"min_decision_margin={args.min_decision_margin}", flush=True)
    print(json.dumps(budgeted_oracle_metadata(oracle_config), sort_keys=True), flush=True)

    train_manifest_out, train_count, train_skipped = _pack_split(
        train_manifest,
        output_root,
        quality_config,
        oracle_config,
        args.reward_scale,
        args.min_halt_reward_range,
        args.min_decision_margin,
        args.shard_size,
        args.num_workers,
        args.log_interval,
    )
    validation_manifest_out, val_count, val_skipped = _pack_split(
        validation_manifest,
        output_root,
        quality_config,
        oracle_config,
        args.reward_scale,
        args.min_halt_reward_range,
        args.min_decision_margin,
        args.shard_size,
        args.num_workers,
        args.log_interval,
    )

    print(f"train_manifest={train_manifest_out}")
    print(f"validation_manifest={validation_manifest_out}")
    print(f"train_episodes={train_count} train_skipped={train_skipped}")
    print(f"validation_episodes={val_count} validation_skipped={val_skipped}")
    print(f"output_root={output_root}")


if __name__ == "__main__":
    main()
