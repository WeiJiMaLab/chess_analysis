from __future__ import annotations

import argparse
import json
import random
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from controller_oracle import compute_oracle_policy, has_strong_optimal_margins
from cts_pretrain import load_encoder_checkpoint
from GNN import TreeEncoderOutput, TreeNN
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from supervised_branch import (
    TeacherSearchConfig,
    build_trimmed_decision_episode_with_halt_rewards,
    load_raw_pretrain_example_paths,
)
from tensorizer import TreeBatch, TreeTensorizer
from tree import SearchTree


@dataclass(frozen=True)
class FittedQEpisode:
    path: str
    snapshots: List[SearchTree]
    halt_rewards: List[float]
    q_targets: torch.Tensor
    oracle_stop_step: int
    oracle_value: float


@dataclass(frozen=True)
class FittedQBatch:
    tree_batch: TreeBatch
    q_targets: torch.Tensor
    paths: List[str]
    path_lengths: List[int]
    halt_rewards: List[List[float]]
    oracle_stop_steps: List[int]
    oracle_values: List[float]
    skipped: int


@dataclass(frozen=True)
class OracleActionNowEpisode:
    path: str
    features: torch.Tensor
    target_advantages: torch.Tensor
    halt_rewards: List[float]
    oracle_stop_step: int
    oracle_value: float


@dataclass(frozen=True)
class MaterializedAdvantageEpisode:
    path: str
    features: torch.Tensor
    target_advantages: torch.Tensor
    halt_rewards: List[float]
    oracle_stop_step: int
    oracle_value: float


@dataclass(frozen=True)
class AdvantageMetrics:
    advantage_mse: float
    mean_abs_advantage_error: float
    sign_accuracy: float
    examples: int


@dataclass(frozen=True)
class GreedyPolicyMetrics:
    exact_stop_step_accuracy: float
    first_action_accuracy: float
    average_return: float
    average_oracle_value: float
    average_regret: float
    average_expansions: float
    evaluated_episodes: int
    skipped_episodes: int


class ComputeAdvantageTreeSearchModel(nn.Module):
    def __init__(
        self,
        *,
        k: int,
        node_feat: int,
        device: str,
        node_embed_hidden: int,
        d_embed: int,
        d_message: int,
        n_heads: int,
        d_att: int,
        q_hidden: int,
    ) -> None:
        super().__init__()
        self.encoder = TreeNN(
            k=k,
            node_feat=node_feat,
            device=device,
            node_embed_hidden=node_embed_hidden,
            d_embed=d_embed,
            d_message=d_message,
            n_heads=n_heads,
            d_att=d_att,
        )
        resolved_device = self.encoder.device
        self.advantage_head = nn.Sequential(
            nn.Linear(self.encoder.d_embed, q_hidden, device=resolved_device),
            nn.ReLU(),
            nn.Linear(q_hidden, 1, device=resolved_device),
        )

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, tree_batch: TreeBatch) -> torch.Tensor:
        encoded: TreeEncoderOutput = self.encoder(tree_batch)
        return self.advantage_head(encoded.root_states).squeeze(-1)


class FixedFeatureEncoder(nn.Module):
    def __init__(self, node_feat: int, device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.d_embed = node_feat

    def forward(self, tree_batch: TreeBatch) -> TreeEncoderOutput:
        node_states = tree_batch.node_features.to(self.device)
        root_states = node_states[tree_batch.root_index.to(self.device)]
        return TreeEncoderOutput(node_states=node_states, root_states=root_states)


class LinearComputeAdvantageModel(nn.Module):
    def __init__(self, node_feat: int, device: str) -> None:
        super().__init__()
        resolved_device = torch.device(device)
        self.encoder = FixedFeatureEncoder(node_feat=node_feat, device=device)
        self.advantage_head = nn.Linear(node_feat, 1, device=resolved_device)

    def freeze_encoder(self) -> None:
        return None

    def forward(self, tree_batch: TreeBatch) -> torch.Tensor:
        encoded = self.encoder(tree_batch)
        return self.advantage_head(encoded.root_states).squeeze(-1)


class FittedQEpisodeDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        quality_config: TeacherSearchConfig,
        continue_cost: float,
        reward_scale: float,
        representation: str,
        min_decision_margin: float,
    ) -> None:
        self.paths = list(paths)
        self.quality_config = quality_config
        self.continue_cost = float(continue_cost)
        self.reward_scale = float(reward_scale)
        self.representation = representation
        self.min_decision_margin = float(min_decision_margin)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> FittedQEpisode | None:
        path = self.paths[index]
        try:
            example = torch.load(path, weights_only=False)
            episode, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(
                example,
                self.quality_config,
            )
        except ValueError:
            return None

        scaled_rewards = [self.reward_scale * reward for reward in halt_rewards]
        if not has_strong_optimal_margins(
            scaled_rewards,
            continue_cost=self.continue_cost,
            min_decision_margin=self.min_decision_margin,
        ):
            return None
        q_targets, oracle_stop_step, oracle_value = bellman_q_targets(
            scaled_rewards,
            self.continue_cost,
        )
        snapshots = list(episode.snapshots)
        if self.representation == "oracle-action-now":
            snapshots = [_oracle_action_now_tree(action) for action in _oracle_actions_from_targets(q_targets)]
        elif self.representation != "tree":
            raise ValueError(f"Unsupported representation: {self.representation}")
        return FittedQEpisode(
            path=path,
            snapshots=snapshots,
            halt_rewards=scaled_rewards,
            q_targets=q_targets,
            oracle_stop_step=oracle_stop_step,
            oracle_value=oracle_value,
        )


class FittedQCollator:
    def __init__(self, tensorizer: TreeTensorizer) -> None:
        self.tensorizer = tensorizer

    def __call__(self, episodes: Sequence[FittedQEpisode | None]) -> FittedQBatch | None:
        valid = [episode for episode in episodes if episode is not None]
        if not valid:
            return None

        trees: List[SearchTree] = []
        q_targets = []
        paths = []
        path_lengths = []
        halt_rewards = []
        oracle_stop_steps = []
        oracle_values = []
        for episode in valid:
            trees.extend(episode.snapshots)
            q_targets.append(episode.q_targets)
            paths.append(episode.path)
            path_lengths.append(len(episode.snapshots))
            halt_rewards.append(episode.halt_rewards)
            oracle_stop_steps.append(episode.oracle_stop_step)
            oracle_values.append(episode.oracle_value)

        return FittedQBatch(
            tree_batch=self.tensorizer.tensorize_forest(trees, validate=False),
            q_targets=torch.cat(q_targets, dim=0),
            paths=paths,
            path_lengths=path_lengths,
            halt_rewards=halt_rewards,
            oracle_stop_steps=oracle_stop_steps,
            oracle_values=oracle_values,
            skipped=len(episodes) - len(valid),
        )


@dataclass(frozen=True)
class PackedControllerEpisode:
    """A single episode extracted from a packed shard, ready for collation."""
    path: str
    step_node_features: List[torch.Tensor]
    step_parent_index: List[torch.Tensor]
    step_edge_parent: List[torch.Tensor]
    step_edge_child: List[torch.Tensor]
    step_edge_slot: List[torch.Tensor]
    step_depth: List[torch.Tensor]
    halt_rewards: List[float]
    q_targets: torch.Tensor
    oracle_stop_step: int
    oracle_value: float


class PackedControllerEpisodeDataset(Dataset):
    """Loads pre-packed controller episode shards and serves individual episodes."""

    def __init__(self, manifest_path: str) -> None:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed shard entries in manifest: {manifest_path}")

        self.shard_paths: List[str] = []
        self.cumulative_sizes: List[int] = []
        total = 0
        for entry in entries:
            num_episodes = int(entry["num_episodes"])
            if num_episodes <= 0:
                continue
            total += num_episodes
            self.shard_paths.append(entry["path"])
            self.cumulative_sizes.append(total)

        if not self.shard_paths:
            raise ValueError(f"All shards empty in manifest: {manifest_path}")

        self._loaded_shard_index: Optional[int] = None
        self._loaded_payload: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _load_shard(self, shard_index: int) -> Dict[str, Any]:
        if self._loaded_shard_index != shard_index:
            payload = torch.load(self.shard_paths[shard_index], weights_only=False)
            if not isinstance(payload, dict) or payload.get("format") != "cts_controller_episode_shard_v1":
                raise ValueError(f"Unexpected shard format: {self.shard_paths[shard_index]}")
            self._loaded_shard_index = shard_index
            self._loaded_payload = payload
        assert self._loaded_payload is not None
        return self._loaded_payload

    def __getitem__(self, index: int) -> PackedControllerEpisode:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        episode_offset = index - shard_start

        payload = self._load_shard(shard_index)

        episode_step_ptr = payload["episode_step_ptr"]
        step_begin = int(episode_step_ptr[episode_offset].item())
        step_end = int(episode_step_ptr[episode_offset + 1].item())
        num_steps = step_end - step_begin

        step_node_ptr = payload["step_node_ptr"]
        step_edge_ptr = payload["step_edge_ptr"]
        feature_names = tuple(payload["feature_names"])

        step_nf = []
        step_pi = []
        step_ep = []
        step_ec = []
        step_es = []
        step_d = []

        for s in range(step_begin, step_end):
            n_start = int(step_node_ptr[s].item())
            n_end = int(step_node_ptr[s + 1].item())
            e_start = int(step_edge_ptr[s].item())
            e_end = int(step_edge_ptr[s + 1].item())

            step_nf.append(payload["node_features"][n_start:n_end])
            step_pi.append(payload["parent_index"][n_start:n_end])
            step_d.append(payload["depth"][n_start:n_end])
            step_ep.append(payload["edge_parent"][e_start:e_end])
            step_ec.append(payload["edge_child"][e_start:e_end])
            step_es.append(payload["edge_slot"][e_start:e_end])

        halt_start = int(episode_step_ptr[episode_offset].item())
        halt_end = int(episode_step_ptr[episode_offset + 1].item())
        halt_rewards_tensor = payload["halt_rewards"][halt_start:halt_end]
        q_targets = payload["q_targets"][halt_start:halt_end]

        return PackedControllerEpisode(
            path=payload["source_paths"][episode_offset],
            step_node_features=step_nf,
            step_parent_index=step_pi,
            step_edge_parent=step_ep,
            step_edge_child=step_ec,
            step_edge_slot=step_es,
            step_depth=step_d,
            halt_rewards=halt_rewards_tensor.tolist(),
            q_targets=q_targets,
            oracle_stop_step=int(payload["oracle_stop_steps"][episode_offset].item()),
            oracle_value=float(payload["oracle_values"][episode_offset].item()),
        )


class PackedControllerCollator:
    """Collates PackedControllerEpisode objects into FittedQBatch."""

    def __call__(self, episodes: Sequence[PackedControllerEpisode]) -> FittedQBatch | None:
        if not episodes:
            return None

        all_node_features = []
        all_parent_index = []
        all_edge_parent = []
        all_edge_child = []
        all_edge_slot = []
        all_depth = []
        all_tree_index = []
        root_index = []
        child_ptr_parts = []

        q_targets_list = []
        paths = []
        path_lengths = []
        halt_rewards = []
        oracle_stop_steps = []
        oracle_values = []

        node_offset = 0
        tree_idx = 0

        for episode in episodes:
            for step_nf, step_pi, step_ep, step_ec, step_es, step_d in zip(
                episode.step_node_features,
                episode.step_parent_index,
                episode.step_edge_parent,
                episode.step_edge_child,
                episode.step_edge_slot,
                episode.step_depth,
            ):
                num_nodes = step_nf.shape[0]

                all_node_features.append(step_nf)
                all_depth.append(step_d)
                all_tree_index.append(torch.full((num_nodes,), tree_idx, dtype=torch.long))

                root_index.append(node_offset)

                pi = step_pi.clone()
                has_parent = pi >= 0
                pi[has_parent] += node_offset
                all_parent_index.append(pi)

                if step_ep.numel() > 0:
                    all_edge_parent.append(step_ep + node_offset)
                    all_edge_child.append(step_ec + node_offset)
                    all_edge_slot.append(step_es)

                node_offset += num_nodes
                tree_idx += 1

            q_targets_list.append(episode.q_targets)
            paths.append(episode.path)
            path_lengths.append(len(episode.step_node_features))
            halt_rewards.append(episode.halt_rewards)
            oracle_stop_steps.append(episode.oracle_stop_step)
            oracle_values.append(episode.oracle_value)

        node_features = torch.cat(all_node_features, dim=0)
        parent_index = torch.cat(all_parent_index, dim=0)
        depth = torch.cat(all_depth, dim=0)
        tree_index = torch.cat(all_tree_index, dim=0)
        root_index_tensor = torch.tensor(root_index, dtype=torch.long)

        if all_edge_parent:
            edge_parent = torch.cat(all_edge_parent, dim=0)
            edge_child = torch.cat(all_edge_child, dim=0)
            edge_slot = torch.cat(all_edge_slot, dim=0)
            children_index = edge_child
            child_counts = torch.bincount(edge_parent, minlength=node_features.shape[0])
            child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long)
            child_ptr[1:] = torch.cumsum(child_counts, dim=0)
        else:
            edge_parent = torch.empty(0, dtype=torch.long)
            edge_child = torch.empty(0, dtype=torch.long)
            edge_slot = torch.empty(0, dtype=torch.long)
            children_index = torch.empty(0, dtype=torch.long)
            child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long)

        tree_batch = TreeBatch(
            node_features=node_features,
            tree_index=tree_index,
            parent_index=parent_index,
            root_index=root_index_tensor,
            edge_parent=edge_parent,
            edge_child=edge_child,
            edge_slot=edge_slot,
            child_ptr=child_ptr,
            children_index=children_index,
            depth=depth,
            feature_names=("packed",),
            batch_size=tree_idx,
            num_nodes=int(node_features.shape[0]),
            num_edges=int(edge_parent.shape[0]),
        )

        return FittedQBatch(
            tree_batch=tree_batch,
            q_targets=torch.cat(q_targets_list, dim=0),
            paths=paths,
            path_lengths=path_lengths,
            halt_rewards=halt_rewards,
            oracle_stop_steps=oracle_stop_steps,
            oracle_values=oracle_values,
            skipped=0,
        )


def bellman_q_targets(
    halt_rewards: Sequence[float],
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


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


def _oracle_action_now_schema() -> NodeFeatureSchema:
    return NodeFeatureSchema.from_ordered_features(
        ["action_0", "action_1"],
        defaults={"action_0": 0.0, "action_1": 0.0},
    )


def _oracle_action_now_tree(action: int) -> SearchTree:
    if action not in (0, 1):
        raise ValueError("oracle-action-now representation requires action in {0, 1}.")
    tree = SearchTree()
    tree.create_root(
        "oracle-action-now-root",
        {
            "action_0": 1.0 if action == 0 else 0.0,
            "action_1": 1.0 if action == 1 else 0.0,
        },
    )
    return tree


def _oracle_actions_from_targets(q_targets: torch.Tensor) -> List[int]:
    target_advantages = _target_advantages(q_targets)
    return [0 if float(advantage) > 0.0 else 1 for advantage in target_advantages]


def _oracle_action_now_features_from_targets(q_targets: torch.Tensor) -> torch.Tensor:
    target_advantages = _target_advantages(q_targets)
    oracle_actions = (target_advantages <= 0.0).to(dtype=torch.long)
    return F.one_hot(oracle_actions, num_classes=2).to(dtype=torch.float32)


def _load_oracle_action_now_episodes(
    paths: Sequence[str],
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    reward_scale: float,
    min_decision_margin: float,
    log_interval: int,
) -> List[OracleActionNowEpisode]:
    episodes: List[OracleActionNowEpisode] = []
    skipped = 0
    started = time.time()
    for example_index, path in enumerate(paths, start=1):
        try:
            example = torch.load(path, weights_only=False)
            _, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(example, quality_config)
        except ValueError:
            skipped += 1
        else:
            scaled_rewards = [reward_scale * reward for reward in halt_rewards]
            if not has_strong_optimal_margins(
                scaled_rewards,
                continue_cost=continue_cost,
                min_decision_margin=min_decision_margin,
            ):
                skipped += 1
            else:
                q_targets, oracle_stop_step, oracle_value = bellman_q_targets(scaled_rewards, continue_cost)
                episodes.append(
                    OracleActionNowEpisode(
                        path=path,
                        features=_oracle_action_now_features_from_targets(q_targets),
                        target_advantages=_target_advantages(q_targets),
                        halt_rewards=scaled_rewards,
                        oracle_stop_step=oracle_stop_step,
                        oracle_value=oracle_value,
                    )
                )

        if log_interval > 0 and (example_index % log_interval == 0 or example_index == len(paths)):
            elapsed = time.time() - started
            print(
                f"materialize_oracle_action_now={example_index}/{len(paths)} "
                f"accepted={len(episodes)} skipped={skipped} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if not episodes:
        raise ValueError("No usable oracle-action-now compute-advantage episodes were produced.")
    return episodes


def _oracle_action_now_tensor_dataset(episodes: Sequence[OracleActionNowEpisode]) -> TensorDataset:
    features = torch.cat([episode.features for episode in episodes], dim=0)
    target_advantages = torch.cat([episode.target_advantages for episode in episodes], dim=0)
    return TensorDataset(features, target_advantages)


def _materialized_tensor_dataset(episodes: Sequence[MaterializedAdvantageEpisode]) -> TensorDataset:
    features = torch.cat([episode.features for episode in episodes], dim=0)
    target_advantages = torch.cat([episode.target_advantages for episode in episodes], dim=0)
    return TensorDataset(features, target_advantages)


def _materialize_tree_encoder_episodes(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    *,
    device: torch.device,
    log_interval: int,
    split_name: str,
) -> List[MaterializedAdvantageEpisode]:
    model.eval()
    materialized: List[MaterializedAdvantageEpisode] = []
    skipped = 0
    snapshots = 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if batch is None:
                continue
            encoded = model.encoder(batch.tree_batch)
            root_states = encoded.root_states.detach().cpu()
            offset = 0
            targets = batch.q_targets.detach().cpu()
            for path, length, halt_rewards, oracle_stop_step, oracle_value in zip(
                batch.paths,
                batch.path_lengths,
                batch.halt_rewards,
                batch.oracle_stop_steps,
                batch.oracle_values,
            ):
                next_offset = offset + length
                episode_features = root_states[offset:next_offset]
                episode_targets = targets[offset:next_offset]
                if episode_features.shape[0] != length:
                    raise ValueError("Materialized encoder feature length mismatch.")
                materialized.append(
                    MaterializedAdvantageEpisode(
                        path=path,
                        features=episode_features,
                        target_advantages=_target_advantages(episode_targets),
                        halt_rewards=halt_rewards,
                        oracle_stop_step=oracle_stop_step,
                        oracle_value=oracle_value,
                    )
                )
                offset = next_offset
            skipped += int(batch.skipped)
            snapshots += int(root_states.shape[0])
            if log_interval > 0 and (batch_index % log_interval == 0 or batch_index == len(loader)):
                elapsed = time.time() - started
                print(
                    f"materialize_{split_name}_encoder_batch={batch_index}/{len(loader)} "
                    f"episodes={len(materialized)} snapshots={snapshots} skipped={skipped} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    if not materialized:
        raise ValueError(f"No usable {split_name} encoder episodes were materialized.")
    return materialized


def _build_tensor_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=device.type == "cuda",
    )


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


def _sample_paths(paths: Sequence[str], max_examples: int, seed: int) -> List[str]:
    selected = list(paths)
    if max_examples > 0 and max_examples < len(selected):
        rng = random.Random(seed)
        selected = rng.sample(selected, max_examples)
    return selected


def _build_loader(
    paths: Sequence[str],
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    reward_scale: float,
    representation: str,
    min_decision_margin: float,
    tensorizer: TreeTensorizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = FittedQEpisodeDataset(
        paths=paths,
        quality_config=quality_config,
        continue_cost=continue_cost,
        reward_scale=reward_scale,
        representation=representation,
        min_decision_margin=min_decision_margin,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=FittedQCollator(tensorizer),
        generator=generator if shuffle else None,
    )


def _target_advantages(targets: torch.Tensor) -> torch.Tensor:
    return targets[:, 0] - targets[:, 1]


def _advantage_loss_components(
    predicted_advantages: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_advantages = _target_advantages(targets)
    advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
    mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))
    sign_accuracy = ((predicted_advantages > 0) == (target_advantages > 0)).float().mean()
    return advantage_mse, mean_abs_advantage_error, sign_accuracy


def _advantage_sign_correct_count(predicted_advantages: torch.Tensor, targets: torch.Tensor) -> int:
    target_advantages = _target_advantages(targets)
    return int(((predicted_advantages > 0) == (target_advantages > 0)).sum().item())


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
) -> AdvantageMetrics:
    model.train()
    total_advantage_mse = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    skipped = 0
    started = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        targets = batch.q_targets.to(device, non_blocking=True)
        predicted_advantages = model(batch.tree_batch)
        advantage_mse, mean_abs_advantage_error, _sign_accuracy = _advantage_loss_components(
            predicted_advantages,
            targets,
        )

        optimizer.zero_grad()
        advantage_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(targets.shape[0])
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += _advantage_sign_correct_count(predicted_advantages.detach(), targets)
        skipped += int(batch.skipped)

        if log_interval > 0 and batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"snapshots={total_examples} skipped_episodes={skipped} "
                f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if total_examples == 0:
        raise ValueError("Training loader produced no valid compute-advantage snapshots.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_advantage_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> AdvantageMetrics:
    model.eval()
    total_advantage_mse = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            targets = batch.q_targets.to(device, non_blocking=True)
            predicted_advantages = model(batch.tree_batch)
            advantage_mse, mean_abs_advantage_error, _sign_accuracy = _advantage_loss_components(
                predicted_advantages,
                targets,
            )
            examples = int(targets.shape[0])
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += _advantage_sign_correct_count(predicted_advantages, targets)
    if total_examples == 0:
        raise ValueError("Evaluation loader produced no valid compute-advantage snapshots.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _train_tensor_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
) -> AdvantageMetrics:
    model.train()
    total_advantage_mse = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    started = time.time()

    for batch_index, (features, target_advantages) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        target_advantages = target_advantages.to(device, non_blocking=True)
        predicted_advantages = model.advantage_head(features).squeeze(-1)
        advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
        mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))

        optimizer.zero_grad()
        advantage_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(target_advantages.shape[0])
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += int(((predicted_advantages.detach() > 0) == (target_advantages > 0)).sum().item())

        if log_interval > 0 and batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"snapshots={total_examples} "
                f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if total_examples == 0:
        raise ValueError("Tensor training loader produced no oracle-action-now snapshots.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_tensor_advantage_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> AdvantageMetrics:
    model.eval()
    total_advantage_mse = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for features, target_advantages in loader:
            features = features.to(device, non_blocking=True)
            target_advantages = target_advantages.to(device, non_blocking=True)
            predicted_advantages = model.advantage_head(features).squeeze(-1)
            advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
            mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))
            examples = int(target_advantages.shape[0])
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((predicted_advantages > 0) == (target_advantages > 0)).sum().item())
    if total_examples == 0:
        raise ValueError("Tensor evaluation loader produced no oracle-action-now snapshots.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _predict_stop_step(
    model: nn.Module,
    tensorizer: TreeTensorizer,
    episode: FittedQEpisode,
) -> int:
    model.eval()
    with torch.inference_mode():
        tree_batch = tensorizer.tensorize_forest(episode.snapshots, validate=False)
        predicted_advantages = model(tree_batch).detach().cpu().tolist()
    for step_index, advantage in enumerate(predicted_advantages):
        if float(advantage) <= 0:
            return step_index
    return len(predicted_advantages) - 1


def evaluate_greedy_policy(
    model: nn.Module,
    paths: Sequence[str],
    quality_config: TeacherSearchConfig,
    continue_cost: float,
    reward_scale: float,
    representation: str,
    min_decision_margin: float,
    tensorizer: TreeTensorizer,
    *,
    log_interval: int,
) -> GreedyPolicyMetrics:
    dataset = FittedQEpisodeDataset(
        paths,
        quality_config,
        continue_cost,
        reward_scale,
        representation,
        min_decision_margin,
    )
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    evaluated = 0
    skipped = 0
    started = time.time()

    for index in range(len(dataset)):
        episode = dataset[index]
        if episode is None:
            skipped += 1
            continue
        predicted_stop = _predict_stop_step(model, tensorizer, episode)
        predicted_return = -continue_cost * predicted_stop + episode.halt_rewards[predicted_stop]

        exact += int(predicted_stop == episode.oracle_stop_step)
        first_action += int((predicted_stop == 0) == (episode.oracle_stop_step == 0))
        total_return += predicted_return
        total_oracle_value += episode.oracle_value
        total_expansions += predicted_stop
        evaluated += 1

        if log_interval > 0 and (index + 1) % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"greedy_eval_progress={index + 1}/{len(dataset)} "
                f"evaluated={evaluated} skipped={skipped} "
                f"exact_stop_step_accuracy={exact / max(evaluated, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no valid fitted-Q episodes.")
    return GreedyPolicyMetrics(
        exact_stop_step_accuracy=exact / evaluated,
        first_action_accuracy=first_action / evaluated,
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle_value / evaluated,
        average_regret=(total_oracle_value - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
        skipped_episodes=skipped,
    )


def evaluate_materialized_greedy_policy(
    model: nn.Module,
    episodes: Sequence[MaterializedAdvantageEpisode] | Sequence[OracleActionNowEpisode],
    continue_cost: float,
    *,
    device: torch.device,
    log_interval: int,
) -> GreedyPolicyMetrics:
    model.eval()
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    started = time.time()

    with torch.inference_mode():
        for index, episode in enumerate(episodes, start=1):
            features = episode.features.to(device, non_blocking=True)
            predicted_advantages = model.advantage_head(features).squeeze(-1).detach().cpu().tolist()
            predicted_stop = len(predicted_advantages) - 1
            for step_index, advantage in enumerate(predicted_advantages):
                if float(advantage) <= 0:
                    predicted_stop = step_index
                    break

            predicted_return = -continue_cost * predicted_stop + episode.halt_rewards[predicted_stop]
            exact += int(predicted_stop == episode.oracle_stop_step)
            first_action += int((predicted_stop == 0) == (episode.oracle_stop_step == 0))
            total_return += predicted_return
            total_oracle_value += episode.oracle_value
            total_expansions += predicted_stop

            if log_interval > 0 and (index % log_interval == 0 or index == len(episodes)):
                elapsed = time.time() - started
                print(
                    f"greedy_eval_progress={index}/{len(episodes)} "
                    f"evaluated={index} skipped=0 "
                    f"exact_stop_step_accuracy={exact / max(index, 1):.3f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    evaluated = len(episodes)
    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no oracle-action-now episodes.")
    return GreedyPolicyMetrics(
        exact_stop_step_accuracy=exact / evaluated,
        first_action_accuracy=first_action / evaluated,
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle_value / evaluated,
        average_regret=(total_oracle_value - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
        skipped_episodes=0,
    )


def _save_checkpoint(path: str, model: nn.Module, metadata: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )


def _linear_advantage_summary(model: LinearComputeAdvantageModel) -> tuple[List[float], float]:
    weight = model.advantage_head.weight.detach().cpu().squeeze(0).tolist()
    bias = float(model.advantage_head.bias.detach().cpu().squeeze(0).item())
    return [float(item) for item in weight], bias


def _build_packed_loader(
    manifest_path: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = PackedControllerEpisodeDataset(manifest_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PackedControllerCollator(),
        generator=generator if shuffle else None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an offline counterfactual compute-advantage controller on generated tree snapshot trajectories."
        )
    )
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--validation-data", default=None)
    parser.add_argument(
        "--packed-train-data",
        default=None,
        help="Path to packed controller episode manifest JSON (train). Mutually exclusive with --train-data.",
    )
    parser.add_argument(
        "--packed-validation-data",
        default=None,
        help="Path to packed controller episode manifest JSON (validation). Mutually exclusive with --validation-data.",
    )
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--representation", choices=["tree", "oracle-action-now"], default="tree")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-cost", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--min-decision-margin",
        type=float,
        default=0.0,
        help="Drop episodes whose oracle stopping decisions have weaker margins than this threshold.",
    )
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    parser.add_argument("--q-hidden", type=int, default=128, help="Hidden width for the compute-advantage head.")
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024, help="Tensor minibatch size after materialization.")
    parser.add_argument("--episode-batch-size", type=int, default=8, help="Raw full-episode batch size for tree materialization.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--greedy-eval-interval", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    use_packed = args.packed_train_data is not None or args.packed_validation_data is not None
    if use_packed:
        if args.packed_train_data is None or args.packed_validation_data is None:
            raise ValueError("--packed-train-data and --packed-validation-data must both be provided.")
        if args.train_data is not None or args.validation_data is not None:
            raise ValueError("Cannot specify both --train-data/--validation-data and --packed-*-data.")
    else:
        if args.train_data is None or args.validation_data is None:
            raise ValueError("Either --train-data/--validation-data or --packed-train-data/--packed-validation-data required.")

    if args.representation == "tree" and not args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is required when --representation tree.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    schema = _feature_schema() if args.representation == "tree" else _oracle_action_now_schema()
    quality_config = _quality_config(args)
    validation_paths: List[str] | None = None
    tensorizer: TreeTensorizer | None = None

    print("[compute_advantage] stage=build_model", flush=True)
    if args.representation == "tree":
        model = ComputeAdvantageTreeSearchModel(
            k=args.k,
            node_feat=len(schema.feature_names),
            device=args.device,
            node_embed_hidden=args.node_embed_hidden,
            d_embed=args.d_embed,
            d_message=args.d_message,
            n_heads=args.n_heads,
            d_att=args.d_att,
            q_hidden=args.q_hidden,
        )
        assert args.encoder_checkpoint is not None
        load_encoder_checkpoint(args.encoder_checkpoint, model.encoder)
        if not args.unfreeze_encoder:
            model.freeze_encoder()
    else:
        model = LinearComputeAdvantageModel(node_feat=len(schema.feature_names), device=args.device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_episodes: List[OracleActionNowEpisode] | List[MaterializedAdvantageEpisode] | None = None
    validation_episodes: List[OracleActionNowEpisode] | List[MaterializedAdvantageEpisode] | None = None

    if use_packed:
        assert args.packed_train_data is not None and args.packed_validation_data is not None
        print(
            f"[compute_advantage] stage=load_packed_data "
            f"train={args.packed_train_data} validation={args.packed_validation_data}",
            flush=True,
        )
        raw_train_loader = _build_packed_loader(
            args.packed_train_data,
            batch_size=args.episode_batch_size,
            shuffle=True,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        raw_validation_loader = _build_packed_loader(
            args.packed_validation_data,
            batch_size=args.episode_batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        print(
            f"[compute_advantage] packed_train_episodes={len(raw_train_loader.dataset)} "
            f"packed_validation_episodes={len(raw_validation_loader.dataset)}",
            flush=True,
        )

        if not args.unfreeze_encoder and isinstance(model, ComputeAdvantageTreeSearchModel):
            print("[compute_advantage] stage=materialize_train_encoder (packed)", flush=True)
            train_episodes = _materialize_tree_encoder_episodes(
                model,
                raw_train_loader,
                device=device,
                log_interval=args.log_interval,
                split_name="train",
            )
            print("[compute_advantage] stage=materialize_validation_encoder (packed)", flush=True)
            validation_episodes = _materialize_tree_encoder_episodes(
                model,
                raw_validation_loader,
                device=device,
                log_interval=args.log_interval,
                split_name="validation",
            )
            train_loader = _build_tensor_loader(
                _materialized_tensor_dataset(train_episodes),
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
                device=device,
            )
            validation_loader = _build_tensor_loader(
                _materialized_tensor_dataset(validation_episodes),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                device=device,
            )
        else:
            train_loader = raw_train_loader
            validation_loader = raw_validation_loader

    else:
        print("[compute_advantage] stage=load_paths", flush=True)
        train_paths = _sample_paths(
            load_raw_pretrain_example_paths(args.train_data), args.max_train_examples, args.seed,
        )
        validation_paths = _sample_paths(
            load_raw_pretrain_example_paths(args.validation_data),
            args.max_validation_examples,
            args.seed + 1,
        )
        print(
            f"[compute_advantage] representation={args.representation} "
            f"train_examples={len(train_paths)} validation_examples={len(validation_paths)}",
            flush=True,
        )
        tensorizer = TreeTensorizer(schema, device="cpu")

        if args.representation == "oracle-action-now":
            print("[compute_advantage] stage=materialize_train", flush=True)
            train_episodes = _load_oracle_action_now_episodes(
                train_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                args.min_decision_margin,
                args.log_interval,
            )
            print("[compute_advantage] stage=materialize_validation", flush=True)
            validation_episodes = _load_oracle_action_now_episodes(
                validation_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                args.min_decision_margin,
                args.log_interval,
            )
            train_loader = _build_tensor_loader(
                _oracle_action_now_tensor_dataset(train_episodes),
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
                device=device,
            )
            validation_loader = _build_tensor_loader(
                _oracle_action_now_tensor_dataset(validation_episodes),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                device=device,
            )
        elif not args.unfreeze_encoder:
            raw_train_loader = _build_loader(
                train_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                args.representation,
                args.min_decision_margin,
                tensorizer,
                batch_size=args.episode_batch_size,
                shuffle=True,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            raw_validation_loader = _build_loader(
                validation_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                args.representation,
                args.min_decision_margin,
                tensorizer,
                batch_size=args.episode_batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            assert isinstance(model, ComputeAdvantageTreeSearchModel)
            print("[compute_advantage] stage=materialize_train_encoder", flush=True)
            train_episodes = _materialize_tree_encoder_episodes(
                model,
                raw_train_loader,
                device=device,
                log_interval=args.log_interval,
                split_name="train",
            )
            print("[compute_advantage] stage=materialize_validation_encoder", flush=True)
            validation_episodes = _materialize_tree_encoder_episodes(
                model,
                raw_validation_loader,
                device=device,
                log_interval=args.log_interval,
                split_name="validation",
            )
            train_loader = _build_tensor_loader(
                _materialized_tensor_dataset(train_episodes),
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed,
                device=device,
            )
            validation_loader = _build_tensor_loader(
                _materialized_tensor_dataset(validation_episodes),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                device=device,
            )
        else:
            train_loader = _build_loader(
                train_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                args.representation,
                args.min_decision_margin,
                tensorizer,
                batch_size=args.episode_batch_size,
                shuffle=True,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            validation_loader = _build_loader(
                validation_paths,
                quality_config,
                args.continue_cost,
                args.reward_scale,
                args.representation,
                args.min_decision_margin,
                tensorizer,
                batch_size=args.episode_batch_size,
                shuffle=False,
                seed=args.seed,
                num_workers=args.num_workers,
            )

    best_validation_advantage_mse = float("inf")
    best_metadata: dict | None = None
    print("[compute_advantage] stage=train_start", flush=True)
    for epoch in range(1, args.epochs + 1):
        if train_episodes is not None:
            train_metrics = _train_tensor_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                max_grad_norm=args.max_grad_norm,
                epoch=epoch,
                log_interval=args.log_interval,
            )
        else:
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                max_grad_norm=args.max_grad_norm,
                epoch=epoch,
                log_interval=args.log_interval,
            )
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_advantage_mse={train_metrics.advantage_mse:.3f} "
            f"train_mean_abs_advantage_error={train_metrics.mean_abs_advantage_error:.3f} "
            f"train_sign_accuracy={train_metrics.sign_accuracy:.3f} "
            f"train_snapshots={train_metrics.examples}",
            flush=True,
        )

        validation_metrics = None
        if args.validation_interval > 0 and (epoch % args.validation_interval == 0 or epoch == args.epochs):
            if validation_episodes is not None:
                validation_metrics = evaluate_tensor_advantage_predictions(model, validation_loader, device=device)
            else:
                validation_metrics = evaluate_advantage_predictions(model, validation_loader, device=device)
            print(
                f"validation_epoch={epoch}/{args.epochs} "
                f"validation_advantage_mse={validation_metrics.advantage_mse:.3f} "
                f"validation_mean_abs_advantage_error={validation_metrics.mean_abs_advantage_error:.3f} "
                f"validation_sign_accuracy={validation_metrics.sign_accuracy:.3f} "
                f"validation_snapshots={validation_metrics.examples}",
                flush=True,
            )
            if validation_metrics.advantage_mse < best_validation_advantage_mse:
                best_validation_advantage_mse = validation_metrics.advantage_mse
                best_metadata = {
                    "stage": "compute_advantage_controller",
                    "epoch": epoch,
                    "validation_advantage_mse": validation_metrics.advantage_mse,
                    "validation_sign_accuracy": validation_metrics.sign_accuracy,
                    "continue_cost": args.continue_cost,
                    "reward_scale": args.reward_scale,
                    "representation": args.representation,
                    "min_decision_margin": args.min_decision_margin,
                    "encoder_checkpoint": args.encoder_checkpoint,
                    "unfreeze_encoder": args.unfreeze_encoder,
                }
                if args.output_checkpoint:
                    print(f"[compute_advantage] stage=save_best path={args.output_checkpoint}", flush=True)
                    _save_checkpoint(args.output_checkpoint, model, best_metadata)

        if args.greedy_eval_interval > 0 and (epoch % args.greedy_eval_interval == 0 or epoch == args.epochs):
            greedy_metrics: GreedyPolicyMetrics | None = None
            if validation_episodes is not None:
                greedy_metrics = evaluate_materialized_greedy_policy(
                    model,
                    validation_episodes,
                    args.continue_cost,
                    device=device,
                    log_interval=args.log_interval,
                )
            elif validation_paths is not None and tensorizer is not None:
                greedy_metrics = evaluate_greedy_policy(
                    model,
                    validation_paths,
                    quality_config,
                    args.continue_cost,
                    args.reward_scale,
                    args.representation,
                    args.min_decision_margin,
                    tensorizer,
                    log_interval=args.log_interval,
                )
            if greedy_metrics is not None:
                print(
                    f"greedy_epoch={epoch}/{args.epochs} "
                    f"exact_stop_step_accuracy={greedy_metrics.exact_stop_step_accuracy:.3f} "
                    f"first_action_accuracy={greedy_metrics.first_action_accuracy:.3f} "
                    f"average_return={greedy_metrics.average_return:.3f} "
                    f"average_oracle_value={greedy_metrics.average_oracle_value:.3f} "
                    f"average_regret={greedy_metrics.average_regret:.3f} "
                    f"average_expansions={greedy_metrics.average_expansions:.3f} "
                    f"evaluated_episodes={greedy_metrics.evaluated_episodes} "
                    f"skipped_episodes={greedy_metrics.skipped_episodes}",
                    flush=True,
                )

    if isinstance(model, LinearComputeAdvantageModel):
        weight, bias = _linear_advantage_summary(model)
        print("feature_names=('action_0', 'action_1')", flush=True)
        print(f"advantage_readout_weight={[round(item, 3) for item in weight]}", flush=True)
        print(f"advantage_readout_bias={bias:.3f}", flush=True)

    if args.output_checkpoint and best_metadata is None:
        _save_checkpoint(
            args.output_checkpoint,
            model,
            {
                "stage": "compute_advantage_controller",
                "epoch": args.epochs,
                "continue_cost": args.continue_cost,
                "reward_scale": args.reward_scale,
                "representation": args.representation,
                "min_decision_margin": args.min_decision_margin,
                "encoder_checkpoint": args.encoder_checkpoint,
                "unfreeze_encoder": args.unfreeze_encoder,
            },
        )
    print("[compute_advantage] stage=done", flush=True)


if __name__ == "__main__":
    main()
