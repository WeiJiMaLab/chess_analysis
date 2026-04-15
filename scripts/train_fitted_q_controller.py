from __future__ import annotations

import argparse
import json
import random
import statistics
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

from budgeted_controller_oracle import (
    BudgetBucket,
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
    budgeted_oracle_metadata,
    return_for_stop_step,
)
from cts_pretrain import load_encoder_checkpoint
from GNN import TreeEncoderOutput, TreeNN
from schema import NodeFeatureSchema, tree_encoder_feature_schema
from tensorizer import TreeBatch


@dataclass(frozen=True)
class FittedQBatch:
    tree_batch: TreeBatch
    target_advantages: torch.Tensor
    tree_sizes: torch.Tensor
    time_budgets: torch.Tensor
    paths: List[str]
    source_paths: List[str]
    path_lengths: List[int]
    halt_rewards: List[List[float]]
    oracle_stop_steps: List[int]
    oracle_values: List[float]
    starting_budgets: List[int]
    budget_bucket_names: List[str]


@dataclass(frozen=True)
class MaterializedAdvantageEpisode:
    path: str
    source_path: str
    features: torch.Tensor
    target_advantages: torch.Tensor
    halt_rewards: List[float]
    tree_sizes: List[int]
    time_budgets: List[int]
    oracle_stop_step: int
    oracle_value: float
    starting_budget: int
    budget_bucket_name: str


@dataclass(frozen=True)
class MaterializedCache:
    episodes: List[MaterializedAdvantageEpisode]
    feature_dataset: TensorDataset


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
        q_hidden_layers: int,
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
        if q_hidden_layers <= 0:
            raise ValueError("q_hidden_layers must be positive.")
        self.advantage_head = _build_advantage_head(
            input_dim=self.encoder.d_embed + 2,
            hidden_dim=q_hidden,
            hidden_layers=q_hidden_layers,
            device=resolved_device,
        )

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def encode_with_state_features(
        self,
        tree_batch: TreeBatch,
        tree_sizes: torch.Tensor,
        time_budgets: torch.Tensor,
    ) -> torch.Tensor:
        encoded: TreeEncoderOutput = self.encoder(tree_batch)
        scalars = torch.stack(
            [
                tree_sizes.to(self.encoder.device, dtype=torch.float32),
                time_budgets.to(self.encoder.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        return torch.cat([encoded.root_states, scalars], dim=-1)

    def forward(
        self,
        tree_batch: TreeBatch,
        tree_sizes: torch.Tensor,
        time_budgets: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode_with_state_features(tree_batch, tree_sizes, time_budgets)
        return self.advantage_head(features).squeeze(-1)


@dataclass(frozen=True)
class PackedControllerEpisode:
    path: str
    source_path: str
    step_node_features: List[torch.Tensor]
    step_parent_index: List[torch.Tensor]
    step_edge_parent: List[torch.Tensor]
    step_edge_child: List[torch.Tensor]
    step_edge_slot: List[torch.Tensor]
    step_depth: List[torch.Tensor]
    halt_rewards: List[float]
    target_advantages: torch.Tensor
    tree_sizes: torch.Tensor
    time_budgets: torch.Tensor
    oracle_stop_step: int
    oracle_value: float
    starting_budget: int
    budget_bucket_name: str


class PackedControllerEpisodeDataset(Dataset):
    def __init__(self, manifest_path: str) -> None:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if manifest.get("format") != "cts_budgeted_controller_episode_manifest_v2":
            raise ValueError(f"Unexpected manifest format: {manifest_path}")

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
            if payload.get("format") != "cts_budgeted_controller_episode_shard_v2":
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

        step_node_ptr = payload["step_node_ptr"]
        step_edge_ptr = payload["step_edge_ptr"]
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

        return PackedControllerEpisode(
            path=payload["episode_keys"][episode_offset],
            source_path=payload["source_paths"][episode_offset],
            step_node_features=step_nf,
            step_parent_index=step_pi,
            step_edge_parent=step_ep,
            step_edge_child=step_ec,
            step_edge_slot=step_es,
            step_depth=step_d,
            halt_rewards=payload["halt_rewards"][step_begin:step_end].tolist(),
            target_advantages=payload["target_advantages"][step_begin:step_end],
            tree_sizes=payload["tree_sizes"][step_begin:step_end],
            time_budgets=payload["time_budgets"][step_begin:step_end],
            oracle_stop_step=int(payload["oracle_stop_steps"][episode_offset].item()),
            oracle_value=float(payload["oracle_values"][episode_offset].item()),
            starting_budget=int(payload["starting_budgets"][episode_offset].item()),
            budget_bucket_name=payload["budget_bucket_names"][episode_offset],
        )


def _build_advantage_head(
    *,
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    device: torch.device,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim, device=device))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1, device=device))
    return nn.Sequential(*layers)


class PackedControllerCollator:
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

        target_advantages_list = []
        tree_sizes_list = []
        time_budgets_list = []
        paths = []
        source_paths = []
        path_lengths = []
        halt_rewards = []
        oracle_stop_steps = []
        oracle_values = []
        starting_budgets = []
        budget_bucket_names = []

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

            target_advantages_list.append(episode.target_advantages)
            tree_sizes_list.append(episode.tree_sizes)
            time_budgets_list.append(episode.time_budgets)
            paths.append(episode.path)
            source_paths.append(episode.source_path)
            path_lengths.append(len(episode.step_node_features))
            halt_rewards.append(episode.halt_rewards)
            oracle_stop_steps.append(episode.oracle_stop_step)
            oracle_values.append(episode.oracle_value)
            starting_budgets.append(episode.starting_budget)
            budget_bucket_names.append(episode.budget_bucket_name)

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
            target_advantages=torch.cat(target_advantages_list, dim=0),
            tree_sizes=torch.cat(tree_sizes_list, dim=0),
            time_budgets=torch.cat(time_budgets_list, dim=0),
            paths=paths,
            source_paths=source_paths,
            path_lengths=path_lengths,
            halt_rewards=halt_rewards,
            oracle_stop_steps=oracle_stop_steps,
            oracle_values=oracle_values,
            starting_budgets=starting_budgets,
            budget_bucket_names=budget_bucket_names,
        )


def _feature_schema() -> NodeFeatureSchema:
    return tree_encoder_feature_schema()


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


def _materialized_tensor_dataset(episodes: Sequence[MaterializedAdvantageEpisode]) -> TensorDataset:
    features = torch.cat([episode.features for episode in episodes], dim=0)
    target_advantages = torch.cat([episode.target_advantages for episode in episodes], dim=0)
    return TensorDataset(features, target_advantages)


def _default_materialized_cache_path(manifest_path: str, encoder_checkpoint: str, split_name: str) -> str:
    manifest = Path(manifest_path)
    encoder_stem = Path(encoder_checkpoint).stem
    manifest_stem = manifest.stem
    return str(manifest.with_name(f"{manifest_stem}.materialized_{split_name}_{encoder_stem}.pt"))


def _serialize_materialized_episodes(episodes: Sequence[MaterializedAdvantageEpisode]) -> Dict[str, Any]:
    episode_ptr = [0]
    all_features = []
    all_target_advantages = []
    paths = []
    source_paths = []
    halt_rewards = []
    tree_sizes = []
    time_budgets = []
    oracle_stop_steps = []
    oracle_values = []
    starting_budgets = []
    budget_bucket_names = []

    for episode in episodes:
        episode_ptr.append(episode_ptr[-1] + int(episode.features.shape[0]))
        all_features.append(episode.features)
        all_target_advantages.append(episode.target_advantages)
        paths.append(episode.path)
        source_paths.append(episode.source_path)
        halt_rewards.append(list(episode.halt_rewards))
        tree_sizes.append(list(episode.tree_sizes))
        time_budgets.append(list(episode.time_budgets))
        oracle_stop_steps.append(int(episode.oracle_stop_step))
        oracle_values.append(float(episode.oracle_value))
        starting_budgets.append(int(episode.starting_budget))
        budget_bucket_names.append(episode.budget_bucket_name)

    return {
        "format": "cts_materialized_advantage_cache_v1",
        "episode_ptr": torch.tensor(episode_ptr, dtype=torch.long),
        "features": torch.cat(all_features, dim=0) if all_features else torch.empty(0, dtype=torch.float32),
        "target_advantages": torch.cat(all_target_advantages, dim=0)
        if all_target_advantages
        else torch.empty(0, dtype=torch.float32),
        "paths": paths,
        "source_paths": source_paths,
        "halt_rewards": halt_rewards,
        "tree_sizes": tree_sizes,
        "time_budgets": time_budgets,
        "oracle_stop_steps": oracle_stop_steps,
        "oracle_values": oracle_values,
        "starting_budgets": starting_budgets,
        "budget_bucket_names": budget_bucket_names,
    }


def _deserialize_materialized_episodes(payload: Dict[str, Any]) -> List[MaterializedAdvantageEpisode]:
    if payload.get("format") != "cts_materialized_advantage_cache_v1":
        raise ValueError("Unexpected materialized cache format.")
    episode_ptr = payload["episode_ptr"]
    features = payload["features"]
    target_advantages = payload["target_advantages"]
    episodes: List[MaterializedAdvantageEpisode] = []
    for index, path in enumerate(payload["paths"]):
        start = int(episode_ptr[index].item())
        end = int(episode_ptr[index + 1].item())
        episodes.append(
            MaterializedAdvantageEpisode(
                path=path,
                source_path=payload["source_paths"][index],
                features=features[start:end],
                target_advantages=target_advantages[start:end],
                halt_rewards=list(payload["halt_rewards"][index]),
                tree_sizes=list(payload["tree_sizes"][index]),
                time_budgets=list(payload["time_budgets"][index]),
                oracle_stop_step=int(payload["oracle_stop_steps"][index]),
                oracle_value=float(payload["oracle_values"][index]),
                starting_budget=int(payload["starting_budgets"][index]),
                budget_bucket_name=payload["budget_bucket_names"][index],
            )
        )
    return episodes


def _save_materialized_cache(
    path: str,
    episodes: Sequence[MaterializedAdvantageEpisode],
    *,
    manifest_path: str,
    encoder_checkpoint: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_materialized_episodes(episodes)
    payload["metadata"] = {
        "manifest_path": str(manifest_path),
        "encoder_checkpoint": str(encoder_checkpoint),
    }
    torch.save(payload, output_path)
    print(f"[compute_advantage] saved_materialized_cache={output_path}", flush=True)


def _load_materialized_cache(
    path: str,
    *,
    manifest_path: str,
    encoder_checkpoint: str,
) -> MaterializedCache:
    payload = torch.load(path, weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("manifest_path") != str(manifest_path):
        raise ValueError(f"Materialized cache manifest mismatch: {path}")
    if metadata.get("encoder_checkpoint") != str(encoder_checkpoint):
        raise ValueError(f"Materialized cache encoder mismatch: {path}")
    episodes = _deserialize_materialized_episodes(payload)
    return MaterializedCache(episodes=episodes, feature_dataset=_materialized_tensor_dataset(episodes))


def _materialize_tree_encoder_episodes(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    *,
    log_interval: int,
    split_name: str,
) -> List[MaterializedAdvantageEpisode]:
    model.eval()
    materialized: List[MaterializedAdvantageEpisode] = []
    snapshots = 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if batch is None:
                continue
            features = model.encode_with_state_features(batch.tree_batch, batch.tree_sizes, batch.time_budgets).detach().cpu()
            targets = batch.target_advantages.detach().cpu()
            offset = 0
            for (
                path,
                source_path,
                length,
                halt_rewards,
                oracle_stop_step,
                oracle_value,
                starting_budget,
                bucket_name,
            ) in zip(
                batch.paths,
                batch.source_paths,
                batch.path_lengths,
                batch.halt_rewards,
                batch.oracle_stop_steps,
                batch.oracle_values,
                batch.starting_budgets,
                batch.budget_bucket_names,
            ):
                next_offset = offset + length
                materialized.append(
                    MaterializedAdvantageEpisode(
                        path=path,
                        source_path=source_path,
                        features=features[offset:next_offset],
                        target_advantages=targets[offset:next_offset],
                        halt_rewards=halt_rewards,
                        tree_sizes=batch.tree_sizes[offset:next_offset].tolist(),
                        time_budgets=batch.time_budgets[offset:next_offset].tolist(),
                        oracle_stop_step=oracle_stop_step,
                        oracle_value=oracle_value,
                        starting_budget=starting_budget,
                        budget_bucket_name=bucket_name,
                    )
                )
                offset = next_offset
            snapshots += int(features.shape[0])
            if log_interval > 0 and (batch_index % log_interval == 0 or batch_index == len(loader)):
                elapsed = time.time() - started
                print(
                    f"materialize_{split_name}_encoder_batch={batch_index}/{len(loader)} "
                    f"episodes={len(materialized)} snapshots={snapshots} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    if not materialized:
        raise ValueError(f"No usable {split_name} episodes were materialized.")
    return materialized


def _advantage_loss_components(
    predicted_advantages: torch.Tensor,
    target_advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
    mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))
    sign_accuracy = ((predicted_advantages > 0) == (target_advantages > 0)).float().mean()
    return advantage_mse, mean_abs_advantage_error, sign_accuracy


def _train_epoch(
    model: ComputeAdvantageTreeSearchModel,
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

    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        targets = batch.target_advantages.to(device, non_blocking=True)
        predicted = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
        advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, targets)

        optimizer.zero_grad()
        advantage_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(targets.shape[0])
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += int(((predicted.detach() > 0) == (targets > 0)).sum().item())

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
        raise ValueError("Training loader produced no valid controller states.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_advantage_predictions(
    model: ComputeAdvantageTreeSearchModel,
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
            targets = batch.target_advantages.to(device, non_blocking=True)
            predicted = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, targets)
            examples = int(targets.shape[0])
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((predicted > 0) == (targets > 0)).sum().item())
    if total_examples == 0:
        raise ValueError("Evaluation loader produced no valid controller states.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _train_tensor_epoch(
    model: ComputeAdvantageTreeSearchModel,
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
        predicted = model.advantage_head(features).squeeze(-1)
        advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, target_advantages)

        optimizer.zero_grad()
        advantage_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(target_advantages.shape[0])
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += int(((predicted.detach() > 0) == (target_advantages > 0)).sum().item())

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
        raise ValueError("Tensor training loader produced no controller states.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_tensor_advantage_predictions(
    model: ComputeAdvantageTreeSearchModel,
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
            predicted = model.advantage_head(features).squeeze(-1)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, target_advantages)
            examples = int(target_advantages.shape[0])
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((predicted > 0) == (target_advantages > 0)).sum().item())
    if total_examples == 0:
        raise ValueError("Tensor evaluation loader produced no controller states.")
    return AdvantageMetrics(
        advantage_mse=total_advantage_mse / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _predict_stop_step(
    model: nn.Module,
    episode: PackedControllerEpisode | MaterializedAdvantageEpisode,
) -> tuple[int, List[float]]:
    model.eval()
    with torch.inference_mode():
        if isinstance(episode, MaterializedAdvantageEpisode):
            predicted_advantages = model.advantage_head(episode.features.to(next(model.parameters()).device)).squeeze(-1)
        else:
            batch = PackedControllerCollator()([episode])
            assert batch is not None
            predicted_advantages = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
        values = predicted_advantages.detach().cpu().tolist()
    stop = len(values) - 1
    for step_index, advantage in enumerate(values):
        if float(advantage) <= 0.0:
            stop = step_index
            break
    return stop, values


def evaluate_packed_greedy_policy(
    model: ComputeAdvantageTreeSearchModel,
    dataset: PackedControllerEpisodeDataset,
    oracle_config: BudgetedOracleConfig,
    *,
    log_interval: int,
    diagnostics_out: List[Dict[str, Any]] | None = None,
) -> GreedyPolicyMetrics:
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    started = time.time()

    for index in range(len(dataset)):
        episode = dataset[index]
        predicted_stop, predicted_advantages = _predict_stop_step(model, episode)
        predicted_return = return_for_stop_step(
            episode.halt_rewards,
            episode.tree_sizes.tolist(),
            episode.time_budgets.tolist(),
            predicted_stop,
            oracle_config,
        )
        regret = episode.oracle_value - predicted_return
        exact += int(predicted_stop == episode.oracle_stop_step)
        first_action += int((predicted_stop == 0) == (episode.oracle_stop_step == 0))
        total_return += predicted_return
        total_oracle_value += episode.oracle_value
        total_expansions += predicted_stop

        if diagnostics_out is not None:
            diagnostics_out.append(
                {
                    "path": episode.path,
                    "source_path": episode.source_path,
                    "episode_length": len(episode.halt_rewards),
                    "starting_budget": episode.starting_budget,
                    "budget_bucket_name": episode.budget_bucket_name,
                    "tree_sizes": episode.tree_sizes.tolist(),
                    "time_budgets": episode.time_budgets.tolist(),
                    "halt_rewards": episode.halt_rewards,
                    "oracle_stop_step": episode.oracle_stop_step,
                    "oracle_value": episode.oracle_value,
                    "predicted_stop_step": predicted_stop,
                    "predicted_value": predicted_return,
                    "regret": regret,
                    "predicted_advantages": predicted_advantages,
                    "target_advantages": episode.target_advantages.tolist(),
                }
            )

        if log_interval > 0 and ((index + 1) % log_interval == 0 or index + 1 == len(dataset)):
            elapsed = time.time() - started
            print(
                f"greedy_eval_progress={index + 1}/{len(dataset)} "
                f"exact_stop_step_accuracy={exact / max(index + 1, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    evaluated = len(dataset)
    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no packed episodes.")
    return GreedyPolicyMetrics(
        exact_stop_step_accuracy=exact / evaluated,
        first_action_accuracy=first_action / evaluated,
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle_value / evaluated,
        average_regret=(total_oracle_value - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
    )


def evaluate_materialized_greedy_policy(
    model: ComputeAdvantageTreeSearchModel,
    episodes: Sequence[MaterializedAdvantageEpisode],
    oracle_config: BudgetedOracleConfig,
    *,
    log_interval: int,
    diagnostics_out: List[Dict[str, Any]] | None = None,
) -> GreedyPolicyMetrics:
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    started = time.time()

    for index, episode in enumerate(episodes, start=1):
        predicted_stop, predicted_advantages = _predict_stop_step(model, episode)
        predicted_return = return_for_stop_step(
            episode.halt_rewards,
            episode.tree_sizes,
            episode.time_budgets,
            predicted_stop,
            oracle_config,
        )
        regret = episode.oracle_value - predicted_return
        exact += int(predicted_stop == episode.oracle_stop_step)
        first_action += int((predicted_stop == 0) == (episode.oracle_stop_step == 0))
        total_return += predicted_return
        total_oracle_value += episode.oracle_value
        total_expansions += predicted_stop

        if diagnostics_out is not None:
            diagnostics_out.append(
                {
                    "path": episode.path,
                    "source_path": episode.source_path,
                    "episode_length": len(episode.halt_rewards),
                    "starting_budget": episode.starting_budget,
                    "budget_bucket_name": episode.budget_bucket_name,
                    "tree_sizes": episode.tree_sizes,
                    "time_budgets": episode.time_budgets,
                    "halt_rewards": episode.halt_rewards,
                    "oracle_stop_step": episode.oracle_stop_step,
                    "oracle_value": episode.oracle_value,
                    "predicted_stop_step": predicted_stop,
                    "predicted_value": predicted_return,
                    "regret": regret,
                    "predicted_advantages": predicted_advantages,
                    "target_advantages": episode.target_advantages.tolist(),
                }
            )

        if log_interval > 0 and (index % log_interval == 0 or index == len(episodes)):
            elapsed = time.time() - started
            print(
                f"greedy_eval_progress={index}/{len(episodes)} "
                f"exact_stop_step_accuracy={exact / max(index, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    evaluated = len(episodes)
    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no materialized episodes.")
    return GreedyPolicyMetrics(
        exact_stop_step_accuracy=exact / evaluated,
        first_action_accuracy=first_action / evaluated,
        average_return=total_return / evaluated,
        average_oracle_value=total_oracle_value / evaluated,
        average_regret=(total_oracle_value - total_return) / evaluated,
        average_expansions=total_expansions / evaluated,
        evaluated_episodes=evaluated,
    )


def _write_diagnostics(diagnostics: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in diagnostics:
            handle.write(json.dumps(record) + "\n")
    regrets = [record["regret"] for record in diagnostics]
    print(f"[diagnostics] wrote {len(diagnostics)} episodes to {path}", flush=True)
    print(
        f"[diagnostics] regret_mean={statistics.mean(regrets):.4f} "
        f"regret_std={statistics.pstdev(regrets):.4f} "
        f"regret_max={max(regrets):.4f}",
        flush=True,
    )


def _save_checkpoint(path: str, model: nn.Module, metadata: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, output_path)


def _validate_packed_manifest_oracle(manifest_path: str, oracle_config: BudgetedOracleConfig) -> None:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_config = budgeted_oracle_config_from_metadata(manifest)
    if manifest_config is None:
        raise ValueError(f"Packed manifest is missing budgeted oracle metadata: {manifest_path}")

    float_fields = (
        "maintenance_scale",
        "maintenance_ref_nodes",
        "maintenance_exponent",
        "time_lambda",
        "time_p",
        "time_tau",
        "timeout_value",
    )
    for field in float_fields:
        if abs(float(getattr(manifest_config, field)) - float(getattr(oracle_config, field))) > 1e-12:
            raise ValueError(f"Packed manifest {field} does not match requested training config.")
    int_fields = ("time_delta", "samples_per_bucket")
    for field in int_fields:
        if int(getattr(manifest_config, field)) != int(getattr(oracle_config, field)):
            raise ValueError(f"Packed manifest {field} does not match requested training config.")
    manifest_buckets = [(bucket.name, bucket.min_time, bucket.max_time) for bucket in manifest_config.budget_buckets]
    requested_buckets = [(bucket.name, bucket.min_time, bucket.max_time) for bucket in oracle_config.budget_buckets]
    if manifest_buckets != requested_buckets:
        raise ValueError("Packed manifest budget buckets do not match requested training config.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a budget-aware offline compute-advantage controller on packed tree snapshots."
    )
    parser.add_argument("--packed-train-data", required=True)
    parser.add_argument("--packed-validation-data", required=True)
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--output-checkpoint")
    parser.add_argument("--materialized-train-cache", default=None)
    parser.add_argument("--materialized-validation-cache", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--node-embed-hidden", type=int, default=128)
    parser.add_argument("--d-embed", type=int, default=128)
    parser.add_argument("--d-message", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-att", type=int, default=32)
    parser.add_argument("--q-hidden", type=int, default=256)
    parser.add_argument("--q-hidden-layers", type=int, default=3)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--greedy-eval-interval", type=int, default=1)
    parser.add_argument("--output-diagnostics", default=None)
    parser.add_argument("--maintenance-scale", type=float, default=0.0025)
    parser.add_argument("--maintenance-ref-nodes", type=float, default=30.0)
    parser.add_argument("--maintenance-exponent", type=float, default=1.1)
    parser.add_argument("--time-lambda", type=float, default=18.537)
    parser.add_argument("--time-p", type=float, default=2.8)
    parser.add_argument("--time-tau", type=float, default=2.5)
    parser.add_argument("--time-delta", type=int, default=1)
    parser.add_argument("--timeout-value", type=float, default=-1.0)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_cache_path = args.materialized_train_cache or _default_materialized_cache_path(
        args.packed_train_data,
        args.encoder_checkpoint,
        "train",
    )
    validation_cache_path = args.materialized_validation_cache or _default_materialized_cache_path(
        args.packed_validation_data,
        args.encoder_checkpoint,
        "validation",
    )
    oracle_config = _oracle_config(args)
    _validate_packed_manifest_oracle(args.packed_train_data, oracle_config)
    _validate_packed_manifest_oracle(args.packed_validation_data, oracle_config)
    schema = _feature_schema()

    print("[compute_advantage] stage=build_model", flush=True)
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
        q_hidden_layers=args.q_hidden_layers,
    )
    load_encoder_checkpoint(args.encoder_checkpoint, model.encoder)
    if not args.unfreeze_encoder:
        model.freeze_encoder()

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        f"[compute_advantage] packed_train={args.packed_train_data} packed_validation={args.packed_validation_data}",
        flush=True,
    )
    train_loader_raw = _build_packed_loader(
        args.packed_train_data,
        batch_size=args.episode_batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validation_loader_raw = _build_packed_loader(
        args.packed_validation_data,
        batch_size=args.episode_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    train_episodes: List[MaterializedAdvantageEpisode] | None = None
    validation_episodes: List[MaterializedAdvantageEpisode] | None = None
    if not args.unfreeze_encoder:
        if Path(train_cache_path).exists():
            print(f"[compute_advantage] stage=load_train_cache path={train_cache_path}", flush=True)
            train_cache = _load_materialized_cache(
                train_cache_path,
                manifest_path=args.packed_train_data,
                encoder_checkpoint=args.encoder_checkpoint,
            )
            train_episodes = train_cache.episodes
            train_feature_dataset = train_cache.feature_dataset
        else:
            print("[compute_advantage] stage=materialize_train_encoder", flush=True)
            train_episodes = _materialize_tree_encoder_episodes(
                model,
                train_loader_raw,
                log_interval=args.log_interval,
                split_name="train",
            )
            _save_materialized_cache(
                train_cache_path,
                train_episodes,
                manifest_path=args.packed_train_data,
                encoder_checkpoint=args.encoder_checkpoint,
            )
            train_feature_dataset = _materialized_tensor_dataset(train_episodes)

        if Path(validation_cache_path).exists():
            print(f"[compute_advantage] stage=load_validation_cache path={validation_cache_path}", flush=True)
            validation_cache = _load_materialized_cache(
                validation_cache_path,
                manifest_path=args.packed_validation_data,
                encoder_checkpoint=args.encoder_checkpoint,
            )
            validation_episodes = validation_cache.episodes
            validation_feature_dataset = validation_cache.feature_dataset
        else:
            print("[compute_advantage] stage=materialize_validation_encoder", flush=True)
            validation_episodes = _materialize_tree_encoder_episodes(
                model,
                validation_loader_raw,
                log_interval=args.log_interval,
                split_name="validation",
            )
            _save_materialized_cache(
                validation_cache_path,
                validation_episodes,
                manifest_path=args.packed_validation_data,
                encoder_checkpoint=args.encoder_checkpoint,
            )
            validation_feature_dataset = _materialized_tensor_dataset(validation_episodes)

        train_loader = DataLoader(
            train_feature_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            pin_memory=device.type == "cuda",
        )
        validation_loader = DataLoader(
            validation_feature_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader = train_loader_raw
        validation_loader = validation_loader_raw

    best_validation_advantage_mse = float("inf")
    best_metadata: dict | None = None
    validation_dataset = PackedControllerEpisodeDataset(args.packed_validation_data)
    print(json.dumps(budgeted_oracle_metadata(oracle_config), sort_keys=True), flush=True)
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
                    "encoder_checkpoint": args.encoder_checkpoint,
                    "unfreeze_encoder": args.unfreeze_encoder,
                    "controller_inputs": ["z_t", "N_t", "T_t"],
                    **budgeted_oracle_metadata(oracle_config),
                }
                if args.output_checkpoint:
                    print(f"[compute_advantage] stage=save_best path={args.output_checkpoint}", flush=True)
                    _save_checkpoint(args.output_checkpoint, model, best_metadata)

        if args.greedy_eval_interval > 0 and (epoch % args.greedy_eval_interval == 0 or epoch == args.epochs):
            diagnostics: List[Dict[str, Any]] | None = [] if (epoch == args.epochs and args.output_diagnostics) else None
            if validation_episodes is not None:
                greedy_metrics = evaluate_materialized_greedy_policy(
                    model,
                    validation_episodes,
                    oracle_config,
                    log_interval=args.log_interval,
                    diagnostics_out=diagnostics,
                )
            else:
                greedy_metrics = evaluate_packed_greedy_policy(
                    model,
                    validation_dataset,
                    oracle_config,
                    log_interval=args.log_interval,
                    diagnostics_out=diagnostics,
                )
            print(
                f"greedy_epoch={epoch}/{args.epochs} "
                f"exact_stop_step_accuracy={greedy_metrics.exact_stop_step_accuracy:.3f} "
                f"first_action_accuracy={greedy_metrics.first_action_accuracy:.3f} "
                f"average_return={greedy_metrics.average_return:.3f} "
                f"average_oracle_value={greedy_metrics.average_oracle_value:.3f} "
                f"average_regret={greedy_metrics.average_regret:.3f} "
                f"average_expansions={greedy_metrics.average_expansions:.3f} "
                f"evaluated_episodes={greedy_metrics.evaluated_episodes}",
                flush=True,
            )
            if diagnostics is not None:
                _write_diagnostics(diagnostics, args.output_diagnostics)

    if args.output_checkpoint and best_metadata is None:
        _save_checkpoint(
            args.output_checkpoint,
            model,
            {
                "stage": "compute_advantage_controller",
                "epoch": args.epochs,
                "encoder_checkpoint": args.encoder_checkpoint,
                "unfreeze_encoder": args.unfreeze_encoder,
                "controller_inputs": ["z_t", "N_t", "T_t"],
                **budgeted_oracle_metadata(oracle_config),
            },
        )
    print("[compute_advantage] stage=done", flush=True)


if __name__ == "__main__":
    main()
