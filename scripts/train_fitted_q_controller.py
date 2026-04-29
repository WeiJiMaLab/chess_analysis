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
class EpisodeMetadata:
    path: str
    source_path: str
    halt_rewards: List[float]
    tree_sizes: List[int]
    time_budgets: List[int]
    target_advantages: List[float]
    oracle_stop_step: int
    oracle_value: float
    starting_budget: int
    budget_bucket_name: str
    num_steps: int


@dataclass(frozen=True)
class MaterializedCache:
    shard_paths: List[str]
    shard_sizes: List[int]
    examples: int


ORACLE_STOP_BIN_BOUNDARIES = torch.tensor([1, 2, 3, 4, 8, 16])
ORACLE_STOP_NUM_BINS = len(ORACLE_STOP_BIN_BOUNDARIES) + 1  # 7


def _compute_inverse_freq_bin_weights(
    cache: MaterializedCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-bin inverse frequency weights from cached oracle_stop_steps.

    Bins: 0, 1, 2, 3, 4-7, 8-15, 16+
    Returns (bin_weights, bin_boundaries) where bin_weights[i] is the loss weight
    for bin i, normalized so the expected weight per snapshot is 1.
    """
    bin_counts = torch.zeros(ORACLE_STOP_NUM_BINS, dtype=torch.long)
    for shard_path in cache.shard_paths:
        payload = torch.load(shard_path, weights_only=False)
        oracle_steps = payload.get("oracle_stop_steps")
        if oracle_steps is None:
            raise ValueError(
                "Materialized cache missing oracle_stop_steps. "
                "Delete the cache and re-materialize to use --inverse-freq-weights."
            )
        bins = torch.bucketize(oracle_steps, ORACLE_STOP_BIN_BOUNDARIES)
        bin_counts += torch.bincount(bins, minlength=ORACLE_STOP_NUM_BINS)
    total = bin_counts.sum().float()
    bin_counts_safe = bin_counts.float().clamp(min=1)
    bin_weights = total / (ORACLE_STOP_NUM_BINS * bin_counts_safe)
    bin_names = ["0", "1", "2", "3", "4-7", "8-15", "16+"]
    for name, count, weight in zip(bin_names, bin_counts.tolist(), bin_weights.tolist()):
        print(f"  bin={name:>4s}  count={count:>8d}  weight={weight:.3f}", flush=True)
    return bin_weights, ORACLE_STOP_BIN_BOUNDARIES


@dataclass(frozen=True)
class AdvantageMetrics:
    total_loss: float
    advantage_mse: float
    sign_bce: float
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
        separate_sign_head: bool = False,
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
        input_dim = self.encoder.d_embed + 2
        self.sign_head: nn.Linear | None = None
        if separate_sign_head:
            backbone_layers: list[nn.Module] = []
            current_dim = input_dim
            for _ in range(q_hidden_layers):
                backbone_layers.append(nn.Linear(current_dim, q_hidden, device=resolved_device))
                backbone_layers.append(nn.ReLU())
                current_dim = q_hidden
            self.advantage_backbone = nn.Sequential(*backbone_layers)
            self.advantage_proj = nn.Linear(q_hidden, 1, device=resolved_device)
            self.sign_head = nn.Linear(q_hidden, 1, device=resolved_device)
            self.advantage_head = None
        else:
            self.advantage_head = _build_advantage_head(
                input_dim=input_dim,
                hidden_dim=q_hidden,
                hidden_layers=q_hidden_layers,
                device=resolved_device,
            )

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def predict_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (advantage, sign_logit). Identical when sign_head is None."""
        if self.sign_head is not None:
            backbone_out = self.advantage_backbone(features)
            advantage = self.advantage_proj(backbone_out).squeeze(-1)
            sign_logit = self.sign_head(backbone_out).squeeze(-1)
            return advantage, sign_logit
        advantage = self.advantage_head(features).squeeze(-1)
        return advantage, advantage

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_with_state_features(tree_batch, tree_sizes, time_budgets)
        return self.predict_from_features(features)


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

        if manifest.get("format") != "cts_budgeted_controller_episode_manifest_v4":
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
            if payload.get("format") != "cts_budgeted_controller_episode_shard_v4":
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
        episode_step_begin = int(episode_step_ptr[episode_offset].item())
        episode_step_end = int(episode_step_ptr[episode_offset + 1].item())
        num_steps = episode_step_end - episode_step_begin

        episode_trajectory_index = payload["episode_trajectory_index"]
        trajectory_index = int(episode_trajectory_index[episode_offset].item())

        trajectory_node_ptr = payload["trajectory_node_ptr"]
        node_begin = int(trajectory_node_ptr[trajectory_index].item())
        node_end = int(trajectory_node_ptr[trajectory_index + 1].item())
        full_node_features = payload["node_features"][node_begin:node_end]
        full_parent_index = payload["parent_index"][node_begin:node_end]
        full_depth = payload["depth"][node_begin:node_end]

        trajectory_edge_ptr = payload["trajectory_edge_ptr"]
        edge_begin = int(trajectory_edge_ptr[trajectory_index].item())
        edge_end = int(trajectory_edge_ptr[trajectory_index + 1].item())
        full_edge_child = payload["edge_child"][edge_begin:edge_end]
        full_edge_slot = payload["edge_slot"][edge_begin:edge_end]

        trajectory_child_ptr_ptr = payload["trajectory_child_ptr_ptr"]
        child_ptr_begin = int(trajectory_child_ptr_ptr[trajectory_index].item())
        child_ptr_end = int(trajectory_child_ptr_ptr[trajectory_index + 1].item())
        full_child_ptr = payload["child_ptr"][child_ptr_begin:child_ptr_end]

        trajectory_expansion_parent_ptr = payload["trajectory_expansion_parent_ptr"]
        expansion_parent_begin = int(trajectory_expansion_parent_ptr[trajectory_index].item())
        expansion_parent_end = int(trajectory_expansion_parent_ptr[trajectory_index + 1].item())
        expansion_parent_ids = payload["expansion_parent_ids"][expansion_parent_begin:expansion_parent_end]

        trajectory_step_ptr = payload["trajectory_step_ptr"]
        trajectory_step_begin = int(trajectory_step_ptr[trajectory_index].item())
        trajectory_step_end = int(trajectory_step_ptr[trajectory_index + 1].item())
        step_node_cutoffs = payload["step_node_cutoffs"][trajectory_step_begin:trajectory_step_end]
        full_halt_rewards = payload["trajectory_halt_rewards"][trajectory_step_begin:trajectory_step_end]
        first_decision_expansion_count = int(payload["first_decision_expansion_counts"][trajectory_index].item())

        step_nf = []
        step_pi = []
        step_ep = []
        step_ec = []
        step_es = []
        step_d = []

        for local_step in range(num_steps):
            node_cutoff = int(step_node_cutoffs[local_step].item())
            expansion_count = first_decision_expansion_count + local_step
            step_nf.append(full_node_features[:node_cutoff])
            step_pi.append(full_parent_index[:node_cutoff])
            step_d.append(full_depth[:node_cutoff])

            active_parents = sorted(int(parent_id) for parent_id in expansion_parent_ids[:expansion_count].tolist())
            edge_parent_parts = []
            edge_child_parts = []
            edge_slot_parts = []
            for parent_id in active_parents:
                local_edge_start = int(full_child_ptr[parent_id].item())
                local_edge_end = int(full_child_ptr[parent_id + 1].item())
                if local_edge_end <= local_edge_start:
                    continue
                count = local_edge_end - local_edge_start
                edge_parent_parts.append(torch.full((count,), parent_id, dtype=torch.long))
                edge_child_parts.append(full_edge_child[local_edge_start:local_edge_end])
                edge_slot_parts.append(full_edge_slot[local_edge_start:local_edge_end])
            if edge_parent_parts:
                step_ep.append(torch.cat(edge_parent_parts, dim=0))
                step_ec.append(torch.cat(edge_child_parts, dim=0))
                step_es.append(torch.cat(edge_slot_parts, dim=0))
            else:
                step_ep.append(torch.empty(0, dtype=torch.long))
                step_ec.append(torch.empty(0, dtype=torch.long))
                step_es.append(torch.empty(0, dtype=torch.long))

        starting_budget = int(payload["starting_budgets"][episode_offset].item())
        time_budgets = torch.arange(
            starting_budget,
            starting_budget - num_steps,
            -1,
            dtype=torch.long,
        )
        halt_rewards = full_halt_rewards[:num_steps].tolist()
        tree_sizes = step_node_cutoffs[:num_steps].to(dtype=torch.long)

        return PackedControllerEpisode(
            path=payload["episode_keys"][episode_offset],
            source_path=payload["trajectory_source_paths"][trajectory_index],
            step_node_features=step_nf,
            step_parent_index=step_pi,
            step_edge_parent=step_ep,
            step_edge_child=step_ec,
            step_edge_slot=step_es,
            step_depth=step_d,
            halt_rewards=halt_rewards,
            target_advantages=payload["target_advantages"][episode_step_begin:episode_step_end],
            tree_sizes=tree_sizes,
            time_budgets=time_budgets,
            oracle_stop_step=int(payload["oracle_stop_steps"][episode_offset].item()),
            oracle_value=float(payload["oracle_values"][episode_offset].item()),
            starting_budget=starting_budget,
            budget_bucket_name=payload["budget_bucket_names"][episode_offset],
        )


def _extract_all_episode_metadata(
    dataset: PackedControllerEpisodeDataset,
) -> List[EpisodeMetadata]:
    """Extract per-episode metadata from packed shards without tree reconstruction."""
    metadata: List[EpisodeMetadata] = []
    for shard_index in range(len(dataset.shard_paths)):
        payload = torch.load(dataset.shard_paths[shard_index], weights_only=False)
        episode_step_ptr = payload["episode_step_ptr"]
        episode_trajectory_index = payload["episode_trajectory_index"]
        trajectory_step_ptr = payload["trajectory_step_ptr"]
        step_node_cutoffs = payload["step_node_cutoffs"]
        trajectory_halt_rewards = payload["trajectory_halt_rewards"]
        target_advantages = payload["target_advantages"]
        oracle_stop_steps = payload["oracle_stop_steps"]
        oracle_values = payload["oracle_values"]
        starting_budgets = payload["starting_budgets"]
        budget_bucket_names = payload["budget_bucket_names"]
        episode_keys = payload["episode_keys"]
        trajectory_source_paths = payload["trajectory_source_paths"]

        num_episodes = len(episode_step_ptr) - 1
        for i in range(num_episodes):
            ep_step_begin = int(episode_step_ptr[i].item())
            ep_step_end = int(episode_step_ptr[i + 1].item())
            num_steps = ep_step_end - ep_step_begin

            trajectory_index = int(episode_trajectory_index[i].item())
            traj_step_begin = int(trajectory_step_ptr[trajectory_index].item())

            starting_budget = int(starting_budgets[i].item())
            metadata.append(EpisodeMetadata(
                path=episode_keys[i],
                source_path=trajectory_source_paths[trajectory_index],
                halt_rewards=trajectory_halt_rewards[traj_step_begin:traj_step_begin + num_steps].tolist(),
                tree_sizes=step_node_cutoffs[traj_step_begin:traj_step_begin + num_steps].tolist(),
                time_budgets=list(range(starting_budget, starting_budget - num_steps, -1)),
                target_advantages=target_advantages[ep_step_begin:ep_step_end].tolist(),
                oracle_stop_step=int(oracle_stop_steps[i].item()),
                oracle_value=float(oracle_values[i].item()),
                starting_budget=starting_budget,
                budget_bucket_name=budget_bucket_names[i],
                num_steps=num_steps,
            ))
    return metadata


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


def _default_materialized_cache_path(manifest_path: str, encoder_checkpoint: str, split_name: str) -> str:
    manifest = Path(manifest_path)
    encoder_stem = Path(encoder_checkpoint).stem
    manifest_stem = manifest.stem
    return str(manifest.with_name(f"{manifest_stem}.materialized_{split_name}_{encoder_stem}.pt"))


def _materialized_tensor_dataset(episodes: Sequence[MaterializedAdvantageEpisode]) -> TensorDataset:
    features = torch.cat([episode.features for episode in episodes], dim=0)
    target_advantages = torch.cat([episode.target_advantages for episode in episodes], dim=0)
    return TensorDataset(features, target_advantages)


def _materialized_cache_shard_dir(path: str) -> Path:
    return Path(f"{path}.d")


def _save_materialized_cache_shards(
    path: str,
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    *,
    manifest_path: str,
    encoder_checkpoint: str,
    log_interval: int,
    split_name: str,
    max_snapshots_per_shard: int,
) -> MaterializedCache:
    output_path = Path(path)
    shard_dir = _materialized_cache_shard_dir(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for stale in shard_dir.glob("shard_*.pt"):
        stale.unlink()
    if output_path.exists():
        output_path.unlink()

    model.eval()
    shard_paths: List[str] = []
    shard_sizes: List[int] = []
    shard_features: List[torch.Tensor] = []
    shard_targets: List[torch.Tensor] = []
    shard_oracle_stop_steps: List[torch.Tensor] = []
    shard_snapshots = 0
    total_snapshots = 0
    total_episodes = 0
    shard_index = 0
    started = time.time()

    def flush_shard() -> None:
        nonlocal shard_features, shard_targets, shard_oracle_stop_steps, shard_snapshots, shard_index
        if not shard_features:
            return
        features = torch.cat(shard_features, dim=0)
        targets = torch.cat(shard_targets, dim=0)
        oracle_steps = torch.cat(shard_oracle_stop_steps, dim=0)
        shard_path = shard_dir / f"shard_{shard_index:05d}.pt"
        torch.save(
            {
                "format": "cts_materialized_advantage_cache_shard_v2",
                "features": features,
                "target_advantages": targets,
                "oracle_stop_steps": oracle_steps,
            },
            shard_path,
        )
        shard_paths.append(str(shard_path))
        shard_sizes.append(int(features.shape[0]))
        shard_features = []
        shard_targets = []
        shard_oracle_stop_steps = []
        shard_snapshots = 0
        shard_index += 1

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if batch is None:
                continue
            features = model.encode_with_state_features(batch.tree_batch, batch.tree_sizes, batch.time_budgets).detach().cpu()
            targets = batch.target_advantages.detach().cpu()
            oracle_steps = torch.tensor(batch.oracle_stop_steps, dtype=torch.int32)
            path_lengths = torch.tensor(batch.path_lengths, dtype=torch.int32)
            oracle_steps_per_snapshot = oracle_steps.repeat_interleave(path_lengths)
            shard_features.append(features)
            shard_targets.append(targets)
            shard_oracle_stop_steps.append(oracle_steps_per_snapshot)
            batch_snapshots = int(features.shape[0])
            shard_snapshots += batch_snapshots
            total_snapshots += batch_snapshots
            total_episodes += len(batch.paths)
            if shard_snapshots >= max_snapshots_per_shard:
                flush_shard()
            if log_interval > 0 and (batch_index % log_interval == 0 or batch_index == len(loader)):
                elapsed = time.time() - started
                print(
                    f"materialize_{split_name}_encoder_batch={batch_index}/{len(loader)} "
                    f"episodes={total_episodes} snapshots={total_snapshots} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    flush_shard()
    if total_snapshots == 0:
        raise ValueError(f"No usable {split_name} episodes were materialized.")
    payload = {
        "format": "cts_materialized_advantage_cache_v2",
        "metadata": {
            "manifest_path": str(manifest_path),
            "encoder_checkpoint": str(encoder_checkpoint),
        },
        "shards": [
            {"path": shard_path, "examples": examples}
            for shard_path, examples in zip(shard_paths, shard_sizes)
        ],
        "examples": total_snapshots,
    }
    torch.save(payload, output_path)
    print(f"[compute_advantage] saved_materialized_cache={output_path}", flush=True)
    return MaterializedCache(shard_paths=shard_paths, shard_sizes=shard_sizes, examples=total_snapshots)


def _load_materialized_cache(
    path: str,
    *,
    manifest_path: str,
    encoder_checkpoint: str,
) -> MaterializedCache:
    payload = torch.load(path, weights_only=False)
    if payload.get("format") != "cts_materialized_advantage_cache_v2":
        raise ValueError("Unexpected materialized cache format.")
    metadata = payload.get("metadata", {})
    if metadata.get("manifest_path") != str(manifest_path):
        raise ValueError(f"Materialized cache manifest mismatch: {path}")
    if metadata.get("encoder_checkpoint") != str(encoder_checkpoint):
        raise ValueError(f"Materialized cache encoder mismatch: {path}")
    shard_paths = [str(entry["path"]) for entry in payload.get("shards", [])]
    shard_sizes = [int(entry["examples"]) for entry in payload.get("shards", [])]
    return MaterializedCache(
        shard_paths=shard_paths,
        shard_sizes=shard_sizes,
        examples=int(payload.get("examples", sum(shard_sizes))),
    )


def _advantage_loss_components(
    predicted_advantages: torch.Tensor,
    target_advantages: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weights is not None:
        per_element_mse = (predicted_advantages - target_advantages) ** 2
        advantage_mse = (per_element_mse * weights).sum() / weights.sum()
        per_element_abs = torch.abs(predicted_advantages - target_advantages)
        mean_abs_advantage_error = (per_element_abs * weights).sum() / weights.sum()
    else:
        advantage_mse = F.mse_loss(predicted_advantages, target_advantages)
        mean_abs_advantage_error = torch.mean(torch.abs(predicted_advantages - target_advantages))
    sign_accuracy = ((predicted_advantages > 0) == (target_advantages > 0)).float().mean()
    return advantage_mse, mean_abs_advantage_error, sign_accuracy


def _sign_auxiliary_loss(
    predicted_advantages: torch.Tensor,
    target_advantages: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    sign_targets = (target_advantages > 0).to(dtype=predicted_advantages.dtype)
    if weights is not None:
        per_element = F.binary_cross_entropy_with_logits(predicted_advantages, sign_targets, reduction="none")
        return (per_element * weights).sum() / weights.sum()
    return F.binary_cross_entropy_with_logits(predicted_advantages, sign_targets)


def _train_epoch(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    sign_loss_weight: float,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
) -> AdvantageMetrics:
    model.train()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    started = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        targets = batch.target_advantages.to(device, non_blocking=True)
        predicted, sign_logits = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
        advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, targets)
        sign_loss = _sign_auxiliary_loss(sign_logits, targets)
        total_loss = advantage_mse + sign_loss_weight * sign_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(targets.shape[0])
        total_loss_sum += float(total_loss.item()) * examples
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_sign_bce += float(sign_loss.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += int(((sign_logits.detach() > 0) == (targets > 0)).sum().item())

        if log_interval > 0 and batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"snapshots={total_examples} "
                f"total_loss={total_loss_sum / max(total_examples, 1):.3f} "
                f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                f"sign_bce={total_sign_bce / max(total_examples, 1):.3f} "
                f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if total_examples == 0:
        raise ValueError("Training loader produced no valid controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_advantage_predictions(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    *,
    device: torch.device,
    sign_loss_weight: float,
) -> AdvantageMetrics:
    model.eval()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            targets = batch.target_advantages.to(device, non_blocking=True)
            predicted, sign_logits = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, targets)
            sign_loss = _sign_auxiliary_loss(sign_logits, targets)
            total_loss = advantage_mse + sign_loss_weight * sign_loss
            examples = int(targets.shape[0])
            total_loss_sum += float(total_loss.item()) * examples
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_sign_bce += float(sign_loss.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((sign_logits > 0) == (targets > 0)).sum().item())
    if total_examples == 0:
        raise ValueError("Evaluation loader produced no valid controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
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
    sign_loss_weight: float,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
) -> AdvantageMetrics:
    model.train()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    started = time.time()

    for batch_index, (features, target_advantages) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        target_advantages = target_advantages.to(device, non_blocking=True)
        predicted, sign_logits = model.predict_from_features(features)
        advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, target_advantages)
        sign_loss = _sign_auxiliary_loss(sign_logits, target_advantages)
        total_loss = advantage_mse + sign_loss_weight * sign_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        examples = int(target_advantages.shape[0])
        total_loss_sum += float(total_loss.item()) * examples
        total_advantage_mse += float(advantage_mse.item()) * examples
        total_sign_bce += float(sign_loss.item()) * examples
        total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
        total_examples += examples
        total_correct += int(((sign_logits.detach() > 0) == (target_advantages > 0)).sum().item())

        if log_interval > 0 and batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"snapshots={total_examples} "
                f"total_loss={total_loss_sum / max(total_examples, 1):.3f} "
                f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                f"sign_bce={total_sign_bce / max(total_examples, 1):.3f} "
                f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if total_examples == 0:
        raise ValueError("Tensor training loader produced no controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_tensor_advantage_predictions(
    model: ComputeAdvantageTreeSearchModel,
    loader: DataLoader,
    *,
    device: torch.device,
    sign_loss_weight: float,
) -> AdvantageMetrics:
    model.eval()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for features, target_advantages in loader:
            features = features.to(device, non_blocking=True)
            target_advantages = target_advantages.to(device, non_blocking=True)
            predicted, sign_logits = model.predict_from_features(features)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, target_advantages)
            sign_loss = _sign_auxiliary_loss(sign_logits, target_advantages)
            total_loss = advantage_mse + sign_loss_weight * sign_loss
            examples = int(target_advantages.shape[0])
            total_loss_sum += float(total_loss.item()) * examples
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_sign_bce += float(sign_loss.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((sign_logits > 0) == (target_advantages > 0)).sum().item())
    if total_examples == 0:
        raise ValueError("Tensor evaluation loader produced no controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def _train_materialized_cache_epoch(
    model: ComputeAdvantageTreeSearchModel,
    cache: MaterializedCache,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
    nontrivial_loss_weight: float,
    max_grad_norm: float,
    epoch: int,
    log_interval: int,
    seed: int,
    bin_weights: torch.Tensor | None = None,
    bin_boundaries: torch.Tensor | None = None,
) -> AdvantageMetrics:
    model.train()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    started = time.time()
    batch_counter = 0
    use_bin_weighting = bin_weights is not None
    use_nontrivial_weighting = not use_bin_weighting and nontrivial_loss_weight != 1.0

    shard_order = list(range(len(cache.shard_paths)))
    random.Random(seed + epoch).shuffle(shard_order)
    for shard_index in shard_order:
        payload = torch.load(cache.shard_paths[shard_index], weights_only=False)
        features = payload["features"]
        target_advantages = payload["target_advantages"]
        shard_oracle = payload.get("oracle_stop_steps")
        if (use_bin_weighting or use_nontrivial_weighting) and shard_oracle is None:
            raise ValueError(
                "Materialized cache missing oracle_stop_steps. "
                "Delete the cache and re-materialize to use loss weighting."
            )
        order = torch.randperm(features.shape[0])
        for start in range(0, int(features.shape[0]), batch_size):
            batch_counter += 1
            batch_index = order[start : start + batch_size]
            batch_features = features[batch_index].to(device, non_blocking=True)
            batch_targets = target_advantages[batch_index].to(device, non_blocking=True)
            if use_bin_weighting:
                batch_bins = torch.bucketize(shard_oracle[batch_index], bin_boundaries)
                weights = bin_weights[batch_bins].to(device, non_blocking=True)
            elif use_nontrivial_weighting:
                batch_oracle = shard_oracle[batch_index]
                weights = torch.where(batch_oracle > 1, nontrivial_loss_weight, 1.0).to(device, non_blocking=True)
            else:
                weights = None
            predicted, sign_logits = model.predict_from_features(batch_features)
            advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, batch_targets, weights)
            sign_loss = _sign_auxiliary_loss(sign_logits, batch_targets, weights)
            total_loss = advantage_mse + sign_loss_weight * sign_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            examples = int(batch_targets.shape[0])
            total_loss_sum += float(total_loss.item()) * examples
            total_advantage_mse += float(advantage_mse.item()) * examples
            total_sign_bce += float(sign_loss.item()) * examples
            total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
            total_examples += examples
            total_correct += int(((sign_logits.detach() > 0) == (batch_targets > 0)).sum().item())

            if log_interval > 0 and batch_counter % log_interval == 0:
                elapsed = time.time() - started
                print(
                    f"epoch={epoch} batch={batch_counter} "
                    f"snapshots={total_examples} "
                    f"total_loss={total_loss_sum / max(total_examples, 1):.3f} "
                    f"advantage_mse={total_advantage_mse / max(total_examples, 1):.3f} "
                    f"sign_bce={total_sign_bce / max(total_examples, 1):.3f} "
                    f"mean_abs_advantage_error={total_mean_abs_advantage_error / max(total_examples, 1):.3f} "
                    f"sign_accuracy={total_correct / max(total_examples, 1):.3f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    if total_examples == 0:
        raise ValueError("Materialized cache training produced no controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
        mean_abs_advantage_error=total_mean_abs_advantage_error / total_examples,
        sign_accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def evaluate_materialized_cache_predictions(
    model: ComputeAdvantageTreeSearchModel,
    cache: MaterializedCache,
    *,
    device: torch.device,
    batch_size: int,
    sign_loss_weight: float,
) -> AdvantageMetrics:
    model.eval()
    total_loss_sum = 0.0
    total_advantage_mse = 0.0
    total_sign_bce = 0.0
    total_mean_abs_advantage_error = 0.0
    total_examples = 0
    total_correct = 0
    with torch.inference_mode():
        for shard_path in cache.shard_paths:
            payload = torch.load(shard_path, weights_only=False)
            features = payload["features"]
            target_advantages = payload["target_advantages"]
            for start in range(0, int(features.shape[0]), batch_size):
                batch_features = features[start : start + batch_size].to(device, non_blocking=True)
                batch_targets = target_advantages[start : start + batch_size].to(device, non_blocking=True)
                predicted, sign_logits = model.predict_from_features(batch_features)
                advantage_mse, mean_abs_advantage_error, _ = _advantage_loss_components(predicted, batch_targets)
                sign_loss = _sign_auxiliary_loss(sign_logits, batch_targets)
                total_loss = advantage_mse + sign_loss_weight * sign_loss
                examples = int(batch_targets.shape[0])
                total_loss_sum += float(total_loss.item()) * examples
                total_advantage_mse += float(advantage_mse.item()) * examples
                total_sign_bce += float(sign_loss.item()) * examples
                total_mean_abs_advantage_error += float(mean_abs_advantage_error.item()) * examples
                total_examples += examples
                total_correct += int(((sign_logits > 0) == (batch_targets > 0)).sum().item())
    if total_examples == 0:
        raise ValueError("Materialized cache evaluation produced no controller states.")
    return AdvantageMetrics(
        total_loss=total_loss_sum / total_examples,
        advantage_mse=total_advantage_mse / total_examples,
        sign_bce=total_sign_bce / total_examples,
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
            device = next(model.parameters()).device
            predicted_advantages, sign_logits = model.predict_from_features(
                episode.features.to(device),
            )
        else:
            batch = PackedControllerCollator()([episode])
            assert batch is not None
            predicted_advantages, sign_logits = model(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
        values = predicted_advantages.detach().cpu().tolist()
    stop = len(values) - 1
    for step_index, v in enumerate(values):
        if float(v) <= 0.0:
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


def evaluate_batched_greedy_policy(
    model: ComputeAdvantageTreeSearchModel,
    dataset: PackedControllerEpisodeDataset,
    cache: MaterializedCache,
    oracle_config: BudgetedOracleConfig,
    *,
    log_interval: int,
    diagnostics_out: List[Dict[str, Any]] | None = None,
    predict_batch_size: int = 65536,
) -> GreedyPolicyMetrics:
    """Greedy eval using materialized cache features in batched forward passes."""
    started = time.time()

    # Phase 1: extract per-episode metadata from packed shards (no tree reconstruction).
    episode_metadata = _extract_all_episode_metadata(dataset)
    assert len(episode_metadata) == len(dataset), (
        f"Metadata extraction returned {len(episode_metadata)} episodes but dataset has {len(dataset)}"
    )
    step_counts = [m.num_steps for m in episode_metadata]
    total_steps = sum(step_counts)

    # Phase 2: batched forward pass over cached features.
    model.eval()
    device = next(model.parameters()).device
    all_predicted: List[torch.Tensor] = []
    with torch.inference_mode():
        for shard_path in cache.shard_paths:
            payload = torch.load(shard_path, weights_only=False)
            features = payload["features"]
            for start in range(0, features.shape[0], predict_batch_size):
                batch_features = features[start:start + predict_batch_size].to(device, non_blocking=True)
                predicted, _ = model.predict_from_features(batch_features)
                all_predicted.append(predicted.detach().cpu())
    all_advantages = torch.cat(all_predicted, dim=0)
    assert all_advantages.shape[0] == total_steps, (
        f"Cache has {all_advantages.shape[0]} steps but packed dataset has {total_steps}"
    )

    # Phase 3: per-episode stop-step + return computation.
    exact = 0
    first_action = 0
    total_return = 0.0
    total_oracle_value = 0.0
    total_expansions = 0
    offset = 0

    for index, meta in enumerate(episode_metadata):
        episode_advantages = all_advantages[offset:offset + meta.num_steps].tolist()
        offset += meta.num_steps

        predicted_stop = len(episode_advantages) - 1
        for step_index, v in enumerate(episode_advantages):
            if v <= 0.0:
                predicted_stop = step_index
                break

        predicted_return = return_for_stop_step(
            meta.halt_rewards,
            meta.tree_sizes,
            meta.time_budgets,
            predicted_stop,
            oracle_config,
        )
        regret = meta.oracle_value - predicted_return
        exact += int(predicted_stop == meta.oracle_stop_step)
        first_action += int((predicted_stop == 0) == (meta.oracle_stop_step == 0))
        total_return += predicted_return
        total_oracle_value += meta.oracle_value
        total_expansions += predicted_stop

        if diagnostics_out is not None:
            diagnostics_out.append(
                {
                    "path": meta.path,
                    "source_path": meta.source_path,
                    "episode_length": meta.num_steps,
                    "starting_budget": meta.starting_budget,
                    "budget_bucket_name": meta.budget_bucket_name,
                    "tree_sizes": meta.tree_sizes,
                    "time_budgets": meta.time_budgets,
                    "halt_rewards": meta.halt_rewards,
                    "oracle_stop_step": meta.oracle_stop_step,
                    "oracle_value": meta.oracle_value,
                    "predicted_stop_step": predicted_stop,
                    "predicted_value": predicted_return,
                    "regret": regret,
                    "predicted_advantages": episode_advantages,
                    "target_advantages": meta.target_advantages,
                }
            )

        if log_interval > 0 and ((index + 1) % log_interval == 0 or index + 1 == len(episode_metadata)):
            elapsed = time.time() - started
            print(
                f"greedy_eval_progress={index + 1}/{len(episode_metadata)} "
                f"exact_stop_step_accuracy={exact / max(index + 1, 1):.3f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    assert offset == total_steps
    evaluated = len(episode_metadata)
    if evaluated == 0:
        raise ValueError("Greedy evaluation produced no episodes.")
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
    parser.add_argument("--materialized-cache-shard-snapshots", type=int, default=250000)
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
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="Minimum LR for cosine decay. 0 disables the scheduler.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--sign-loss-weight", type=float, default=1.0)
    parser.add_argument("--nontrivial-loss-weight", type=float, default=1.0,
                        help="Upweight loss on non-trivial episodes (oracle_stop_step > 1). 1.0 = uniform.")
    parser.add_argument("--inverse-freq-weights", action="store_true",
                        help="Weight loss by inverse bin frequency. Overrides --nontrivial-loss-weight.")
    parser.add_argument("--separate-sign-head", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--greedy-eval-interval", type=int, default=1)
    parser.add_argument("--output-diagnostics", default=None)
    parser.add_argument("--maintenance-scale", type=float, default=0.0)
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
        separate_sign_head=args.separate_sign_head,
    )
    load_encoder_checkpoint(args.encoder_checkpoint, model.encoder)
    if not args.unfreeze_encoder:
        model.freeze_encoder()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.min_lr > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr,
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

    train_cache: MaterializedCache | None = None
    validation_cache: MaterializedCache | None = None
    if not args.unfreeze_encoder:
        if Path(train_cache_path).exists():
            print(f"[compute_advantage] stage=load_train_cache path={train_cache_path}", flush=True)
            train_cache = _load_materialized_cache(
                train_cache_path,
                manifest_path=args.packed_train_data,
                encoder_checkpoint=args.encoder_checkpoint,
            )
        else:
            print("[compute_advantage] stage=materialize_train_encoder", flush=True)
            train_cache = _save_materialized_cache_shards(
                train_cache_path,
                model,
                train_loader_raw,
                manifest_path=args.packed_train_data,
                encoder_checkpoint=args.encoder_checkpoint,
                log_interval=args.log_interval,
                split_name="train",
                max_snapshots_per_shard=args.materialized_cache_shard_snapshots,
            )
        if Path(validation_cache_path).exists():
            print(f"[compute_advantage] stage=load_validation_cache path={validation_cache_path}", flush=True)
            validation_cache = _load_materialized_cache(
                validation_cache_path,
                manifest_path=args.packed_validation_data,
                encoder_checkpoint=args.encoder_checkpoint,
            )
        else:
            print("[compute_advantage] stage=materialize_validation_encoder", flush=True)
            validation_cache = _save_materialized_cache_shards(
                validation_cache_path,
                model,
                validation_loader_raw,
                manifest_path=args.packed_validation_data,
                encoder_checkpoint=args.encoder_checkpoint,
                log_interval=args.log_interval,
                split_name="validation",
                max_snapshots_per_shard=args.materialized_cache_shard_snapshots,
            )
    else:
        train_loader = train_loader_raw
        validation_loader = validation_loader_raw

    best_greedy_regret = float("inf")
    best_metadata: dict | None = None
    validation_dataset = PackedControllerEpisodeDataset(args.packed_validation_data)
    print(json.dumps(budgeted_oracle_metadata(oracle_config), sort_keys=True), flush=True)

    bin_weights: torch.Tensor | None = None
    bin_boundaries: torch.Tensor | None = None
    if args.inverse_freq_weights and train_cache is not None:
        print("[compute_advantage] stage=compute_inverse_freq_weights", flush=True)
        bin_weights, bin_boundaries = _compute_inverse_freq_bin_weights(train_cache)

    print("[compute_advantage] stage=train_start", flush=True)

    for epoch in range(1, args.epochs + 1):
        if train_cache is not None:
            train_metrics = _train_materialized_cache_epoch(
                model,
                train_cache,
                optimizer,
                device=device,
                batch_size=args.batch_size,
                sign_loss_weight=args.sign_loss_weight,
                nontrivial_loss_weight=args.nontrivial_loss_weight,
                max_grad_norm=args.max_grad_norm,
                epoch=epoch,
                log_interval=args.log_interval,
                seed=args.seed,
                bin_weights=bin_weights,
                bin_boundaries=bin_boundaries,
            )
        else:
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                sign_loss_weight=args.sign_loss_weight,
                max_grad_norm=args.max_grad_norm,
                epoch=epoch,
                log_interval=args.log_interval,
            )
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_total_loss={train_metrics.total_loss:.3f} "
            f"train_advantage_mse={train_metrics.advantage_mse:.3f} "
            f"train_sign_bce={train_metrics.sign_bce:.3f} "
            f"train_mean_abs_advantage_error={train_metrics.mean_abs_advantage_error:.3f} "
            f"train_sign_accuracy={train_metrics.sign_accuracy:.3f} "
            f"train_snapshots={train_metrics.examples}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()

        if args.validation_interval > 0 and (epoch % args.validation_interval == 0 or epoch == args.epochs):
            if validation_cache is not None:
                validation_metrics = evaluate_materialized_cache_predictions(
                    model,
                    validation_cache,
                    device=device,
                    batch_size=args.batch_size,
                    sign_loss_weight=args.sign_loss_weight,
                )
            else:
                validation_metrics = evaluate_advantage_predictions(
                    model,
                    validation_loader,
                    device=device,
                    sign_loss_weight=args.sign_loss_weight,
                )
            print(
                f"validation_epoch={epoch}/{args.epochs} "
                f"validation_total_loss={validation_metrics.total_loss:.3f} "
                f"validation_advantage_mse={validation_metrics.advantage_mse:.3f} "
                f"validation_sign_bce={validation_metrics.sign_bce:.3f} "
                f"validation_mean_abs_advantage_error={validation_metrics.mean_abs_advantage_error:.3f} "
                f"validation_sign_accuracy={validation_metrics.sign_accuracy:.3f} "
                f"validation_snapshots={validation_metrics.examples}",
                flush=True,
            )
        if args.greedy_eval_interval > 0 and (epoch % args.greedy_eval_interval == 0 or epoch == args.epochs):
            diagnostics: List[Dict[str, Any]] | None = [] if (epoch == args.epochs and args.output_diagnostics) else None
            if validation_cache is not None:
                greedy_metrics = evaluate_batched_greedy_policy(
                    model,
                    validation_dataset,
                    validation_cache,
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
            if greedy_metrics.average_regret < best_greedy_regret:
                best_greedy_regret = greedy_metrics.average_regret
                best_metadata = {
                    "stage": "compute_advantage_controller",
                    "epoch": epoch,
                    "average_regret": greedy_metrics.average_regret,
                    "average_return": greedy_metrics.average_return,
                    "exact_stop_step_accuracy": greedy_metrics.exact_stop_step_accuracy,
                    "encoder_checkpoint": args.encoder_checkpoint,
                    "unfreeze_encoder": args.unfreeze_encoder,
                    "sign_loss_weight": args.sign_loss_weight,
                    "nontrivial_loss_weight": args.nontrivial_loss_weight,
                    "inverse_freq_weights": args.inverse_freq_weights,
                    "separate_sign_head": args.separate_sign_head,
                    "controller_inputs": ["z_t", "N_t", "T_t"],
                    **budgeted_oracle_metadata(oracle_config),
                }
                if args.output_checkpoint:
                    print(f"[compute_advantage] stage=save_best regret={best_greedy_regret:.6f} path={args.output_checkpoint}", flush=True)
                    _save_checkpoint(args.output_checkpoint, model, best_metadata)
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
