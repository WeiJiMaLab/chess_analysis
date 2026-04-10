from __future__ import annotations

import copy
import json
import math
import os
import random
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GNN import ChildWdlModel, NodeValueModel
from tensorizer import (
    TensorizedTreeExample,
    TreeTensorizer,
    collate_tensorized_examples,
    edge_wdl_targets_from_node_features,
)
from tree import ExpansionChild, SearchTree


@dataclass(frozen=True)
class TeacherSearchConfig:
    max_depth: int
    search_budget: int
    c_puct: float = 1.0
    prior_feature: str = "prior"
    value_feature: str = "value"
    target_normalization_version: str = "v1"
    search_config_id: str = "default"

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if self.search_budget <= 0:
            raise ValueError("search_budget must be positive.")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be non-negative.")


@dataclass(frozen=True)
class NodeBudgetDistribution:
    min_nodes: int
    max_nodes: int

    def __post_init__(self) -> None:
        if self.min_nodes <= 0:
            raise ValueError("min_nodes must be positive.")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be >= min_nodes.")

    def sample(self, rng: Optional[random.Random] = None) -> int:
        rng = rng or random
        if self.min_nodes == self.max_nodes:
            return self.min_nodes
        log_min = math.log(self.min_nodes)
        log_max = math.log(self.max_nodes)
        sampled = math.exp(rng.uniform(log_min, log_max))
        return max(self.min_nodes, min(self.max_nodes, int(round(sampled))))


class TreeExpansionProvider(ABC):
    @abstractmethod
    def root_features(self, fen: str) -> Mapping[str, float]:
        raise NotImplementedError

    @abstractmethod
    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        raise NotImplementedError

    def root_metadata(self, fen: str) -> Mapping[str, Any]:
        return {}

    def provider_metadata(self) -> Mapping[str, Any]:
        return {}

    def clear_caches(self) -> None:
        pass


@dataclass
class EdgeStats:
    visit_count: int = 0
    total_value: float = 0.0
    q_value: float = 0.0


@dataclass
class TeacherSearchResult:
    node_target_values: List[float]
    edge_stats: Dict[Tuple[int, int], EdgeStats]


@dataclass
class GeneratedTree:
    tree: SearchTree
    edge_stats: Dict[Tuple[int, int], EdgeStats]
    sampled_node_budget: int
    num_expansions: int


@dataclass
class PretrainExample:
    tree: SearchTree
    node_target_values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_target_values = [float(value) for value in self.node_target_values]
        self.metadata = dict(self.metadata)
        if len(self.node_target_values) != self.tree.num_nodes():
            raise ValueError("node_target_values must match tree.num_nodes().")


def save_pretrain_examples(path: str, examples: Sequence[PretrainExample]) -> None:
    torch.save(list(examples), path)


def load_pretrain_examples(path: str) -> List[PretrainExample]:
    return torch.load(path, weights_only=False)


def _pretrain_example_output_path(directory: str, root_position_id: str, index: int) -> str:
    safe_root_position_id = str(root_position_id).replace("/", "_")
    return os.path.join(directory, f"{index:06d}_{safe_root_position_id}.pt")


def save_pretrain_example_to_directory(directory: str, example: PretrainExample, index: int) -> str:
    os.makedirs(directory, exist_ok=True)
    root_position_id = str(example.metadata.get("root_position_id", f"example_{index}"))
    path = _pretrain_example_output_path(directory, root_position_id, index)
    torch.save(example, path)
    return path


def save_pretrain_examples_to_directory(directory: str, examples: Sequence[PretrainExample]) -> List[str]:
    os.makedirs(directory, exist_ok=True)
    saved_paths = []
    for index, example in enumerate(examples):
        saved_paths.append(save_pretrain_example_to_directory(directory, example, index))
    return saved_paths


def load_pretrain_examples_from_directory(directory: str) -> List[PretrainExample]:
    examples = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".pt"):
            continue
        path = os.path.join(directory, filename)
        examples.append(torch.load(path, weights_only=False))
    return examples


class PretrainExampleDirectoryDataset(Sequence[PretrainExample]):
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.paths = [
            os.path.join(directory, filename)
            for filename in sorted(os.listdir(directory))
            if filename.endswith(".pt")
        ]
        if not self.paths:
            raise ValueError(f"No pretrain examples found in directory: {directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PretrainExample:
        return torch.load(self.paths[index], weights_only=False)


class PretrainExamplePathDataset(Sequence[PretrainExample]):
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("At least one pretrain example path is required.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PretrainExample:
        return torch.load(self.paths[index], weights_only=False)


class PackedPretrainShardDataset(Sequence[PretrainExample]):
    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed shard entries found in manifest: {manifest_path}")

        self.paths: List[str] = []
        self.cumulative_sizes: List[int] = []
        total = 0
        for entry in entries:
            path = entry["path"]
            num_examples = int(entry["num_examples"])
            if num_examples <= 0:
                raise ValueError(f"Packed shard entry has non-positive num_examples: {entry}")
            total += num_examples
            self.paths.append(path)
            self.cumulative_sizes.append(total)

        self._loaded_shard_index: Optional[int] = None
        self._loaded_examples: Optional[List[PretrainExample]] = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> PretrainExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        example_offset = index - shard_start

        if self._loaded_shard_index != shard_index:
            payload = torch.load(self.paths[shard_index], weights_only=False)
            if not isinstance(payload, dict) or "examples" not in payload:
                raise ValueError(f"Packed shard file has unexpected format: {self.paths[shard_index]}")
            self._loaded_shard_index = shard_index
            self._loaded_examples = payload["examples"]

        assert self._loaded_examples is not None
        return self._loaded_examples[example_offset]


class PackedTensorizedShardDataset(Sequence[TensorizedTreeExample]):
    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed shard entries found in manifest: {manifest_path}")

        self.paths: List[str] = []
        self.cumulative_sizes: List[int] = []
        total = 0
        for entry in entries:
            path = entry["path"]
            num_examples = int(entry["num_examples"])
            if num_examples <= 0:
                raise ValueError(f"Packed shard entry has non-positive num_examples: {entry}")
            total += num_examples
            self.paths.append(path)
            self.cumulative_sizes.append(total)

        self._loaded_shard_index: Optional[int] = None
        self._loaded_payload: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _open_tensorized_v1_payload(self, shard_path: str) -> Dict[str, Any]:
        payload = torch.load(shard_path, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "cts_tensorized_pretrain_shard_v1":
            raise ValueError(f"Packed tensorized shard file has unexpected format: {shard_path}")
        if "edge_slot" not in payload:
            raise ValueError(
                f"Packed tensorized shard is missing edge_slot data: {shard_path}. "
                "Canonical slot order cannot be recovered from legacy tensorized shards; re-pack from raw trees."
            )
        return payload

    def _open_tensorized_v2_payload(self, shard_dir: str) -> Dict[str, Any]:
        metadata_path = os.path.join(shard_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("format") != "cts_tensorized_pretrain_shard_v2":
            raise ValueError(f"Packed tensorized shard directory has unexpected format: {shard_dir}")

        def _mmap_array(filename: str):
            return np.load(os.path.join(shard_dir, filename), mmap_mode="r")

        return {
            "format": metadata["format"],
            "feature_names": metadata["feature_names"],
            "node_ptr": _mmap_array("node_ptr.npy"),
            "edge_ptr": _mmap_array("edge_ptr.npy"),
            "node_features": _mmap_array("node_features.npy"),
            "parent_index": _mmap_array("parent_index.npy"),
            "edge_parent": _mmap_array("edge_parent.npy"),
            "edge_child": _mmap_array("edge_child.npy"),
            "edge_slot": _mmap_array("edge_slot.npy"),
            "edge_wdl_targets": _mmap_array("edge_wdl_targets.npy"),
            "depth": _mmap_array("depth.npy"),
            "node_targets": _mmap_array("node_targets.npy"),
        }

    def _load_shard_payload(self, shard_path: str) -> Dict[str, Any]:
        if os.path.isdir(shard_path):
            return self._open_tensorized_v2_payload(shard_path)
        return self._open_tensorized_v1_payload(shard_path)

    @staticmethod
    def _slice_to_tensor(array, start: int, end: int) -> torch.Tensor:
        sliced = array[start:end]
        if isinstance(sliced, torch.Tensor):
            return sliced
        return torch.from_numpy(np.array(sliced, copy=True))

    def __getitem__(self, index: int) -> TensorizedTreeExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        example_offset = index - shard_start

        if self._loaded_shard_index != shard_index:
            payload = self._load_shard_payload(self.paths[shard_index])
            self._loaded_shard_index = shard_index
            self._loaded_payload = payload

        assert self._loaded_payload is not None
        payload = self._loaded_payload
        node_ptr = payload["node_ptr"]
        edge_ptr = payload["edge_ptr"]
        node_start = int(node_ptr[example_offset])
        node_end = int(node_ptr[example_offset + 1])
        edge_start = int(edge_ptr[example_offset])
        edge_end = int(edge_ptr[example_offset + 1])
        feature_names = tuple(payload["feature_names"])
        return TensorizedTreeExample(
            node_features=self._slice_to_tensor(payload["node_features"], node_start, node_end),
            parent_index=self._slice_to_tensor(payload["parent_index"], node_start, node_end),
            edge_parent=self._slice_to_tensor(payload["edge_parent"], edge_start, edge_end),
            edge_child=self._slice_to_tensor(payload["edge_child"], edge_start, edge_end),
            edge_slot=self._slice_to_tensor(payload["edge_slot"], edge_start, edge_end),
            depth=self._slice_to_tensor(payload["depth"], node_start, node_end),
            edge_wdl_targets=(
                self._slice_to_tensor(payload["edge_wdl_targets"], edge_start, edge_end)
                if "edge_wdl_targets" in payload
                else edge_wdl_targets_from_node_features(
                    self._slice_to_tensor(payload["node_features"], node_start, node_end),
                    self._slice_to_tensor(payload["edge_child"], edge_start, edge_end),
                    feature_names,
                )
            ),
            node_targets=self._slice_to_tensor(payload["node_targets"], node_start, node_end),
            feature_names=feature_names,
        )

    @staticmethod
    def collate_fn(batch: Sequence[TensorizedTreeExample]) -> tuple[Any, Any]:
        return collate_tensorized_examples(batch)


def load_pretrain_example_dataset(path: str) -> Sequence[PretrainExample]:
    if os.path.isdir(path):
        return PretrainExampleDirectoryDataset(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrain dataset path does not exist: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_format = manifest.get("format")
        if manifest_format in {"cts_tensorized_pretrain_manifest_v1", "cts_tensorized_pretrain_manifest_v2"}:
            return PackedTensorizedShardDataset(path)
        return PackedPretrainShardDataset(path)

    with open(path, "r", encoding="utf-8") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    if not paths:
        raise ValueError(f"No pretrain example paths found in manifest: {path}")
    return PretrainExamplePathDataset(paths)


def load_raw_pretrain_example_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        paths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if filename.endswith(".pt")
        ]
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pretrain dataset path does not exist: {path}")
        if path.endswith(".json"):
            raise ValueError(
                "Raw pretrain example paths are required for controller training. "
                "Use a directory of .pt examples or a text manifest of raw example paths, not a packed JSON manifest."
            )
        with open(path, "r", encoding="utf-8") as handle:
            paths = [line.strip() for line in handle if line.strip()]
    if not paths:
        raise ValueError(f"No raw pretrain examples found at: {path}")
    return paths

def edge_child_wdl_targets(tree_batch) -> torch.Tensor:
    if getattr(tree_batch, "edge_wdl_targets", None) is not None:
        return tree_batch.edge_wdl_targets
    feature_names = tree_batch.feature_names
    missing = [name for name in ("wdl_win", "wdl_draw", "wdl_loss") if name not in feature_names]
    if missing:
        raise KeyError(f"Tree batch is missing WDL feature columns {missing}.")
    wdl_indices = [feature_names.index(name) for name in ("wdl_win", "wdl_draw", "wdl_loss")]
    edge_child = tree_batch.edge_child
    if edge_child.numel() == 0:
        return tree_batch.node_features.new_empty((0, 3))

    edge_targets = tree_batch.node_features.index_select(0, edge_child)
    edge_targets = edge_targets[:, wdl_indices]
    edge_targets = edge_targets.clamp_min(0.0)
    target_mass = edge_targets.sum(dim=-1, keepdim=True)
    return edge_targets / target_mass.clamp_min(1e-12)


def save_encoder_checkpoint(path: str, encoder, metadata: Optional[Mapping[str, Any]] = None) -> None:
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_encoder_checkpoint(path: str, encoder, map_location: str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return metadata


def save_policy_value_checkpoint(path: str, model, metadata: Optional[Mapping[str, Any]] = None) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def build_tree_from_provider(
    root_fen: str,
    provider: TreeExpansionProvider,
    config: TeacherSearchConfig,
) -> SearchTree:
    tree = SearchTree()
    root_id = tree.create_root(
        fen=root_fen,
        scalar_features=provider.root_features(root_fen),
        metadata=provider.root_metadata(root_fen),
    )

    queue = deque([root_id])
    while queue:
        node_id = queue.popleft()
        node = tree.get_node(node_id)
        if node.depth >= config.max_depth or node.is_terminal:
            continue

        children = list(provider.expand_node(node.fen, node.depth))
        if not children:
            continue

        child_ids = tree.add_children(node_id, children)
        queue.extend(child_ids)

    return tree


def normalize_prior_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []

    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Prior scores must be finite.")

    if all(score >= 0.0 for score in scores):
        score_sum = sum(scores)
        if score_sum > 0.0:
            return [score / score_sum for score in scores]

    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    exp_sum = sum(exp_scores)
    return [score / exp_sum for score in exp_scores]


def _prepare_children(
    children: Sequence[ExpansionChild],
    prior_feature: str,
    remaining_slots: Optional[int] = None,
) -> List[ExpansionChild]:
    if remaining_slots is None:
        selected_children = list(children)
    else:
        selected_children = list(children[:remaining_slots])
    if not selected_children:
        return []

    raw_scores = [float(child.scalar_features[prior_feature]) for child in selected_children]
    normalized_priors = normalize_prior_scores(raw_scores)

    return [
        ExpansionChild(
            move_uci=child.move_uci,
            fen=child.fen,
            scalar_features={**child.scalar_features, prior_feature: prior},
            metadata=child.metadata,
            is_terminal=child.is_terminal,
        )
        for child, prior in zip(selected_children, normalized_priors)
    ]


def _has_expandable_frontier(tree: SearchTree, config: TeacherSearchConfig) -> bool:
    for node in tree.iter_nodes():
        if node.is_terminal or node.is_expanded:
            continue
        if node.depth >= config.max_depth:
            continue
        return True
    return False


def _select_leaf_by_puct(
    tree: SearchTree,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
    config: TeacherSearchConfig,
) -> Tuple[int, List[Tuple[int, int]]]:
    if tree.root_id is None:
        raise ValueError("Tree must contain a root.")

    path: List[Tuple[int, int]] = []
    node_id = tree.root_id
    while True:
        node = tree.get_node(node_id)
        if node.is_terminal or not node.is_expanded:
            return node_id, path

        child_ids = tree.child_ids(node_id)
        if not child_ids:
            return node_id, path

        keyed_child_stats = [(child_id, edge_stats[(node_id, child_id)]) for child_id in child_ids]
        total_visits = sum(stats.visit_count for _, stats in keyed_child_stats)
        parent_visit_scale = config.c_puct * math.sqrt(total_visits + 1.0)
        best_child_id = child_ids[0]
        best_score = float("-inf")
        for child_id, stats in keyed_child_stats:
            child = tree.get_node(child_id)
            prior = float(child.scalar_features.get(config.prior_feature, 0.0))
            q_value = stats.q_value if stats.visit_count > 0 else 0.0
            exploration = prior * parent_visit_scale / (1.0 + stats.visit_count)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_child_id = child_id

        path.append((node_id, best_child_id))
        node_id = best_child_id


def _backpropagate_path(
    edge_stats: Dict[Tuple[int, int], EdgeStats],
    path: Sequence[Tuple[int, int]],
    leaf_value: float,
) -> None:
    value = leaf_value
    for parent_id, child_id in reversed(path):
        value = -value
        stats = edge_stats[(parent_id, child_id)]
        stats.visit_count += 1
        stats.total_value += value
        stats.q_value = stats.total_value / stats.visit_count


def _backup_target_from_child_q(
    tree: SearchTree,
    node_id: int,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
    value_feature: str,
) -> float:
    child_ids = tree.child_ids(node_id)
    if not child_ids:
        return _static_node_value(tree, node_id, value_feature)

    keyed_child_stats = [edge_stats[(node_id, child_id)] for child_id in child_ids]
    total_visits = sum(stats.visit_count for stats in keyed_child_stats)
    if total_visits == 0:
        return _static_node_value(tree, node_id, value_feature)

    weighted_sum = sum(stats.visit_count * stats.q_value for stats in keyed_child_stats)
    return weighted_sum / total_visits


def generate_partial_tree_from_provider(
    root_fen: str,
    provider: TreeExpansionProvider,
    config: TeacherSearchConfig,
    node_budget_distribution: NodeBudgetDistribution,
    rng: Optional[random.Random] = None,
) -> GeneratedTree:
    rng = rng or random.Random()
    sampled_node_budget = node_budget_distribution.sample(rng)

    tree = SearchTree()
    root_id = tree.create_root(
        fen=root_fen,
        scalar_features=provider.root_features(root_fen),
        metadata=provider.root_metadata(root_fen),
    )
    edge_stats: Dict[Tuple[int, int], EdgeStats] = {}
    num_expansions = 0

    while num_expansions < sampled_node_budget and _has_expandable_frontier(tree, config):
        node_id, path = _select_leaf_by_puct(tree, edge_stats, config)
        node = tree.get_node(node_id)
        leaf_value = _static_node_value(tree, node_id, config.value_feature)

        if node.is_terminal or node.depth >= config.max_depth:
            node.is_terminal = True
            _backpropagate_path(edge_stats, path, leaf_value)
            continue

        raw_children = provider.expand_node(node.fen, node.depth)
        children = _prepare_children(
            raw_children,
            prior_feature=config.prior_feature,
        )
        if not children:
            node.is_terminal = True
            _backpropagate_path(edge_stats, path, leaf_value)
            continue

        child_ids = tree.add_children(node_id, children)
        for child_id in child_ids:
            edge_stats[(node_id, child_id)] = EdgeStats()
        num_expansions += 1
        _backpropagate_path(edge_stats, path, leaf_value)

    return GeneratedTree(
        tree=tree,
        edge_stats=edge_stats,
        sampled_node_budget=sampled_node_budget,
        num_expansions=num_expansions,
    )


def consolidate_generated_tree(
    generated_tree: GeneratedTree,
    config: TeacherSearchConfig,
) -> TeacherSearchResult:
    tree = generated_tree.tree
    tree.validate()
    node_target_values = [
        float(
            _backup_target_from_child_q(
                tree,
                node.node_id,
                generated_tree.edge_stats,
                config.value_feature,
            )
        )
        for node in tree.iter_nodes()
    ]

    return TeacherSearchResult(
        node_target_values=node_target_values,
        edge_stats=dict(generated_tree.edge_stats),
    )


def _static_node_value(tree: SearchTree, node_id: int, value_feature: str) -> float:
    node = tree.get_node(node_id)
    if value_feature not in node.scalar_features:
        raise KeyError(f"Node {node_id} is missing feature '{value_feature}'.")
    return float(node.scalar_features[value_feature])


def compute_teacher_targets(
    tree: SearchTree,
    config: TeacherSearchConfig,
    *,
    validate: bool = True,
) -> TeacherSearchResult:
    if validate:
        tree.validate()
    if tree.root_id is None:
        raise ValueError("Tree must contain a root.")

    edge_stats: Dict[Tuple[int, int], EdgeStats] = {}
    for node in tree.iter_nodes():
        for child_id in tree.child_ids(node.node_id):
            edge_stats[(node.node_id, child_id)] = EdgeStats()

    for _ in range(config.search_budget):
        node_id, path = _select_leaf_by_puct(tree, edge_stats, config)
        leaf_value = _static_node_value(tree, node_id, config.value_feature)
        _backpropagate_path(edge_stats, path, leaf_value)

    node_target_values = [
        float(_backup_target_from_child_q(tree, node.node_id, edge_stats, config.value_feature))
        for node in tree.iter_nodes()
    ]

    return TeacherSearchResult(node_target_values=node_target_values, edge_stats=edge_stats)


def build_pretrain_example(
    root_fen: str,
    provider: TreeExpansionProvider,
    config: TeacherSearchConfig,
    node_budget_distribution: Optional[NodeBudgetDistribution] = None,
    rng: Optional[random.Random] = None,
    root_position_id: Optional[str] = None,
) -> PretrainExample:
    if node_budget_distribution is None:
        tree = build_tree_from_provider(root_fen=root_fen, provider=provider, config=config)
        teacher_result = compute_teacher_targets(tree, config)
    else:
        generated_tree = generate_partial_tree_from_provider(
            root_fen=root_fen,
            provider=provider,
            config=config,
            node_budget_distribution=node_budget_distribution,
            rng=rng,
        )
        tree = generated_tree.tree
        teacher_result = consolidate_generated_tree(generated_tree, config)

    metadata = {
        "root_position_id": root_position_id or root_fen,
        "search_config_id": config.search_config_id,
        "provider_metadata": dict(provider.provider_metadata()),
        "target_generation_version": config.target_normalization_version,
    }
    return PretrainExample(
        tree=tree,
        node_target_values=teacher_result.node_target_values,
        metadata=metadata,
    )


@dataclass(frozen=True)
class SupervisedPretrainConfig:
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    root_loss_weight: float = 1.0
    epochs: int = 5
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass(frozen=True)
class ChildWdlPretrainConfig:
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 5
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True


@dataclass
class SupervisedMetrics:
    total_loss: float
    node_mse: float
    root_mse: float
    num_examples: int


@dataclass
class ChildWdlMetrics:
    total_loss: float
    num_examples: int
    num_supervised_edges: int


class SupervisedPretrainer:
    def __init__(
        self,
        model: NodeValueModel,
        tensorizer: TreeTensorizer,
        train_examples: Sequence[PretrainExample],
        validation_examples: Sequence[PretrainExample],
        config: SupervisedPretrainConfig,
    ) -> None:
        self.model = model
        self.tensorizer = tensorizer
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_validation_loss = float("inf")
        self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())

    def _iter_batches(self, examples: Sequence[PretrainExample], shuffle: bool):
        collate_fn = getattr(examples, "collate_fn", list)
        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            dataloader_kwargs["persistent_workers"] = self.config.persistent_workers
        return DataLoader(examples, **dataloader_kwargs)

    def _target_tensor(self, examples: Sequence[PretrainExample], device: torch.device) -> torch.Tensor:
        targets: List[float] = []
        for example in examples:
            targets.extend(example.node_target_values)
        return torch.tensor(targets, dtype=torch.float32, device=device)

    def _run_epoch(
        self,
        examples: Sequence[PretrainExample],
        training: bool,
        batch_progress_callback: Optional[
            Callable[[str, int, int, int, float, float, float], None]
        ] = None,
    ) -> SupervisedMetrics:
        if training:
            self.model.train()
        else:
            self.model.eval()

        metric_device = self.model.encoder.device
        total_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        node_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        root_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        total_examples = 0

        batches = self._iter_batches(examples, shuffle=training and self.config.shuffle)
        total_batches = len(batches)
        phase = "train" if training else "validation"
        for batch_index, batch_data in enumerate(batches, start=1):
            device = self.model.encoder.device
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                tree_batch, targets = batch_data
                targets = targets.to(device)
            else:
                batch_examples = batch_data
                tree_batch = self.tensorizer.tensorize_forest([example.tree for example in batch_examples])
                targets = self._target_tensor(batch_examples, device=device)
            root_index = tree_batch.root_index.to(device)

            with torch.set_grad_enabled(training):
                output = self.model(tree_batch)
                node_mse = F.mse_loss(output.node_values, targets)
                root_mse = F.mse_loss(output.node_values[root_index], targets[root_index])
                total_loss = node_mse + self.config.root_loss_weight * root_mse

                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            batch_size = int(tree_batch.batch_size)
            total_examples += batch_size
            total_loss_sum += total_loss.detach() * batch_size
            node_loss_sum += node_mse.detach() * batch_size
            root_loss_sum += root_mse.detach() * batch_size
            if batch_progress_callback is not None:
                total_loss_value = float(total_loss.item())
                node_mse_value = float(node_mse.item())
                root_mse_value = float(root_mse.item())
                batch_progress_callback(
                    phase,
                    batch_index,
                    total_batches,
                    total_examples,
                    total_loss_value,
                    node_mse_value,
                    root_mse_value,
                )

        if total_examples == 0:
            return SupervisedMetrics(total_loss=0.0, node_mse=0.0, root_mse=0.0, num_examples=0)

        return SupervisedMetrics(
            total_loss=float((total_loss_sum / total_examples).item()),
            node_mse=float((node_loss_sum / total_examples).item()),
            root_mse=float((root_loss_sum / total_examples).item()),
            num_examples=total_examples,
        )

    def train_epoch(
        self,
        batch_progress_callback: Optional[Callable[[str, int, int, int, float, float, float], None]] = None,
    ) -> SupervisedMetrics:
        return self._run_epoch(
            self.train_examples,
            training=True,
            batch_progress_callback=batch_progress_callback,
        )

    def validate(
        self,
        batch_progress_callback: Optional[Callable[[str, int, int, int, float, float, float], None]] = None,
    ) -> SupervisedMetrics:
        return self._run_epoch(
            self.validation_examples,
            training=False,
            batch_progress_callback=batch_progress_callback,
        )

    def fit(
        self,
        progress_callback: Optional[Callable[[int, SupervisedMetrics, SupervisedMetrics], None]] = None,
        batch_progress_callback: Optional[Callable[[str, int, int, int, float, float, float], None]] = None,
    ) -> List[Dict[str, SupervisedMetrics]]:
        history: List[Dict[str, SupervisedMetrics]] = []
        for epoch_index in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(batch_progress_callback=batch_progress_callback)
            validation_metrics = self.validate(batch_progress_callback=batch_progress_callback)
            history.append({"train": train_metrics, "validation": validation_metrics})
            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
            if progress_callback is not None:
                progress_callback(epoch_index, train_metrics, validation_metrics)
        self.model.encoder.load_state_dict(self.best_encoder_state)
        return history

    def save_best_encoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        save_encoder_checkpoint(path, self.model.encoder, metadata=metadata)


class ChildWdlPretrainer:
    def __init__(
        self,
        model: ChildWdlModel,
        tensorizer: TreeTensorizer,
        train_examples: Sequence[PretrainExample],
        validation_examples: Sequence[PretrainExample],
        config: ChildWdlPretrainConfig,
    ) -> None:
        self.model = model
        self.tensorizer = tensorizer
        self.train_examples = train_examples
        self.validation_examples = validation_examples
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_validation_loss = float("inf")
        self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())

    def _iter_batches(self, examples: Sequence[PretrainExample], shuffle: bool):
        collate_fn = getattr(examples, "collate_fn", list)
        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_fn,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.config.prefetch_factor
            dataloader_kwargs["persistent_workers"] = self.config.persistent_workers
        return DataLoader(examples, **dataloader_kwargs)

    def _tree_batch_from_batch_data(self, batch_data):
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            tree_batch, _ = batch_data
            return tree_batch
        return self.tensorizer.tensorize_forest([example.tree for example in batch_data])

    def _run_epoch(
        self,
        examples: Sequence[PretrainExample],
        training: bool,
        batch_progress_callback: Optional[Callable[[str, int, int, int, int, float], None]] = None,
    ) -> ChildWdlMetrics:
        if training:
            self.model.train()
        else:
            self.model.eval()

        metric_device = self.model.encoder.device
        total_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        total_examples = 0
        total_supervised_edges = 0

        batches = self._iter_batches(examples, shuffle=training and self.config.shuffle)
        total_batches = len(batches)
        phase = "train" if training else "validation"
        for batch_index, batch_data in enumerate(batches, start=1):
            tree_batch = self._tree_batch_from_batch_data(batch_data)
            edge_targets = edge_child_wdl_targets(tree_batch).to(self.model.encoder.device)
            if edge_targets.numel() == 0:
                continue

            with torch.set_grad_enabled(training):
                output = self.model(tree_batch)
                log_probs = F.log_softmax(output.edge_logits, dim=-1)
                per_edge_loss = -(edge_targets * log_probs).sum(dim=-1)
                total_loss = per_edge_loss.mean()

                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            batch_size = int(tree_batch.batch_size)
            num_supervised_edges = int(edge_targets.shape[0])
            total_examples += batch_size
            total_supervised_edges += num_supervised_edges
            total_loss_sum += total_loss.detach() * num_supervised_edges
            if batch_progress_callback is not None:
                batch_progress_callback(
                    phase,
                    batch_index,
                    total_batches,
                    total_examples,
                    total_supervised_edges,
                    float(total_loss.item()),
                )

        if total_supervised_edges == 0:
            return ChildWdlMetrics(total_loss=0.0, num_examples=total_examples, num_supervised_edges=0)

        return ChildWdlMetrics(
            total_loss=float((total_loss_sum / total_supervised_edges).item()),
            num_examples=total_examples,
            num_supervised_edges=total_supervised_edges,
        )

    def train_epoch(
        self,
        batch_progress_callback: Optional[Callable[[str, int, int, int, int, float], None]] = None,
    ) -> ChildWdlMetrics:
        return self._run_epoch(
            self.train_examples,
            training=True,
            batch_progress_callback=batch_progress_callback,
        )

    def validate(
        self,
        batch_progress_callback: Optional[Callable[[str, int, int, int, int, float], None]] = None,
    ) -> ChildWdlMetrics:
        return self._run_epoch(
            self.validation_examples,
            training=False,
            batch_progress_callback=batch_progress_callback,
        )

    def fit(
        self,
        progress_callback: Optional[Callable[[int, ChildWdlMetrics, ChildWdlMetrics], None]] = None,
        batch_progress_callback: Optional[Callable[[str, int, int, int, int, float], None]] = None,
    ) -> List[Dict[str, ChildWdlMetrics]]:
        history: List[Dict[str, ChildWdlMetrics]] = []
        for epoch_index in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(batch_progress_callback=batch_progress_callback)
            validation_metrics = self.validate(batch_progress_callback=batch_progress_callback)
            history.append({"train": train_metrics, "validation": validation_metrics})
            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
            if progress_callback is not None:
                progress_callback(epoch_index, train_metrics, validation_metrics)
        self.model.encoder.load_state_dict(self.best_encoder_state)
        return history

    def save_best_encoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        save_encoder_checkpoint(path, self.model.encoder, metadata=metadata)
