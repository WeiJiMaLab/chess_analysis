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
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GNN import ChildWdlModel
from tensorizer import TensorizedTreeExample, TreeTensorizer, collate_tensorized_examples
from tree import ExpansionChild, SearchNode, SearchTree
from cts_uci_common import append_move_to_position_spec


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


def _normalize_wdl_target(target: Sequence[float]) -> Tuple[float, float, float]:
    if len(target) != 3:
        raise ValueError(f"WDL target must have length 3, got {len(target)}.")
    values = [max(0.0, float(value)) for value in target]
    total = sum(values)
    if total <= 0.0:
        raise ValueError("WDL target must have positive mass.")
    return (values[0] / total, values[1] / total, values[2] / total)


def _flip_wdl_target(target: Sequence[float]) -> Tuple[float, float, float]:
    win, draw, loss = _normalize_wdl_target(target)
    return (loss, draw, win)


@dataclass
class EdgeStats:
    visit_count: int = 0
    total_value: float = 0.0
    q_value: float = 0.0
    total_wdl: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_wdl: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class TeacherSearchResult:
    node_target_values: List[float]
    edge_stats: Dict[Tuple[int, int], EdgeStats]
    edge_target_wdls: Dict[Tuple[int, int], Tuple[float, float, float]]


@dataclass
class GeneratedTree:
    tree: SearchTree
    edge_stats: Dict[Tuple[int, int], EdgeStats]
    sampled_node_budget: int
    num_expansions: int
    oracle_trace_expansion_counts: List[int] = field(default_factory=list)
    oracle_root_moves: List[str] = field(default_factory=list)
    oracle_root_q_trace: List[List[float]] = field(default_factory=list)
    oracle_best_move_trace: List[str] = field(default_factory=list)


@dataclass
class PretrainExample:
    tree: SearchTree
    node_target_values: List[float]
    edge_wdl_targets: Dict[Tuple[int, int], Tuple[float, float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    oracle_trace_expansion_counts: List[int] = field(default_factory=list)
    oracle_root_moves: List[str] = field(default_factory=list)
    oracle_root_q_trace: List[List[float]] = field(default_factory=list)
    oracle_best_move_trace: List[str] = field(default_factory=list)
    oracle_final_root_q_values: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_target_values = [float(value) for value in self.node_target_values]
        self.edge_wdl_targets = {
            (int(parent_id), int(child_id)): _normalize_wdl_target(target)
            for (parent_id, child_id), target in dict(self.edge_wdl_targets).items()
        }
        self.metadata = dict(self.metadata)
        self.oracle_trace_expansion_counts = [int(value) for value in self.oracle_trace_expansion_counts]
        self.oracle_root_moves = [str(move) for move in self.oracle_root_moves]
        self.oracle_root_q_trace = [
            [float(value) for value in row]
            for row in self.oracle_root_q_trace
        ]
        self.oracle_best_move_trace = [str(move) for move in self.oracle_best_move_trace]
        self.oracle_final_root_q_values = {
            str(move): float(value)
            for move, value in dict(self.oracle_final_root_q_values).items()
        }
        if len(self.node_target_values) != self.tree.num_nodes():
            raise ValueError("node_target_values must match tree.num_nodes().")
        if self.edge_wdl_targets and len(self.edge_wdl_targets) != self.tree.num_edges():
            raise ValueError("edge_wdl_targets must match tree.num_edges().")
        if self.oracle_trace_expansion_counts or self.oracle_root_q_trace or self.oracle_best_move_trace:
            if not self.oracle_root_moves:
                raise ValueError("oracle_root_moves must be provided when oracle traces are stored.")
            if len(self.oracle_trace_expansion_counts) != len(self.oracle_root_q_trace):
                raise ValueError("oracle_trace_expansion_counts and oracle_root_q_trace must have the same length.")
            if len(self.oracle_trace_expansion_counts) != len(self.oracle_best_move_trace):
                raise ValueError("oracle_trace_expansion_counts and oracle_best_move_trace must have the same length.")
            if any(len(row) != len(self.oracle_root_moves) for row in self.oracle_root_q_trace):
                raise ValueError("Each oracle_root_q_trace row must align with oracle_root_moves.")
            if any(move not in self.oracle_root_moves for move in self.oracle_best_move_trace):
                raise ValueError("oracle_best_move_trace contains a move outside oracle_root_moves.")
            if any(count <= 0 for count in self.oracle_trace_expansion_counts):
                raise ValueError("oracle trace expansion counts must be positive.")
            if any(
                right <= left
                for left, right in zip(self.oracle_trace_expansion_counts, self.oracle_trace_expansion_counts[1:])
            ):
                raise ValueError("oracle trace expansion counts must be strictly increasing.")
        if self.oracle_final_root_q_values and self.oracle_root_moves:
            if set(self.oracle_final_root_q_values) != set(self.oracle_root_moves):
                raise ValueError("oracle_final_root_q_values must align with oracle_root_moves.")

RAW_PRETRAIN_FORMAT = "cts_raw_pretrain_example_v2"


def _ordered_feature_names_from_tree(tree: SearchTree) -> Tuple[str, ...]:
    ordered: List[str] = []
    seen = set()
    for node in tree.iter_nodes():
        for name in node.scalar_features:
            if name not in seen:
                seen.add(name)
                ordered.append(str(name))
    return tuple(ordered)


def _dense_node_feature_tensor(tree: SearchTree, feature_names: Sequence[str]) -> torch.Tensor:
    rows: List[List[float]] = []
    for node in tree.iter_nodes():
        row = []
        for feature_name in feature_names:
            value = node.scalar_features.get(feature_name, float("nan"))
            row.append(float(value))
        rows.append(row)
    if not rows:
        return torch.empty((0, len(feature_names)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _child_ptr_and_children_index(tree: SearchTree) -> Tuple[torch.Tensor, torch.Tensor]:
    # Children are stored in UCI-lexicographic order so that edge_slot derived
    # from `arange(count_per_parent)` matches the canonical slot assignment used
    # by TreeTensorizer._sorted_child_ids_with_slots. Engine-dependent insertion
    # order is non-canonical; sorting here makes the on-disk representation stable.
    child_ptr = [0]
    children_index: List[int] = []
    for node in tree.iter_nodes():
        child_ids = tree.child_ids(node.node_id)
        sorted_child_ids = sorted(
            child_ids,
            key=lambda cid: (tree.get_node(cid).incoming_move_uci or "", cid),
        )
        children_index.extend(int(child_id) for child_id in sorted_child_ids)
        child_ptr.append(len(children_index))
    return (
        torch.tensor(child_ptr, dtype=torch.int32),
        torch.tensor(children_index, dtype=torch.int32),
    )


def _edge_wdl_target_tensor_for_tree(
    tree: SearchTree,
    edge_wdl_targets: Mapping[Tuple[int, int], Sequence[float]],
) -> torch.Tensor:
    rows: List[Tuple[float, float, float]] = []
    for node in tree.iter_nodes():
        for child_id in tree.child_ids(node.node_id):
            edge_key = (node.node_id, child_id)
            if edge_key not in edge_wdl_targets:
                raise KeyError(f"Missing edge WDL target for edge {edge_key}.")
            rows.append(_normalize_wdl_target(edge_wdl_targets[edge_key]))
    if not rows:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _compact_position_spec_payload(tree: SearchTree) -> Dict[str, Any]:
    nodes = list(tree.iter_nodes())
    if not nodes:
        return {"root_position_spec": None, "position_specs": None}
    root_position_spec = nodes[0].fen
    reconstructable = True
    for node in nodes:
        if node.parent_id is None:
            if node.fen != root_position_spec:
                reconstructable = False
                break
            continue
        if node.incoming_move_uci is None:
            reconstructable = False
            break
        parent_fen = nodes[node.parent_id].fen
        if append_move_to_position_spec(parent_fen, node.incoming_move_uci) != node.fen:
            reconstructable = False
            break
    if reconstructable:
        return {"root_position_spec": root_position_spec, "position_specs": None}
    return {
        "root_position_spec": root_position_spec,
        "position_specs": [node.fen for node in nodes],
    }


@dataclass(frozen=True)
class RawPretrainExampleRecord:
    root_position_spec: Optional[str]
    incoming_moves: List[Optional[str]]
    feature_names: Tuple[str, ...]
    node_features: torch.Tensor
    parent_index: torch.Tensor
    child_ptr: torch.Tensor
    children_index: torch.Tensor
    depth: torch.Tensor
    is_terminal: torch.Tensor
    is_expanded: torch.Tensor
    node_targets: torch.Tensor
    edge_wdl_targets: torch.Tensor
    metadata: Dict[str, Any]
    sparse_node_metadata: List[Tuple[int, Dict[str, Any]]]
    oracle_trace_expansion_counts: torch.Tensor
    oracle_root_moves: List[str]
    oracle_root_q_trace: torch.Tensor
    oracle_best_move_index: torch.Tensor
    oracle_final_root_q_values: torch.Tensor
    position_specs: Optional[List[str]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "incoming_moves", [None if move is None else str(move) for move in self.incoming_moves])
        object.__setattr__(self, "feature_names", tuple(str(name) for name in self.feature_names))
        object.__setattr__(self, "node_features", self.node_features.to(dtype=torch.float32, device="cpu"))
        object.__setattr__(self, "parent_index", self.parent_index.to(dtype=torch.int32, device="cpu"))
        object.__setattr__(self, "child_ptr", self.child_ptr.to(dtype=torch.int32, device="cpu"))
        object.__setattr__(self, "children_index", self.children_index.to(dtype=torch.int32, device="cpu"))
        object.__setattr__(self, "depth", self.depth.to(dtype=torch.int16, device="cpu"))
        object.__setattr__(self, "is_terminal", self.is_terminal.to(dtype=torch.bool, device="cpu"))
        object.__setattr__(self, "is_expanded", self.is_expanded.to(dtype=torch.bool, device="cpu"))
        object.__setattr__(self, "node_targets", self.node_targets.to(dtype=torch.float32, device="cpu"))
        object.__setattr__(self, "edge_wdl_targets", self.edge_wdl_targets.to(dtype=torch.float32, device="cpu"))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "sparse_node_metadata",
            [(int(node_id), dict(node_metadata)) for node_id, node_metadata in self.sparse_node_metadata],
        )
        object.__setattr__(
            self,
            "oracle_trace_expansion_counts",
            self.oracle_trace_expansion_counts.to(dtype=torch.int32, device="cpu"),
        )
        object.__setattr__(self, "oracle_root_moves", [str(move) for move in self.oracle_root_moves])
        object.__setattr__(self, "oracle_root_q_trace", self.oracle_root_q_trace.to(dtype=torch.float32, device="cpu"))
        object.__setattr__(
            self,
            "oracle_best_move_index",
            self.oracle_best_move_index.to(dtype=torch.int32, device="cpu"),
        )
        object.__setattr__(
            self,
            "oracle_final_root_q_values",
            self.oracle_final_root_q_values.to(dtype=torch.float32, device="cpu"),
        )
        if self.position_specs is not None:
            object.__setattr__(self, "position_specs", [str(spec) for spec in self.position_specs])
        self._validate()

    def _validate(self) -> None:
        num_nodes = int(self.parent_index.shape[0])
        if self.node_features.shape != (num_nodes, len(self.feature_names)):
            raise ValueError("node_features must align with feature_names and parent_index.")
        if num_nodes > 0 and int(self.parent_index[0].item()) != -1:
            raise ValueError("The root node must have parent_index -1.")
        for node_id, parent_id in enumerate(self.parent_index.tolist()):
            if node_id == 0:
                continue
            if int(parent_id) < 0 or int(parent_id) >= node_id:
                raise ValueError("parent_index must be topologically ordered with exactly one root.")
        if self.depth.shape != (num_nodes,):
            raise ValueError("depth must align with parent_index.")
        if self.is_terminal.shape != (num_nodes,):
            raise ValueError("is_terminal must align with parent_index.")
        if self.is_expanded.shape != (num_nodes,):
            raise ValueError("is_expanded must align with parent_index.")
        if self.node_targets.shape != (num_nodes,):
            raise ValueError("node_targets must align with parent_index.")
        if self.child_ptr.shape != (num_nodes + 1,):
            raise ValueError("child_ptr must have length num_nodes + 1.")
        if int(self.child_ptr[0].item()) != 0:
            raise ValueError("child_ptr must start at 0.")
        if bool((self.child_ptr[1:] < self.child_ptr[:-1]).any()):
            raise ValueError("child_ptr must be nondecreasing.")
        if int(self.child_ptr[-1].item()) != int(self.children_index.shape[0]):
            raise ValueError("child_ptr must end at len(children_index).")
        if num_nodes > 0 and bool(((self.children_index < 0) | (self.children_index >= num_nodes)).any()):
            raise ValueError("children_index entries must reference valid node ids.")
        if self.edge_wdl_targets.shape != (int(self.children_index.shape[0]), 3):
            raise ValueError("edge_wdl_targets must align with children_index.")
        if self.position_specs is None and self.root_position_spec is None and num_nodes > 0:
            raise ValueError("Either root_position_spec or position_specs must be provided.")
        if self.position_specs is not None and len(self.position_specs) != num_nodes:
            raise ValueError("position_specs must align with parent_index.")
        if self.oracle_root_q_trace.shape != (int(self.oracle_trace_expansion_counts.shape[0]), len(self.oracle_root_moves)):
            raise ValueError("oracle_root_q_trace must align with oracle traces and root moves.")
        if self.oracle_best_move_index.shape != self.oracle_trace_expansion_counts.shape:
            raise ValueError("oracle_best_move_index must align with oracle traces.")
        if self.oracle_final_root_q_values.shape != (len(self.oracle_root_moves),):
            raise ValueError("oracle_final_root_q_values must align with oracle_root_moves.")
        if len(self.oracle_root_moves) > 0 and self.oracle_best_move_index.numel() > 0:
            if bool(((self.oracle_best_move_index < 0) | (self.oracle_best_move_index >= len(self.oracle_root_moves))).any()):
                raise ValueError("oracle_best_move_index entries must reference valid oracle_root_moves.")

    @classmethod
    def from_example(cls, example: PretrainExample) -> "RawPretrainExampleRecord":
        tree = example.tree
        feature_names = _ordered_feature_names_from_tree(tree)
        child_ptr, children_index = _child_ptr_and_children_index(tree)
        position_payload = _compact_position_spec_payload(tree)
        oracle_best_move_index = torch.tensor(
            [example.oracle_root_moves.index(move) for move in example.oracle_best_move_trace],
            dtype=torch.int32,
        ) if example.oracle_best_move_trace else torch.empty((0,), dtype=torch.int32)
        oracle_final_root_q_values = torch.tensor(
            [float(example.oracle_final_root_q_values[move]) for move in example.oracle_root_moves],
            dtype=torch.float32,
        ) if example.oracle_root_moves else torch.empty((0,), dtype=torch.float32)
        return cls(
            root_position_spec=position_payload["root_position_spec"],
            incoming_moves=[node.incoming_move_uci for node in tree.iter_nodes()],
            feature_names=feature_names,
            node_features=_dense_node_feature_tensor(tree, feature_names),
            parent_index=torch.tensor(
                [-1 if node.parent_id is None else int(node.parent_id) for node in tree.iter_nodes()],
                dtype=torch.int32,
            ),
            child_ptr=child_ptr,
            children_index=children_index,
            depth=torch.tensor([int(node.depth) for node in tree.iter_nodes()], dtype=torch.int16),
            is_terminal=torch.tensor([bool(node.is_terminal) for node in tree.iter_nodes()], dtype=torch.bool),
            is_expanded=torch.tensor([bool(node.is_expanded) for node in tree.iter_nodes()], dtype=torch.bool),
            node_targets=torch.tensor(example.node_target_values, dtype=torch.float32),
            edge_wdl_targets=_edge_wdl_target_tensor_for_tree(tree, example.edge_wdl_targets),
            metadata=dict(example.metadata),
            sparse_node_metadata=[
                (int(node.node_id), dict(node.metadata))
                for node in tree.iter_nodes()
                if node.metadata
            ],
            oracle_trace_expansion_counts=torch.tensor(example.oracle_trace_expansion_counts, dtype=torch.int32),
            oracle_root_moves=list(example.oracle_root_moves),
            oracle_root_q_trace=(
                torch.tensor(example.oracle_root_q_trace, dtype=torch.float32)
                if example.oracle_root_q_trace
                else torch.empty((0, len(example.oracle_root_moves)), dtype=torch.float32)
            ),
            oracle_best_move_index=oracle_best_move_index,
            oracle_final_root_q_values=oracle_final_root_q_values,
            position_specs=position_payload["position_specs"],
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "format": RAW_PRETRAIN_FORMAT,
            "root_position_spec": self.root_position_spec,
            "incoming_moves": list(self.incoming_moves),
            "feature_names": list(self.feature_names),
            "node_features": self.node_features,
            "parent_index": self.parent_index,
            "child_ptr": self.child_ptr,
            "children_index": self.children_index,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "is_expanded": self.is_expanded,
            "node_targets": self.node_targets,
            "edge_wdl_targets": self.edge_wdl_targets,
            "metadata": dict(self.metadata),
            "sparse_node_metadata": list(self.sparse_node_metadata),
            "oracle_trace_expansion_counts": self.oracle_trace_expansion_counts,
            "oracle_root_moves": list(self.oracle_root_moves),
            "oracle_root_q_trace": self.oracle_root_q_trace,
            "oracle_best_move_index": self.oracle_best_move_index,
            "oracle_final_root_q_values": self.oracle_final_root_q_values,
            "position_specs": list(self.position_specs) if self.position_specs is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RawPretrainExampleRecord":
        if payload.get("format") != RAW_PRETRAIN_FORMAT:
            raise ValueError(f"Expected raw pretrain format {RAW_PRETRAIN_FORMAT}, got {payload.get('format')!r}.")
        return cls(
            root_position_spec=payload.get("root_position_spec"),
            incoming_moves=list(payload["incoming_moves"]),
            feature_names=tuple(payload["feature_names"]),
            node_features=payload["node_features"],
            parent_index=payload["parent_index"],
            child_ptr=payload["child_ptr"],
            children_index=payload["children_index"],
            depth=payload["depth"],
            is_terminal=payload["is_terminal"],
            is_expanded=payload["is_expanded"],
            node_targets=payload["node_targets"],
            edge_wdl_targets=payload["edge_wdl_targets"],
            metadata=dict(payload.get("metadata", {})),
            sparse_node_metadata=list(payload.get("sparse_node_metadata", [])),
            oracle_trace_expansion_counts=payload["oracle_trace_expansion_counts"],
            oracle_root_moves=list(payload.get("oracle_root_moves", [])),
            oracle_root_q_trace=payload["oracle_root_q_trace"],
            oracle_best_move_index=payload["oracle_best_move_index"],
            oracle_final_root_q_values=payload["oracle_final_root_q_values"],
            position_specs=payload.get("position_specs"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RawPretrainExampleRecord":
        payload = torch.load(path, weights_only=False)
        return cls.from_payload(payload)

    def save(self, path: str | Path) -> None:
        torch.save(self.to_payload(), path)

    def _resolved_position_specs(self) -> List[str]:
        if self.position_specs is not None:
            return list(self.position_specs)
        if self.root_position_spec is None:
            raise ValueError("root_position_spec is required when position_specs are omitted.")
        position_specs: List[str] = []
        for node_id, (parent_id, move) in enumerate(zip(self.parent_index.tolist(), self.incoming_moves)):
            if parent_id < 0:
                position_specs.append(str(self.root_position_spec))
            else:
                if move is None:
                    raise ValueError("Non-root node is missing incoming move.")
                position_specs.append(append_move_to_position_spec(position_specs[parent_id], move))
        return position_specs

    def _node_scalar_feature_dicts(self) -> List[Dict[str, float]]:
        features: List[Dict[str, float]] = []
        for row in self.node_features.tolist():
            scalar_features = {}
            for feature_name, value in zip(self.feature_names, row):
                if not math.isnan(float(value)):
                    scalar_features[str(feature_name)] = float(value)
            features.append(scalar_features)
        return features

    def to_pretrain_example(self) -> PretrainExample:
        position_specs = self._resolved_position_specs()
        node_scalar_features = self._node_scalar_feature_dicts()
        metadata_by_node = {int(node_id): dict(node_metadata) for node_id, node_metadata in self.sparse_node_metadata}
        tree = SearchTree()
        tree.root_id = 0 if len(position_specs) > 0 else None
        tree._nodes = []
        tree._children = {node_id: [] for node_id in range(len(position_specs))}
        for node_id, (parent_id, move, fen, depth, terminal, expanded, scalar_features) in enumerate(
            zip(
                self.parent_index.tolist(),
                self.incoming_moves,
                position_specs,
                self.depth.tolist(),
                self.is_terminal.tolist(),
                self.is_expanded.tolist(),
                node_scalar_features,
            )
        ):
            resolved_parent = None if int(parent_id) < 0 else int(parent_id)
            tree._nodes.append(
                SearchNode(
                    node_id=node_id,
                    parent_id=resolved_parent,
                    incoming_move_uci=None if move is None else str(move),
                    fen=str(fen),
                    depth=int(depth),
                    is_terminal=bool(terminal),
                    is_expanded=bool(expanded),
                    scalar_features=scalar_features,
                    metadata=dict(metadata_by_node.get(node_id, {})),
                )
            )
        children_index = self.children_index.tolist()
        child_ptr = self.child_ptr.tolist()
        for node_id in range(len(position_specs)):
            tree._children[node_id] = [int(child_id) for child_id in children_index[child_ptr[node_id]:child_ptr[node_id + 1]]]
        edge_wdl_targets: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        edge_row = 0
        for parent_id in range(len(position_specs)):
            for child_id in tree.child_ids(parent_id):
                target = tuple(float(value) for value in self.edge_wdl_targets[edge_row].tolist())
                edge_wdl_targets[(parent_id, child_id)] = _normalize_wdl_target(target)
                edge_row += 1
        oracle_best_move_trace = [
            self.oracle_root_moves[int(index)]
            for index in self.oracle_best_move_index.tolist()
        ]
        oracle_final_root_q_values = {
            move: float(value)
            for move, value in zip(self.oracle_root_moves, self.oracle_final_root_q_values.tolist())
        }
        return PretrainExample(
            tree=tree,
            node_target_values=self.node_targets.tolist(),
            edge_wdl_targets=edge_wdl_targets,
            metadata=dict(self.metadata),
            oracle_trace_expansion_counts=self.oracle_trace_expansion_counts.tolist(),
            oracle_root_moves=list(self.oracle_root_moves),
            oracle_root_q_trace=self.oracle_root_q_trace.tolist(),
            oracle_best_move_trace=oracle_best_move_trace,
            oracle_final_root_q_values=oracle_final_root_q_values,
        )

    def to_tensorized_tree_example(self, schema) -> TensorizedTreeExample:
        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        num_nodes = int(self.parent_index.shape[0])
        node_features = torch.empty((num_nodes, len(schema.feature_names)), dtype=schema.dtype)
        for column_index, feature_name in enumerate(schema.feature_names):
            default_value = float(schema.defaults.get(feature_name, 0.0))
            if feature_name not in feature_index:
                node_features[:, column_index] = default_value
                continue
            source_column = self.node_features[:, feature_index[feature_name]].to(dtype=schema.dtype)
            if torch.isnan(source_column).any():
                source_column = torch.where(
                    torch.isnan(source_column),
                    torch.full_like(source_column, default_value),
                    source_column,
                )
            node_features[:, column_index] = source_column
        counts = (self.child_ptr[1:] - self.child_ptr[:-1]).to(dtype=torch.long)
        edge_parent = torch.repeat_interleave(torch.arange(num_nodes, dtype=torch.long), counts)
        edge_slot_parts = [
            torch.arange(int(count.item()), dtype=torch.long)
            for count in counts
            if int(count.item()) > 0
        ]
        edge_slot = torch.cat(edge_slot_parts, dim=0) if edge_slot_parts else torch.empty((0,), dtype=torch.long)
        return TensorizedTreeExample(
            node_features=node_features,
            parent_index=self.parent_index.to(dtype=torch.long),
            edge_parent=edge_parent,
            edge_child=self.children_index.to(dtype=torch.long),
            edge_slot=edge_slot,
            depth=self.depth.to(dtype=torch.long),
            node_targets=self.node_targets.to(dtype=torch.float32),
            feature_names=tuple(schema.feature_names),
            edge_wdl_targets=self.edge_wdl_targets.to(dtype=torch.float32),
        )


def save_pretrain_example(path: str, example: PretrainExample) -> None:
    RawPretrainExampleRecord.from_example(example).save(path)


RAW_PRETRAIN_LIST_FORMAT = "cts_raw_pretrain_example_list_v2"


def save_pretrain_examples(path: str, examples: Sequence[PretrainExample]) -> None:
    torch.save(
        {
            "format": RAW_PRETRAIN_LIST_FORMAT,
            "examples": [RawPretrainExampleRecord.from_example(example).to_payload() for example in examples],
        },
        path,
    )


def load_pretrain_examples(path: str) -> List[PretrainExample]:
    payload = torch.load(path, weights_only=False)
    if payload.get("format") != RAW_PRETRAIN_LIST_FORMAT:
        raise ValueError(f"Expected {RAW_PRETRAIN_LIST_FORMAT} at {path}.")
    return [RawPretrainExampleRecord.from_payload(example_payload).to_pretrain_example() for example_payload in payload["examples"]]


def load_raw_pretrain_record(path: str) -> RawPretrainExampleRecord:
    return RawPretrainExampleRecord.load(path)


def load_pretrain_example(path: str) -> PretrainExample:
    return load_raw_pretrain_record(path).to_pretrain_example()


def _pretrain_example_output_path(directory: str, root_position_id: str, index: int) -> str:
    safe_root_position_id = str(root_position_id).replace("/", "_")
    return os.path.join(directory, f"{index:06d}_{safe_root_position_id}.pt")


def save_pretrain_example_to_directory(directory: str, example: PretrainExample, index: int) -> str:
    os.makedirs(directory, exist_ok=True)
    root_position_id = str(example.metadata.get("root_position_id", f"example_{index}"))
    path = _pretrain_example_output_path(directory, root_position_id, index)
    save_pretrain_example(path, example)
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
        examples.append(load_pretrain_example(path))
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
        return load_pretrain_example(self.paths[index])


class PretrainExamplePathDataset(Sequence[PretrainExample]):
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("At least one pretrain example path is required.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PretrainExample:
        return load_pretrain_example(self.paths[index])


class PackedTensorizedShardDataset(Sequence[TensorizedTreeExample]):
    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed shard entries found in manifest: {manifest_path}")

        # Preload all shards and flatten into individual examples to avoid
        # shard-thrashing under shuffled access and view-pinning of large shard
        # storages.  Total memory is ~20KB per example (vs ~600MB per shard).
        self._examples: List[TensorizedTreeExample] = []
        for entry in entries:
            path = entry["path"]
            payload = torch.load(path, weights_only=False)
            if not isinstance(payload, dict) or payload.get("format") != "cts_tensorized_pretrain_shard_v1":
                raise ValueError(f"Packed tensorized shard file has unexpected format: {path}")
            node_ptr = payload["node_ptr"]
            edge_ptr = payload["edge_ptr"]
            feature_names = tuple(payload["feature_names"])
            edge_wdl_targets = payload.get("edge_wdl_targets")
            num_examples = int(entry["num_examples"])
            for i in range(num_examples):
                ns = int(node_ptr[i].item())
                ne = int(node_ptr[i + 1].item())
                es = int(edge_ptr[i].item())
                ee = int(edge_ptr[i + 1].item())
                if "edge_slot" in payload:
                    edge_slot = payload["edge_slot"][es:ee].clone()
                else:
                    edge_slot = torch.arange(ee - es, dtype=torch.long)
                self._examples.append(TensorizedTreeExample(
                    node_features=payload["node_features"][ns:ne].clone(),
                    parent_index=payload["parent_index"][ns:ne].clone(),
                    edge_parent=payload["edge_parent"][es:ee].clone(),
                    edge_child=payload["edge_child"][es:ee].clone(),
                    edge_slot=edge_slot,
                    depth=payload["depth"][ns:ne].clone(),
                    node_targets=payload["node_targets"][ns:ne].clone(),
                    feature_names=feature_names,
                    edge_wdl_targets=edge_wdl_targets[es:ee].clone() if edge_wdl_targets is not None else None,
                ))
            del payload

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TensorizedTreeExample:
        return self._examples[index]

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
        if manifest_format == "cts_tensorized_pretrain_manifest_v1":
            return PackedTensorizedShardDataset(path)
        raise ValueError(f"Unsupported pretrain manifest format: {manifest_format!r}")

    with open(path, "r", encoding="utf-8") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    if not paths:
        raise ValueError(f"No pretrain example paths found in manifest: {path}")
    return PretrainExamplePathDataset(paths)


def load_raw_pretrain_example_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        paths = [
            str(candidate)
            for candidate in sorted(Path(path).rglob("*.pt"))
            if candidate.is_file()
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
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    return dict(checkpoint.get("metadata", {}))


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
    leaf_wdl: Optional[Sequence[float]] = None,
) -> None:
    value = leaf_value
    wdl = None if leaf_wdl is None else _normalize_wdl_target(leaf_wdl)
    for parent_id, child_id in reversed(path):
        value = -value
        stats = edge_stats[(parent_id, child_id)]
        stats.visit_count += 1
        stats.total_value += value
        stats.q_value = stats.total_value / stats.visit_count
        if wdl is not None:
            wdl = _flip_wdl_target(wdl)
            total_wdl = tuple(stats.total_wdl[index] + wdl[index] for index in range(3))
            stats.total_wdl = total_wdl
            stats.mean_wdl = tuple(component / stats.visit_count for component in total_wdl)


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


def _backup_target_from_child_wdl(
    tree: SearchTree,
    node_id: int,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
) -> Tuple[float, float, float]:
    child_ids = tree.child_ids(node_id)
    if not child_ids:
        return _static_node_wdl(tree, node_id)

    keyed_child_stats = [edge_stats[(node_id, child_id)] for child_id in child_ids]
    total_visits = sum(stats.visit_count for stats in keyed_child_stats)
    if total_visits == 0:
        return _static_node_wdl(tree, node_id)

    weighted = [0.0, 0.0, 0.0]
    for stats in keyed_child_stats:
        for index, value in enumerate(stats.mean_wdl):
            weighted[index] += stats.visit_count * value
    return tuple(component / total_visits for component in weighted)


def _edge_target_wdls_from_edge_stats(
    tree: SearchTree,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    edge_targets: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for node in tree.iter_nodes():
        for child_id in tree.child_ids(node.node_id):
            edge_key = (node.node_id, child_id)
            stats = edge_stats[edge_key]
            if stats.visit_count > 0 and sum(stats.mean_wdl) > 0.0:
                edge_targets[edge_key] = _normalize_wdl_target(stats.mean_wdl)
            else:
                child_wdl = _maybe_static_node_wdl(tree, child_id)
                if child_wdl is None:
                    return {}
                edge_targets[edge_key] = _flip_wdl_target(child_wdl)
    return edge_targets


def _root_q_values_from_edge_stats(
    tree: SearchTree,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
) -> Dict[str, float]:
    root_id = tree.root_id
    if root_id is None:
        raise ValueError("Tree must contain a root.")

    root_q_values: Dict[str, float] = {}
    for child_id in tree.root_children():
        move_uci = tree.get_node(child_id).incoming_move_uci
        if move_uci is None:
            raise ValueError(f"Root child {child_id} is missing an incoming move.")
        stats = edge_stats.get((root_id, child_id))
        root_q_values[str(move_uci)] = float(stats.q_value) if stats is not None else 0.0
    return root_q_values


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
    oracle_trace_expansion_counts: List[int] = []
    oracle_root_moves: List[str] = []
    oracle_root_q_trace: List[List[float]] = []
    oracle_best_move_trace: List[str] = []

    def _record_oracle_root_trace() -> None:
        nonlocal oracle_root_moves
        if tree.root_id is None:
            return
        root_children = tree.root_children()
        if not root_children:
            return
        if not oracle_root_moves:
            oracle_root_moves = []
            for child_id in root_children:
                move_uci = tree.get_node(child_id).incoming_move_uci
                if move_uci is None:
                    raise ValueError(f"Root child {child_id} is missing an incoming move.")
                oracle_root_moves.append(str(move_uci))
        root_q_values = _root_q_values_from_edge_stats(tree, edge_stats)
        row = [float(root_q_values[move]) for move in oracle_root_moves]
        best_move = oracle_root_moves[max(range(len(row)), key=row.__getitem__)]
        oracle_trace_expansion_counts.append(num_expansions)
        oracle_root_q_trace.append(row)
        oracle_best_move_trace.append(best_move)

    while num_expansions < sampled_node_budget and _has_expandable_frontier(tree, config):
        node_id, path = _select_leaf_by_puct(tree, edge_stats, config)
        node = tree.get_node(node_id)
        leaf_value = _static_node_value(tree, node_id, config.value_feature)
        leaf_wdl = _maybe_static_node_wdl(tree, node_id)

        if node.is_terminal or node.depth >= config.max_depth:
            node.is_terminal = True
            _backpropagate_path(edge_stats, path, leaf_value, leaf_wdl)
            continue

        raw_children = provider.expand_node(node.fen, node.depth)
        children = _prepare_children(
            raw_children,
            prior_feature=config.prior_feature,
        )
        if not children:
            node.is_terminal = True
            _backpropagate_path(edge_stats, path, leaf_value, leaf_wdl)
            continue

        child_ids = tree.add_children(node_id, children)
        for child_id in child_ids:
            edge_stats[(node_id, child_id)] = EdgeStats()
        num_expansions += 1
        _backpropagate_path(edge_stats, path, leaf_value, leaf_wdl)
        _record_oracle_root_trace()

    return GeneratedTree(
        tree=tree,
        edge_stats=edge_stats,
        sampled_node_budget=sampled_node_budget,
        num_expansions=num_expansions,
        oracle_trace_expansion_counts=oracle_trace_expansion_counts,
        oracle_root_moves=oracle_root_moves,
        oracle_root_q_trace=oracle_root_q_trace,
        oracle_best_move_trace=oracle_best_move_trace,
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
        edge_target_wdls=_edge_target_wdls_from_edge_stats(tree, generated_tree.edge_stats),
    )


def _static_node_value(tree: SearchTree, node_id: int, value_feature: str) -> float:
    node = tree.get_node(node_id)
    if value_feature not in node.scalar_features:
        raise KeyError(f"Node {node_id} is missing feature '{value_feature}'.")
    return float(node.scalar_features[value_feature])


def _static_node_wdl(tree: SearchTree, node_id: int) -> Tuple[float, float, float]:
    node = tree.get_node(node_id)
    try:
        return _normalize_wdl_target(
            (
                node.scalar_features["wdl_win"],
                node.scalar_features["wdl_draw"],
                node.scalar_features["wdl_loss"],
            )
        )
    except KeyError as exc:
        raise KeyError(f"Node {node_id} is missing WDL feature {exc.args[0]!r}.") from exc


def _maybe_static_node_wdl(tree: SearchTree, node_id: int) -> Optional[Tuple[float, float, float]]:
    try:
        return _static_node_wdl(tree, node_id)
    except KeyError:
        return None


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
        leaf_wdl = _maybe_static_node_wdl(tree, node_id)
        _backpropagate_path(edge_stats, path, leaf_value, leaf_wdl)

    node_target_values = [
        float(_backup_target_from_child_q(tree, node.node_id, edge_stats, config.value_feature))
        for node in tree.iter_nodes()
    ]

    return TeacherSearchResult(
        node_target_values=node_target_values,
        edge_stats=edge_stats,
        edge_target_wdls=_edge_target_wdls_from_edge_stats(tree, edge_stats),
    )


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
        "edge_wdl_target_generation_version": "search_consolidated_edge_wdl_v1",
    }
    oracle_trace_expansion_counts: List[int] = []
    oracle_root_moves: List[str] = []
    oracle_root_q_trace: List[List[float]] = []
    oracle_best_move_trace: List[str] = []
    oracle_final_root_q_values: Dict[str, float] = {}
    if node_budget_distribution is not None:
        oracle_trace_expansion_counts = list(generated_tree.oracle_trace_expansion_counts)
        oracle_root_moves = list(generated_tree.oracle_root_moves)
        oracle_root_q_trace = [list(row) for row in generated_tree.oracle_root_q_trace]
        oracle_best_move_trace = list(generated_tree.oracle_best_move_trace)
        oracle_final_root_q_values = _root_q_values_from_edge_stats(tree, generated_tree.edge_stats)
        metadata["oracle_trace_generation_version"] = "generated_search_root_q_v1"
    return PretrainExample(
        tree=tree,
        node_target_values=teacher_result.node_target_values,
        edge_wdl_targets=teacher_result.edge_target_wdls,
        metadata=metadata,
        oracle_trace_expansion_counts=oracle_trace_expansion_counts,
        oracle_root_moves=oracle_root_moves,
        oracle_root_q_trace=oracle_root_q_trace,
        oracle_best_move_trace=oracle_best_move_trace,
        oracle_final_root_q_values=oracle_final_root_q_values,
    )


def prefix_expansion_count_schedule(tree: SearchTree) -> List[int]:
    if tree.root_id is None:
        raise ValueError("Tree must contain a root.")
    return list(range(0, len(tree.ordered_expansion_parent_ids()) + 1))


def sample_prefix_expansion_count_for_node_budget(
    tree: SearchTree,
    min_nodes: int,
    max_nodes: int,
    rng: Optional[random.Random] = None,
) -> int:
    if min_nodes <= 0:
        raise ValueError("min_nodes must be positive.")
    if max_nodes < min_nodes:
        raise ValueError("max_nodes must be >= min_nodes.")

    rng = rng or random.Random()
    counts = prefix_expansion_count_schedule(tree)
    eligible = [count for count in counts if min_nodes <= count <= max_nodes]
    if not eligible:
        raise ValueError(
            f"Tree with {len(tree.ordered_expansion_parent_ids())} expanded nodes has no root prefix whose expanded-node count falls in [{min_nodes}, {max_nodes}]."
        )

    if min_nodes == max_nodes:
        target_nodes = float(min_nodes)
    else:
        log_min = math.log(min_nodes)
        log_max = math.log(max_nodes)
        target_nodes = math.exp(rng.uniform(log_min, log_max))

    return min(
        eligible,
        key=lambda count: (
            abs(math.log(count) - math.log(target_nodes)),
            count,
        ),
    )


def derive_prefix_pretrain_example(
    source_example: PretrainExample,
    config: TeacherSearchConfig,
    min_nodes: int,
    max_nodes: int,
    rng: Optional[random.Random] = None,
) -> PretrainExample:
    rng = rng or random.Random()
    prefix_expansion_count = sample_prefix_expansion_count_for_node_budget(
        source_example.tree,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        rng=rng,
    )
    prefix_tree = source_example.tree.clone_expansion_prefix(prefix_expansion_count)
    teacher_result = compute_teacher_targets(prefix_tree, config)
    prefix_total_node_count = prefix_tree.num_nodes()

    source_root_position_id = str(source_example.metadata.get("root_position_id", "unknown_root"))
    metadata = dict(source_example.metadata)
    metadata.update(
        {
            "root_position_id": f"{source_root_position_id}__prefix_expanded_{prefix_expansion_count}",
            "source_root_position_id": source_root_position_id,
            "source_num_nodes": source_example.tree.num_nodes(),
            "source_expanded_node_count": len(source_example.tree.ordered_expansion_parent_ids()),
            "prefix_expansion_count": prefix_expansion_count,
            "prefix_total_node_count": prefix_total_node_count,
            "prefix_expanded_node_budget_min": min_nodes,
            "prefix_expanded_node_budget_max": max_nodes,
            "prefix_target_generation_version": "search_consolidated_edge_wdl_prefix_v1",
        }
    )
    return PretrainExample(
        tree=prefix_tree,
        node_target_values=teacher_result.node_target_values,
        edge_wdl_targets=teacher_result.edge_target_wdls,
        metadata=metadata,
    )


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
class ChildWdlMetrics:
    total_loss: float
    num_examples: int
    num_supervised_edges: int
    target_entropy: float = 0.0
    loss_gap: float = 0.0


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
        self.best_decoder_state = copy.deepcopy(self.model.child_wdl_head.state_dict())

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

    def _edge_target_tensor(self, examples: Sequence[PretrainExample], device: torch.device) -> torch.Tensor:
        targets = []
        for example in examples:
            if not example.edge_wdl_targets:
                raise ValueError(
                    "Child-WDL pretraining requires raw examples with search-consolidated edge_wdl_targets."
                )
            targets.append(self.tensorizer.edge_wdl_target_tensor(example.tree, example.edge_wdl_targets))
        if not targets:
            return torch.empty((0, 3), dtype=torch.float32, device=device)
        return torch.cat(targets, dim=0).to(device)

    def _tree_batch_and_edge_targets(self, batch_data, device: torch.device) -> Tuple[Any, torch.Tensor]:
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            tree_batch, _node_targets = batch_data
            if tree_batch.edge_wdl_targets is None:
                raise ValueError(
                    "Packed tensorized child-WDL batches require edge_wdl_targets; repack the dataset with edge targets."
                )
            return tree_batch, tree_batch.edge_wdl_targets.to(device)

        batch_examples = batch_data
        tree_batch = self.tensorizer.tensorize_forest([example.tree for example in batch_examples])
        edge_targets = self._edge_target_tensor(batch_examples, device=device)
        return tree_batch, edge_targets

    def _run_epoch(
        self,
        examples: Sequence[PretrainExample],
        epoch_index: int,
        training: bool,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> ChildWdlMetrics:
        if training:
            self.model.train()
        else:
            self.model.eval()

        metric_device = self.model.encoder.device
        total_loss_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        target_entropy_sum = torch.zeros((), dtype=torch.float32, device=metric_device)
        total_examples = 0
        total_supervised_edges = 0

        batches = self._iter_batches(examples, shuffle=training and self.config.shuffle)
        total_batches = len(batches)
        phase = "train" if training else "validation"
        for batch_index, batch_data in enumerate(batches, start=1):
            tree_batch, edge_targets = self._tree_batch_and_edge_targets(batch_data, device=self.model.encoder.device)
            if edge_targets.numel() == 0:
                continue

            with torch.set_grad_enabled(training):
                output = self.model(tree_batch)
                log_probs = F.log_softmax(output.edge_logits, dim=-1)
                per_edge_loss = -(edge_targets * log_probs).sum(dim=-1)
                total_loss = per_edge_loss.mean()
                target_log_probs = edge_targets.clamp_min(1e-12).log()
                per_edge_target_entropy = -(edge_targets * target_log_probs).sum(dim=-1)
                target_entropy = per_edge_target_entropy.mean()
                loss_gap = total_loss - target_entropy

                if training:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            batch_size = int(tree_batch.batch_size)
            num_supervised_edges = int(edge_targets.shape[0])
            total_examples += batch_size
            total_supervised_edges += num_supervised_edges
            total_loss_sum += total_loss.detach() * num_supervised_edges
            target_entropy_sum += target_entropy.detach() * num_supervised_edges
            if batch_progress_callback is not None:
                batch_progress_callback(
                    epoch_index,
                    phase,
                    batch_index,
                    total_batches,
                    total_examples,
                    total_supervised_edges,
                    float(total_loss.item()),
                    float(target_entropy.item()),
                    float(loss_gap.item()),
                )

        if total_supervised_edges == 0:
            return ChildWdlMetrics(total_loss=0.0, num_examples=total_examples, num_supervised_edges=0)

        mean_total_loss = float((total_loss_sum / total_supervised_edges).item())
        mean_target_entropy = float((target_entropy_sum / total_supervised_edges).item())
        return ChildWdlMetrics(
            total_loss=mean_total_loss,
            num_examples=total_examples,
            num_supervised_edges=total_supervised_edges,
            target_entropy=mean_target_entropy,
            loss_gap=mean_total_loss - mean_target_entropy,
        )

    def train_epoch(
        self,
        epoch_index: int = 1,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> ChildWdlMetrics:
        return self._run_epoch(
            self.train_examples,
            epoch_index,
            training=True,
            batch_progress_callback=batch_progress_callback,
        )

    def validate(
        self,
        epoch_index: int = 1,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
    ) -> ChildWdlMetrics:
        return self._run_epoch(
            self.validation_examples,
            epoch_index,
            training=False,
            batch_progress_callback=batch_progress_callback,
        )

    def save_training_state(self, path: str, epoch: int) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_encoder_state": self.best_encoder_state,
            "best_decoder_state": self.best_decoder_state,
            "best_validation_loss": self.best_validation_loss,
            "epoch": epoch,
        }, path)

    def load_training_state(self, path: str) -> int:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_encoder_state = state["best_encoder_state"]
        self.best_decoder_state = state["best_decoder_state"]
        self.best_validation_loss = state["best_validation_loss"]
        return int(state["epoch"])

    def fit(
        self,
        progress_callback: Optional[Callable[[int, ChildWdlMetrics, ChildWdlMetrics], None]] = None,
        batch_progress_callback: Optional[Callable[[int, str, int, int, int, int, float, float, float], None]] = None,
        start_epoch: int = 1,
        resume_path: Optional[str] = None,
    ) -> List[Dict[str, ChildWdlMetrics]]:
        history: List[Dict[str, ChildWdlMetrics]] = []
        for epoch_index in range(start_epoch, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch_index, batch_progress_callback=batch_progress_callback)
            validation_metrics = self.validate(epoch_index, batch_progress_callback=batch_progress_callback)
            history.append({"train": train_metrics, "validation": validation_metrics})
            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.best_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
                self.best_decoder_state = copy.deepcopy(self.model.child_wdl_head.state_dict())
            if progress_callback is not None:
                progress_callback(epoch_index, train_metrics, validation_metrics)
            if resume_path is not None:
                self.save_training_state(resume_path, epoch_index)
        self.model.encoder.load_state_dict(self.best_encoder_state)
        self.model.child_wdl_head.load_state_dict(self.best_decoder_state)
        return history

    def save_best_encoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        save_encoder_checkpoint(path, self.model.encoder, metadata=metadata)

    def save_best_decoder(self, path: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        torch.save({
            "decoder_state_dict": self.model.child_wdl_head.state_dict(),
            "metadata": dict(metadata or {}),
        }, path)
