"""Teacher search and pretrain-example format/IO.

Extracted out of the monolithic ``cts_pretrain.py`` as part of the migration
to the ``cts`` package layout. This module owns:

- The on-disk pretrain example format (``cts_raw_pretrain_example_v3`` and
  its list counterpart) plus the save/load helpers.
- The teacher PUCT search machinery that converts a partially-expanded tree
  into node value targets and per-edge WDL targets
  (``generate_partial_tree_from_provider``, ``compute_teacher_targets``,
  ``build_pretrain_example``).
- The prefix-snapshot helpers used by the encoder pretraining data chain
  (``prefix_expansion_count_schedule``, ``derive_prefix_pretrain_example``).
- The dataset wrappers (``PretrainExampleDirectoryDataset``,
  ``PackedTensorizedShardDataset``) consumed by the training loop.

The training loop itself lives in :mod:`cts.train.gnn_pretrain`.

Slot ordering is load-bearing: ``_child_ptr_and_children_index`` sorts
children by UCI move string and must agree with
``cts.core.tensorizer._sorted_child_ids_with_slots``.
"""

from __future__ import annotations

import json
import math
import os
import random
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from cts.core.schema import (
    TEACHER_NODETARGETS_FEATURE_NAMES,
    TREE_ENCODER_FEATURE_NAMES,
    tree_encoder_feature_schema,
)
from cts.core.tensorizer import (
    TensorizedTreeExample,
    collate_tensorized_examples,
    collate_tensorized_examples_for_nodetargets,
)
from cts.core.tree import ExpansionChild, SearchNode, SearchTree
from cts.core.providers.common import append_move_to_position_spec
from cts.core.providers.base import TreeExpansionProvider


@dataclass(frozen=True)
class TeacherSearchConfig:
    """PUCT hyperparameters for the teacher search that produces pretraining targets.

    The same config is used both during tree generation (drives leaf selection
    in ``generate_partial_tree_from_provider``) and during target consolidation
    (drives the post-hoc backup pass in ``compute_teacher_targets``).
    """

    max_depth: int  # plies-from-root cap; deeper nodes are treated as terminal during search
    search_budget: int  # number of PUCT simulations in ``compute_teacher_targets``
    c_puct: float = 1.0  # PUCT exploration constant
    prior_feature: str = "prior"  # name of the per-child prior feature on each node
    value_feature: str = "value"  # name of the per-node scalar value feature backed up at leaves
    target_normalization_version: str = "v1"  # stamped into metadata so consumers can detect format drift
    search_config_id: str = "default"  # short label identifying this config in metadata

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if self.search_budget <= 0:
            raise ValueError("search_budget must be positive.")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be non-negative.")


@dataclass(frozen=True)
class NodeBudgetDistribution:
    """Log-uniform distribution over the per-tree expanded-node budget."""

    min_nodes: int  # smallest expansion count to sample (inclusive)
    max_nodes: int  # largest expansion count to sample (inclusive)

    def __post_init__(self) -> None:
        if self.min_nodes <= 0:
            raise ValueError("min_nodes must be positive.")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be >= min_nodes.")

    def sample(self, rng: Optional[random.Random] = None) -> int:
        """Return a node budget drawn log-uniformly from ``[min_nodes, max_nodes]``."""
        rng = rng or random
        if self.min_nodes == self.max_nodes:
            return self.min_nodes
        # Log-uniform sampling gives equal weight to each order of magnitude
        # of tree size, which matches the distribution of trees we want to
        # train on (lots of small trees, fewer large ones).
        log_min = math.log(self.min_nodes)
        log_max = math.log(self.max_nodes)
        sampled = math.exp(rng.uniform(log_min, log_max))
        return max(self.min_nodes, min(self.max_nodes, int(round(sampled))))


def _normalize_wdl_target(target: Sequence[float]) -> Tuple[float, float, float]:
    """Clip-negatives + L1-normalize a length-3 WDL vector. Raises on zero mass."""
    if len(target) != 3:
        raise ValueError(f"WDL target must have length 3, got {len(target)}.")
    values = [max(0.0, float(value)) for value in target]
    total = sum(values)
    if total <= 0.0:
        raise ValueError("WDL target must have positive mass.")
    return (values[0] / total, values[1] / total, values[2] / total)


def _flip_wdl_target(target: Sequence[float]) -> Tuple[float, float, float]:
    """Swap win/loss to convert a child's WDL into its parent's side-to-move WDL."""
    win, draw, loss = _normalize_wdl_target(target)
    return (loss, draw, win)


@dataclass
class EdgeStats:
    """Per-edge running statistics accumulated during PUCT rollouts."""

    visit_count: int = 0  # number of PUCT visits that traversed this edge
    total_value: float = 0.0  # sum of side-corrected leaf values backed up through this edge
    total_wdl: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # sum of side-corrected leaf WDLs

    @property
    def q_value(self) -> float:
        """Action-value estimate: ``total_value / visit_count`` (0 when unvisited)."""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    @property
    def mean_wdl(self) -> Tuple[float, float, float]:
        """Mean per-component WDL: ``total_wdl / visit_count`` (zeros when unvisited)."""
        if self.visit_count == 0:
            return (0.0, 0.0, 0.0)
        return (
            self.total_wdl[0] / self.visit_count,
            self.total_wdl[1] / self.visit_count,
            self.total_wdl[2] / self.visit_count,
        )


@dataclass
class TeacherSearchResult:
    """Targets produced by the teacher search over an already-built tree."""

    node_target_values: List[float]  # per-node scalar value target (indexed by node_id)
    edge_stats: Dict[Tuple[int, int], EdgeStats]  # per-edge stats keyed by (parent_id, child_id)
    edge_target_wdls: Dict[Tuple[int, int], Tuple[float, float, float]]  # per-edge WDL pretraining target


@dataclass
class GeneratedTree:
    """Output of ``generate_partial_tree_from_provider`` — tree + per-step oracle trace.

    The oracle trace records the root's best move and per-child Q-values after
    every expansion, enabling downstream consumers to study how the teacher's
    preference evolved with more search.
    """

    tree: SearchTree  # the partially-expanded tree
    edge_stats: Dict[Tuple[int, int], EdgeStats]  # PUCT stats accumulated during generation
    sampled_node_budget: int  # the budget drawn from ``NodeBudgetDistribution`` for this tree
    num_expansions: int  # how many expansions actually happened (may be < budget if frontier exhausts)
    oracle_trace_expansion_counts: List[int] = field(default_factory=list)  # expansion-step indices at which the trace was sampled
    oracle_root_moves: List[str] = field(default_factory=list)  # canonical move ordering for the trace columns
    oracle_root_q_trace: List[List[float]] = field(default_factory=list)  # per-step Q-values aligned with oracle_root_moves
    oracle_best_move_trace: List[str] = field(default_factory=list)  # per-step argmax move
    oracle_root_visits_trace: List[List[int]] = field(default_factory=list)  # per-step visit counts aligned with oracle_root_moves


@dataclass
class PretrainExample:
    """The in-memory training example consumed by the encoder pretrainer.

    Topology pretraining uses **node-wise** supervision (``value_gap`` per node).
    Optional per-edge WDL targets remain for the legacy Child-WDL pretrain path.
    """

    tree: SearchTree  # the teacher-built tree
    node_target_values: List[float]  # one target per node, aligned with tree.iter_nodes()
    edge_wdl_targets: Dict[Tuple[int, int], Tuple[float, float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    oracle_trace_expansion_counts: List[int] = field(default_factory=list)
    oracle_root_moves: List[str] = field(default_factory=list)
    oracle_root_q_trace: List[List[float]] = field(default_factory=list)
    oracle_best_move_trace: List[str] = field(default_factory=list)
    oracle_final_root_q_values: Dict[str, float] = field(default_factory=dict)
    value_gap: List[float] = field(default_factory=list)
    policy_drift: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Normalize and validate at the boundary so the rest of the pipeline
        # can assume canonical types and shapes.
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
        self.value_gap = [float(value) for value in self.value_gap]
        self.policy_drift = [float(value) for value in self.policy_drift]
        # --- Shape/coverage invariants ---
        num_nodes = self.tree.num_nodes()
        if self.value_gap and len(self.value_gap) != num_nodes:
            raise ValueError("value_gap must match tree.num_nodes().")
        if self.policy_drift and len(self.policy_drift) != num_nodes:
            raise ValueError("policy_drift must match tree.num_nodes().")
        if len(self.node_target_values) != num_nodes:
            raise ValueError("node_target_values must match tree.num_nodes().")
        if self.edge_wdl_targets and len(self.edge_wdl_targets) != self.tree.num_edges():
            raise ValueError("edge_wdl_targets must match tree.num_edges().")
        # --- Oracle trace consistency: either everything's empty, or all
        # the trace fields are present and mutually aligned. ---
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

# Disk format tag for raw (un-packed) pretrain examples. The ``v2`` bump
# was made when ``_child_ptr_and_children_index`` switched to UCI-sorted
# child order — older files using insertion order are incompatible.
# ``v5`` is node-only topology supervision: per-node ``value_gap`` (no ``n_visits`` column,
# no edge WDL targets in the default generation path).
RAW_PRETRAIN_FORMAT = "cts_raw_pretrain_example_v5"
RAW_PRETRAIN_LEGACY_FORMATS = frozenset(
    {
        "cts_raw_pretrain_example_v2",
        "cts_raw_pretrain_example_v3",
        "cts_raw_pretrain_example_v4",
        RAW_PRETRAIN_FORMAT,
    }
)
RAW_PRETRAIN_V3_FORMAT = "cts_raw_pretrain_example_v3"

# Teacher ``value`` scalars are win-loss in ``[-1, 1]``; multiply by this for centipawn targets.
VALUE_SCALAR_TO_CENTIPAWNS = 100.0


def _ordered_feature_names_from_tree(tree: SearchTree) -> Tuple[str, ...]:
    """Collect every feature name observed in the tree, in first-seen order.

    Used when serializing a tree to disk so the column order is recorded in
    the file rather than implicit. Consumers project this back onto the
    encoder schema at load time.
    """
    ordered: List[str] = []
    seen = set()
    for node in tree.iter_nodes():
        for name in node.scalar_features:
            if name not in seen:
                seen.add(name)
                ordered.append(str(name))
    return tuple(ordered)


def _node_visit_counts_from_edge_stats(
    tree: SearchTree,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
) -> List[int]:
    """Map PUCT edge visit counts to per-node visit totals.

    Non-root nodes inherit the visit count on the incoming parent→child edge.
    The root uses the sum of outgoing edge visits (total rollouts through the root).
    """
    counts = [0] * tree.num_nodes()
    for node in tree.iter_nodes():
        node_id = node.node_id
        if node.parent_id is None:
            counts[node_id] = sum(
                edge_stats.get((node_id, child_id), EdgeStats()).visit_count
                for child_id in tree.child_ids(node_id)
            )
        else:
            counts[node_id] = edge_stats.get((node.parent_id, node_id), EdgeStats()).visit_count
    return counts


def scalar_value_to_centipawns(value: float) -> float:
    """Map a side-correct ``value`` scalar (win-loss in ``[-1, 1]``) to centipawns."""
    return float(VALUE_SCALAR_TO_CENTIPAWNS * value)


def child_q_from_parent_perspective(
    tree: SearchTree,
    parent_id: int,
    child_id: int,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
    *,
    value_feature: str = "value",
) -> float:
    """Return the parent-side Q estimate for edge ``parent_id → child_id``.

    Uses visit-mean edge ``q_value`` when the edge was traversed during search;
    otherwise falls back to ``-static_child_value`` (negamax flip).
    """
    stats = edge_stats.get((parent_id, child_id))
    if stats is not None and stats.visit_count > 0:
        return float(stats.q_value)
    return float(-_static_node_value(tree, child_id, value_feature))


def node_value_gap_centipawns(
    tree: SearchTree,
    node_id: int,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
    *,
    value_feature: str = "value",
) -> float:
    """Centipawn gap between the best and second-best child Q at ``node_id``.

    Returns ``float('nan')`` for leaves and nodes with fewer than two children.
    Ties among the top two children yield ``0.0``.
    """
    child_ids = tree.child_ids(node_id)
    if len(child_ids) < 2:
        return float("nan")
    child_q = [
        child_q_from_parent_perspective(
            tree,
            node_id,
            child_id,
            edge_stats,
            value_feature=value_feature,
        )
        for child_id in child_ids
    ]
    child_q.sort(reverse=True)
    best_q, second_q = child_q[0], child_q[1]
    return scalar_value_to_centipawns(best_q) - scalar_value_to_centipawns(second_q)


def node_value_gap_features(
    tree: SearchTree,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
    *,
    value_feature: str = "value",
) -> List[float]:
    """Per-node value-gap vector aligned with ``tree.iter_nodes()``."""
    return [
        node_value_gap_centipawns(tree, node.node_id, edge_stats, value_feature=value_feature)
        for node in tree.iter_nodes()
    ]


def _subtree_topology_for_node(tree: SearchTree, node_id: int) -> Tuple[int, int, int]:
    """Return ``(nodes_below, max_breadth_relative, max_depth_relative)`` for one node.

    ``nodes_below`` counts strict descendants. ``max_depth_relative`` is the
    maximum ply distance to any descendant. ``max_breadth_relative`` is the widest
    depth slice in the subtree below this node (not counting siblings above).
    """
    nodes_below = 0
    max_rel_depth = 0
    breadth_by_level: Dict[int, int] = {}
    stack: List[Tuple[int, int]] = [(child_id, 1) for child_id in tree.child_ids(node_id)]
    while stack:
        current_id, rel_depth = stack.pop()
        nodes_below += 1
        max_rel_depth = max(max_rel_depth, rel_depth)
        breadth_by_level[rel_depth] = breadth_by_level.get(rel_depth, 0) + 1
        for child_id in tree.child_ids(current_id):
            stack.append((child_id, rel_depth + 1))
    max_breadth_relative = max(breadth_by_level.values()) if breadth_by_level else 0
    return nodes_below, max_breadth_relative, max_rel_depth


def _subtree_topology_features(tree: SearchTree) -> Tuple[List[int], List[int], List[int]]:
    """Compute per-node subtree topology stats aligned with ``tree.iter_nodes()``."""
    nodes_below: List[int] = []
    max_breadth_relative: List[int] = []
    max_depth_relative: List[int] = []
    for node in tree.iter_nodes():
        below, breadth, rel_depth = _subtree_topology_for_node(tree, node.node_id)
        nodes_below.append(below)
        max_breadth_relative.append(breadth)
        max_depth_relative.append(rel_depth)
    return nodes_below, max_breadth_relative, max_depth_relative


def scaled_teacher_node_matrix(record: "RawPretrainExampleRecord") -> torch.Tensor:
    """Stack and scale selected teacher targets defined in TEACHER_NODETARGETS_FEATURE_NAMES."""
    num_nodes = int(record.parent_index.shape[0])
    tensors = []
    for name in TEACHER_NODETARGETS_FEATURE_NAMES:
        column = getattr(record, name)
        if int(column.shape[0]) != num_nodes:
            raise ValueError(f"{name} length {column.shape[0]} != num_nodes={num_nodes}.")
        tensors.append(column.to(dtype=torch.float32))
    return torch.stack(tensors, dim=-1)


def _subtree_topology_from_csr(
    parent_index: Sequence[int],
    child_ptr: Sequence[int],
    children_index: Sequence[int],
) -> Tuple[List[int], List[int], List[int]]:
    """CSR variant of ``_subtree_topology_features`` for legacy record loads."""
    num_nodes = len(parent_index)
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    for parent_id in range(num_nodes):
        children[parent_id] = [int(children_index[i]) for i in range(int(child_ptr[parent_id]), int(child_ptr[parent_id + 1]))]

    nodes_below: List[int] = []
    max_breadth_relative: List[int] = []
    max_depth_relative: List[int] = []
    for node_id in range(num_nodes):
        below = 0
        max_rel_depth = 0
        breadth_by_level: Dict[int, int] = {}
        stack = [(child_id, 1) for child_id in children[node_id]]
        while stack:
            current_id, rel_depth = stack.pop()
            below += 1
            max_rel_depth = max(max_rel_depth, rel_depth)
            breadth_by_level[rel_depth] = breadth_by_level.get(rel_depth, 0) + 1
            for child_id in children[current_id]:
                stack.append((child_id, rel_depth + 1))
        nodes_below.append(below)
        max_breadth_relative.append(max(breadth_by_level.values()) if breadth_by_level else 0)
        max_depth_relative.append(max_rel_depth)
    return nodes_below, max_breadth_relative, max_depth_relative


def _dense_node_feature_tensor(tree: SearchTree, feature_names: Sequence[str]) -> torch.Tensor:
    """Materialize the per-node feature matrix in ``feature_names`` column order.

    Missing features become NaN rather than 0 so consumers can distinguish
    "feature absent from this node" from "feature is exactly zero".
    """
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
    """Build the CSR-style (child_ptr, children_index) pair in UCI-sorted slot order.

    Children are stored in UCI-lexicographic order so that ``edge_slot`` derived
    from ``arange(count_per_parent)`` matches the canonical slot assignment used
    by ``cts.core.tensorizer._sorted_child_ids_with_slots``. Engine-dependent insertion
    order is non-canonical; sorting here is what makes the on-disk representation
    stable across providers. This is the fix that motivated the ``v2`` format bump.
    """
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
    edge_wdl_targets: Optional[Mapping[Tuple[int, int], Sequence[float]]],
) -> torch.Tensor:
    """Stack per-edge WDL targets in canonical edge order, or NaN rows when absent."""
    rows: List[Tuple[float, float, float]] = []
    for node in tree.iter_nodes():
        for child_id in tree.child_ids(node.node_id):
            edge_key = (node.node_id, child_id)
            if edge_wdl_targets and edge_key in edge_wdl_targets:
                rows.append(_normalize_wdl_target(edge_wdl_targets[edge_key]))
            else:
                rows.append((float("nan"), float("nan"), float("nan")))
    if not rows:
        return torch.empty((0, 3), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _record_has_edge_wdl_targets(edge_wdl_targets: torch.Tensor) -> bool:
    return edge_wdl_targets.numel() > 0 and bool(torch.isfinite(edge_wdl_targets).any().item())


def _compact_position_spec_payload(tree: SearchTree) -> Dict[str, Any]:
    """Decide whether to store just the root FEN or every node's FEN.

    If every non-root node's FEN can be reconstructed by applying its
    incoming UCI move to its parent's FEN, we save only the root spec —
    consumers can replay the moves at load time. Otherwise the per-node
    list is required.
    """
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
    """On-disk representation of a single ``PretrainExample``.

    Flattens a ``SearchTree`` into tensor columns (parent_index, child_ptr,
    children_index, etc.) so the file is dense, self-describing, and quick
    to load. ``__post_init__`` normalizes every field to the canonical dtype
    and device on construction; ``_validate`` enforces shape invariants.
    """

    root_position_spec: Optional[str]  # FEN of the root; None only for empty trees
    incoming_moves: List[Optional[str]]  # UCI move into each node (None for root)
    feature_names: Tuple[str, ...]  # column order of ``node_features``
    node_features: torch.Tensor  # [N, F] dense features; missing values are NaN
    parent_index: torch.Tensor  # [N] int32; -1 for root
    child_ptr: torch.Tensor  # [N+1] int32 CSR pointer
    children_index: torch.Tensor  # [num_edges] int32; flat child id list in UCI-sorted slot order
    depth: torch.Tensor  # [N] int16; plies from root
    is_terminal: torch.Tensor  # [N] bool
    is_expanded: torch.Tensor  # [N] bool
    node_targets: torch.Tensor  # [N] float32 scalar value targets
    edge_wdl_targets: torch.Tensor  # [num_edges, 3] float32; NaN when edge supervision omitted
    value_gap: torch.Tensor  # [N] float32 centipawn gap (NaN when undefined)
    policy_drift: torch.Tensor  # [N] float32 policy drift (NaN when undefined)
    metadata: Dict[str, Any]  # free-form tree-level metadata
    sparse_node_metadata: List[Tuple[int, Dict[str, Any]]]  # only nodes with non-empty metadata get a row
    oracle_trace_expansion_counts: torch.Tensor  # [T] int32; expansion-step indices for the oracle trace
    oracle_root_moves: List[str]  # canonical move ordering for trace columns
    oracle_root_q_trace: torch.Tensor  # [T, len(oracle_root_moves)] float32
    oracle_best_move_index: torch.Tensor  # [T] int32; index into oracle_root_moves
    oracle_final_root_q_values: torch.Tensor  # [len(oracle_root_moves)] float32
    position_specs: Optional[List[str]] = None  # set only when FENs aren't reconstructable from moves

    def __post_init__(self) -> None:
        # Frozen dataclass: rebind every field via object.__setattr__ to its
        # canonical dtype/device. Defensive copies prevent caller-side mutation
        # of mutable fields (lists, dicts) after construction.
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
        object.__setattr__(self, "value_gap", self.value_gap.to(dtype=torch.float32, device="cpu"))
        object.__setattr__(self, "policy_drift", self.policy_drift.to(dtype=torch.float32, device="cpu"))
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
        """Defensive checks that every tensor's shape matches the rest of the record.

        Caught early these errors are obvious; caught late (e.g. inside a
        DataLoader worker) they're cryptic, so we spend the cycles here.
        """
        num_nodes = int(self.parent_index.shape[0])
        if self.node_features.shape != (num_nodes, len(self.feature_names)):
            raise ValueError("node_features must align with feature_names and parent_index.")
        if num_nodes > 0 and int(self.parent_index[0].item()) != -1:
            raise ValueError("The root node must have parent_index -1.")
        # Topological ordering: every non-root node points back to an earlier index.
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
        if self.value_gap.shape != (num_nodes,):
            raise ValueError("value_gap must align with parent_index.")
        if self.policy_drift.shape != (num_nodes,):
            raise ValueError("policy_drift must align with parent_index.")
        # CSR pointer validity.
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
        # Oracle trace shape checks.
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
        """Build a record from an in-memory ``PretrainExample`` for serialization."""
        tree = example.tree
        feature_names = _ordered_feature_names_from_tree(tree)
        child_ptr, children_index = _child_ptr_and_children_index(tree)
        position_payload = _compact_position_spec_payload(tree)
        # Convert the oracle best-move trace from strings to indices into
        # oracle_root_moves so it stores efficiently as an int tensor.
        oracle_best_move_index = torch.tensor(
            [example.oracle_root_moves.index(move) for move in example.oracle_best_move_trace],
            dtype=torch.int32,
        ) if example.oracle_best_move_trace else torch.empty((0,), dtype=torch.int32)
        oracle_final_root_q_values = torch.tensor(
            [float(example.oracle_final_root_q_values[move]) for move in example.oracle_root_moves],
            dtype=torch.float32,
        ) if example.oracle_root_moves else torch.empty((0,), dtype=torch.float32)
        if example.value_gap:
            value_gap = torch.tensor(example.value_gap, dtype=torch.float32)
        else:
            value_gap = torch.full((tree.num_nodes(),), float("nan"), dtype=torch.float32)
        if example.policy_drift:
            policy_drift = torch.tensor(example.policy_drift, dtype=torch.float32)
        else:
            policy_drift = torch.full((tree.num_nodes(),), float("nan"), dtype=torch.float32)
        edge_wdl_tensor = _edge_wdl_target_tensor_for_tree(
            tree,
            example.edge_wdl_targets if example.edge_wdl_targets else None,
        )
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
            edge_wdl_targets=edge_wdl_tensor,
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
            value_gap=value_gap,
            policy_drift=policy_drift,
        )

    def to_payload(self) -> Dict[str, Any]:
        """Return a dict payload ready for ``torch.save``. Mirrors ``from_payload``."""
        payload: Dict[str, Any] = {
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
            "value_gap": self.value_gap,
            "policy_drift": self.policy_drift,
            "metadata": dict(self.metadata),
            "sparse_node_metadata": list(self.sparse_node_metadata),
            "oracle_trace_expansion_counts": self.oracle_trace_expansion_counts,
            "oracle_root_moves": list(self.oracle_root_moves),
            "oracle_root_q_trace": self.oracle_root_q_trace,
            "oracle_best_move_index": self.oracle_best_move_index,
            "oracle_final_root_q_values": self.oracle_final_root_q_values,
            "position_specs": list(self.position_specs) if self.position_specs is not None else None,
        }
        if _record_has_edge_wdl_targets(self.edge_wdl_targets):
            payload["edge_wdl_targets"] = self.edge_wdl_targets
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RawPretrainExampleRecord":
        """Reconstruct a record from a loaded payload dict. Rejects mismatched format tags."""
        record_format = payload.get("format")
        if record_format not in RAW_PRETRAIN_LEGACY_FORMATS:
            raise ValueError(
                f"Expected raw pretrain format in {sorted(RAW_PRETRAIN_LEGACY_FORMATS)}, got {record_format!r}."
            )
        parent_index = payload["parent_index"]
        child_ptr = payload["child_ptr"]
        children_index = payload["children_index"]
        num_nodes = int(parent_index.shape[0])
        num_edges = int(children_index.shape[0])
        if "value_gap" in payload:
            value_gap = payload["value_gap"]
        else:
            value_gap = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
        if "policy_drift" in payload:
            policy_drift = payload["policy_drift"]
        else:
            policy_drift = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
        if "edge_wdl_targets" in payload:
            edge_wdl_targets = payload["edge_wdl_targets"]
        else:
            edge_wdl_targets = torch.full((num_edges, 3), float("nan"), dtype=torch.float32)
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
            edge_wdl_targets=edge_wdl_targets,
            value_gap=value_gap,
            policy_drift=policy_drift,
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
        """Load a single record from disk."""
        payload = torch.load(path, weights_only=False)
        return cls.from_payload(payload)

    def save(self, path: str | Path) -> None:
        """Save this record to disk via ``torch.save``."""
        torch.save(self.to_payload(), path)

    def _resolved_position_specs(self) -> List[str]:
        """Return one FEN per node, reconstructing them from moves if needed."""
        if self.position_specs is not None:
            return list(self.position_specs)
        if self.root_position_spec is None:
            raise ValueError("root_position_spec is required when position_specs are omitted.")
        # Compact path: replay each non-root node's incoming move on its
        # parent's already-resolved FEN. Topological order (enforced in
        # _validate) means the parent is always resolved first.
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
        """Convert the dense feature matrix back to per-node dicts, dropping NaN entries."""
        features: List[Dict[str, float]] = []
        for row in self.node_features.tolist():
            scalar_features = {}
            for feature_name, value in zip(self.feature_names, row):
                if not math.isnan(float(value)):
                    scalar_features[str(feature_name)] = float(value)
            features.append(scalar_features)
        return features

    def to_pretrain_example(self) -> PretrainExample:
        """Rehydrate a record back into a ``PretrainExample`` for in-memory use.

        Bypasses ``SearchTree.add_children`` and writes the private ``_nodes``
        and ``_children`` storage directly — necessary because the stored
        tree is already in its final shape and we don't want to re-run the
        append-only invariants step by step.
        """
        position_specs = self._resolved_position_specs()
        node_scalar_features = self._node_scalar_feature_dicts()
        metadata_by_node = {int(node_id): dict(node_metadata) for node_id, node_metadata in self.sparse_node_metadata}
        if not position_specs:
            raise ValueError("RawPretrainExampleRecord has no nodes; cannot rebuild a tree from it.")
        # --- Rebuild the tree by bypassing __init__ ---
        # The packed record is already in its final shape, so we don't want
        # __init__ to validate + insert a fresh root. We construct the
        # underlying object directly and fill _nodes / _children below.
        tree = object.__new__(SearchTree)
        tree.root_id = 0
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
        # Slice each parent's child block out of the flat children_index.
        children_index = self.children_index.tolist()
        child_ptr = self.child_ptr.tolist()
        for node_id in range(len(position_specs)):
            tree._children[node_id] = [int(child_id) for child_id in children_index[child_ptr[node_id]:child_ptr[node_id + 1]]]
        # --- Rebuild optional edge WDL targets (Child-WDL path only) ---
        edge_wdl_targets: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        if _record_has_edge_wdl_targets(self.edge_wdl_targets):
            edge_row = 0
            for parent_id in range(len(position_specs)):
                for child_id in tree.child_ids(parent_id):
                    target = tuple(float(value) for value in self.edge_wdl_targets[edge_row].tolist())
                    edge_wdl_targets[(parent_id, child_id)] = _normalize_wdl_target(target)
                    edge_row += 1
        # --- Rebuild oracle trace fields ---
        oracle_best_move_trace = [
            self.oracle_root_moves[int(index)]
            for index in self.oracle_best_move_index.tolist()
        ]
        oracle_final_root_q_values = {
            move: float(value)
            for move, value in zip(self.oracle_root_moves, self.oracle_final_root_q_values.tolist())
        }
        value_gap = self.value_gap.tolist()
        if bool(torch.isnan(self.value_gap).all()):
            value_gap = node_value_gap_features(tree, {}, value_feature="value")
        policy_drift = self.policy_drift.tolist()
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
            value_gap=value_gap,
            policy_drift=policy_drift,
        )

    def to_tensorized_tree_example(
        self,
        schema,
        *,
        node_targets: bool = False,
        topology_supervision_shard: bool = False,
    ) -> TensorizedTreeExample:
        """Fill ``node_features`` (and optionally ``topology_targets``) exactly as configured by packing."""

        expected = tree_encoder_feature_schema(node_targets=node_targets)
        if tuple(schema.feature_names) != tuple(expected.feature_names):
            raise ValueError(
                "Record tensorization schema mismatch: "
                f"got columns {schema.feature_names!r}; expected "
                f"{tuple(expected.feature_names)!r} for "
                f"node_targets={node_targets}. "
                "Rebuild with the packing schema from preprocess_gnn.pack."
            )
        scaled_topo: Optional[torch.Tensor] = None
        if node_targets or topology_supervision_shard:
            scaled_topo = scaled_teacher_node_matrix(self)

        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        num_nodes = int(self.parent_index.shape[0])
        # Allocate the output matrix and fill column by column, defaulting
        # missing columns and NaN cells to the schema default.
        node_features = torch.empty((num_nodes, len(schema.feature_names)), dtype=schema.dtype)
        for column_index, feature_name in enumerate(schema.feature_names):
            default_value = float(schema.defaults.get(feature_name, 0.0))
            if feature_name in TEACHER_NODETARGETS_FEATURE_NAMES:
                if node_targets and scaled_topo is not None:
                    topo_col_idx = TEACHER_NODETARGETS_FEATURE_NAMES.index(feature_name)
                    node_features[:, column_index] = scaled_topo[:, topo_col_idx].to(dtype=schema.dtype)
                else:
                     node_features[:, column_index] = default_value
                continue
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
        topology_targets = scaled_topo if topology_supervision_shard else None
        # Derive edge_parent / edge_slot from child_ptr: each parent contributes
        # `count` edges, and edge_slot is 0..count-1 within that block.
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
            edge_wdl_targets=(
                self.edge_wdl_targets.to(dtype=torch.float32)
                if _record_has_edge_wdl_targets(self.edge_wdl_targets)
                else None
            ),
            nodetargets_targets=topology_targets,
        )


def save_pretrain_example(path: str, example: PretrainExample) -> None:
    """Save a single example to ``path`` via the raw record format."""
    RawPretrainExampleRecord.from_example(example).save(path)


# Disk format tag for a list of raw pretrain examples in one file. Used when
# shipping shards rather than one-file-per-example.
RAW_PRETRAIN_LIST_FORMAT = "cts_raw_pretrain_example_list_v2"


def save_pretrain_examples(path: str, examples: Sequence[PretrainExample]) -> None:
    """Save many examples to a single file as a list-format payload."""
    torch.save(
        {
            "format": RAW_PRETRAIN_LIST_FORMAT,
            "examples": [RawPretrainExampleRecord.from_example(example).to_payload() for example in examples],
        },
        path,
    )


def load_pretrain_examples(path: str) -> List[PretrainExample]:
    """Load a list-format file and rehydrate every example."""
    payload = torch.load(path, weights_only=False)
    if payload.get("format") != RAW_PRETRAIN_LIST_FORMAT:
        raise ValueError(f"Expected {RAW_PRETRAIN_LIST_FORMAT} at {path}.")
    return [RawPretrainExampleRecord.from_payload(example_payload).to_pretrain_example() for example_payload in payload["examples"]]


def load_raw_pretrain_record(path: str) -> RawPretrainExampleRecord:
    """Load a single raw record without rehydrating it into a ``PretrainExample``."""
    return RawPretrainExampleRecord.load(path)


def load_pretrain_example(path: str) -> PretrainExample:
    """Load and rehydrate a single example from a per-file record."""
    return load_raw_pretrain_record(path).to_pretrain_example()


def _pretrain_example_output_path(directory: str, root_position_id: str, index: int) -> str:
    """Build the per-example filename. ``index`` is zero-padded so directory listings sort naturally."""
    safe_root_position_id = str(root_position_id).replace("/", "_")
    return os.path.join(directory, f"{index:06d}_{safe_root_position_id}.pt")


def save_pretrain_example_to_directory(directory: str, example: PretrainExample, index: int) -> str:
    """Save one example into ``directory`` using the indexed filename convention."""
    os.makedirs(directory, exist_ok=True)
    root_position_id = str(example.metadata.get("root_position_id", f"example_{index}"))
    path = _pretrain_example_output_path(directory, root_position_id, index)
    save_pretrain_example(path, example)
    return path


def save_pretrain_examples_to_directory(directory: str, examples: Sequence[PretrainExample]) -> List[str]:
    """Save every example into ``directory`` and return the list of written paths."""
    os.makedirs(directory, exist_ok=True)
    saved_paths = []
    for index, example in enumerate(examples):
        saved_paths.append(save_pretrain_example_to_directory(directory, example, index))
    return saved_paths


def load_pretrain_examples_from_directory(directory: str) -> List[PretrainExample]:
    """Load every ``*.pt`` file in ``directory`` as a ``PretrainExample`` (sorted by filename)."""
    examples = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".pt"):
            continue
        path = os.path.join(directory, filename)
        examples.append(load_pretrain_example(path))
    return examples


class PretrainExampleDirectoryDataset(Sequence[PretrainExample]):
    """Lazy-loading dataset backed by a directory of per-example ``.pt`` files."""

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
    """Lazy-loading dataset backed by an explicit list of file paths."""

    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("At least one pretrain example path is required.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PretrainExample:
        return load_pretrain_example(self.paths[index])


class PackedTensorizedShardDataset(Sequence[TensorizedTreeExample]):
    """Dataset over pre-tensorized, pre-packed encoder shards.

    Shards are the fastest path for training: they store the tensorized form
    (encoder schema columns, ``edge_parent``, ``edge_slot``, etc.) in CSR
    blocks so loading is one ``torch.load`` per shard rather than one rebuild
    per example. The constructor flattens every shard up-front so subsequent
    ``__getitem__`` is O(1) and shuffled access doesn't thrash shards.
    """

    def __init__(self, manifest_path: str) -> None:
        self.manifest_path = manifest_path
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        entries = manifest.get("entries", [])
        if not entries:
            raise ValueError(f"No packed shard entries found in manifest: {manifest_path}")

        self.expect_topology_targets = bool(manifest.get("topology_targets", False)) or bool(manifest.get("nodetargets_targets", False))

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
            shard_topology = payload.get("topology_targets")
            num_examples = int(entry["num_examples"])
            # Slice each example's nodes/edges out of the shard's flat tensors
            # and clone so the slice can outlive the shard payload (which we
            # drop right after).
            for i in range(num_examples):
                ns = int(node_ptr[i].item())
                ne = int(node_ptr[i + 1].item())
                es = int(edge_ptr[i].item())
                ee = int(edge_ptr[i + 1].item())
                if "edge_slot" in payload:
                    edge_slot = payload["edge_slot"][es:ee].clone()
                else:
                    # Older shards didn't store edge_slot explicitly; derive
                    # the within-parent slot id by assuming the canonical
                    # consecutive ordering inside each parent block.
                    edge_slot = torch.arange(ee - es, dtype=torch.long)
                node_feat_block = payload["node_features"][ns:ne].clone()
                topology_slice = None
                if shard_topology is not None:
                    topology_slice = shard_topology[ns:ne].clone()
                elif self.expect_topology_targets:
                    tail_need = tuple(TEACHER_NODETARGETS_FEATURE_NAMES)
                    if (
                        node_feat_block.shape[1] >= len(TREE_ENCODER_FEATURE_NAMES) + len(tail_need)
                        and tuple(feature_names[-len(tail_need) :]) == tail_need
                    ):
                        topology_slice = node_feat_block[:, -len(tail_need) :].clone()
                    else:
                        raise ValueError(
                            f"Manifest requires nodetargets_targets but shard {path} has no "
                            "nodetargets_targets key and node_features tail does not match "
                            f"{tail_need!r}; repack or use shards with nodetargets_targets written."
                        )
                elif (
                    shard_topology is None
                    and node_feat_block.shape[1] >= len(TREE_ENCODER_FEATURE_NAMES) + len(TEACHER_NODETARGETS_FEATURE_NAMES)
                    and tuple(feature_names[-len(TEACHER_NODETARGETS_FEATURE_NAMES) :]) == tuple(TEACHER_NODETARGETS_FEATURE_NAMES)
                ):
                    # Optional: hydrate supervision from wide features for callers that omit explicit tensor.
                    topology_slice = node_feat_block[:, -len(TEACHER_NODETARGETS_FEATURE_NAMES) :].clone()
                self._examples.append(TensorizedTreeExample(
                    node_features=node_feat_block,
                    parent_index=payload["parent_index"][ns:ne].clone(),
                    edge_parent=payload["edge_parent"][es:ee].clone(),
                    edge_child=payload["edge_child"][es:ee].clone(),
                    edge_slot=edge_slot,
                    depth=payload["depth"][ns:ne].clone(),
                    node_targets=payload["node_targets"][ns:ne].clone(),
                    feature_names=feature_names,
                    edge_wdl_targets=edge_wdl_targets[es:ee].clone() if edge_wdl_targets is not None else None,
                    nodetargets_targets=topology_slice,
                ))
            del payload

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TensorizedTreeExample:
        return self._examples[index]

    @staticmethod
    def collate_fn(batch: Sequence[TensorizedTreeExample]) -> tuple[Any, Any]:
        """DataLoader collate that defers to the canonical batcher in ``tensorizer``."""
        return collate_tensorized_examples(batch)

    @staticmethod
    def collate_fn_nodetargets(batch: Sequence[TensorizedTreeExample]) -> tuple[Any, Any]:
        """Collate for node-targets supervised encoder pretraining."""
        return collate_tensorized_examples_for_nodetargets(batch)


def load_pretrain_example_dataset(path: str) -> Sequence[PretrainExample]:
    """Resolve a dataset spec to the right dataset class.

    Accepts:
        - a directory of per-example ``.pt`` files (``PretrainExampleDirectoryDataset``);
        - a JSON manifest of packed tensorized shards (``PackedTensorizedShardDataset``);
        - a text manifest listing one example path per line (``PretrainExamplePathDataset``).
    """
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
    """Resolve a path spec to a flat list of raw ``.pt`` example paths.

    Controller training needs raw examples (not packed shards) because the
    controller's snapshot replay machinery uses ``PretrainExample.tree``
    directly. Rejecting packed JSON manifests here surfaces that mismatch
    early rather than letting it explode mid-training.
    """
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



def build_tree_from_provider(
    root_fen: str,
    provider: TreeExpansionProvider,
    config: TeacherSearchConfig,
) -> SearchTree:
    """BFS-expand a tree from ``root_fen`` up to ``config.max_depth``.

    Used by the no-budget branch of ``build_pretrain_example``: just enumerate
    every position the provider knows about, capped by depth. No PUCT here —
    the search is run later in ``compute_teacher_targets``.
    """
    tree = SearchTree(
        root_fen=root_fen,
        root_scalar_features=provider.root_features(root_fen),
        root_metadata=provider.root_metadata(root_fen),
    )
    root_id = tree.root_id

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
    """Normalize a vector of prior scores to a probability distribution.

    If every score is non-negative and at least one is positive, treat them
    as counts/probabilities and rescale by the sum. Otherwise fall back to
    softmax — this is the right thing for engines that emit log-policies
    or raw logits.
    """
    if not scores:
        return []

    if any(not math.isfinite(score) for score in scores):
        raise ValueError("Prior scores must be finite.")

    if all(score >= 0.0 for score in scores):
        score_sum = sum(scores)
        if score_sum > 0.0:
            return [score / score_sum for score in scores]

    # Numerically stable softmax for negative or mixed-sign scores.
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    exp_sum = sum(exp_scores)
    return [score / exp_sum for score in exp_scores]


def _prepare_children(
    children: Sequence[ExpansionChild],
    prior_feature: str,
    remaining_slots: Optional[int] = None,
) -> List[ExpansionChild]:
    """Normalize priors across a child list, optionally truncating to a budget.

    The provider may emit raw priors; the teacher expects them normalized
    within each parent's children. This is the chokepoint where that
    normalization happens.
    """
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
    """True iff any leaf can still be expanded under the depth cap."""
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
    """Walk down the tree picking the PUCT-max child until reaching a leaf.

    Returns the leaf node id and the path of (parent_id, child_id) edges
    taken to reach it. PUCT score is ``Q + c_puct * prior * sqrt(N) / (1+n)``,
    the AlphaZero formulation.
    """
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

        # Compute per-child PUCT score and pick the argmax.
        # Formula: score = Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)
        # where N_parent is the sum of child visit counts (= visits to this
        # node from its children's perspective) and N_child is this child's
        # visit count. The `+ 1.0` in `(1.0 + N_child)` keeps the exploration
        # term finite for unvisited children (N_child=0) and the `+ 1.0` in
        # `sqrt(total_visits + 1.0)` does the same for the root before any
        # backups land.
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
    """Walk back up the path applying side-to-move flips and updating edge stats.

    Each step up the tree negates the value (the perspective flips between
    plies) and swaps win/loss in the WDL triple.
    """
    value = leaf_value
    wdl = None if leaf_wdl is None else _normalize_wdl_target(leaf_wdl)
    for parent_id, child_id in reversed(path):
        value = -value
        stats = edge_stats[(parent_id, child_id)]
        stats.visit_count += 1
        stats.total_value += value
        if wdl is not None:
            wdl = _flip_wdl_target(wdl)
            stats.total_wdl = tuple(stats.total_wdl[index] + wdl[index] for index in range(3))


def _backup_target_from_child_q(
    tree: SearchTree,
    node_id: int,
    edge_stats: Mapping[Tuple[int, int], EdgeStats],
    value_feature: str,
) -> float:
    """Visit-weighted mean of child Q-values, falling back to the static value at leaves."""
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
    """Visit-weighted mean child WDL, with fallback to the static node WDL.

    Not currently called from the public pipeline — kept for symmetry with
    ``_backup_target_from_child_q`` and for analyses that want a WDL-style
    node target.
    """
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
    """Produce the per-edge WDL training targets.

    If the edge was visited, the target is the visit-mean WDL accumulated
    during search. If it was never visited, fall back to the child's static
    WDL (flipped to the parent's side). If even that fallback is unavailable
    (older trees without WDL features), return an empty dict so the caller
    knows WDL supervision can't be produced for this tree.
    """
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
    """Map root child move (UCI) → current Q-value. Used by the oracle trace."""
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
    """Build a tree by running PUCT until a sampled expansion budget is hit.

    This is the standard generation path: draw a node budget from
    ``node_budget_distribution``, then alternate PUCT-selection of a leaf,
    provider-driven expansion of that leaf, and full-path backprop. The
    oracle trace is captured after each expansion so we have the teacher's
    decision evolution recorded alongside the final tree.
    """
    rng = rng or random.Random()
    sampled_node_budget = node_budget_distribution.sample(rng)

    # --- Initialize tree with just the root ---
    tree = SearchTree(
        root_fen=root_fen,
        root_scalar_features=provider.root_features(root_fen),
        root_metadata=provider.root_metadata(root_fen),
    )
    root_id = tree.root_id
    edge_stats: Dict[Tuple[int, int], EdgeStats] = {}
    num_expansions = 0
    oracle_trace_expansion_counts: List[int] = []
    oracle_root_moves: List[str] = []
    oracle_root_q_trace: List[List[float]] = []
    oracle_best_move_trace: List[str] = []
    oracle_root_visits_trace: List[List[int]] = []

    def _record_oracle_root_trace() -> None:
        """Snapshot the root's current per-move Q-values after each expansion."""
        nonlocal oracle_root_moves
        if tree.root_id is None:
            return
        root_children = tree.root_children()
        if not root_children:
            return
        # On the first call, freeze the canonical move ordering used for
        # every subsequent trace row.
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
        
        visits_row = [int(edge_stats.get((tree.root_id, child_id), EdgeStats()).visit_count) for child_id in root_children]
        oracle_root_visits_trace.append(visits_row)

    # --- Main PUCT loop: pick a leaf, expand or terminate, backprop ---
    while num_expansions < sampled_node_budget and _has_expandable_frontier(tree, config):
        node_id, path = _select_leaf_by_puct(tree, edge_stats, config)
        node = tree.get_node(node_id)
        leaf_value = _static_node_value(tree, node_id, config.value_feature)
        leaf_wdl = _maybe_static_node_wdl(tree, node_id)

        # Depth-capped or already-terminal leaves don't expand: we just
        # backprop their static value and mark them terminal so future
        # selections won't visit them.
        if node.is_terminal or node.depth >= config.max_depth:
            node.is_terminal = True
            _backpropagate_path(edge_stats, path, leaf_value, leaf_wdl)
            continue

        raw_children = provider.expand_node(node.fen, node.depth)
        children = _prepare_children(
            raw_children,
            prior_feature=config.prior_feature,
        )
        # Provider yielded nothing (real mate/stalemate or engine quirk):
        # treat as a true terminal and back up the static value.
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
        oracle_root_visits_trace=oracle_root_visits_trace,
    )


def consolidate_generated_tree(
    generated_tree: GeneratedTree,
    config: TeacherSearchConfig,
) -> TeacherSearchResult:
    """Convert per-step edge stats from generation into final pretraining targets.

    Wraps up the output of ``generate_partial_tree_from_provider`` by
    computing per-node value targets (visit-weighted child Q) and per-edge
    WDL targets (visit-mean or static fallback). The tree itself is
    validated as a sanity check.
    """
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
    """Read the static value feature off a node. Raises if it's missing."""
    node = tree.get_node(node_id)
    if value_feature not in node.scalar_features:
        raise KeyError(f"Node {node_id} is missing feature '{value_feature}'.")
    return float(node.scalar_features[value_feature])


def _static_node_wdl(tree: SearchTree, node_id: int) -> Tuple[float, float, float]:
    """Read the static WDL features off a node. Raises if any of the three is missing."""
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
    """Like ``_static_node_wdl`` but returns None instead of raising on missing WDL features."""
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
    """Run PUCT over an already-built tree and produce the pretraining targets.

    Unlike ``generate_partial_tree_from_provider``, this does not expand the
    tree — it just runs ``config.search_budget`` simulations over the
    existing structure to estimate Q-values and visit-mean WDLs on every
    edge, then backs those up to per-node value targets.
    """
    if validate:
        tree.validate()
    if tree.root_id is None:
        raise ValueError("Tree must contain a root.")

    # Initialize zero-stats for every existing edge.
    edge_stats: Dict[Tuple[int, int], EdgeStats] = {}
    for node in tree.iter_nodes():
        for child_id in tree.child_ids(node.node_id):
            edge_stats[(node.node_id, child_id)] = EdgeStats()

    # Standard PUCT: select a leaf, evaluate its static features, backprop.
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
    *,
    include_edge_wdl_targets: bool = False,
) -> PretrainExample:
    """End-to-end: build a tree, run teacher search, and return a ``PretrainExample``.

    Two modes:
        - ``node_budget_distribution is None``: BFS-expand to ``max_depth``,
          then run a fixed-budget PUCT consolidation. No oracle trace.
        - ``node_budget_distribution is not None``: PUCT-driven generation
          up to a sampled budget, with per-step oracle trace recording.

    Args:
        root_position_id: optional human-readable id for the root (defaults to the FEN).
    """
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
    if include_edge_wdl_targets:
        metadata["edge_wdl_target_generation_version"] = "search_consolidated_edge_wdl_v1"
    # Oracle trace only exists in the budget-driven mode.
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
    value_gap = node_value_gap_features(tree, teacher_result.edge_stats, value_feature=config.value_feature)
    
    policy_drift = [float("nan")] * tree.num_nodes()
    if node_budget_distribution is not None and tree.root_id is not None and generated_tree.oracle_root_visits_trace:
        # Enforce n_min = 10 (first point at which early policy is estimable)
        n_min = 10
        early_visits = None
        for visits_row in generated_tree.oracle_root_visits_trace:
            if sum(visits_row) >= n_min:
                early_visits = visits_row
                break
        if early_visits is not None:
            late_visits = [
                int(generated_tree.edge_stats.get((tree.root_id, child_id), EdgeStats()).visit_count)
                for child_id in tree.root_children()
            ]
            from cts.data.preprocess_gnn.policy_drift import compute_policy_drift
            drift_m = compute_policy_drift(
                node_id=tree.root_id,
                early_visits=early_visits,
                late_visits=late_visits,
                n_min=n_min
            )
            if drift_m is not None:
                policy_drift[tree.root_id] = drift_m.kl_divergence

    return PretrainExample(
        tree=tree,
        node_target_values=teacher_result.node_target_values,
        edge_wdl_targets=teacher_result.edge_target_wdls if include_edge_wdl_targets else {},
        metadata=metadata,
        oracle_trace_expansion_counts=oracle_trace_expansion_counts,
        oracle_root_moves=oracle_root_moves,
        oracle_root_q_trace=oracle_root_q_trace,
        oracle_best_move_trace=oracle_best_move_trace,
        oracle_final_root_q_values=oracle_final_root_q_values,
        value_gap=value_gap,
        policy_drift=policy_drift,
    )


def prefix_expansion_count_schedule(tree: SearchTree) -> List[int]:
    """Return ``[0, 1, ..., total_expansions]`` — every valid prefix length."""
    if tree.root_id is None:
        raise ValueError("Tree must contain a root.")
    return list(range(0, len(tree.ordered_expansion_parent_ids()) + 1))


def sample_prefix_expansion_count_for_node_budget(
    tree: SearchTree,
    min_nodes: int,
    max_nodes: int,
    rng: Optional[random.Random] = None,
) -> int:
    """Pick an expansion prefix length whose expanded-node count lies in ``[min_nodes, max_nodes]``.

    Target is drawn log-uniformly in the same way ``NodeBudgetDistribution``
    samples, and we pick the eligible prefix length whose log is closest to
    the target. Ties broken by smaller prefix.
    """
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
    *,
    include_edge_wdl_targets: bool = False,
) -> PretrainExample:
    """Snapshot a source example to a prefix and re-run teacher search on it.

    Used to augment training data: from a single fully-built tree, derive
    smaller-tree pretrain examples that retrace the teacher's intermediate
    states. The prefix tree carries provenance metadata pointing back to
    the source.
    """
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
        edge_wdl_targets=teacher_result.edge_target_wdls if include_edge_wdl_targets else {},
        metadata=metadata,
        value_gap=node_value_gap_features(
            prefix_tree,
            teacher_result.edge_stats,
            value_feature=config.value_feature,
        ),
    )

