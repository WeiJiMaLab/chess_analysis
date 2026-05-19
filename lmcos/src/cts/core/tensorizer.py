"""Packed-tensor adapter between ``SearchTree`` and the tree encoder.

This module converts the dict-of-dicts ``SearchTree`` representation into
the flat tensor batches the encoder/attention layers consume. Trees are
concatenated along the node axis with a ``tree_index`` mapping (rather
than padded), and parent→child edges are stored in CSR form via
``child_ptr`` and ``children_index``. The slot assignment used for the
encoder's positional encoding is UCI-lexicographic and is the canonical
ordering — it must agree byte-for-byte with the order used by the
pretraining packer in ``cts_pretrain._child_ptr_and_children_index``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch

from .schema import (
    TEACHER_TOPOLOGY_FEATURE_NAMES,
    TREE_ENCODER_FEATURE_NAMES,
    NodeFeatureSchema,
)
from .tree import SearchTree


@dataclass
class TreeBatch:
    """Packed-tensor batch of one or more trees consumed by the encoder.

    All trees in the batch are concatenated along the node axis; the
    per-row ``tree_index`` recovers which tree each node belongs to.
    Edges are stored both as flat parent/child id arrays (for the
    attention layer's scatter ops) and in CSR form via ``child_ptr`` /
    ``children_index`` (for the decoder and per-parent lookups).
    """

    node_features: torch.Tensor  # [N, F] per-node feature rows in schema order
    tree_index: torch.Tensor  # [N] long; tree_index[i] is the batch index of node i
    parent_index: torch.Tensor  # [N] long; node's parent in the concatenated batch, -1 for roots
    root_index: torch.Tensor  # [batch_size] long; global node id of each tree's root
    edge_parent: torch.Tensor  # [E] long; global parent id per edge
    edge_child: torch.Tensor  # [E] long; global child id per edge
    edge_slot: torch.Tensor  # [E] long; UCI-lex slot index of the child under its parent
    child_ptr: torch.Tensor  # [N+1] long; CSR row pointer into children_index
    children_index: torch.Tensor  # [E] long; CSR-packed global child ids
    depth: torch.Tensor  # [N] long; plies from each node's root
    feature_names: Tuple[str, ...]  # column order matching node_features; mirrors the schema
    batch_size: int  # number of trees concatenated into this batch
    num_nodes: int  # N
    num_edges: int  # E
    edge_wdl_targets: Optional[torch.Tensor] = None  # [E, 3] normalized per-edge WDL targets when training the edge head
    topology_targets: Optional[torch.Tensor] = None  # [N, 4] per-node topology supervision when topology-pretraining


@dataclass
class TensorizedTreeExample:
    """A single tree pre-tensorized for training (carries node-level targets).

    Produced by ``tensorize_tree_with_targets`` and later concatenated
    by ``collate_tensorized_examples`` into a ``TreeBatch``. Holds only
    the per-tree (not yet batch-offset-adjusted) tensors plus the
    training targets.
    """

    node_features: torch.Tensor  # [n_nodes, F]
    parent_index: torch.Tensor  # [n_nodes] long; local parent ids, -1 for the root
    edge_parent: torch.Tensor  # [n_edges] long; local parent ids per edge
    edge_child: torch.Tensor  # [n_edges] long; local child ids per edge
    edge_slot: torch.Tensor  # [n_edges] long; UCI-lex slot of the child
    depth: torch.Tensor  # [n_nodes] long
    node_targets: torch.Tensor  # [n_nodes] training targets aligned to node order
    feature_names: Tuple[str, ...]  # mirrors the schema's column order
    edge_wdl_targets: Optional[torch.Tensor] = None  # [n_edges, 3] normalized WDL targets, when provided
    topology_targets: Optional[torch.Tensor] = None  # [n_nodes, 4] topology supervision, when provided


@dataclass
class TensorizedTreeObservation:
    """A single tree pre-tensorized for inference (no targets).

    Used as the input format to the controller's online tree evaluation:
    same shape as ``TensorizedTreeExample`` minus training-only fields.
    """

    node_features: torch.Tensor  # [n_nodes, F]
    parent_index: torch.Tensor  # [n_nodes] long; local parent ids, -1 for the root
    edge_parent: torch.Tensor  # [n_edges] long; local parent ids per edge
    edge_child: torch.Tensor  # [n_edges] long; local child ids per edge
    edge_slot: torch.Tensor  # [n_edges] long; UCI-lex slot of the child
    depth: torch.Tensor  # [n_nodes] long
    root_index: int  # local node id of the root (typically 0)
    feature_names: Tuple[str, ...]  # mirrors the schema's column order


def _sorted_child_ids_with_slots(tree: SearchTree, parent_id: int) -> list[tuple[int, int]]:
    """Return ``(child_id, slot)`` pairs in UCI-lexicographic order.

    The slot ordering is load-bearing: the encoder's per-slot
    positional embedding indexes into this slot id, and the
    pretraining packer (``cts_pretrain._child_ptr_and_children_index``)
    uses the same UCI sort. Any divergence here silently desyncs the
    positional encoding from the data it was trained on.
    """
    ordered_children = []
    for child_id in tree.child_ids(parent_id):
        move_uci = tree.get_node(child_id).incoming_move_uci
        if move_uci is None:
            raise ValueError(f"Child node {child_id} is missing incoming_move_uci.")
        ordered_children.append((move_uci, child_id))
    ordered_children.sort()
    return [(child_id, slot) for slot, (_, child_id) in enumerate(ordered_children)]


def edge_wdl_target_tensor(
    tree: SearchTree,
    edge_wdl_targets: Mapping[tuple[int, int], Sequence[float]],
    *,
    schema: NodeFeatureSchema,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """Stack per-edge WDL targets in the canonical edge order.

    The output row order matches the (parent, child) order produced
    by ``_sorted_child_ids_with_slots``, so it lines up 1-to-1 with
    ``edge_parent`` / ``edge_child``. Each row is clamped to the
    non-negative orthant and renormalized to sum to one.

    Args:
        tree: source tree (its child ordering defines the row order).
        edge_wdl_targets: ``(parent_id, child_id) -> (w, d, l)``;
            every realized edge must have a target.
        schema: feature schema (unused here but kept for signature parity).
        device: target device for the produced tensor.
    """
    device = torch.device(device)
    target_rows = []
    for node in tree.iter_nodes():
        for child_id, _slot in _sorted_child_ids_with_slots(tree, node.node_id):
            edge_key = (node.node_id, child_id)
            if edge_key not in edge_wdl_targets:
                raise KeyError(f"Missing edge WDL target for edge {edge_key}.")
            target = torch.tensor(edge_wdl_targets[edge_key], dtype=torch.float32, device=device)
            if target.shape != (3,):
                raise ValueError(f"Edge WDL target for edge {edge_key} must have shape (3,), got {tuple(target.shape)}.")
            target = target.clamp_min(0.0)
            target_mass = target.sum()
            if float(target_mass.item()) <= 0.0:
                raise ValueError(f"Edge WDL target for edge {edge_key} must have positive mass.")
            target_rows.append(target / target_mass)
    if not target_rows:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    return torch.stack(target_rows, dim=0)


def tensorize_tree(
    tree: SearchTree,
    *,
    schema: NodeFeatureSchema,
    device: Union[torch.device, str] = "cpu",
    validate: bool = False,
) -> TreeBatch:
    """Convenience wrapper: tensorize a single tree as a batch of one."""
    return tensorize_forest([tree], schema=schema, device=device, validate=validate)


def tensorize_tree_observation(
    tree: SearchTree,
    *,
    schema: NodeFeatureSchema,
    device: Union[torch.device, str] = "cpu",
    validate: bool = False,
) -> TensorizedTreeObservation:
    """Tensorize a single tree without batch-offset adjustments (inference path).

    Args:
        tree: tree to tensorize; must have a root.
        schema: feature schema that fixes the column order of
            ``node_features`` and the tensor dtype.
        device: target device for the produced tensors.
        validate: run ``tree.validate()`` first; turn off only in
            hot paths where the tree is known to be well-formed.
    """
    device = torch.device(device)
    if validate:
        tree.validate()

    node_feature_rows = []
    parent_index = []
    depth = []
    edge_parent = []
    edge_child = []
    edge_slot = []

    # Single traversal: emit one node row, then enumerate the node's
    # children in slot order so the edge arrays land in canonical order.
    for node in tree.iter_nodes():
        node_feature_rows.append(schema.vectorize_tuple(node.scalar_features))
        parent_index.append(-1 if node.parent_id is None else node.parent_id)
        depth.append(node.depth)
        for child_id, slot in _sorted_child_ids_with_slots(tree, node.node_id):
            edge_parent.append(node.node_id)
            edge_child.append(child_id)
            edge_slot.append(slot)

    return TensorizedTreeObservation(
        node_features=torch.tensor(node_feature_rows, dtype=schema.dtype, device=device),
        parent_index=torch.tensor(parent_index, dtype=torch.long, device=device),
        edge_parent=torch.tensor(edge_parent, dtype=torch.long, device=device),
        edge_child=torch.tensor(edge_child, dtype=torch.long, device=device),
        edge_slot=torch.tensor(edge_slot, dtype=torch.long, device=device),
        depth=torch.tensor(depth, dtype=torch.long, device=device),
        root_index=tree.root_id,
        feature_names=schema.feature_names,
    )


def tensorize_forest(
    trees: Sequence[SearchTree],
    *,
    schema: NodeFeatureSchema,
    device: Union[torch.device, str] = "cpu",
    validate: bool = False,
) -> TreeBatch:
    """Concatenate multiple trees into a single ``TreeBatch``.

    Per-tree node ids are remapped into the global batch index space
    by adding a running ``node_offset``. The CSR ``child_ptr`` is
    built incrementally: one ``child_ptr`` entry is appended per
    node, recording the running edge count after that node's
    children have been emitted.

    Args:
        trees: sequence of trees; must be non-empty and each must have a root.
        schema: feature schema that fixes the column order of
            ``node_features`` and the tensor dtype.
        device: target device for the produced tensors.
        validate: run ``tree.validate()`` on each tree before tensorizing.
    """
    if not trees:
        raise ValueError("Cannot tensorize an empty forest.")

    device = torch.device(device)
    node_feature_rows = []
    tree_index = []
    parent_index = []
    root_index = []
    depth = []
    edge_parent = []
    edge_child = []
    edge_slot = []
    child_ptr = [0]
    children_index = []

    node_offset = 0
    edge_offset = 0
    for batch_idx, tree in enumerate(trees):
        if validate:
            tree.validate()

        root_index.append(node_offset + tree.root_id)
        for node in tree.iter_nodes():
            global_node_id = node_offset + node.node_id
            node_feature_rows.append(schema.vectorize_tuple(node.scalar_features))
            tree_index.append(batch_idx)
            parent_index.append(-1 if node.parent_id is None else node_offset + node.parent_id)
            depth.append(node.depth)

            # Emit this node's outgoing edges in canonical slot order,
            # mirroring the slot assignment into edge_slot and the
            # CSR-style children_index list.
            child_id_slots = _sorted_child_ids_with_slots(tree, node.node_id)
            for child_id, slot in child_id_slots:
                edge_parent.append(global_node_id)
                edge_child.append(node_offset + child_id)
                edge_slot.append(slot)
                children_index.append(node_offset + child_id)
                edge_offset += 1
            # One child_ptr entry per node, recording the cumulative
            # edge count after this node's children.
            child_ptr.append(edge_offset)

        node_offset += tree.num_nodes()

    node_features_tensor = torch.tensor(
        node_feature_rows,
        dtype=schema.dtype,
        device=device,
    )
    long_device = device
    return TreeBatch(
        node_features=node_features_tensor,
        tree_index=torch.tensor(tree_index, dtype=torch.long, device=long_device),
        parent_index=torch.tensor(parent_index, dtype=torch.long, device=long_device),
        root_index=torch.tensor(root_index, dtype=torch.long, device=long_device),
        edge_parent=torch.tensor(edge_parent, dtype=torch.long, device=long_device),
        edge_child=torch.tensor(edge_child, dtype=torch.long, device=long_device),
        edge_slot=torch.tensor(edge_slot, dtype=torch.long, device=long_device),
        child_ptr=torch.tensor(child_ptr, dtype=torch.long, device=long_device),
        children_index=torch.tensor(children_index, dtype=torch.long, device=long_device),
        depth=torch.tensor(depth, dtype=torch.long, device=long_device),
        feature_names=schema.feature_names,
        batch_size=len(trees),
        num_nodes=len(node_feature_rows),
        num_edges=len(edge_parent),
    )


def tensorize_tree_with_targets(
    tree: SearchTree,
    node_target_values: Sequence[float],
    schema: NodeFeatureSchema,
    device: Union[torch.device, str] = "cpu",
    edge_wdl_targets: Optional[Mapping[tuple[int, int], Sequence[float]]] = None,
) -> TensorizedTreeExample:
    """Tensorize a tree together with per-node training targets.

    Returns the per-tree (not yet collated) format; concatenate via
    ``collate_tensorized_examples`` to build a training batch.

    Args:
        tree: source tree.
        node_target_values: one scalar target per node, in node-id order.
        schema: feature schema for the node feature columns.
        device: target device for the result tensors.
        edge_wdl_targets: optional per-edge WDL targets keyed by
            ``(parent_id, child_id)``; passed through ``edge_wdl_target_tensor``.
    """
    tree_batch = tensorize_tree(tree, schema=schema, device=device)
    return TensorizedTreeExample(
        node_features=tree_batch.node_features,
        parent_index=tree_batch.parent_index,
        edge_parent=tree_batch.edge_parent,
        edge_child=tree_batch.edge_child,
        edge_slot=tree_batch.edge_slot,
        depth=tree_batch.depth,
        node_targets=torch.tensor(node_target_values, dtype=torch.float32, device=tree_batch.node_features.device),
        feature_names=tree_batch.feature_names,
        edge_wdl_targets=(
            edge_wdl_target_tensor(tree, edge_wdl_targets, schema=schema, device=device)
            if edge_wdl_targets is not None
            else None
        ),
    )


def collate_tensorized_examples(examples: Sequence[TensorizedTreeExample]) -> tuple[TreeBatch, torch.Tensor]:
    """Concatenate per-tree examples into one batched ``TreeBatch`` + target vector.

    Adjusts every local node id by the running ``node_offset`` so ids
    refer to positions in the concatenated tensor, then rebuilds the CSR
    ``child_ptr`` from the per-parent child counts.

    Args:
        examples: sequence of per-tree tensorized examples; all must
            share the same ``feature_names``.

    Returns:
        ``(tree_batch, node_targets)`` where ``node_targets`` is the
        flat per-node target vector aligned to ``tree_batch.node_features``.
    """
    if not examples:
        raise ValueError("Cannot collate an empty batch.")

    feature_names = examples[0].feature_names
    node_features_parts = []
    parent_index_parts = []
    edge_parent_parts = []
    edge_child_parts = []
    edge_slot_parts = []
    depth_parts = []
    target_parts = []
    edge_target_parts = []
    tree_index_parts = []
    root_index = []

    node_offset = 0
    for batch_idx, example in enumerate(examples):
        if example.feature_names != feature_names:
            raise ValueError("All tensorized examples must share the same feature schema.")
        num_nodes = int(example.node_features.shape[0])
        node_features_parts.append(example.node_features)
        depth_parts.append(example.depth)
        target_parts.append(example.node_targets)
        tree_index_parts.append(
            torch.full((num_nodes,), batch_idx, dtype=torch.long, device=example.node_features.device)
        )
        # In per-example form, root_index is implicitly 0; in the batched
        # form it becomes node_offset (start of this tree's node block).
        root_index.append(node_offset)

        # Shift only the entries that point at a real parent; leave the
        # -1 sentinels (roots) alone.
        parent_index = example.parent_index.clone()
        has_parent = parent_index >= 0
        parent_index[has_parent] += node_offset
        parent_index_parts.append(parent_index)

        if example.edge_parent.numel() > 0:
            edge_parent_parts.append(example.edge_parent + node_offset)
            edge_child_parts.append(example.edge_child + node_offset)
            edge_slot_parts.append(example.edge_slot)
            if example.edge_wdl_targets is not None:
                edge_target_parts.append(example.edge_wdl_targets)

        node_offset += num_nodes

    node_features = torch.cat(node_features_parts, dim=0)
    parent_index = torch.cat(parent_index_parts, dim=0)
    depth = torch.cat(depth_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    tree_index = torch.cat(tree_index_parts, dim=0)
    root_index_tensor = torch.tensor(root_index, dtype=torch.long, device=node_features.device)

    if edge_parent_parts:
        edge_parent = torch.cat(edge_parent_parts, dim=0)
        edge_child = torch.cat(edge_child_parts, dim=0)
        edge_slot = torch.cat(edge_slot_parts, dim=0)
        children_index = edge_child
        # Rebuild CSR row pointer from per-parent child counts. bincount
        # over edge_parent gives one count per node; cumsum produces the
        # CSR offsets, prefixed by an implicit 0 at child_ptr[0].
        child_counts = torch.bincount(edge_parent, minlength=node_features.shape[0])
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long, device=node_features.device)
        child_ptr[1:] = torch.cumsum(child_counts, dim=0)
    else:
        # Edgeless batch (all trees are root-only): emit empty edge
        # tensors and a degenerate all-zero CSR.
        edge_parent = torch.empty(0, dtype=torch.long, device=node_features.device)
        edge_child = torch.empty(0, dtype=torch.long, device=node_features.device)
        edge_slot = torch.empty(0, dtype=torch.long, device=node_features.device)
        children_index = torch.empty(0, dtype=torch.long, device=node_features.device)
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long, device=node_features.device)

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
        feature_names=feature_names,
        batch_size=len(examples),
        num_nodes=int(node_features.shape[0]),
        num_edges=int(edge_parent.shape[0]),
        edge_wdl_targets=(
            # Only forward edge targets if every contributing example
            # supplied them; otherwise the batch would silently drop
            # rows, which would misalign with edge_parent / edge_child.
            torch.cat(edge_target_parts, dim=0)
            if edge_target_parts and len(edge_target_parts) == len(edge_parent_parts)
            else None
        ),
    )
    return tree_batch, targets


def collate_tensorized_examples_for_topology(
    examples: Sequence[TensorizedTreeExample],
) -> tuple[TreeBatch, torch.Tensor]:
    """Stack trees for Encoder+TopologyHead training: encoder input is five cols; supervision is four."""
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    encoder_width = len(TREE_ENCODER_FEATURE_NAMES)
    encoder_examples: list[TensorizedTreeExample] = []
    topology_parts: list[torch.Tensor] = []
    for example in examples:
        topology_row = example.topology_targets
        num_feat = int(example.node_features.shape[1])
        if topology_row is None and num_feat >= len(TREE_ENCODER_FEATURE_NAMES) + len(
            TEACHER_TOPOLOGY_FEATURE_NAMES
        ):
            tail = tuple(example.feature_names[-len(TEACHER_TOPOLOGY_FEATURE_NAMES) :])
            if tail == TEACHER_TOPOLOGY_FEATURE_NAMES:
                topology_row = example.node_features[:, -len(TEACHER_TOPOLOGY_FEATURE_NAMES) :]

        if topology_row is None:
            raise ValueError(
                "Every example must expose topology_targets (field or trailing node_features columns); "
                "repack with topology_supervision_shard=true."
            )

        if num_feat < encoder_width:
            raise ValueError(
                f"node_features has {num_feat} columns; expected at least {encoder_width} for the encoder."
            )
        node_features = example.node_features
        if num_feat > encoder_width:
            node_features = node_features[:, :encoder_width]
        encoder_examples.append(
            replace(
                example,
                node_features=node_features,
                feature_names=TREE_ENCODER_FEATURE_NAMES,
            )
        )
        topology_parts.append(topology_row)

    tree_batch, _node_targets = collate_tensorized_examples(encoder_examples)
    topology_targets = torch.cat(topology_parts, dim=0)
    tree_batch = replace(
        tree_batch,
        feature_names=TREE_ENCODER_FEATURE_NAMES,
        topology_targets=topology_targets,
    )
    return tree_batch, topology_targets


def collate_tensorized_observations(observations: Sequence[TensorizedTreeObservation]) -> TreeBatch:
    """Concatenate per-tree observations into a batched ``TreeBatch`` (inference).

    Mirror of ``collate_tensorized_examples`` for the no-targets path:
    same id remapping and CSR reconstruction, but carries the per-tree
    ``root_index`` through (observations don't assume the root is at
    local id 0, so it must be tracked explicitly).
    """
    if not observations:
        raise ValueError("Cannot collate an empty batch.")

    feature_names = observations[0].feature_names
    node_features_parts = []
    parent_index_parts = []
    edge_parent_parts = []
    edge_child_parts = []
    edge_slot_parts = []
    depth_parts = []
    tree_index_parts = []
    root_index = []

    node_offset = 0
    for batch_idx, observation in enumerate(observations):
        if observation.feature_names != feature_names:
            raise ValueError("All tensorized observations must share the same feature schema.")
        num_nodes = int(observation.node_features.shape[0])
        node_features_parts.append(observation.node_features)
        depth_parts.append(observation.depth)
        tree_index_parts.append(
            torch.full((num_nodes,), batch_idx, dtype=torch.long, device=observation.node_features.device)
        )
        # Honor the per-observation root_index rather than assuming 0,
        # since the observation type carries it explicitly.
        root_index.append(node_offset + int(observation.root_index))

        parent_index = observation.parent_index.clone()
        has_parent = parent_index >= 0
        parent_index[has_parent] += node_offset
        parent_index_parts.append(parent_index)

        if observation.edge_parent.numel() > 0:
            edge_parent_parts.append(observation.edge_parent + node_offset)
            edge_child_parts.append(observation.edge_child + node_offset)
            edge_slot_parts.append(observation.edge_slot)

        node_offset += num_nodes

    node_features = torch.cat(node_features_parts, dim=0)
    parent_index = torch.cat(parent_index_parts, dim=0)
    depth = torch.cat(depth_parts, dim=0)
    tree_index = torch.cat(tree_index_parts, dim=0)
    root_index_tensor = torch.tensor(root_index, dtype=torch.long, device=node_features.device)

    if edge_parent_parts:
        edge_parent = torch.cat(edge_parent_parts, dim=0)
        edge_child = torch.cat(edge_child_parts, dim=0)
        edge_slot = torch.cat(edge_slot_parts, dim=0)
        children_index = edge_child
        # CSR pointer rebuilt from per-parent edge counts; see the
        # collate_tensorized_examples branch for the rationale.
        child_counts = torch.bincount(edge_parent, minlength=node_features.shape[0])
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long, device=node_features.device)
        child_ptr[1:] = torch.cumsum(child_counts, dim=0)
    else:
        edge_parent = torch.empty(0, dtype=torch.long, device=node_features.device)
        edge_child = torch.empty(0, dtype=torch.long, device=node_features.device)
        edge_slot = torch.empty(0, dtype=torch.long, device=node_features.device)
        children_index = torch.empty(0, dtype=torch.long, device=node_features.device)
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long, device=node_features.device)

    return TreeBatch(
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
        feature_names=feature_names,
        batch_size=len(observations),
        num_nodes=int(node_features.shape[0]),
        num_edges=int(edge_parent.shape[0]),
    )
