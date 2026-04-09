from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import torch

from schema import NodeFeatureSchema
from tree import SearchTree


DEFAULT_CHILD_SLOT_COUNT = 9


@dataclass
class TreeBatch:
    node_features: torch.Tensor
    tree_index: torch.Tensor
    parent_index: torch.Tensor
    root_index: torch.Tensor
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_slot: torch.Tensor
    child_ptr: torch.Tensor
    children_index: torch.Tensor
    depth: torch.Tensor
    feature_names: Tuple[str, ...]
    batch_size: int
    num_nodes: int
    num_edges: int


@dataclass
class TensorizedTreeExample:
    node_features: torch.Tensor
    parent_index: torch.Tensor
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_slot: torch.Tensor
    depth: torch.Tensor
    node_targets: torch.Tensor
    feature_names: Tuple[str, ...]


@dataclass
class TensorizedTreeObservation:
    node_features: torch.Tensor
    parent_index: torch.Tensor
    edge_parent: torch.Tensor
    edge_child: torch.Tensor
    edge_slot: torch.Tensor
    depth: torch.Tensor
    root_index: int
    feature_names: Tuple[str, ...]


class TreeTensorizer:
    def __init__(
        self,
        schema: NodeFeatureSchema,
        device: Union[torch.device, str] = "cpu",
        child_slot_count: int = DEFAULT_CHILD_SLOT_COUNT,
    ) -> None:
        if child_slot_count < 2:
            raise ValueError("child_slot_count must be at least 2 so the final slot can act as overflow.")
        self.schema = schema
        self.device = torch.device(device)
        self.child_slot_count = int(child_slot_count)

    def _sorted_child_ids_with_slots(self, tree: SearchTree, parent_id: int) -> list[tuple[int, int]]:
        ordered_children = []
        for child_id in tree.child_ids(parent_id):
            move_uci = tree.get_node(child_id).incoming_move_uci
            if move_uci is None:
                raise ValueError(f"Child node {child_id} is missing incoming_move_uci.")
            ordered_children.append((move_uci, child_id))
        ordered_children.sort(key=lambda item: item[0])

        overflow_slot = self.child_slot_count - 1
        return [
            (child_id, min(slot_index, overflow_slot))
            for slot_index, (_, child_id) in enumerate(ordered_children)
        ]

    def tensorize_tree(self, tree: SearchTree, *, validate: bool = True) -> TreeBatch:
        return self.tensorize_forest([tree], validate=validate)

    def tensorize_tree_observation(
        self,
        tree: SearchTree,
        *,
        validate: bool = True,
    ) -> TensorizedTreeObservation:
        if validate:
            tree.validate()
        if tree.root_id is None:
            raise ValueError("Tree must have a root.")

        node_feature_rows = []
        parent_index = []
        depth = []
        edge_parent = []
        edge_child = []
        edge_slot = []

        for node in tree.iter_nodes():
            node_feature_rows.append(self.schema.vectorize_tuple(node.scalar_features))
            parent_index.append(-1 if node.parent_id is None else node.parent_id)
            depth.append(node.depth)
            for child_id, slot_index in self._sorted_child_ids_with_slots(tree, node.node_id):
                edge_parent.append(node.node_id)
                edge_child.append(child_id)
                edge_slot.append(slot_index)

        return TensorizedTreeObservation(
            node_features=torch.tensor(node_feature_rows, dtype=self.schema.dtype, device=self.device),
            parent_index=torch.tensor(parent_index, dtype=torch.long, device=self.device),
            edge_parent=torch.tensor(edge_parent, dtype=torch.long, device=self.device),
            edge_child=torch.tensor(edge_child, dtype=torch.long, device=self.device),
            edge_slot=torch.tensor(edge_slot, dtype=torch.long, device=self.device),
            depth=torch.tensor(depth, dtype=torch.long, device=self.device),
            root_index=tree.root_id,
            feature_names=self.schema.feature_names,
        )

    def tensorize_forest(self, trees: Sequence[SearchTree], *, validate: bool = True) -> TreeBatch:
        if not trees:
            raise ValueError("Cannot tensorize an empty forest.")

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
            if tree.root_id is None:
                raise ValueError("Each tree must have a root.")

            root_index.append(node_offset + tree.root_id)
            for node in tree.iter_nodes():
                global_node_id = node_offset + node.node_id
                node_feature_rows.append(self.schema.vectorize_tuple(node.scalar_features))
                tree_index.append(batch_idx)
                parent_index.append(-1 if node.parent_id is None else node_offset + node.parent_id)
                depth.append(node.depth)

                child_specs = self._sorted_child_ids_with_slots(tree, node.node_id)
                for child_id, slot_index in child_specs:
                    edge_parent.append(global_node_id)
                    edge_child.append(node_offset + child_id)
                    edge_slot.append(slot_index)
                    children_index.append(node_offset + child_id)
                    edge_offset += 1
                child_ptr.append(edge_offset)

            node_offset += tree.num_nodes()

        node_features_tensor = torch.tensor(
            node_feature_rows,
            dtype=self.schema.dtype,
            device=self.device,
        )
        long_device = self.device
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
            feature_names=self.schema.feature_names,
            batch_size=len(trees),
            num_nodes=len(node_feature_rows),
            num_edges=len(edge_parent),
        )


def tensorize_tree_with_targets(
    tree: SearchTree,
    node_target_values: Sequence[float],
    schema: NodeFeatureSchema,
    device: Union[torch.device, str] = "cpu",
) -> TensorizedTreeExample:
    tree_batch = TreeTensorizer(schema=schema, device=device).tensorize_tree(tree)
    return TensorizedTreeExample(
        node_features=tree_batch.node_features,
        parent_index=tree_batch.parent_index,
        edge_parent=tree_batch.edge_parent,
        edge_child=tree_batch.edge_child,
        edge_slot=tree_batch.edge_slot,
        depth=tree_batch.depth,
        node_targets=torch.tensor(node_target_values, dtype=torch.float32, device=tree_batch.node_features.device),
        feature_names=tree_batch.feature_names,
    )


def collate_tensorized_examples(examples: Sequence[TensorizedTreeExample]) -> tuple[TreeBatch, torch.Tensor]:
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
        root_index.append(node_offset)

        parent_index = example.parent_index.clone()
        has_parent = parent_index >= 0
        parent_index[has_parent] += node_offset
        parent_index_parts.append(parent_index)

        if example.edge_parent.numel() > 0:
            edge_parent_parts.append(example.edge_parent + node_offset)
            edge_child_parts.append(example.edge_child + node_offset)
            edge_slot_parts.append(example.edge_slot)

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
        child_counts = torch.bincount(edge_parent, minlength=node_features.shape[0])
        child_ptr = torch.zeros(node_features.shape[0] + 1, dtype=torch.long, device=node_features.device)
        child_ptr[1:] = torch.cumsum(child_counts, dim=0)
    else:
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
    )
    return tree_batch, targets


def collate_tensorized_observations(observations: Sequence[TensorizedTreeObservation]) -> TreeBatch:
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
