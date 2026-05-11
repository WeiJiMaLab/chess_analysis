from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional


def _normalize_scalar_features(features: Mapping[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for name, value in features.items():
        if not isinstance(name, str):
            raise TypeError("Feature names must be strings.")
        if not isinstance(value, (int, float)):
            raise TypeError(f"Feature '{name}' must be numeric.")
        normalized[name] = float(value)
    return normalized


@dataclass
class SearchNode:
    node_id: int
    parent_id: Optional[int]
    incoming_move_uci: Optional[str]
    fen: str
    depth: int
    is_terminal: bool
    is_expanded: bool
    scalar_features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionChild:
    move_uci: str
    fen: str
    scalar_features: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False

    def __post_init__(self) -> None:
        self.scalar_features = _normalize_scalar_features(self.scalar_features)
        self.metadata = dict(self.metadata)


@dataclass
class SearchTree:
    root_id: Optional[int] = None
    _nodes: List[SearchNode] = field(default_factory=list, init=False, repr=False)
    _children: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)

    def create_root(
        self,
        fen: str,
        scalar_features: Mapping[str, float],
        metadata: Optional[Mapping[str, Any]] = None,
        is_terminal: bool = False,
    ) -> int:
        if self.root_id is not None:
            raise ValueError("Root node already exists.")

        node_id = len(self._nodes)
        node = SearchNode(
            node_id=node_id,
            parent_id=None,
            incoming_move_uci=None,
            fen=fen,
            depth=0,
            is_terminal=is_terminal,
            is_expanded=False,
            scalar_features=_normalize_scalar_features(scalar_features),
            metadata=dict(metadata or {}),
        )
        self._nodes.append(node)
        self._children[node_id] = []
        self.root_id = node_id
        return node_id

    def add_children(self, parent_id: int, children: Iterable[ExpansionChild]) -> List[int]:
        parent = self.get_node(parent_id)
        if parent.is_terminal:
            raise ValueError("Cannot expand a terminal node.")
        if parent.is_expanded:
            raise ValueError(f"Node {parent_id} has already been expanded.")

        child_specs = self._validated_child_specs(parent_id, children)

        child_ids: List[int] = []
        for child in child_specs:
            node_id = len(self._nodes)
            node = SearchNode(
                node_id=node_id,
                parent_id=parent_id,
                incoming_move_uci=child.move_uci,
                fen=child.fen,
                depth=parent.depth + 1,
                is_terminal=child.is_terminal,
                is_expanded=False,
                scalar_features=dict(child.scalar_features),
                metadata=dict(child.metadata),
            )
            self._nodes.append(node)
            self._children[node_id] = []
            child_ids.append(node_id)

        self._children[parent_id].extend(child_ids)
        parent.is_expanded = True
        return child_ids

    def _validated_child_specs(self, parent_id: int, children: Iterable[ExpansionChild]) -> List[ExpansionChild]:
        child_specs = list(children)
        seen_moves = set()
        for child in child_specs:
            if child.move_uci in seen_moves:
                raise ValueError(f"Duplicate child move '{child.move_uci}' for parent {parent_id}.")
            seen_moves.add(child.move_uci)
        return child_specs

    def get_node(self, node_id: int) -> SearchNode:
        if node_id < 0 or node_id >= len(self._nodes):
            raise KeyError(f"Unknown node_id: {node_id}")
        return self._nodes[node_id]

    def children(self, node_id: int) -> List[int]:
        self.get_node(node_id)
        return list(self._children[node_id])

    def child_ids(self, node_id: int) -> List[int]:
        self.get_node(node_id)
        return self._children[node_id]

    def root_children(self) -> List[int]:
        if self.root_id is None:
            raise ValueError("Tree has no root node.")
        return self.children(self.root_id)

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return sum(len(child_ids) for child_ids in self._children.values())

    def iter_nodes(self) -> Iterator[SearchNode]:
        return iter(self._nodes)

    def ordered_expansion_parent_ids(self) -> List[int]:
        expansion_parents = []
        for node in self.iter_nodes():
            child_ids = self.child_ids(node.node_id)
            if node.is_expanded and child_ids:
                expansion_parents.append((child_ids[0], node.node_id))
        expansion_parents.sort()
        return [node_id for _, node_id in expansion_parents]

    def clone_expansion_prefix(self, expansion_count: int) -> "SearchTree":
        if self.root_id is None:
            raise ValueError("Tree must contain a root.")

        expansion_parent_ids = self.ordered_expansion_parent_ids()
        if expansion_count < 0 or expansion_count > len(expansion_parent_ids):
            raise ValueError(
                f"expansion_count must be between 0 and {len(expansion_parent_ids)}, got {expansion_count}."
            )

        cloned = SearchTree()
        root = self.get_node(self.root_id)
        new_root_id = cloned.create_root(
            fen=root.fen,
            scalar_features=root.scalar_features,
            metadata=root.metadata,
            is_terminal=root.is_terminal,
        )
        node_id_map = {self.root_id: new_root_id}

        for old_parent_id in expansion_parent_ids[:expansion_count]:
            if old_parent_id not in node_id_map:
                raise ValueError(f"Expansion parent {old_parent_id} is unavailable in the cloned tree.")
            new_parent_id = node_id_map[old_parent_id]
            old_child_ids = self.child_ids(old_parent_id)
            child_specs = self._child_specs_for_parent(old_parent_id)
            new_child_ids = cloned.add_children(new_parent_id, child_specs)
            for old_child_id, new_child_id in zip(old_child_ids, new_child_ids):
                node_id_map[old_child_id] = new_child_id

        return cloned

    def clone(self) -> "SearchTree":
        return self.clone_expansion_prefix(len(self.ordered_expansion_parent_ids()))

    def _child_specs_for_parent(self, parent_id: int) -> List[ExpansionChild]:
        child_specs: List[ExpansionChild] = []
        for child_id in self.child_ids(parent_id):
            child_node = self.get_node(child_id)
            if child_node.incoming_move_uci is None:
                raise ValueError(f"Expanded node {child_id} is missing an incoming move.")
            child_specs.append(
                ExpansionChild(
                    move_uci=child_node.incoming_move_uci,
                    fen=child_node.fen,
                    scalar_features=child_node.scalar_features,
                    metadata=child_node.metadata,
                    is_terminal=child_node.is_terminal,
                )
            )
        return child_specs

    def validate(self) -> None:
        if self.root_id is None:
            raise ValueError("Tree must contain a root node.")
        if self.root_id != 0:
            raise ValueError("Root node id must be 0 in the append-only tree.")
        if len(self._nodes) != len(self._children):
            raise ValueError("Node and child index storage are inconsistent.")

        referenced_children = set()
        for node in self._nodes:
            if node.node_id >= len(self._nodes):
                raise ValueError(f"Invalid node id {node.node_id}.")
            if node.node_id != self._nodes[node.node_id].node_id:
                raise ValueError("Node ids must match insertion order.")

            child_ids = self._children.get(node.node_id)
            if child_ids is None:
                raise ValueError(f"Missing child list for node {node.node_id}.")

            if node.parent_id is None:
                if node.node_id != self.root_id:
                    raise ValueError("Only the root node may have no parent.")
                if node.depth != 0:
                    raise ValueError("Root node depth must be zero.")
                if node.incoming_move_uci is not None:
                    raise ValueError("Root node cannot have an incoming move.")
            else:
                parent = self.get_node(node.parent_id)
                if node.node_id not in self._children[parent.node_id]:
                    raise ValueError(f"Parent {parent.node_id} does not reference child {node.node_id}.")
                if node.depth != parent.depth + 1:
                    raise ValueError(f"Node {node.node_id} has an inconsistent depth.")
                if node.node_id in referenced_children:
                    raise ValueError(f"Node {node.node_id} is referenced by multiple parents.")
                referenced_children.add(node.node_id)

            if node.is_expanded and node.is_terminal:
                raise ValueError(f"Node {node.node_id} cannot be terminal and expanded.")
            if node.is_expanded and not child_ids:
                raise ValueError(f"Expanded node {node.node_id} must have at least one child.")
