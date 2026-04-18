from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from cts_uci_common import append_move_to_position_spec


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

    def __getstate__(self) -> Dict[str, Any]:
        compact_position_specs = self._compact_position_specs_for_state()
        scalar_feature_names: Optional[List[str]]
        scalar_feature_values: Optional[List[List[float]]]
        node_scalar_features: Optional[List[Dict[str, float]]]
        if not self._nodes:
            scalar_feature_names = []
            scalar_feature_values = []
            node_scalar_features = None
        else:
            first_names = list(self._nodes[0].scalar_features.keys())
            if all(list(node.scalar_features.keys()) == first_names for node in self._nodes):
                scalar_feature_names = list(first_names)
                scalar_feature_values = [
                    [float(node.scalar_features[name]) for name in scalar_feature_names]
                    for node in self._nodes
                ]
                node_scalar_features = None
            else:
                scalar_feature_names = None
                scalar_feature_values = None
                node_scalar_features = [dict(node.scalar_features) for node in self._nodes]

        sparse_metadata = [
            (node.node_id, dict(node.metadata))
            for node in self._nodes
            if node.metadata
        ]
        return {
            "__format__": "search_tree_v2",
            "root_id": self.root_id,
            "parent_ids": [node.parent_id for node in self._nodes],
            "incoming_moves": [node.incoming_move_uci for node in self._nodes],
            "root_position_spec": compact_position_specs["root_position_spec"],
            "fens": compact_position_specs["fens"],
            "depths": [node.depth for node in self._nodes],
            "is_terminal": [node.is_terminal for node in self._nodes],
            "is_expanded": [node.is_expanded for node in self._nodes],
            "scalar_feature_names": scalar_feature_names,
            "scalar_feature_values": scalar_feature_values,
            "node_scalar_features": node_scalar_features,
            "sparse_metadata": sparse_metadata,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        if state.get("__format__") != "search_tree_v2":
            self.__dict__.update(state)
            return

        root_id = state["root_id"]
        parent_ids = list(state["parent_ids"])
        incoming_moves = list(state["incoming_moves"])
        root_position_spec = state.get("root_position_spec")
        fens = state.get("fens")
        depths = list(state["depths"])
        is_terminal = list(state["is_terminal"])
        is_expanded = list(state["is_expanded"])
        scalar_feature_names = state.get("scalar_feature_names")
        scalar_feature_values = state.get("scalar_feature_values")
        node_scalar_features = state.get("node_scalar_features")
        sparse_metadata_items = state.get("sparse_metadata", [])

        metadata_by_node = {int(node_id): dict(metadata) for node_id, metadata in sparse_metadata_items}

        self.root_id = int(root_id) if root_id is not None else None
        self._nodes = []
        self._children = {}

        resolved_fens = self._restore_position_specs(
            parent_ids=parent_ids,
            incoming_moves=incoming_moves,
            root_position_spec=root_position_spec,
            fens=fens,
        )

        if scalar_feature_names is not None:
            scalar_feature_names = [str(name) for name in scalar_feature_names]
            resolved_scalar_features = [
                {
                    name: float(value)
                    for name, value in zip(scalar_feature_names, row)
                }
                for row in scalar_feature_values
            ]
        else:
            resolved_scalar_features = [
                {str(name): float(value) for name, value in dict(features).items()}
                for features in node_scalar_features
            ]

        for node_id, (parent_id, move, fen, depth, terminal, expanded, scalar_features) in enumerate(
            zip(
                parent_ids,
                incoming_moves,
                resolved_fens,
                depths,
                is_terminal,
                is_expanded,
                resolved_scalar_features,
            )
        ):
            self._nodes.append(
                SearchNode(
                    node_id=node_id,
                    parent_id=int(parent_id) if parent_id is not None else None,
                    incoming_move_uci=str(move) if move is not None else None,
                    fen=str(fen),
                    depth=int(depth),
                    is_terminal=bool(terminal),
                    is_expanded=bool(expanded),
                    scalar_features=scalar_features,
                    metadata=dict(metadata_by_node.get(node_id, {})),
                )
            )
            self._children[node_id] = []

        for node in self._nodes:
            if node.parent_id is not None:
                self._children[node.parent_id].append(node.node_id)

    def _compact_position_specs_for_state(self) -> Dict[str, Any]:
        if not self._nodes:
            return {"root_position_spec": None, "fens": []}

        root_position_spec = self._nodes[0].fen
        if self._position_specs_match_parent_move_encoding(root_position_spec):
            return {"root_position_spec": root_position_spec, "fens": None}
        return {"root_position_spec": None, "fens": [node.fen for node in self._nodes]}

    def _position_specs_match_parent_move_encoding(self, root_position_spec: str) -> bool:
        if self.root_id is None:
            return False
        for node in self._nodes:
            if node.parent_id is None:
                if node.fen != root_position_spec:
                    return False
                continue
            if node.incoming_move_uci is None:
                return False
            parent_fen = self._nodes[node.parent_id].fen
            if append_move_to_position_spec(parent_fen, node.incoming_move_uci) != node.fen:
                return False
        return True

    @staticmethod
    def _restore_position_specs(
        parent_ids: List[Optional[int]],
        incoming_moves: List[Optional[str]],
        root_position_spec: Optional[str],
        fens: Optional[List[str]],
    ) -> List[str]:
        if fens is not None:
            return [str(fen) for fen in fens]
        if root_position_spec is None:
            raise ValueError("Serialized tree must include either fens or root_position_spec.")

        resolved_fens: List[str] = []
        for node_id, (parent_id, move) in enumerate(zip(parent_ids, incoming_moves)):
            if parent_id is None:
                resolved_fens.append(str(root_position_spec))
                continue
            if parent_id >= node_id:
                raise ValueError("Serialized parent_ids must be topologically ordered.")
            if move is None:
                raise ValueError("Non-root serialized nodes must include an incoming move.")
            resolved_fens.append(append_move_to_position_spec(resolved_fens[parent_id], str(move)))
        return resolved_fens

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

    def best_root_child(self, value_feature: str = "value") -> int:
        root_children = self.root_children()
        if not root_children:
            raise ValueError("Root has no children to choose from.")

        def child_value(child_id: int) -> float:
            node = self.get_node(child_id)
            if value_feature not in node.scalar_features:
                raise KeyError(f"Child node {child_id} is missing feature '{value_feature}'.")
            return node.scalar_features[value_feature]

        return max(root_children, key=child_value)

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
