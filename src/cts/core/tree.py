"""``SearchTree`` is append-only: node ids are assigned in insertion order and
never change, so downstream code can reconstruct expansion history from id order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional


def _normalize_scalar_features(features: Mapping[str, float]) -> Dict[str, float]:
    """Validate and copy a feature mapping into a fresh ``dict[str, float]``.

    Used at every entry point that accepts user-supplied features so a
    bad value fails at the boundary rather than deep inside a tensor op.
    """
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
    """A single node in a ``SearchTree``.

    ``node_id`` is the node's insertion-order index in the owning tree.
    ``incoming_move_uci`` is the move that produced this node from its
    parent (None for the root). ``is_expanded`` flips to True the first
    time children are added; expansion is one-shot per node.
    """

    node_id: int  # insertion-order index in the owning tree; stable for the tree's lifetime
    parent_id: Optional[int]  # parent's node_id; None only for the root
    incoming_move_uci: Optional[str]  # UCI move that produced this node from its parent; None only for the root
    fen: str  # FEN string of the position at this node
    depth: int  # plies from the root (root has depth 0)
    is_terminal: bool  # True if the position has no legal moves (checkmate or stalemate)
    is_expanded: bool  # True once children have been added; one-shot per node
    scalar_features: Dict[str, float]  # per-node features (priors, WDL, value, etc.); keys per schema
    metadata: Dict[str, Any] = field(default_factory=dict)  # free-form extras (timings, provenance, etc.) not consumed by the encoder


class ExpansionChild(NamedTuple):
    """Specification for a single child to be appended to a parent.

    Used as the input format to ``SearchTree.add_children`` and as the
    return type of ``TreeExpansionProvider.expand_node``. Holds everything
    needed to materialize a ``SearchNode`` except the parent-derived fields
    (id, parent_id, depth) which the tree assigns at insertion. Validation
    of ``scalar_features`` happens at consumption in ``add_children`` rather
    than at construction — keeps the type cheap and immutable.
    """

    move_uci: str  # UCI move that leads from the parent to this child
    fen: str  # FEN string of the position after the move
    scalar_features: Dict[str, float]  # per-node features for the child (priors, WDL, etc.)
    metadata: Optional[Dict[str, Any]] = None  # optional free-form extras; None is normalized to {} at consumption
    is_terminal: bool = False  # True if the child position has no legal moves


class SearchTree:
    """Append-only search tree keyed by insertion-order node ids.

    The append-only invariant is load-bearing: ``ordered_expansion_parent_ids``
    and the snapshot/clone machinery rely on the fact that a parent's
    children occupy a contiguous block of node ids assigned at the moment
    of expansion. Mutating ``_nodes`` or ``_children`` outside the provided
    methods will break those invariants.

    The root node is materialized in ``__init__``; there is no separate
    ``create_root`` step. ``root_id`` is always 0.
    """

    root_id: int = 0  # always 0 — the root is the first node, installed at construction

    def __init__(
        self,
        root_fen: str,
        root_scalar_features: Mapping[str, float],
        root_metadata: Optional[Mapping[str, Any]] = None,
        root_is_terminal: bool = False,
    ) -> None:
        """Build a tree with the root node already installed.

        Args:
            root_fen: FEN string of the root position.
            root_scalar_features: per-node features for the root (priors, WDL, etc.).
            root_metadata: optional free-form extras stored on the root node.
            root_is_terminal: True if the root position has no legal moves.
        """
        self._nodes: List[SearchNode] = []  # nodes in insertion order; index == node_id
        self._children: Dict[int, List[int]] = {}  # node_id → ordered child ids
        root = SearchNode(
            node_id=0,
            parent_id=None,
            incoming_move_uci=None,
            fen=root_fen,
            depth=0,
            is_terminal=root_is_terminal,
            is_expanded=False,
            scalar_features=_normalize_scalar_features(root_scalar_features),
            metadata=dict(root_metadata or {}),
        )
        self._nodes.append(root)
        self._children[0] = []

    def add_children(self, parent_id: int, children: Iterable[ExpansionChild]) -> List[int]:
        """Expand ``parent_id`` with the given children and return their new ids.

        One-shot: a node can be expanded at most once. Terminal nodes
        cannot be expanded. Children are appended in the order they appear
        in ``children``, so child id ordering reflects the caller's ordering
        (e.g. UCI lexicographic, when that's the convention being followed).
        Each child's ``scalar_features`` is validated + copied into a fresh
        dict here, so callers can safely reuse / mutate the dicts they pass
        in afterwards.

        Args:
            parent_id: id of the node to expand; must not be terminal or already expanded.
            children: child specs to install; duplicate ``move_uci`` is rejected.
        """
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
                scalar_features=_normalize_scalar_features(child.scalar_features),
                metadata=dict(child.metadata or {}),
            )
            self._nodes.append(node)
            self._children[node_id] = []
            child_ids.append(node_id)

        self._children[parent_id].extend(child_ids)
        parent.is_expanded = True
        return child_ids

    def _validated_child_specs(self, parent_id: int, children: Iterable[ExpansionChild]) -> List[ExpansionChild]:
        """Materialize the iterable and reject duplicate moves under the same parent."""
        child_specs = list(children)
        seen_moves = set()
        for child in child_specs:
            if child.move_uci in seen_moves:
                raise ValueError(f"Duplicate child move '{child.move_uci}' for parent {parent_id}.")
            seen_moves.add(child.move_uci)
        return child_specs

    def get_node(self, node_id: int) -> SearchNode:
        """Return the node with the given id, or raise ``KeyError`` if out of range."""
        if node_id < 0 or node_id >= len(self._nodes):
            raise KeyError(f"Unknown node_id: {node_id}")
        return self._nodes[node_id]

    def children(self, node_id: int) -> List[int]:
        """Return a fresh copy of the child id list. Safe to mutate."""
        return list(self._children[node_id])

    def child_ids(self, node_id: int) -> List[int]:
        """Return the live child id list. Faster than ``children`` but DO NOT mutate."""
        return self._children[node_id]

    def root_children(self) -> List[int]:
        """Return a copy of the root's child id list."""
        return self.children(self.root_id)

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return sum(len(child_ids) for child_ids in self._children.values())

    def iter_nodes(self) -> Iterator[SearchNode]:
        """Iterate nodes in insertion order (which equals id order)."""
        return iter(self._nodes)

    def ordered_expansion_parent_ids(self) -> List[int]:
        """Return parent ids in the order they were expanded.

        Because children are assigned contiguous ids at the moment of
        expansion, the first child's id is a monotonic timestamp of when
        the parent was expanded. Sorting by ``(first_child_id, parent_id)``
        reconstructs the original expansion sequence.
        """
        expansion_parents = []
        for node in self.iter_nodes():
            child_ids = self.child_ids(node.node_id)
            if node.is_expanded and child_ids:
                expansion_parents.append((child_ids[0], node.node_id))
        expansion_parents.sort()
        return [node_id for _, node_id in expansion_parents]

    def clone_expansion_prefix(self, expansion_count: int) -> "SearchTree":
        """Return a new tree containing only the first ``expansion_count`` expansions.

        This is the core snapshotting primitive used by the controller's
        per-snapshot training data: take a fully-built tree, replay just
        the first N expansions, and treat the result as the controller's
        state at expansion step N. Always replays in the original
        expansion order via ``ordered_expansion_parent_ids``.

        Args:
            expansion_count: number of expansions to replay; must be in
                ``[0, total_expansions]``. 0 returns a root-only tree.
        """
        expansion_parent_ids = self.ordered_expansion_parent_ids()
        if expansion_count < 0 or expansion_count > len(expansion_parent_ids):
            raise ValueError(
                f"expansion_count must be between 0 and {len(expansion_parent_ids)}, got {expansion_count}."
            )

        root = self.get_node(self.root_id)
        cloned = SearchTree(
            root_fen=root.fen,
            root_scalar_features=root.scalar_features,
            root_metadata=root.metadata,
            root_is_terminal=root.is_terminal,
        )
        # Map original ids → new ids as we replay so we can look up the
        # cloned parent for each expansion. The mapping grows monotonically.
        node_id_map = {self.root_id: cloned.root_id}

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
        """Return a deep copy of the entire tree (all expansions replayed)."""
        return self.clone_expansion_prefix(len(self.ordered_expansion_parent_ids()))

    def _child_specs_for_parent(self, parent_id: int) -> List[ExpansionChild]:
        """Reconstruct ``ExpansionChild`` records from an already-expanded parent."""
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
        """Defensive consistency check used in tests and after deserialization.

        Verifies every invariant the rest of the code assumes: root id is 0,
        parent/child links are bidirectional, depths are consistent, no node
        is referenced by multiple parents, and terminal/expanded states
        don't contradict each other.
        """
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
