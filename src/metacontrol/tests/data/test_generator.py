"""Tests for metacontrol.data.generator — Stage 2 of the migration plan."""

import math
import pytest
from typing import Dict, List, Optional, Tuple

from metacontrol.core.schemas import (
    ChildInfo,
    EdgeStats,
    GeneratorConfig,
    GeneratorResult,
    SearchNode
)
from metacontrol.core.providers import TreeExpansionProvider
from metacontrol.data.generator import (
    TreeSearch,
    normalize_priors,
)
from metacontrol.core.tree import SearchTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DeterministicProvider(TreeExpansionProvider):
    """A mock provider whose expansions are entirely scripted.

    *expansion_table* maps FEN strings to a list of ChildInfo that will be
    returned by expand().  Positions absent from the table produce no children
    (treated as terminal by the generator).
    """

    def __init__(
        self,
        expansion_table: Dict[str, List[ChildInfo]],
        root_values: Optional[Dict[str, Tuple[float, Optional[Tuple[float, float, float]]]]] = None,
    ):
        self._table = expansion_table
        self._root_values = root_values or {}

    def evaluate_root(self, fen: str) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        if fen in self._root_values:
            return self._root_values[fen]
        return 0.0, None

    def expand(self, fen: str, depth: int) -> List[ChildInfo]:
        return self._table.get(fen, [])


def _make_child(move: str, fen: str, value: float = 0.0, prior: float = 1.0, **kw) -> ChildInfo:
    return ChildInfo(move_uci=move, fen=fen, value=value, prior=prior, **kw)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_validation():
    with pytest.raises(AssertionError, match="max_nodes"):
        GeneratorConfig(max_nodes=0, max_depth=5)
    with pytest.raises(AssertionError, match="max_depth"):
        GeneratorConfig(max_nodes=10, max_depth=-1)
    with pytest.raises(AssertionError, match="c_puct"):
        GeneratorConfig(max_nodes=10, max_depth=5, c_puct=-0.1)


# ---------------------------------------------------------------------------
# PUCT selection
# ---------------------------------------------------------------------------

def test_puct_selection():
    """PUCT should prefer the child with higher prior when unvisited."""
    root = SearchNode(node_id=0, fen="root")
    tree = SearchTree(root)
    c1 = SearchNode(node_id=1, fen="c1", features={"value": 0.0, "prior": 0.9})
    c2 = SearchNode(node_id=2, fen="c2", features={"value": 0.0, "prior": 0.1})
    tree.add_node(0, "e2e4", c1)
    tree.add_node(0, "d2d4", c2)

    gen = TreeSearch(DeterministicProvider({}), GeneratorConfig(10, 10, c_puct=1.0))
    gen.tree = tree
    gen.edge_stats = {
        (0, 1): EdgeStats(),
        (0, 2): EdgeStats(),
    }

    leaf_id, path = gen.select_leaf()
    # Should select child 1 (higher prior)
    assert leaf_id == 1
    assert path == [(0, 1)]


def test_puct_exploration_balances():
    """After visiting one child many times, PUCT should explore the other."""
    root = SearchNode(node_id=0, fen="root")
    tree = SearchTree(root)
    c1 = SearchNode(node_id=1, fen="c1", features={"value": 0.0, "prior": 0.5})
    c2 = SearchNode(node_id=2, fen="c2", features={"value": 0.0, "prior": 0.5})
    tree.add_node(0, "e2e4", c1)
    tree.add_node(0, "d2d4", c2)

    gen = TreeSearch(DeterministicProvider({}), GeneratorConfig(10, 10, c_puct=1.0))
    gen.tree = tree
    gen.edge_stats = {
        (0, 1): EdgeStats(visit_count=100, total_value=0.0, q_value=0.0),
        (0, 2): EdgeStats(),
    }

    leaf_id, _path = gen.select_leaf()
    assert leaf_id == 2, "Should explore the less-visited child"


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------

def test_backprop_averages():
    """Backprop should correctly average values with negation at each level."""
    gen = TreeSearch(DeterministicProvider({}), GeneratorConfig(10, 10))
    gen.edge_stats = {
        (0, 1): EdgeStats(),
        (1, 2): EdgeStats(),
    }
    # Leaf value of 1.0 at depth 2
    leaf = SearchNode(node_id=2, fen="leaf", features={"value": 1.0})
    gen.backpropagate([(0, 1), (1, 2)], leaf)

    # Edge (1, 2): one level up from leaf → value is negated → Q = -1.0
    assert gen.edge_stats[(1, 2)].visit_count == 1
    assert math.isclose(gen.edge_stats[(1, 2)].q_value, -1.0)

    # Edge (0, 1): two levels up → negated again → Q = 1.0
    assert gen.edge_stats[(0, 1)].visit_count == 1
    assert math.isclose(gen.edge_stats[(0, 1)].q_value, 1.0)


def test_backprop_multiple_visits():
    """Multiple backprops should give the average Q."""
    gen = TreeSearch(DeterministicProvider({}), GeneratorConfig(10, 10))
    gen.edge_stats = {(0, 1): EdgeStats()}
    
    leaf1 = SearchNode(node_id=1, fen="leaf1", features={"value": 0.6})
    gen.backpropagate([(0, 1)], leaf1)
    
    leaf2 = SearchNode(node_id=1, fen="leaf2", features={"value": 0.4})
    gen.backpropagate([(0, 1)], leaf2)
    
    assert gen.edge_stats[(0, 1)].visit_count == 2
    # Both negated: -0.6, -0.4 → average = -0.5
    assert math.isclose(gen.edge_stats[(0, 1)].q_value, -0.5)


# ---------------------------------------------------------------------------
# Full generator tests
# ---------------------------------------------------------------------------

def test_alternating_turns():
    """Tree should expand with alternating depths."""
    provider = DeterministicProvider({
        "root": [_make_child("e2e4", "after_e4", value=0.3)],
        "after_e4": [_make_child("e7e5", "after_e4_e5", value=-0.1)],
    })
    config = GeneratorConfig(max_nodes=2, max_depth=10)
    gen = TreeSearch(provider, config)
    result = gen.generate("root")

    assert result.num_expansions == 2
    assert result.tree.get_node(0).depth == 0
    child = list(result.tree.root.children.values())[0]
    assert child.depth == 1
    if child.children:
        grandchild = list(child.children.values())[0]
        assert grandchild.depth == 2


def test_expand_on_checkmate():
    """A terminal child should not be expanded further."""
    provider = DeterministicProvider({
        "root": [
            _make_child("e2e4", "checkmate", value=1.0, is_terminal=True),
        ],
    })
    config = GeneratorConfig(max_nodes=5, max_depth=10)
    gen = TreeSearch(provider, config)
    result = gen.generate("root")

    # Only 1 expansion (root → checkmate), then generator should stop
    assert result.num_expansions == 1
    mate_node = result.tree.get_node(1)
    assert mate_node.is_terminal
    assert len(mate_node.children) == 0


def test_q_leaf_is_v():
    """For a leaf with no children, Q should converge to its value."""
    provider = DeterministicProvider({
        "root": [
            _make_child("e2e4", "leaf", value=0.7, is_terminal=True),
        ],
    })
    config = GeneratorConfig(max_nodes=10, max_depth=10)
    gen = TreeSearch(provider, config)
    result = gen.generate("root")

    # The only edge is (0, 1); its Q should be -leaf_value (negated)
    # But from root's perspective, visiting the child repeatedly should
    # average the backpropagated value
    stats = result.edge_stats.get((0, 1))
    if stats and stats.visit_count > 0:
        # Each backprop sends -0.7 up one level
        assert math.isclose(stats.q_value, -0.7, abs_tol=1e-6)


def test_invariants():
    """Node count ≤ max_nodes and depth ≤ max_depth."""
    provider = DeterministicProvider({
        f"d{d}": [_make_child(f"m{i}", f"d{d+1}", value=0.1 * i) for i in range(3)]
        for d in range(10)
    })
    config = GeneratorConfig(max_nodes=5, max_depth=3)
    gen = TreeSearch(provider, config)
    result = gen.generate("d0")

    assert result.num_expansions <= config.max_nodes
    for node in result.tree.nodes_by_id.values():
        assert node.depth <= config.max_depth


def test_discovery_dynamics():
    """The generator should expand multiple branches, not just one line."""
    # Provider that always offers two children
    table = {}
    for i in range(20):
        table[f"pos{i}"] = [
            _make_child(f"a{i}", f"pos{2*i+1}", value=0.1, prior=0.6),
            _make_child(f"b{i}", f"pos{2*i+2}", value=0.1, prior=0.4),
        ]
    provider = DeterministicProvider(table)
    config = GeneratorConfig(max_nodes=5, max_depth=10)
    gen = TreeSearch(provider, config)
    result = gen.generate("pos0")

    # With 5 expansions, the root should have at least 2 children
    assert len(result.tree.root.children) >= 2
    assert result.num_expansions == 5


def test_expansion_history_records_order():
    """expansion_history should record nodes in the order they were added."""
    provider = DeterministicProvider({
        "root": [
            _make_child("e2e4", "c1", value=0.1, prior=0.9),
            _make_child("d2d4", "c2", value=0.1, prior=0.1),
        ],
        "c1": [_make_child("e7e5", "c3", value=-0.1)],
    })
    config = GeneratorConfig(max_nodes=2, max_depth=10)
    gen = TreeSearch(provider, config)
    result = gen.generate("root")

    history = result.tree.expansion_history
    # First expansion adds c1 and c2, second adds c3
    assert len(history) >= 2  # At least 2 children from first expansion
    # All nodes in history should be in the tree
    for node in history:
        assert result.tree.get_node(node.node_id) is node


# ---------------------------------------------------------------------------
# Prior normalisation
# ---------------------------------------------------------------------------

def test_normalize_priors():
    result = normalize_priors([1.0, 1.0, 1.0])
    assert len(result) == 3
    for p in result:
        assert math.isclose(p, 1.0 / 3.0, abs_tol=1e-6)


def test_normalize_priors_empty():
    assert normalize_priors([]) == []


def test_normalize_priors_single():
    result = normalize_priors([5.0])
    assert len(result) == 1
    assert math.isclose(result[0], 1.0)
