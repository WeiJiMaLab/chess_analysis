import math
import pytest
import os
import chess
from typing import Dict, List, Optional, Tuple

from metacontrol.core.schemas import (
    ChildInfo,
    EdgeStats,
    GeneratorConfig,
    SearchNode
)
from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider
from metacontrol.data.generator import (
    TreeSearch,
    normalize_priors,
)
from metacontrol.core.tree import SearchTree

# Real Engine Paths (Princeton Della)
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
STOCKFISH_BIN = "/home/hl4291/stockfish-sf_15/src/stockfish"

@pytest.fixture
def provider():
    if os.path.exists(LC0_BIN):
        return LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=128)
    if os.path.exists(STOCKFISH_BIN):
        return StockfishExpansionProvider(STOCKFISH_BIN, nodes=1000)
    pytest.skip("No real engines found for generator tests.")

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_validation():
    with pytest.raises(AssertionError, match="max_nodes"):
        GeneratorConfig(max_nodes=0, max_depth=5)
    with pytest.raises(AssertionError, match="max_depth"):
        GeneratorConfig(max_nodes=10, max_depth=-1)

# ---------------------------------------------------------------------------
# PUCT selection (Structural)
# ---------------------------------------------------------------------------

def test_puct_selection_structural():
    """PUCT should prefer the child with higher prior when unvisited."""
    root = SearchNode(node_id=0, fen="root")
    tree = SearchTree(root)
    c1 = SearchNode(node_id=1, fen="c1", features={"value": 0.0, "prior": 0.9})
    c2 = SearchNode(node_id=2, fen="c2", features={"value": 0.0, "prior": 0.1})
    tree.add_node(0, "e2e4", c1)
    tree.add_node(0, "d2d4", c2)

    # Use any real provider (path doesn't matter for structural selection)
    gen = TreeSearch(None, GeneratorConfig(10, 10, c_puct=1.0))
    gen.tree = tree
    gen.edge_stats = { (0, 1): EdgeStats(), (0, 2): EdgeStats() }

    leaf_id, path = gen.select_leaf()
    assert leaf_id == 1
    assert path == [(0, 1)]

def test_puct_exploration_balances_structural():
    """After visiting one child many times, PUCT should explore the other."""
    root = SearchNode(node_id=0, fen="root")
    tree = SearchTree(root)
    c1 = SearchNode(node_id=1, fen="c1", features={"value": 0.0, "prior": 0.5})
    c2 = SearchNode(node_id=2, fen="c2", features={"value": 0.0, "prior": 0.5})
    tree.add_node(0, "e2e4", c1)
    tree.add_node(0, "d2d4", c2)

    gen = TreeSearch(None, GeneratorConfig(10, 10, c_puct=1.0))
    gen.tree = tree
    gen.edge_stats = {
        (0, 1): EdgeStats(visit_count=100, total_value=0.0, q_value=0.0),
        (0, 2): EdgeStats(),
    }

    leaf_id, _ = gen.select_leaf()
    assert leaf_id == 2

# ---------------------------------------------------------------------------
# Backpropagation (Structural)
# ---------------------------------------------------------------------------

def test_backprop_averages_structural():
    gen = TreeSearch(None, GeneratorConfig(10, 10))
    gen.edge_stats = { (0, 1): EdgeStats(), (1, 2): EdgeStats() }
    leaf = SearchNode(node_id=2, fen="leaf", features={"value": 1.0})
    gen.backpropagate([(0, 1), (1, 2)], leaf)

    assert gen.edge_stats[(1, 2)].visit_count == 1
    assert math.isclose(gen.edge_stats[(1, 2)].q_value, -1.0)
    assert gen.edge_stats[(0, 1)].visit_count == 1
    assert math.isclose(gen.edge_stats[(0, 1)].q_value, 1.0)

# ---------------------------------------------------------------------------
# Full generator tests (Real Engine)
# ---------------------------------------------------------------------------

def test_real_generator_expands(provider):
    config = GeneratorConfig(max_nodes=5, max_depth=10)
    gen = TreeSearch(provider, config)
    result = gen.generate(chess.STARTING_FEN)

    assert result.num_expansions == 5
    assert len(result.tree.nodes_by_id) > 5
    # Check tree integrity
    for node in result.tree.nodes_by_id.values():
        assert node.fen is not None

def test_real_generator_respects_max_depth(provider):
    config = GeneratorConfig(max_nodes=50, max_depth=2)
    gen = TreeSearch(provider, config)
    result = gen.generate(chess.STARTING_FEN)

    for node in result.tree.nodes_by_id.values():
        assert node.depth <= 2

# ---------------------------------------------------------------------------
# Prior normalisation
# ---------------------------------------------------------------------------

def test_normalize_priors():
    result = normalize_priors([1.0, 1.0, 1.0])
    assert len(result) == 3
    for p in result:
        assert math.isclose(p, 1.0 / 3.0, abs_tol=1e-6)
