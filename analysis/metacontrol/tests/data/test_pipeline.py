import math
import pytest
import os
import chess
from typing import List

from metacontrol.core.tree import SearchTree
from metacontrol.core.schemas import ChessFeatureSchema, GeneratorConfig
from metacontrol.core.tensorizer import TreeTensorizer
from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider
from metacontrol.data.generator import TreeSearch
from metacontrol.data.targets_mc import derive_snapshots

# Real Engine Paths (Princeton Della)
LC0_BIN = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
STOCKFISH_BIN = "/home/hl4291/stockfish-sf_15/src/stockfish"

@pytest.fixture
def provider():
    if os.path.exists(LC0_BIN):
        return LC0ExpansionProvider(LC0_BIN, LC0_WEIGHTS, nodes=100)
    if os.path.exists(STOCKFISH_BIN):
        return StockfishExpansionProvider(STOCKFISH_BIN, nodes=1000)
    pytest.skip("No real engines found for pipeline tests.")

# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

def test_pipeline_fen_to_snapshots(provider):
    """FEN -> SearchGenerator -> derive_snapshots -> valid advantage targets."""
    config = GeneratorConfig(max_nodes=5, max_depth=4, c_puct=1.0)
    generator = TreeSearch(provider, config)

    result = generator.generate(chess.STARTING_FEN)

    assert result.num_expansions > 0
    assert len(result.tree.nodes_by_id) > 1
    
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=0.05)
    assert len(snapshots) == len(result.tree.search_expansion_history) + 1
    assert len(result.tree.search_expansion_history) == result.num_expansions
    assert all(snap.expansion_index == i for i, snap in enumerate(snapshots))

    for snap in snapshots:
        assert math.isclose(snap.advantage, snap.continue_value - snap.halt_reward, abs_tol=1e-10)

def test_pipeline_tensorization(provider):
    """Generated trees should be tensorizable."""
    config = GeneratorConfig(max_nodes=4, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate(chess.STARTING_FEN)

    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    batch = tensorizer.tensorize(result.tree)

    assert batch.num_nodes == len(result.tree.nodes_by_id)
    assert batch.num_edges > 0
    assert batch.node_features.shape[0] == batch.num_nodes

def test_pipeline_zero_cost_advantage_nonneg(provider):
    """With zero continue cost, advantages should be non-negative."""
    config = GeneratorConfig(max_nodes=6, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate(chess.STARTING_FEN)
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=0.0)

    for snap in snapshots:
        assert snap.advantage >= -1e-10

def test_pipeline_high_cost_should_halt_early(provider):
    """With very high continue cost, the DP should prefer halting immediately."""
    config = GeneratorConfig(max_nodes=5, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate(chess.STARTING_FEN)
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=10.0)

    for snap in snapshots:
        assert snap.advantage <= 1e-10
