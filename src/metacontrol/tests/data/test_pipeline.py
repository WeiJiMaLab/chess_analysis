"""End-to-end pipeline tests — Stage 4 of the migration plan.

Tests the full flow:  FEN → SearchGenerator → derive_snapshots → tensorize.
"""

import math
import pytest
from typing import Dict, List, Optional, Tuple

from metacontrol.core.tree import SearchNode, SearchTree
from metacontrol.core.schema import ChessFeatureSchema
from metacontrol.core.tensorizer import TreeTensorizer
from metacontrol.data.generator import (
    ChildInfo,
    GeneratorConfig,
    TreeSearch,
    TreeExpansionProvider,
)
from metacontrol.data.targets_mc import derive_snapshots


# ---------------------------------------------------------------------------
# Mock provider for integration tests
# ---------------------------------------------------------------------------

class IntegrationProvider(TreeExpansionProvider):
    """A provider that generates a small but non-trivial tree structure.

    The tree has a branching factor of 2 at each level, with deterministic
    values and priors to make the search reproducible.
    """

    def evaluate_root(self, fen: str) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        return 0.0, (0.33, 0.34, 0.33)

    def expand(self, fen: str, depth: int) -> List[ChildInfo]:
        # Generate two children deterministically based on the FEN
        return [
            ChildInfo(
                move_uci=f"m{depth}a",
                fen=f"{fen}/a{depth}",
                value=0.1 * (depth + 1),
                prior=0.6,
                wdl=(0.4, 0.3, 0.3),
            ),
            ChildInfo(
                move_uci=f"m{depth}b",
                fen=f"{fen}/b{depth}",
                value=-0.1 * (depth + 1),
                prior=0.4,
                wdl=(0.3, 0.3, 0.4),
            ),
        ]


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

def test_pipeline_fen_to_snapshots():
    """FEN → SearchGenerator → derive_snapshots → valid advantage targets."""
    provider = IntegrationProvider()
    config = GeneratorConfig(max_nodes=5, max_depth=4, c_puct=1.0)
    generator = TreeSearch(provider, config)

    result = generator.generate("startpos")

    # The tree should have been grown
    assert result.num_expansions > 0
    assert len(result.tree.nodes_by_id) > 1
    assert len(result.tree.expansion_history) > 0

    # Derive snapshots
    snapshots = derive_snapshots(
        result.tree,
        result.edge_stats,
        continue_cost=0.05,
    )

    assert len(snapshots) == len(result.tree.expansion_history) + 1
    assert all(snap.expansion_index == i for i, snap in enumerate(snapshots))

    # Verify advantage consistency
    for snap in snapshots:
        assert math.isclose(
            snap.advantage,
            snap.continue_value - snap.halt_reward,
            abs_tol=1e-10,
        )

    # Last snapshot should have non-positive advantage (forced halt)
    assert snapshots[-1].advantage <= 1e-10


def test_pipeline_snapshots_have_valid_halt_rewards():
    """Halt rewards should be finite and reasonable."""
    provider = IntegrationProvider()
    config = GeneratorConfig(max_nodes=8, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate("startpos")
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=0.02)

    for snap in snapshots:
        assert math.isfinite(snap.halt_reward), f"Non-finite halt at step {snap.expansion_index}"
        assert math.isfinite(snap.continue_value)
        assert math.isfinite(snap.advantage)


def test_pipeline_tensorization():
    """Generated trees should be tensorizable."""
    provider = IntegrationProvider()
    config = GeneratorConfig(max_nodes=4, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate("startpos")

    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    batch = tensorizer.tensorize(result.tree)

    assert batch.num_nodes == len(result.tree.nodes_by_id)
    assert batch.num_edges > 0
    assert batch.batch_size == 1
    assert batch.node_features.shape[0] == batch.num_nodes
    assert batch.node_features.shape[1] == schema.num_features


def test_pipeline_multiple_trees_tensorization():
    """Multiple generated trees can be batched together."""
    provider = IntegrationProvider()
    config = GeneratorConfig(max_nodes=3, max_depth=2)
    generator = TreeSearch(provider, config)

    trees = []
    for fen in ["pos1", "pos2", "pos3"]:
        result = generator.generate(fen)
        trees.append(result.tree)

    schema = ChessFeatureSchema()
    tensorizer = TreeTensorizer(schema)
    batch = tensorizer.tensorize_forest(trees)

    expected_nodes = sum(len(t.nodes_by_id) for t in trees)
    assert batch.num_nodes == expected_nodes
    assert batch.batch_size == 3


def test_pipeline_zero_cost_advantage_nonneg():
    """With zero continue cost, advantages should be non-negative
    (continuing is free, so it's always at least as good as halting)."""
    provider = IntegrationProvider()
    config = GeneratorConfig(max_nodes=6, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate("startpos")
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=0.0)

    for snap in snapshots:
        assert snap.advantage >= -1e-10, (
            f"Negative advantage {snap.advantage} at step {snap.expansion_index} "
            f"with zero cost (halt={snap.halt_reward}, cont={snap.continue_value})"
        )


def test_pipeline_high_cost_should_halt_early():
    """With very high continue cost, the DP should prefer halting immediately."""
    provider = IntegrationProvider()
    config = GeneratorConfig(max_nodes=5, max_depth=3)
    generator = TreeSearch(provider, config)
    result = generator.generate("startpos")
    snapshots = derive_snapshots(result.tree, result.edge_stats, continue_cost=10.0)

    # With cost=10, continuing is never worth it
    for snap in snapshots:
        assert snap.advantage <= 1e-10, (
            f"Positive advantage {snap.advantage} at step {snap.expansion_index} "
            f"despite very high cost"
        )
