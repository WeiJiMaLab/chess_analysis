"""Shared helpers/fixtures for the batched-tree-generation parity tests.

These tests cover the §4 "Tests we need" table in
``reports/batched-tree-generation.md`` for the new ``cts.data.batched_gen``
subpackage. This module holds the small, fast, synthetic helpers reused across
``test_batched_gen_*.py``; it is imported by the other files rather than via
``conftest.py`` (which the contract forbids us from touching).

Nothing here drives a real engine, GPU, or network — everything is built on a
deterministic ``MockEvaluator`` and tiny FEN lists.

Public symbols under test (the BINDING CONTRACT this exercises):
  cts.data.batched_gen.evaluator
    - PositionEval(priors: dict[str,float], value: float, wdl: tuple[float,float,float])
    - Evaluator (abc) with .evaluate(fens) -> list[PositionEval], .clear_caches()
    - MockEvaluator(table=None)
    - CachedEvaluator(inner, cache_path)
    - evaluator_to_provider(ev) -> TreeExpansionProvider
  cts.data.batched_gen.search
    - generate_trees_batched(fens, evaluator, search_config,
        node_budget_distribution, *, rng, max_concurrent=1024,
        root_position_ids=None) -> list[PretrainExample]
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import torch

from cts.core.tree import SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    PretrainExample,
    TeacherSearchConfig,
    build_pretrain_example,
)

# --------------------------------------------------------------------------- #
# Fixed, tiny FEN list. Deliberately short games / forced lines so the
# generated trees stay small and the tests run fast.
# --------------------------------------------------------------------------- #
SMALL_FEN_LIST: Tuple[str, ...] = (
    # Standard opening position.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # After 1. e4.
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # A simple king-and-pawn endgame (few legal moves -> small branching).
    "8/8/8/3k4/8/3K4/4P3/8 w - - 0 1",
    # A middlegame-ish position with moderate branching.
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
)


def make_config() -> TeacherSearchConfig:
    """A small but non-trivial PUCT config used across the parity tests."""
    return TeacherSearchConfig(
        max_depth=3,
        search_budget=8,
        c_puct=1.25,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id="batched-gen-test",
    )


def make_budget(min_nodes: int = 6, max_nodes: int = 12) -> NodeBudgetDistribution:
    """A small log-uniform node-budget distribution for fast trees."""
    return NodeBudgetDistribution(min_nodes=min_nodes, max_nodes=max_nodes)


def fresh_rng(seed: int = 1234) -> random.Random:
    """A fresh seeded RNG so each consumer controls budget sampling."""
    return random.Random(seed)


# --------------------------------------------------------------------------- #
# Parity comparison utilities.
# --------------------------------------------------------------------------- #
def _trees_structurally_identical(a: SearchTree, b: SearchTree) -> None:
    """Assert two SearchTrees are byte-identical in topology + node features."""
    assert a.num_nodes() == b.num_nodes(), (
        f"node count differs: {a.num_nodes()} != {b.num_nodes()}"
    )
    assert a.num_edges() == b.num_edges(), (
        f"edge count differs: {a.num_edges()} != {b.num_edges()}"
    )
    for na, nb in zip(a.iter_nodes(), b.iter_nodes()):
        assert na.node_id == nb.node_id
        assert na.parent_id == nb.parent_id, f"parent_id @ {na.node_id}"
        assert na.incoming_move_uci == nb.incoming_move_uci, f"move @ {na.node_id}"
        assert na.fen == nb.fen, f"fen @ {na.node_id}"
        assert na.depth == nb.depth, f"depth @ {na.node_id}"
        assert na.is_terminal == nb.is_terminal, f"terminal @ {na.node_id}"
        assert na.is_expanded == nb.is_expanded, f"expanded @ {na.node_id}"
        # Node features must match key-for-key, value-for-value (bit-exact).
        assert set(na.scalar_features) == set(nb.scalar_features), (
            f"feature keys @ {na.node_id}: "
            f"{set(na.scalar_features) ^ set(nb.scalar_features)}"
        )
        for key in na.scalar_features:
            va = na.scalar_features[key]
            vb = nb.scalar_features[key]
            if math.isnan(va) or math.isnan(vb):
                assert math.isnan(va) and math.isnan(vb), f"feature {key} @ {na.node_id}"
            else:
                assert va == vb, f"feature {key} @ {na.node_id}: {va} != {vb}"
    # Child id ordering (insertion order) must be identical.
    for node in a.iter_nodes():
        assert a.child_ids(node.node_id) == b.child_ids(node.node_id), (
            f"child order @ {node.node_id}"
        )


def _seq_equal(a: Sequence[float], b: Sequence[float], *, label: str) -> None:
    """Bit-exact equality for a flat numeric sequence, NaN-aware."""
    assert len(a) == len(b), f"{label}: length {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = float(x), float(y)
        if math.isnan(x) or math.isnan(y):
            assert math.isnan(x) and math.isnan(y), f"{label}[{i}]: {x} vs {y}"
        else:
            assert x == y, f"{label}[{i}]: {x} != {y}"


def assert_examples_byte_identical(
    got: PretrainExample,
    expected: PretrainExample,
    *,
    compare_metadata: bool = False,
) -> None:
    """Assert two ``PretrainExample`` objects are byte-identical.

    Covers tree topology + node features, node_target_values, edge_wdl_targets,
    and every oracle-trace array. ``metadata`` is compared only when asked
    (provider_metadata legitimately differs between the legacy provider path
    and the batched evaluator path).
    """
    _trees_structurally_identical(got.tree, expected.tree)

    _seq_equal(got.node_target_values, expected.node_target_values, label="node_target_values")
    _seq_equal(got.value_gap, expected.value_gap, label="value_gap")
    _seq_equal(got.policy_drift, expected.policy_drift, label="policy_drift")

    # Edge WDL targets: identical key set, each triple bit-exact.
    assert set(got.edge_wdl_targets) == set(expected.edge_wdl_targets), "edge_wdl key set"
    for key in got.edge_wdl_targets:
        _seq_equal(
            got.edge_wdl_targets[key],
            expected.edge_wdl_targets[key],
            label=f"edge_wdl_targets[{key}]",
        )

    # Oracle trace arrays.
    assert got.oracle_trace_expansion_counts == expected.oracle_trace_expansion_counts, (
        "oracle_trace_expansion_counts"
    )
    assert got.oracle_root_moves == expected.oracle_root_moves, "oracle_root_moves"
    assert got.oracle_best_move_trace == expected.oracle_best_move_trace, "oracle_best_move_trace"
    assert len(got.oracle_root_q_trace) == len(expected.oracle_root_q_trace), "q_trace rows"
    for i, (rg, re_) in enumerate(zip(got.oracle_root_q_trace, expected.oracle_root_q_trace)):
        _seq_equal(rg, re_, label=f"oracle_root_q_trace[{i}]")

    assert set(got.oracle_final_root_q_values) == set(expected.oracle_final_root_q_values), (
        "oracle_final_root_q_values key set"
    )
    for move in got.oracle_final_root_q_values:
        gv = got.oracle_final_root_q_values[move]
        ev = expected.oracle_final_root_q_values[move]
        assert gv == ev, f"oracle_final_root_q_values[{move}]: {gv} != {ev}"

    if compare_metadata:
        assert got.metadata == expected.metadata, "metadata"


def legacy_example(
    fen: str,
    mock,  # MockEvaluator
    config: TeacherSearchConfig,
    budget: NodeBudgetDistribution,
    *,
    seed: int,
    root_position_id: str,
) -> PretrainExample:
    """Drive the LEGACY ``build_pretrain_example`` through the same fixed eval table.

    Uses ``evaluator_to_provider`` so the legacy sequential loop sees exactly
    the same priors/value/WDL the batched loop sees — this is what makes the
    L1 (search-logic) comparison apples-to-apples.
    """
    from cts.data.batched_gen.evaluator import evaluator_to_provider

    provider = evaluator_to_provider(mock)
    return build_pretrain_example(
        fen,
        provider,
        config,
        budget,
        rng=random.Random(seed),
        root_position_id=root_position_id,
    )
