"""T-tgt / T-nan: target tensors and output health for the batched loop.

Report §4 IDs covered:
  - T-tgt: ``edge_wdl_targets`` (visit-weighted, perspective-flipped), node
           values, and ``target_advantages`` (value_gap) match the legacy path
           within a tight tolerance.
  - T-nan: batched outputs contain no NaN/inf where finite values are required,
           and the serialized record has the expected shapes/dtypes.

Uses a deterministic ``MockEvaluator`` and the small fixed FEN list; no
engine/GPU/network.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord

from cts.data.batched_gen.evaluator import MockEvaluator
from cts.data.batched_gen.search import generate_trees_batched

from test_batched_gen_common import (
    SMALL_FEN_LIST,
    legacy_example,
    make_budget,
    make_config,
)

_TOL = 1e-9  # L1 is bit-exact for the same evaluator; this leaves headroom.


@pytest.mark.parametrize("fen", SMALL_FEN_LIST)
def test_node_values_and_edge_wdl_within_tol(fen: str) -> None:
    """T-tgt: node value targets + edge WDL targets match legacy within tol."""
    mock = MockEvaluator()
    config = make_config()
    budget = make_budget()
    seed = 21
    root_id = f"tgt::{fen}"

    expected = legacy_example(fen, mock, config, budget, seed=seed, root_position_id=root_id)
    got = generate_trees_batched(
        [fen], mock, config, budget, rng=random.Random(seed), root_position_ids=[root_id]
    )[0]

    assert len(got.node_target_values) == len(expected.node_target_values)
    for g, e in zip(got.node_target_values, expected.node_target_values):
        assert abs(g - e) <= _TOL, f"node value {g} vs {e}"

    assert set(got.edge_wdl_targets) == set(expected.edge_wdl_targets)
    for key, g_triple in got.edge_wdl_targets.items():
        e_triple = expected.edge_wdl_targets[key]
        # WDL targets are L1-normalized -> each component in [0,1], sums to ~1.
        assert sum(g_triple) == pytest.approx(1.0, abs=1e-6), f"wdl mass @ {key}"
        for gc, ec in zip(g_triple, e_triple):
            assert abs(gc - ec) <= 1e-6, f"wdl @ {key}: {gc} vs {ec}"


@pytest.mark.parametrize("fen", SMALL_FEN_LIST)
def test_target_advantages_value_gap_within_tol(fen: str) -> None:
    """T-tgt: per-node value_gap (the centipawn 'target advantage') matches legacy.

    ``value_gap`` is NaN for leaves / single-child nodes by construction, so we
    compare NaN-aware and require finite entries to match within tol.
    """
    mock = MockEvaluator()
    config = make_config()
    budget = make_budget()
    seed = 87
    root_id = f"adv::{fen}"

    expected = legacy_example(fen, mock, config, budget, seed=seed, root_position_id=root_id)
    got = generate_trees_batched(
        [fen], mock, config, budget, rng=random.Random(seed), root_position_ids=[root_id]
    )[0]

    assert len(got.value_gap) == len(expected.value_gap)
    for g, e in zip(got.value_gap, expected.value_gap):
        if math.isnan(g) or math.isnan(e):
            assert math.isnan(g) and math.isnan(e)
        else:
            assert abs(g - e) <= 1e-6


@pytest.mark.parametrize("fen", SMALL_FEN_LIST)
def test_no_nan_inf_in_required_fields(fen: str) -> None:
    """T-nan: node values + oracle Q traces + WDL targets are all finite."""
    mock = MockEvaluator()
    config = make_config()
    budget = make_budget()
    got = generate_trees_batched(
        [fen], mock, config, budget, rng=random.Random(3)
    )[0]

    assert all(math.isfinite(v) for v in got.node_target_values), "node values finite"
    for row in got.oracle_root_q_trace:
        assert all(math.isfinite(v) for v in row), "oracle q trace finite"
    for triple in got.edge_wdl_targets.values():
        assert all(math.isfinite(c) for c in triple), "edge wdl finite"
    # Node features that must be finite for downstream PUCT: value + prior.
    for node in got.tree.iter_nodes():
        assert math.isfinite(node.scalar_features["value"]), f"value @ {node.node_id}"
        if node.parent_id is not None:
            assert math.isfinite(node.scalar_features["prior"]), f"prior @ {node.node_id}"


@pytest.mark.parametrize("fen", SMALL_FEN_LIST)
def test_record_shapes_and_dtypes(fen: str) -> None:
    """T-nan: serializing the batched example yields a record with valid shapes/dtypes.

    Round-tripping through ``RawPretrainExampleRecord`` exercises every shape
    invariant the on-disk format requires (``_validate``), so a clean
    conversion is itself a strong shape/dtype check on the batched output.
    """
    mock = MockEvaluator()
    config = make_config()
    budget = make_budget()
    example = generate_trees_batched(
        [fen], mock, config, budget, rng=random.Random(11)
    )[0]

    record = RawPretrainExampleRecord.from_example(example)  # runs _validate()
    num_nodes = example.tree.num_nodes()
    assert record.parent_index.shape == (num_nodes,)
    assert record.parent_index.dtype == torch.int32
    assert record.node_targets.shape == (num_nodes,)
    assert record.node_targets.dtype == torch.float32
    assert record.child_ptr.shape == (num_nodes + 1,)
    assert record.depth.dtype == torch.int16
    assert record.is_terminal.dtype == torch.bool
    assert record.is_expanded.dtype == torch.bool
    assert not torch.isnan(record.node_targets).any(), "node targets must be finite"

    # Rehydrate and confirm node count survives the round trip.
    rehydrated = record.to_pretrain_example()
    assert rehydrated.tree.num_nodes() == num_nodes
