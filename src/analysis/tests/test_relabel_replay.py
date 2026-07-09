"""TDD sanity tests for ``analysis.relabel_replay`` -- the standalone reimplementation
of PUCT's accumulate-along-root-path backup used to relabel node values (see
module docstring in ``relabel_replay.py`` and ``saturation_hypothesis.md``).

Two layers:
  1. A hand-worked synthetic tree (``test_replay_synthetic_multilevel_backup``) where
     every edge's expected Q is computed by hand, including a MULTI-LEVEL backup
     (a grandchild's expansion propagating through TWO edges with alternating sign)
     -- the part most likely to be subtly wrong.
  2. A REAL tree from the ``puctvalue_md36`` corpus, fed its own ORIGINAL (unmodified)
     "value" column: the replay must reproduce the tree's own saved
     ``oracle_final_root_q_values``/``oracle_root_q_trace`` (near-bit-for-bit; only
     float16-storage + summation-order slop separates them) before we trust it on
     relabeled values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.relabel_replay import (
    build_trajectory,
    ordered_expansion_parent_ids,
    replay_puct_backup,
    saturation_fraction,
    tanh_cp_value,
)

_CORPUS_DIR = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/trees/puctvalue_md36")


def _sample_real_tree_path() -> Path | None:
    if not _CORPUS_DIR.is_dir():
        return None
    for p in sorted(_CORPUS_DIR.glob("*.pt"))[:5]:
        return p
    return None


# ===========================================================================
# 1. Hand-worked synthetic tree
# ===========================================================================
def _synthetic_tree():
    """8-node tree, hand-designed so a multi-level backup is exercised:

        0(root) -> 1,2,3
        1 -> 4,5
        4 -> 6,7

    Node 2 is a TERMINAL leaf (backed up once, no children). Nodes 3,5,6,7 are
    plain untouched frontier leaves (never backed up). Nodes 0,1,4 are the three
    real expansions, in that causal order.
    """
    parent_index = np.array([-1, 0, 0, 0, 1, 1, 4, 4], dtype=np.int64)
    child_ptr = np.array([0, 3, 5, 5, 5, 7, 7, 7, 7], dtype=np.int64)
    children_index = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    is_expanded = np.array([True, True, False, False, True, False, False, False])
    is_terminal = np.array([False, False, True, False, False, False, False, False])
    node_values = np.array([0.0, 0.5, -0.3, 0.9, 0.1, -0.6, 0.2, -0.4], dtype=np.float64)
    depth = np.array([0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    return parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values, depth


def test_ordered_expansion_parent_ids_synthetic():
    parent_index, child_ptr, children_index, is_expanded, _, _, _ = _synthetic_tree()
    assert ordered_expansion_parent_ids(parent_index, child_ptr, children_index, is_expanded) == [0, 1, 4]


def test_replay_synthetic_multilevel_backup():
    parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values, _ = _synthetic_tree()
    result = replay_puct_backup(parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values)

    assert result.root_children.tolist() == [1, 2, 3]
    assert result.expansion_parent_ids.tolist() == [0, 1, 4]
    assert result.root_q_trace.shape == (3, 3)

    # Step 0: right after root's own expansion, before anything else has been
    # backed up -- every child is at its FPU seed (= -own static value).
    np.testing.assert_allclose(result.root_q_trace[0], [-0.5, 0.3, -0.9], atol=1e-12)

    # Step 1: node 1's own expansion backs up ITS value once -- with exactly one
    # visit this coincides with the FPU seed, so the row is unchanged. Node 2's
    # terminal backup (interleaved between steps 0 and 1 by node-id ordering)
    # also coincides with its own FPU seed for the same one-visit reason.
    np.testing.assert_allclose(result.root_q_trace[1], [-0.5, 0.3, -0.9], atol=1e-12)

    # Step 2: node 4's expansion backs up ITS OWN value through TWO edges
    # (4->1 and 1->0), flipping sign each time. Edge (0,1) now has TWO visits
    # (from node 1's own backup AND node 4's), averaging away from the pure FPU
    # value -- this is the one row that actually exercises multi-level backup.
    # visit1 contributed -0.5 (=-v1), visit2 contributed +v4=+0.1 (double flip
    # over two edges) -> mean = (-0.5 + 0.1) / 2 = -0.2.
    np.testing.assert_allclose(result.root_q_trace[2], [-0.2, 0.3, -0.9], atol=1e-12)

    np.testing.assert_allclose(result.final_root_q, [-0.2, 0.3, -0.9], atol=1e-12)
    assert result.best_move_index.tolist() == [1, 1, 1]  # child "2" (value 0.3) wins every step


def test_replay_ignores_untouched_frontier_nodes():
    """Node 3 (plain untouched leaf) and nodes 5,6,7 never enter node_values in a
    way that affects anything -- corrupting their values must not change the result."""
    parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values, _ = _synthetic_tree()
    corrupted = node_values.copy()
    corrupted[[3, 5, 6, 7]] = 999.0  # these must never be read by backup
    a = replay_puct_backup(parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values)
    b = replay_puct_backup(parent_index, child_ptr, children_index, is_expanded, is_terminal, corrupted)
    np.testing.assert_allclose(a.final_root_q, [-0.2, 0.3, -0.9], atol=1e-12)
    assert a.final_root_q[0] == b.final_root_q[0] and a.final_root_q[1] == b.final_root_q[1]
    # Only move index 2 (node 3, the untouched frontier root child) reads node_values[3]
    # directly as its FPU seed -- that ONE entry is allowed to change.
    assert a.final_root_q[2] != b.final_root_q[2]


def test_build_trajectory_synthetic():
    parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values, depth = _synthetic_tree()
    traj = build_trajectory(parent_index, child_ptr, children_index, is_expanded, is_terminal, depth, node_values)
    assert traj is not None
    assert traj["num_steps"] == 3  # root is expansion_parent_ids[0] -> root_rank=0, no trimming
    # tree_sizes: cumulative node count after each real expansion (root:1+3=4, node1:+2=6, node4:+2=8)
    assert traj["tree_sizes"] == [4, 6, 8]
    # halt_rewards[t] = final_root_q[best_move_index[t]] = final_root_q[1] = 0.3 for every step
    # (move index 1 was the argmax at every intermediate step in this synthetic tree)
    np.testing.assert_allclose(traj["halt_rewards"], [0.3, 0.3, 0.3], atol=1e-12)
    assert traj["time_budgets"] == [3, 2, 1]


# ===========================================================================
# 2. Real-tree bit-for-bit (near-) reproduction with ORIGINAL values
# ===========================================================================
@pytest.mark.skipif(_sample_real_tree_path() is None, reason="puctvalue_md36 corpus not available on this machine")
def test_replay_reproduces_real_tree_original_values():
    path = _sample_real_tree_path()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_names = list(payload["feature_names"])
    value_col = feature_names.index("value")
    node_features = payload["node_features"].to(torch.float64).numpy()
    node_values = node_features[:, value_col]

    result = replay_puct_backup(
        payload["parent_index"].numpy(),
        payload["child_ptr"].numpy(),
        payload["children_index"].numpy(),
        payload["is_expanded"].numpy(),
        payload["is_terminal"].numpy(),
        node_values,
    )

    orig_final_q = payload["oracle_final_root_q_values"].to(torch.float64).numpy()
    orig_trace = payload["oracle_root_q_trace"].to(torch.float64).numpy()
    orig_best_move = payload["oracle_best_move_index"].numpy()
    orig_moves = list(payload["oracle_root_moves"])

    # ``oracle_root_moves`` reflects the LIVE generation-time child order (whatever
    # order the provider returned moves in); ``children_index`` reflects a
    # UCI-string-sorted canonical order applied later at tensorization
    # (``_child_ptr_and_children_index`` docstring). The two orderings can differ
    # -- reindex by MOVE STRING (not array position) to compare apples-to-apples.
    incoming_moves = payload["incoming_moves"]
    replay_move_to_col = {incoming_moves[node_id]: col for col, node_id in enumerate(result.root_children.tolist())}
    reindex = [replay_move_to_col[move] for move in orig_moves]
    replay_trace_reindexed = result.root_q_trace[:, reindex]
    replay_final_q_reindexed = result.final_root_q[reindex]

    assert replay_trace_reindexed.shape == orig_trace.shape
    # float16 on-disk storage of node_features means values were quantized to
    # ~1e-3 relative precision before backup even starts; summed over up to ~96
    # backups the tolerance below is generous but still far tighter than any
    # signal we care about (saturated values differ from unsaturated by O(0.1-1)).
    np.testing.assert_allclose(replay_trace_reindexed, orig_trace, atol=5e-3)
    np.testing.assert_allclose(replay_final_q_reindexed, orig_final_q, atol=5e-3)

    # argmax should agree at (near-)every step; original ties are broken via a
    # per-tree RNG we don't replay, so allow a small mismatch rate rather than
    # requiring exact equality. Compare in the ORIGINAL move-index space.
    replay_best_move_orig_space = np.array([reindex.index(c) for c in result.best_move_index.tolist()])
    mismatch_rate = float(np.mean(replay_best_move_orig_space != orig_best_move))
    assert mismatch_rate < 0.05, f"best-move argmax mismatch rate too high: {mismatch_rate:.3f}"


@pytest.mark.skipif(_sample_real_tree_path() is None, reason="puctvalue_md36 corpus not available on this machine")
def test_build_trajectory_matches_production_on_real_tree():
    """build_trajectory(..., original 'value' column) must match
    cts.data.preprocess_mc.pack.build_compact_trajectory's halt_rewards/tree_sizes
    on a real tree -- the two are meant to be the same computation, one via live
    tree-search replay bookkeeping, one via this standalone replay."""
    from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
    from cts.data.preprocess_mc.pack import build_compact_trajectory

    path = _sample_real_tree_path()
    record = RawPretrainExampleRecord.load(str(path))
    prod_traj = build_compact_trajectory(record, source_path=str(path))
    assert prod_traj is not None

    value_col = list(record.feature_names).index("value")
    node_values = record.node_features.to(torch.float64).numpy()[:, value_col]
    my_traj = build_trajectory(
        record.parent_index.numpy(), record.child_ptr.numpy(), record.children_index.numpy(),
        record.is_expanded.numpy(), record.is_terminal.numpy(), record.depth.numpy(), node_values,
    )
    assert my_traj is not None
    assert my_traj["num_steps"] == prod_traj["num_steps"]
    assert my_traj["tree_sizes"] == prod_traj["tree_sizes"].tolist()
    np.testing.assert_allclose(my_traj["halt_rewards"], prod_traj["halt_rewards"], atol=5e-3)


# ===========================================================================
# 3. Relabeling helpers
# ===========================================================================
def test_tanh_cp_value_bounded_and_monotone():
    cp = np.array([-2000.0, -300.0, 0.0, 300.0, 2000.0])
    v = tanh_cp_value(cp, temperature=600.0)
    assert np.all(v > -1.0) and np.all(v < 1.0)
    assert np.all(np.diff(v) > 0)  # monotone increasing in cp
    np.testing.assert_allclose(v[2], 0.0, atol=1e-12)


def test_tanh_cp_value_lower_temperature_saturates_more():
    cp = np.linspace(-1500, 1500, 501)
    low_t = tanh_cp_value(cp, temperature=150.0)
    high_t = tanh_cp_value(cp, temperature=1200.0)
    assert saturation_fraction(low_t) > saturation_fraction(high_t)


def test_saturation_fraction_basic():
    values = np.array([1.0, -1.0, 0.995, 0.5, -0.2, 0.0])
    assert saturation_fraction(values, threshold=0.99) == pytest.approx(3 / 6)
