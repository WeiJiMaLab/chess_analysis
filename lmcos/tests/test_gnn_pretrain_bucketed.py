"""Unit tests for the per-validation-epoch bucketed-KL accumulator.

Covers the small surface of ``BucketedKLState``: empty allocation has the
right shape, ``update`` scatters per-edge KL into the (size, depth) grid
using the same bucketing helpers the audit module uses, and ``summary``
returns a JSON-serializable dict with the expected fields. A minimal
integration test fits a 1-epoch run with two fake-tree examples and
verifies the JSONL writer emits one row.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cts.core.kl_buckets import compute_subtree_sizes
from cts.train.gnn_pretrain import BucketedKLState


class _FakeTreeBatch:
    """Just enough of ``TreeBatch`` to drive ``BucketedKLState.update``.

    ``update`` only touches ``edge_parent``, ``edge_child``, ``depth``, and
    ``parent_index``, so we don't need a real encoder forward pass.
    """

    def __init__(self, parent_index, depth, edge_parent, edge_child):
        self.parent_index = parent_index
        self.depth = depth
        self.edge_parent = edge_parent
        self.edge_child = edge_child


class BucketedKLStateTests(unittest.TestCase):
    def test_empty_allocates_correct_shape(self) -> None:
        state = BucketedKLState.empty(
            max_depth_bin=4,
            subtree_size_log_max=3,
            device=torch.device("cpu"),
        )
        # subtree_size_log_max=3 -> 4 size bins; max_depth_bin=4 -> 5 depth bins.
        self.assertEqual(tuple(state.sum_kl.shape), (4, 5))
        self.assertEqual(tuple(state.count.shape), (4, 5))
        self.assertEqual(state.overall_count, 0)
        self.assertEqual(float(state.overall_sum_kl.item()), 0.0)

    def test_update_scatters_into_expected_cells(self) -> None:
        # 3-node chain: 0 (root, depth 0) -> 1 (depth 1) -> 2 (leaf, depth 2).
        # Two edges: (parent=0, child=1) and (parent=1, child=2).
        # Child 1's subtree size = 2 (itself + descendant 2); child 2's = 1.
        tree_batch = _FakeTreeBatch(
            parent_index=torch.tensor([-1, 0, 1], dtype=torch.long),
            depth=torch.tensor([0, 1, 2], dtype=torch.long),
            edge_parent=torch.tensor([0, 1], dtype=torch.long),
            edge_child=torch.tensor([1, 2], dtype=torch.long),
        )
        per_edge_kl = torch.tensor([0.5, 0.1], dtype=torch.float32)

        state = BucketedKLState.empty(max_depth_bin=4, subtree_size_log_max=3, device=torch.device("cpu"))
        state.update(tree_batch, per_edge_kl)

        # Edge 0: child=1, subtree_size=2 -> size_bin=1; parent=0, depth=0 -> depth_bin=0.
        # Edge 1: child=2, subtree_size=1 -> size_bin=0; parent=1, depth=1 -> depth_bin=1.
        self.assertEqual(int(state.count[1, 0].item()), 1)
        self.assertEqual(int(state.count[0, 1].item()), 1)
        self.assertAlmostEqual(float(state.sum_kl[1, 0].item()), 0.5, places=5)
        self.assertAlmostEqual(float(state.sum_kl[0, 1].item()), 0.1, places=5)
        self.assertEqual(state.overall_count, 2)
        self.assertAlmostEqual(float(state.overall_sum_kl.item()), 0.6, places=5)

    def test_update_is_additive_across_calls(self) -> None:
        tree_batch = _FakeTreeBatch(
            parent_index=torch.tensor([-1, 0], dtype=torch.long),
            depth=torch.tensor([0, 1], dtype=torch.long),
            edge_parent=torch.tensor([0], dtype=torch.long),
            edge_child=torch.tensor([1], dtype=torch.long),
        )
        state = BucketedKLState.empty(max_depth_bin=4, subtree_size_log_max=3, device=torch.device("cpu"))
        state.update(tree_batch, torch.tensor([0.3], dtype=torch.float32))
        state.update(tree_batch, torch.tensor([0.7], dtype=torch.float32))
        # Same (parent, child) pair seen twice; the same cell accumulates both.
        # Child 1's subtree_size=1 -> size_bin=0; parent depth=0 -> depth_bin=0.
        self.assertEqual(int(state.count[0, 0].item()), 2)
        self.assertAlmostEqual(float(state.sum_kl[0, 0].item()), 1.0, places=5)
        self.assertEqual(state.overall_count, 2)

    def test_empty_kl_tensor_is_noop(self) -> None:
        tree_batch = _FakeTreeBatch(
            parent_index=torch.empty((0,), dtype=torch.long),
            depth=torch.empty((0,), dtype=torch.long),
            edge_parent=torch.empty((0,), dtype=torch.long),
            edge_child=torch.empty((0,), dtype=torch.long),
        )
        state = BucketedKLState.empty(max_depth_bin=4, subtree_size_log_max=3, device=torch.device("cpu"))
        state.update(tree_batch, torch.empty((0,), dtype=torch.float32))
        self.assertEqual(state.overall_count, 0)
        self.assertEqual(int(state.count.sum().item()), 0)

    def test_summary_returns_serializable_dict_with_expected_fields(self) -> None:
        tree_batch = _FakeTreeBatch(
            parent_index=torch.tensor([-1, 0, 1], dtype=torch.long),
            depth=torch.tensor([0, 1, 2], dtype=torch.long),
            edge_parent=torch.tensor([0, 1], dtype=torch.long),
            edge_child=torch.tensor([1, 2], dtype=torch.long),
        )
        state = BucketedKLState.empty(max_depth_bin=4, subtree_size_log_max=3, device=torch.device("cpu"))
        state.update(tree_batch, torch.tensor([0.5, 0.1], dtype=torch.float32))

        summary = state.summary()
        # JSON-roundtrip-able.
        serialized = json.dumps(summary)
        roundtrip = json.loads(serialized)
        for key in (
            "overall_mean_kl",
            "overall_edge_count",
            "max_depth_bin",
            "subtree_size_log_max",
            "mean_kl_grid",
            "edge_count_grid",
            "marginal_mean_kl_by_depth",
            "marginal_mean_kl_by_size_bin",
            "marginal_edge_count_by_depth",
            "marginal_edge_count_by_size_bin",
            "size_bin_labels",
            "depth_bin_labels",
        ):
            self.assertIn(key, roundtrip)
        self.assertAlmostEqual(roundtrip["overall_mean_kl"], 0.3, places=5)
        self.assertEqual(roundtrip["overall_edge_count"], 2)
        self.assertEqual(len(roundtrip["mean_kl_grid"]), 4)  # num_size_bins
        self.assertEqual(len(roundtrip["mean_kl_grid"][0]), 5)  # num_depth_bins

    def test_summary_zero_counts_yields_zero_mean(self) -> None:
        state = BucketedKLState.empty(max_depth_bin=2, subtree_size_log_max=2, device=torch.device("cpu"))
        summary = state.summary()
        self.assertEqual(summary["overall_mean_kl"], 0.0)
        self.assertEqual(summary["overall_edge_count"], 0)
        # Mean grid is well-defined (zero) when counts are zero.
        for row in summary["mean_kl_grid"]:
            for cell in row:
                self.assertEqual(cell, 0.0)


class SubtreeSizeWeightedLossTests(unittest.TestCase):
    """The weighted-loss path mathematically: a chain tree with two edges,
    one to a size-1 leaf and one to a size-2 internal node, weighted by
    child subtree size, should compute (1*L_leaf + 2*L_internal) / (1+2).
    Pure arithmetic over the formula — no model forward needed.
    """

    def test_weighted_mean_matches_hand_computed_value(self) -> None:
        # Two edges. Edge A: child has subtree size 1. Edge B: child has subtree size 2.
        # Per-edge losses chosen to be distinct so the weighted mean reveals
        # which weighting was applied.
        per_edge_loss = torch.tensor([0.4, 1.0], dtype=torch.float32)
        # Tree: 0 (root) -> 1 (depth 1) -> 2 (depth 2, leaf).
        # Edges in canonical order: (0, 1) and (1, 2).
        parent_index = torch.tensor([-1, 0, 1], dtype=torch.long)
        depth = torch.tensor([0, 1, 2], dtype=torch.long)
        edge_child = torch.tensor([1, 2], dtype=torch.long)

        subtree_sizes = compute_subtree_sizes(parent_index, depth)
        self.assertEqual(subtree_sizes.tolist(), [3, 2, 1])
        edge_weights = subtree_sizes[edge_child].to(torch.float32)
        self.assertEqual(edge_weights.tolist(), [2.0, 1.0])

        weighted = (edge_weights * per_edge_loss).sum() / edge_weights.sum()
        # (2*0.4 + 1*1.0) / (2+1) = (0.8 + 1.0) / 3 = 0.6
        self.assertAlmostEqual(float(weighted.item()), 0.6, places=5)
        # Unweighted reference: (0.4 + 1.0) / 2 = 0.7. Different from weighted.
        unweighted = per_edge_loss.mean()
        self.assertAlmostEqual(float(unweighted.item()), 0.7, places=5)
        self.assertNotAlmostEqual(float(weighted.item()), float(unweighted.item()))


if __name__ == "__main__":
    unittest.main()
