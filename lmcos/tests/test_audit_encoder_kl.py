"""Unit tests for the per-edge KL audit's topology and binning helpers.

These are the only pieces of ``cts.analysis.audit_encoder_kl`` that can be
exercised without an encoder checkpoint and a packed dataset: subtree-size
accumulation across a manually constructed parent_index/depth pair, plus
the log2 and depth-clamp binners.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cts.analysis.audit_encoder_kl import (
    _V1OrV2PretrainExamplePathDataset,
    _build_audit_dataset,
    _compute_subtree_sizes,
    _depth_bin,
    _depth_bin_labels,
    _load_resume_checkpoint,
    _load_v1_or_v2_pretrain_example,
    _resume_path,
    _save_resume_checkpoint,
    _size_bin,
    _size_bin_labels,
)
from cts.core.kl_buckets import (
    num_children_bin,
    num_children_bin_labels,
    num_children_per_node,
)
from cts.core.tensorizer import edge_wdl_target_tensor, tensorize_tree
from cts.core.schema import tree_encoder_feature_schema
from cts.core.tree import ExpansionChild, SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    PretrainExample,
    RawPretrainExampleRecord,
)


class SubtreeSizeTests(unittest.TestCase):
    def test_chain(self) -> None:
        # Linear chain: 0 -> 1 -> 2 -> 3. Subtree sizes are 4, 3, 2, 1.
        parent_index = torch.tensor([-1, 0, 1, 2], dtype=torch.long)
        depth = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        sizes = _compute_subtree_sizes(parent_index, depth)
        self.assertEqual(sizes.tolist(), [4, 3, 2, 1])

    def test_branching_tree(self) -> None:
        # Tree:
        #   0
        #   ├── 1
        #   │   ├── 3
        #   │   └── 4
        #   └── 2
        # Subtree sizes: 0 -> 5, 1 -> 3, 2 -> 1, 3 -> 1, 4 -> 1.
        parent_index = torch.tensor([-1, 0, 0, 1, 1], dtype=torch.long)
        depth = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)
        sizes = _compute_subtree_sizes(parent_index, depth)
        self.assertEqual(sizes.tolist(), [5, 3, 1, 1, 1])

    def test_batched_forest_two_disjoint_trees(self) -> None:
        # Two trees concatenated: indices 0–2 are tree A, 3–4 are tree B.
        # parent_index uses -1 for both roots; subtree sizes should not bleed
        # across the disjoint partitions because B's parent_index pointers
        # stay inside its own block.
        parent_index = torch.tensor([-1, 0, 1, -1, 3], dtype=torch.long)
        depth = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        sizes = _compute_subtree_sizes(parent_index, depth)
        self.assertEqual(sizes.tolist(), [3, 2, 1, 2, 1])

    def test_single_node(self) -> None:
        parent_index = torch.tensor([-1], dtype=torch.long)
        depth = torch.tensor([0], dtype=torch.long)
        sizes = _compute_subtree_sizes(parent_index, depth)
        self.assertEqual(sizes.tolist(), [1])


class BinningTests(unittest.TestCase):
    def test_size_bin_log2_buckets(self) -> None:
        sizes = torch.tensor([1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 128, 500], dtype=torch.long)
        # max_bin=6 -> bins 0..6; sizes >= 64 fold into bin 6.
        bins = _size_bin(sizes, max_bin=6)
        expected = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6]
        self.assertEqual(bins.tolist(), expected)

    def test_size_bin_zero_clamps_to_leaf(self) -> None:
        # Defensive: subtree_size should never be < 1, but clamp guards it.
        sizes = torch.tensor([0, 1, 2], dtype=torch.long)
        bins = _size_bin(sizes, max_bin=4)
        self.assertEqual(bins.tolist(), [0, 0, 1])

    def test_depth_bin_clamps(self) -> None:
        depths = torch.tensor([0, 3, 11, 12, 15], dtype=torch.long)
        bins = _depth_bin(depths, max_bin=12)
        self.assertEqual(bins.tolist(), [0, 3, 11, 12, 12])

    def test_size_labels(self) -> None:
        labels = _size_bin_labels(max_bin=4)
        self.assertEqual(labels, ["1", "2–3", "4–7", "8–15", "≥16"])

    def test_depth_labels(self) -> None:
        labels = _depth_bin_labels(max_bin=3)
        self.assertEqual(labels, ["0", "1", "2", "≥3"])


class NumChildrenBucketingTests(unittest.TestCase):
    def test_num_children_per_node_counts_edges_correctly(self) -> None:
        # Two parents (nodes 0 and 1): node 0 has 3 children, node 1 has 1.
        edge_parent = torch.tensor([0, 0, 0, 1], dtype=torch.long)
        counts = num_children_per_node(edge_parent, num_nodes=5)
        self.assertEqual(counts.tolist(), [3, 1, 0, 0, 0])

    def test_num_children_bin_clamps(self) -> None:
        counts = torch.tensor([0, 1, 5, 39, 40, 100], dtype=torch.long)
        bins = num_children_bin(counts, max_bin=40)
        self.assertEqual(bins.tolist(), [0, 1, 5, 39, 40, 40])

    def test_num_children_labels(self) -> None:
        labels = num_children_bin_labels(max_bin=3)
        self.assertEqual(labels, ["0", "1", "2", "≥3"])


class V1OrV2LoaderTests(unittest.TestCase):
    """The v2 bump tightened the on-disk children_index ordering; the audit's
    in-memory tensorize path applies canonical UCI sorting regardless, so v1
    raw .pt records load fine via the audit's retag-aware loader.
    """

    @staticmethod
    def _build_example_with_two_children() -> PretrainExample:
        tree = SearchTree(
            root_fen="root",
            root_scalar_features={
                "value": 0.0, "wdl_win": 0.4, "wdl_draw": 0.4,
                "wdl_loss": 0.2, "wdl_var": 0.01,
            },
        )
        # Children added in UCI-sorted order (d2d4 < e2e4), matching the convention
        # producers like lc0 follow. ``from_example`` has a separate pre-existing
        # bug where it sorts ``children_index`` but not ``edge_wdl_targets``;
        # production records stay internally consistent because their insertion
        # order already matches the canonical UCI sort.
        tree.add_children(
            tree.root_id,
            [
                ExpansionChild(
                    move_uci="d2d4",
                    fen="child-1",
                    scalar_features={
                        "value": 0.1, "wdl_win": 0.4, "wdl_draw": 0.4,
                        "wdl_loss": 0.2, "wdl_var": 0.03,
                    },
                ),
                ExpansionChild(
                    move_uci="e2e4",
                    fen="child-2",
                    scalar_features={
                        "value": 0.3, "wdl_win": 0.6, "wdl_draw": 0.3,
                        "wdl_loss": 0.1, "wdl_var": 0.02,
                    },
                ),
            ],
        )
        # node_id 1 ↔ d2d4, node_id 2 ↔ e2e4.
        return PretrainExample(
            tree=tree,
            node_target_values=[0.0, 0.1, 0.3],
            edge_wdl_targets={
                (0, 1): (0.4, 0.4, 0.2),
                (0, 2): (0.6, 0.3, 0.1),
            },
        )

    def test_v1_retag_round_trips_to_pretrain_example(self) -> None:
        example = self._build_example_with_two_children()
        # Build a v2 payload via the canonical converter, then retag it to v1
        # to simulate a legacy raw .pt file that the strict-v2 from_payload check
        # would otherwise reject.
        record = RawPretrainExampleRecord.from_example(example)
        v1_payload = record.to_payload()
        v1_payload["format"] = "cts_raw_pretrain_example_v1"

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as handle:
            torch.save(v1_payload, handle.name)
            loaded = _load_v1_or_v2_pretrain_example(handle.name)

        self.assertEqual(loaded.tree.num_nodes(), example.tree.num_nodes())
        self.assertEqual(loaded.tree.num_edges(), example.tree.num_edges())
        # node_target_values goes through float32 round-trip, so compare with
        # tolerance.
        self.assertEqual(len(loaded.node_target_values), len(example.node_target_values))
        for got, want in zip(loaded.node_target_values, example.node_target_values):
            self.assertAlmostEqual(got, want, places=5)
        # edge_wdl_targets is a (parent_id, child_id)-keyed dict, so it's order-independent.
        self.assertEqual(set(loaded.edge_wdl_targets.keys()), set(example.edge_wdl_targets.keys()))
        for key in example.edge_wdl_targets:
            for got, want in zip(loaded.edge_wdl_targets[key], example.edge_wdl_targets[key]):
                self.assertAlmostEqual(got, want, places=5)

    def test_v1_retag_tensorizes_to_canonical_uci_slot_order(self) -> None:
        """The audit relies on the in-memory tensorizer's UCI sort to neutralize
        any v1 ``children_index`` order; verify by tensorizing the round-tripped
        example and checking edge_slot is the canonical (UCI-sorted) sequence
        (d2d4 < e2e4 lexicographically, so slot 0 -> child of d2d4)."""
        example = self._build_example_with_two_children()
        record = RawPretrainExampleRecord.from_example(example)
        v1_payload = record.to_payload()
        v1_payload["format"] = "cts_raw_pretrain_example_v1"

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as handle:
            torch.save(v1_payload, handle.name)
            loaded = _load_v1_or_v2_pretrain_example(handle.name)

        schema = tree_encoder_feature_schema()
        tensorized = tensorize_tree(loaded.tree, schema=schema)
        # Two edges from root; slot 0 corresponds to d2d4 (UCI-lex smaller).
        self.assertEqual(tensorized.edge_slot.tolist(), [0, 1])
        edge_targets = edge_wdl_target_tensor(
            loaded.tree, loaded.edge_wdl_targets, schema=schema,
        ).tolist()
        # Row 0 is d2d4's target (win=0.4); row 1 is e2e4's (win=0.6).
        self.assertAlmostEqual(edge_targets[0][0], 0.4)
        self.assertAlmostEqual(edge_targets[1][0], 0.6)

    def test_v2_payload_loads_unchanged(self) -> None:
        example = self._build_example_with_two_children()
        record = RawPretrainExampleRecord.from_example(example)
        v2_payload = record.to_payload()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as handle:
            torch.save(v2_payload, handle.name)
            loaded = _load_v1_or_v2_pretrain_example(handle.name)

        self.assertEqual(loaded.tree.num_nodes(), example.tree.num_nodes())

    def test_unknown_format_tag_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as handle:
            torch.save({"format": "something_else"}, handle.name)
            with self.assertRaises(ValueError):
                _load_v1_or_v2_pretrain_example(handle.name)

    def test_text_manifest_dispatches_to_v1_dataset(self) -> None:
        example = self._build_example_with_two_children()
        record = RawPretrainExampleRecord.from_example(example)
        v1_payload = record.to_payload()
        v1_payload["format"] = "cts_raw_pretrain_example_v1"

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = f"{tmpdir}/example.pt"
            torch.save(v1_payload, pt_path)
            manifest_path = f"{tmpdir}/manifest.txt"
            with open(manifest_path, "w") as handle:
                handle.write(pt_path + "\n")
            dataset = _build_audit_dataset(manifest_path)
            self.assertIsInstance(dataset, _V1OrV2PretrainExamplePathDataset)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].tree.num_nodes(), example.tree.num_nodes())


class ResumeCheckpointTests(unittest.TestCase):
    """Round-trip of the resume-checkpoint save/load + identity guards."""

    @staticmethod
    def _common_save_kwargs(path):
        return dict(
            path=path,
            sum_kl=torch.tensor([[0.1, 0.2], [0.3, 0.0]], dtype=torch.float64),
            count=torch.tensor([[3, 5], [7, 0]], dtype=torch.long),
            sum_kl_nc=torch.tensor([[0.05, 0.15], [0.25, 0.0], [0.0, 0.35]], dtype=torch.float64),
            count_nc=torch.tensor([[1, 2], [3, 0], [0, 4]], dtype=torch.long),
            overall_sum_kl=torch.tensor(0.6, dtype=torch.float64),
            overall_count=15,
            next_batch_index=42,
            manifest_path="/data/manifest.txt",
            encoder_checkpoint="/ckpt/encoder.pt",
            decoder_checkpoint="/ckpt/decoder.pt",
            num_size_bins=2,
            num_depth_bins=2,
            num_nc_bins=3,
            max_depth_bin=12,
            subtree_size_log_max=7,
            num_children_max_bin=2,
        )

    def test_resume_path_appends_suffix(self) -> None:
        self.assertEqual(
            str(_resume_path("/tmp/audit_train.json")),
            "/tmp/audit_train.json.resume.pt",
        )

    def test_save_then_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json.resume.pt"
            kwargs = self._common_save_kwargs(path)
            _save_resume_checkpoint(**kwargs)
            self.assertTrue(path.is_file())
            state = _load_resume_checkpoint(
                path,
                device=torch.device("cpu"),
                expected_manifest=kwargs["manifest_path"],
                expected_encoder=kwargs["encoder_checkpoint"],
                expected_decoder=kwargs["decoder_checkpoint"],
                expected_num_size_bins=kwargs["num_size_bins"],
                expected_num_depth_bins=kwargs["num_depth_bins"],
                expected_num_nc_bins=kwargs["num_nc_bins"],
            )
            self.assertIsNotNone(state)
            self.assertTrue(torch.equal(state["sum_kl"], kwargs["sum_kl"]))
            self.assertTrue(torch.equal(state["count"], kwargs["count"]))
            self.assertTrue(torch.equal(state["sum_kl_nc"], kwargs["sum_kl_nc"]))
            self.assertTrue(torch.equal(state["count_nc"], kwargs["count_nc"]))
            self.assertEqual(state["overall_count"], kwargs["overall_count"])
            self.assertEqual(state["next_batch_index"], kwargs["next_batch_index"])

    def test_missing_checkpoint_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nope.pt"
            state = _load_resume_checkpoint(
                path,
                device=torch.device("cpu"),
                expected_manifest="/x",
                expected_encoder="/y",
                expected_decoder="/z",
                expected_num_size_bins=1,
                expected_num_depth_bins=1,
                expected_num_nc_bins=1,
            )
            self.assertIsNone(state)

    def test_mismatched_identity_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json.resume.pt"
            kwargs = self._common_save_kwargs(path)
            _save_resume_checkpoint(**kwargs)
            with self.assertRaises(ValueError):
                _load_resume_checkpoint(
                    path,
                    device=torch.device("cpu"),
                    expected_manifest="/different/manifest.txt",
                    expected_encoder=kwargs["encoder_checkpoint"],
                    expected_decoder=kwargs["decoder_checkpoint"],
                    expected_num_size_bins=kwargs["num_size_bins"],
                    expected_num_depth_bins=kwargs["num_depth_bins"],
                    expected_num_nc_bins=kwargs["num_nc_bins"],
                )

    def test_mismatched_bin_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json.resume.pt"
            kwargs = self._common_save_kwargs(path)
            _save_resume_checkpoint(**kwargs)
            with self.assertRaises(ValueError):
                _load_resume_checkpoint(
                    path,
                    device=torch.device("cpu"),
                    expected_manifest=kwargs["manifest_path"],
                    expected_encoder=kwargs["encoder_checkpoint"],
                    expected_decoder=kwargs["decoder_checkpoint"],
                    expected_num_size_bins=99,  # mismatch
                    expected_num_depth_bins=kwargs["num_depth_bins"],
                    expected_num_nc_bins=kwargs["num_nc_bins"],
                )

    def test_mismatched_nc_bin_shape_raises(self) -> None:
        """num_children grid shape mismatch must also raise — the bug this fix addresses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json.resume.pt"
            kwargs = self._common_save_kwargs(path)
            _save_resume_checkpoint(**kwargs)
            with self.assertRaises(ValueError):
                _load_resume_checkpoint(
                    path,
                    device=torch.device("cpu"),
                    expected_manifest=kwargs["manifest_path"],
                    expected_encoder=kwargs["encoder_checkpoint"],
                    expected_decoder=kwargs["decoder_checkpoint"],
                    expected_num_size_bins=kwargs["num_size_bins"],
                    expected_num_depth_bins=kwargs["num_depth_bins"],
                    expected_num_nc_bins=99,  # mismatch
                )

    def test_v1_resume_payload_rejected(self) -> None:
        """A v1-format resume file (no num_children grid) must not load against v2 code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json.resume.pt"
            # Hand-craft a v1 payload (the old format, no num_children fields).
            v1_payload = {
                "format": "cts_audit_encoder_kl_resume_v1",
                "manifest_path": "/data/manifest.txt",
                "encoder_checkpoint": "/ckpt/encoder.pt",
                "decoder_checkpoint": "/ckpt/decoder.pt",
                "max_depth_bin": 12,
                "subtree_size_log_max": 7,
                "num_size_bins": 2,
                "num_depth_bins": 2,
                "sum_kl": torch.zeros((2, 2), dtype=torch.float64),
                "count": torch.zeros((2, 2), dtype=torch.long),
                "overall_sum_kl": torch.zeros((), dtype=torch.float64),
                "overall_count": 0,
                "next_batch_index": 0,
            }
            torch.save(v1_payload, path)
            with self.assertRaises(ValueError):
                _load_resume_checkpoint(
                    path,
                    device=torch.device("cpu"),
                    expected_manifest="/data/manifest.txt",
                    expected_encoder="/ckpt/encoder.pt",
                    expected_decoder="/ckpt/decoder.pt",
                    expected_num_size_bins=2,
                    expected_num_depth_bins=2,
                    expected_num_nc_bins=3,
                )

    def test_atomic_save_does_not_leak_tmp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.json.resume.pt"
            _save_resume_checkpoint(**self._common_save_kwargs(path))
            self.assertTrue(path.is_file())
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            self.assertFalse(tmp_path.exists())


if __name__ == "__main__":
    unittest.main()
