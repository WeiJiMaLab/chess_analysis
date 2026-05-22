"""Unit test for the ``rewrite_compact`` v1-dict load fix.

The shipped ``_load_record_from_source`` originally dispatched v1 dicts to
``RawPretrainExampleRecord.from_payload``, which is strict-v2 and rejected
v1. The fix retag v1 dicts to v2 before delegating, since the v2 disk-format
bump only enforced canonical UCI-sorted ``children_index`` ordering — a
property production lc0-generated v1 records already satisfy.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cts.core.tree import ExpansionChild, SearchTree
from cts.data.preprocess_gnn.rewrite_compact import _load_record_from_source
from cts.data.preprocess_gnn.teacher_targets import (
    PretrainExample,
    RAW_PRETRAIN_FORMAT,
    RawPretrainExampleRecord,
)


def _build_example_with_two_children() -> PretrainExample:
    """Two UCI-ordered children so ``from_example`` produces a self-consistent v2 payload."""
    tree = SearchTree(
        root_fen="root",
        root_scalar_features={
            "value": 0.0, "wdl_win": 0.4, "wdl_draw": 0.4,
            "wdl_loss": 0.2, "wdl_var": 0.01,
        },
    )
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
    return PretrainExample(
        tree=tree,
        node_target_values=[0.0, 0.1, 0.3],
        edge_wdl_targets={
            (0, 1): (0.4, 0.4, 0.2),
            (0, 2): (0.6, 0.3, 0.1),
        },
    )


class RewriteCompactLegacyLoadTests(unittest.TestCase):
    def test_v1_dict_payload_is_accepted_and_loaded(self) -> None:
        """The fix: a dict tagged ``cts_raw_pretrain_example_v1`` loads successfully."""
        example = _build_example_with_two_children()
        # Build a v2 payload via the canonical converter, then retag as v1
        # to simulate a legacy raw .pt file on disk.
        record = RawPretrainExampleRecord.from_example(example)
        v1_payload = record.to_payload()
        v1_payload["format"] = "cts_raw_pretrain_example_v1"

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "example.pt"
            torch.save(v1_payload, source)
            loaded = _load_record_from_source(source)

        # Round-tripped through retag and loaded successfully into a
        # RawPretrainExampleRecord. Shape checks are enough — full content
        # round-trip is covered by the audit module's tests.
        self.assertIsInstance(loaded, RawPretrainExampleRecord)
        self.assertEqual(int(loaded.parent_index.shape[0]), 3)
        self.assertEqual(int(loaded.edge_wdl_targets.shape[0]), 2)

    def test_v2_dict_payload_still_loads(self) -> None:
        example = _build_example_with_two_children()
        record = RawPretrainExampleRecord.from_example(example)
        v2_payload = record.to_payload()
        self.assertEqual(v2_payload["format"], RAW_PRETRAIN_FORMAT)

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "example.pt"
            torch.save(v2_payload, source)
            loaded = _load_record_from_source(source)

        self.assertIsInstance(loaded, RawPretrainExampleRecord)
        self.assertEqual(int(loaded.parent_index.shape[0]), 3)

    def test_unknown_format_tag_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "example.pt"
            torch.save({"format": "something_else"}, source)
            with self.assertRaises(ValueError):
                _load_record_from_source(source)


if __name__ == "__main__":
    unittest.main()
