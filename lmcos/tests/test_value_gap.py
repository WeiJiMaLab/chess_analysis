"""Unit tests for per-node child value-gap (centipawn) topology targets."""

from __future__ import annotations

import math
import tempfile
import unittest

import torch

from cts.core.tree import ExpansionChild, SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    RAW_PRETRAIN_FORMAT,
    EdgeStats,
    PretrainExample,
    RawPretrainExampleRecord,
    TeacherSearchConfig,
    VALUE_SCALAR_TO_CENTIPAWNS,
    build_pretrain_example,
    child_q_from_parent_perspective,
    node_value_gap_centipawns,
    node_value_gap_features,
    save_pretrain_example,
    scaled_teacher_topology_matrix,
    scalar_value_to_centipawns,
)
from cts.core.providers.base import TreeExpansionProvider


def _wdl_features(win: float, draw: float, loss: float, prior: float = 1.0) -> dict[str, float]:
    total = win + draw + loss
    p_win = win / total
    p_draw = draw / total
    p_loss = loss / total
    value = p_win - p_loss
    return {
        "value": value,
        "prior": prior,
        "wdl_win": p_win,
        "wdl_draw": p_draw,
        "wdl_loss": p_loss,
        "wdl_var": (p_win + p_loss) - value * value,
    }


def _edge_q(parent_id: int, child_id: int, q: float, *, visits: int = 8) -> tuple[tuple[int, int], EdgeStats]:
    return (parent_id, child_id), EdgeStats(visit_count=visits, total_value=q * visits)


def _two_child_tree() -> SearchTree:
    tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
    tree.add_children(
        0,
        [
            ExpansionChild("a", "a", _wdl_features(0.6, 0.2, 0.2, prior=0.7)),
            ExpansionChild("b", "b", _wdl_features(0.2, 0.4, 0.4, prior=0.3)),
        ],
    )
    return tree


class ValueGapTests(unittest.TestCase):
    def test_scalar_value_to_centipawns(self) -> None:
        self.assertAlmostEqual(scalar_value_to_centipawns(0.5), 50.0)
        self.assertAlmostEqual(scalar_value_to_centipawns(-0.25), -25.0)

    def test_equal_children_yield_zero_gap(self) -> None:
        tree = _two_child_tree()
        edge_stats = dict(
            [
                _edge_q(0, 1, 0.4),
                _edge_q(0, 2, 0.4),
            ]
        )
        gap = node_value_gap_centipawns(tree, 0, edge_stats)
        self.assertEqual(gap, 0.0)

    def test_leaf_gap_is_undefined(self) -> None:
        tree = _two_child_tree()
        edge_stats: dict[tuple[int, int], EdgeStats] = {}
        self.assertTrue(math.isnan(node_value_gap_centipawns(tree, 1, edge_stats)))
        self.assertTrue(math.isnan(node_value_gap_centipawns(tree, 2, edge_stats)))

    def test_single_child_internal_node_is_undefined(self) -> None:
        tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree.add_children(0, [ExpansionChild("only", "only", _wdl_features(0.55, 0.2, 0.25))])
        self.assertTrue(math.isnan(node_value_gap_centipawns(tree, 0, {})))

    def test_checkmate_like_gap_is_large(self) -> None:
        tree = _two_child_tree()
        edge_stats = dict(
            [
                _edge_q(0, 1, 1.0),
                _edge_q(0, 2, -0.95),
            ]
        )
        gap = node_value_gap_centipawns(tree, 0, edge_stats)
        expected = scalar_value_to_centipawns(1.0) - scalar_value_to_centipawns(-0.95)
        self.assertAlmostEqual(gap, expected)
        self.assertGreater(gap, 150.0)

    def test_gap_uses_parent_perspective_via_negamax_static_fallback(self) -> None:
        """Static child values are from the child side-to-move; parent Q flips sign."""
        tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree.add_children(
            0,
            [
                ExpansionChild("a", "a", {"value": 0.8, "prior": 1.0, "wdl_win": 0.9, "wdl_draw": 0.0, "wdl_loss": 0.1, "wdl_var": 0.1}),
                ExpansionChild("b", "b", {"value": 0.7, "prior": 1.0, "wdl_win": 0.85, "wdl_draw": 0.0, "wdl_loss": 0.15, "wdl_var": 0.1}),
            ],
        )
        # Correct parent Q: -0.8 and -0.7 → gap = 10 cp.
        gap = node_value_gap_centipawns(tree, 0, {})
        self.assertAlmostEqual(gap, 10.0)
        # Wrong-side (child POV) gap would be +10 cp with reversed ordering — not what we want.
        wrong_gap = scalar_value_to_centipawns(0.8) - scalar_value_to_centipawns(0.7)
        self.assertAlmostEqual(wrong_gap, 10.0)
        # Demonstrate inversion matters when child ordering differs from parent Q ordering:
        tree2 = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree2.add_children(
            0,
            [
                ExpansionChild("a", "a", {"value": -0.2, "prior": 1.0, "wdl_win": 0.4, "wdl_draw": 0.0, "wdl_loss": 0.6, "wdl_var": 0.1}),
                ExpansionChild("b", "b", {"value": -0.5, "prior": 1.0, "wdl_win": 0.25, "wdl_draw": 0.0, "wdl_loss": 0.75, "wdl_var": 0.1}),
            ],
        )
        gap2 = node_value_gap_centipawns(tree2, 0, {})
        self.assertAlmostEqual(gap2, 30.0)

    def test_visit_weighted_q_overrides_static_fallback(self) -> None:
        tree = _two_child_tree()
        edge_stats = dict(
            [
                _edge_q(0, 1, 0.9),
                _edge_q(0, 2, -0.2),
            ]
        )
        gap = node_value_gap_centipawns(tree, 0, edge_stats)
        self.assertAlmostEqual(gap, scalar_value_to_centipawns(0.9) - scalar_value_to_centipawns(-0.2))

    def test_gap_uses_second_best_not_third(self) -> None:
        tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree.add_children(
            0,
            [
                ExpansionChild("a", "a", _wdl_features(0.6, 0.2, 0.2)),
                ExpansionChild("b", "b", _wdl_features(0.55, 0.2, 0.25)),
                ExpansionChild("c", "c", _wdl_features(0.2, 0.4, 0.4)),
            ],
        )
        edge_stats = dict(
            [
                _edge_q(0, 1, 0.8),
                _edge_q(0, 2, 0.5),
                _edge_q(0, 3, -0.9),
            ]
        )
        gap = node_value_gap_centipawns(tree, 0, edge_stats)
        self.assertAlmostEqual(gap, scalar_value_to_centipawns(0.8) - scalar_value_to_centipawns(0.5))

    def test_deeper_node_respects_side_to_move_perspective(self) -> None:
        tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree.add_children(0, [ExpansionChild("mid", "mid", _wdl_features(0.45, 0.2, 0.35))])
        tree.add_children(
            1,
            [
                ExpansionChild("d1", "d1", _wdl_features(0.9, 0.05, 0.05)),
                ExpansionChild("d2", "d2", _wdl_features(0.85, 0.05, 0.1)),
            ],
        )
        edge_stats = dict(
            [
                _edge_q(1, 2, 0.6),
                _edge_q(1, 3, 0.55),
            ]
        )
        gap = node_value_gap_centipawns(tree, 1, edge_stats)
        self.assertAlmostEqual(gap, 5.0)
        self.assertAlmostEqual(
            child_q_from_parent_perspective(tree, 1, 2, edge_stats),
            0.6,
        )

    def test_node_value_gap_features_align_with_tree_nodes(self) -> None:
        tree = _two_child_tree()
        edge_stats = dict([_edge_q(0, 1, 0.5), _edge_q(0, 2, 0.1)])
        gaps = node_value_gap_features(tree, edge_stats)
        self.assertEqual(len(gaps), tree.num_nodes())
        self.assertAlmostEqual(gaps[0], 40.0)
        self.assertTrue(all(math.isnan(g) for g in gaps[1:]))

    def test_build_pretrain_example_populates_value_gap(self) -> None:
        class _Provider(TreeExpansionProvider):
            def root_features(self, fen: str) -> dict[str, float]:
                return _wdl_features(0.35, 0.3, 0.35)

            def expand_node(self, fen: str, depth: int, max_children: int | None = None):
                if fen == "root":
                    return [
                        ExpansionChild("a", "a", _wdl_features(0.6, 0.2, 0.2, prior=0.7)),
                        ExpansionChild("b", "b", _wdl_features(0.2, 0.4, 0.4, prior=0.3)),
                    ]
                if fen == "a":
                    return [ExpansionChild("a1", "a1", _wdl_features(0.55, 0.2, 0.25))]
                return []

            def provider_metadata(self) -> dict[str, str]:
                return {"provider": "value-gap-test"}

        config = TeacherSearchConfig(
            max_depth=2,
            search_budget=4,
            c_puct=1.0,
            prior_feature="prior",
            value_feature="value",
            target_normalization_version="v1",
            search_config_id="value-gap-test",
        )
        example = build_pretrain_example("root", _Provider(), config)
        self.assertEqual(len(example.value_gap), example.tree.num_nodes())
        self.assertFalse(all(math.isnan(g) for g in example.value_gap))
        self.assertTrue(any(math.isnan(g) for g in example.value_gap))

    def test_v4_record_round_trip_preserves_value_gap(self) -> None:
        tree = _two_child_tree()
        edge_stats = dict([_edge_q(0, 1, 0.7), _edge_q(0, 2, 0.2)])
        gaps = node_value_gap_features(tree, edge_stats)
        example = PretrainExample(
            tree=tree,
            node_target_values=[0.0] * tree.num_nodes(),
            value_gap=gaps,
        )
        record = RawPretrainExampleRecord.from_example(example)
        self.assertEqual(record.to_payload()["format"], RAW_PRETRAIN_FORMAT)
        matrix = scaled_teacher_topology_matrix(record)
        self.assertEqual(tuple(matrix.shape), (tree.num_nodes(), 2))
        self.assertAlmostEqual(float(matrix[0, 0].item()), 50.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/example.pt"
            save_pretrain_example(path, example)
            loaded = RawPretrainExampleRecord.load(path).to_pretrain_example()
        self.assertEqual(len(loaded.value_gap), tree.num_nodes())
        self.assertAlmostEqual(loaded.value_gap[0], 50.0)
        self.assertTrue(all(math.isnan(g) for g in loaded.value_gap[1:]))

    def test_two_siblings_share_best_value_yields_zero_gap(self) -> None:
        tree = _two_child_tree()
        edge_stats = dict(
            [
                _edge_q(0, 1, 0.6),
                _edge_q(0, 2, 0.6),
            ]
        )
        gap = node_value_gap_centipawns(tree, 0, edge_stats)
        self.assertEqual(gap, 0.0)

    def test_two_siblings_share_suboptimal_value_normal_gap(self) -> None:
        # Parent has 3 children: A (best, 0.8), B (second-best, 0.3), C (third-best, 0.3)
        tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree.add_children(
            0,
            [
                ExpansionChild("a", "a", _wdl_features(0.8, 0.1, 0.1)),
                ExpansionChild("b", "b", _wdl_features(0.3, 0.35, 0.35)),
                ExpansionChild("c", "c", _wdl_features(0.3, 0.35, 0.35)),
            ],
        )
        edge_stats = dict(
            [
                _edge_q(0, 1, 0.8),
                _edge_q(0, 2, 0.3),
                _edge_q(0, 3, 0.3),
            ]
        )
        gap = node_value_gap_centipawns(tree, 0, edge_stats)
        self.assertAlmostEqual(gap, 50.0) # 0.8 - 0.3 = 0.5 (which is 50 centipawns)

    def test_single_child_gap_is_nan(self) -> None:
        tree = SearchTree(root_fen="root", root_scalar_features=_wdl_features(0.35, 0.3, 0.35))
        tree.add_children(0, [ExpansionChild("only", "only", _wdl_features(0.5, 0.25, 0.25))])
        self.assertTrue(math.isnan(node_value_gap_centipawns(tree, 0, {})))


if __name__ == "__main__":
    unittest.main()
