"""Unit tests for the ``controller_inputs`` projection in MetaController.

Covers:
- The validator rejects empty, unknown, and duplicate input lists.
- ``_head_input_dim`` matches the configured input dimensions.
- ``_select_features`` slices the canonical ``[z_t, N_t, T_t]`` layout
  correctly for each subset.
- End-to-end predict_from_features on a small encoder produces the right
  output shape for each subset.
"""

from __future__ import annotations

import unittest

import torch

from cts.models.mc import (
    CONTROLLER_INPUT_NAMES,
    MetaController,
    validate_controller_inputs,
)


D_EMBED = 4


def _make_controller(inputs):
    """Tiny encoder + controller with ``d_embed=4`` so feature slicing is easy to assert."""
    return MetaController(
        k=1,
        node_feat=5,
        device="cpu",
        node_embed_hidden=8,
        d_embed=D_EMBED,
        d_message=4,
        n_heads=1,
        d_att=2,
        hidden_dim=8,
        hidden_layers=1,
        controller_inputs=inputs,
    )


class ValidateControllerInputsTests(unittest.TestCase):
    def test_canonical_default_accepted(self) -> None:
        self.assertEqual(
            validate_controller_inputs(["z_t", "N_t", "T_t"]),
            ("z_t", "N_t", "T_t"),
        )

    def test_subset_accepted(self) -> None:
        self.assertEqual(validate_controller_inputs(["z_t", "T_t"]), ("z_t", "T_t"))
        self.assertEqual(validate_controller_inputs(["z_t"]), ("z_t",))

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_controller_inputs([])

    def test_unknown_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_controller_inputs(["z_t", "garbage"])

    def test_duplicates_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_controller_inputs(["z_t", "z_t"])
        with self.assertRaises(ValueError):
            validate_controller_inputs(["z_t", "N_t", "N_t"])


class HeadInputDimTests(unittest.TestCase):
    def test_all_three(self) -> None:
        self.assertEqual(MetaController._head_input_dim(128, ("z_t", "N_t", "T_t")), 130)

    def test_zt_plus_tt(self) -> None:
        self.assertEqual(MetaController._head_input_dim(128, ("z_t", "T_t")), 129)

    def test_zt_only(self) -> None:
        self.assertEqual(MetaController._head_input_dim(128, ("z_t",)), 128)

    def test_no_z_scalars_only(self) -> None:
        self.assertEqual(MetaController._head_input_dim(128, ("N_t", "T_t")), 2)
        self.assertEqual(MetaController._head_input_dim(128, ("T_t",)), 1)


class SelectFeaturesTests(unittest.TestCase):
    """Hand-crafted feature tensor with distinct sentinel values per column,
    so the slicing's correctness is decidable cell-by-cell.
    """

    @staticmethod
    def _sentinel_features() -> torch.Tensor:
        # Batch of 2; first row: z_t = [10, 11, 12, 13], N_t = 99, T_t = 77.
        # Second row: z_t = [20, 21, 22, 23], N_t = 199, T_t = 177.
        return torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0, 99.0, 77.0],
                [20.0, 21.0, 22.0, 23.0, 199.0, 177.0],
            ]
        )

    def test_all_three_passes_through(self) -> None:
        m = _make_controller(["z_t", "N_t", "T_t"])
        out = m._select_features(self._sentinel_features())
        self.assertEqual(tuple(out.shape), (2, 6))
        self.assertTrue(torch.equal(out, self._sentinel_features()))

    def test_zt_plus_tt_drops_nt(self) -> None:
        m = _make_controller(["z_t", "T_t"])
        out = m._select_features(self._sentinel_features())
        self.assertEqual(tuple(out.shape), (2, 5))
        # z_t cols then T_t col
        expected = torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0, 77.0],
                [20.0, 21.0, 22.0, 23.0, 177.0],
            ]
        )
        self.assertTrue(torch.equal(out, expected))

    def test_zt_only_drops_both_scalars(self) -> None:
        m = _make_controller(["z_t"])
        out = m._select_features(self._sentinel_features())
        self.assertEqual(tuple(out.shape), (2, 4))
        expected = torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ]
        )
        self.assertTrue(torch.equal(out, expected))

    def test_canonical_order_independent_of_config_order(self) -> None:
        """``controller_inputs=['T_t', 'z_t']`` should produce the same output as
        ``['z_t', 'T_t']`` — selection follows canonical order, not the config order."""
        m_canonical = _make_controller(["z_t", "T_t"])
        m_reversed = _make_controller(["T_t", "z_t"])
        features = self._sentinel_features()
        self.assertTrue(
            torch.equal(m_canonical._select_features(features), m_reversed._select_features(features))
        )


class PredictFromFeaturesShapeTests(unittest.TestCase):
    def test_shape_matches_each_input_subset(self) -> None:
        features = torch.randn(3, D_EMBED + 2)
        for inputs in (
            ["z_t", "N_t", "T_t"],
            ["z_t", "T_t"],
            ["z_t", "N_t"],
            ["z_t"],
            ["N_t", "T_t"],
            ["T_t"],
        ):
            with self.subTest(inputs=inputs):
                m = _make_controller(inputs)
                advantage, sign = m.predict_from_features(features)
                self.assertEqual(tuple(advantage.shape), (3,))
                self.assertEqual(tuple(sign.shape), (3,))

    def test_default_constructor_uses_all_three(self) -> None:
        # No controller_inputs passed → CONTROLLER_INPUT_NAMES default.
        m = MetaController(
            k=1, node_feat=5, device="cpu",
            node_embed_hidden=8, d_embed=D_EMBED, d_message=4,
            n_heads=1, d_att=2, hidden_dim=8, hidden_layers=1,
        )
        self.assertEqual(m.controller_inputs, CONTROLLER_INPUT_NAMES)


if __name__ == "__main__":
    unittest.main()
