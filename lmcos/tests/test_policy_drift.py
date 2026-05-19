"""Unit tests for GNN tree search policy drift and KL divergence metrics."""

from __future__ import annotations

import math
import unittest

from cts.data.preprocess_gnn.policy_drift import (
    compute_policy_drift,
    kl_divergence,
    softmax_policy,
)


class PolicyDriftTests(unittest.TestCase):
    def test_softmax_policy_tied_visits(self) -> None:
        """Tied child visits must yield uniform probability distribution."""
        probs = softmax_policy([5, 5])
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)

        probs_triple = softmax_policy([20, 20, 20])
        self.assertEqual(len(probs_triple), 3)
        self.assertAlmostEqual(probs_triple[0], 1.0 / 3.0)
        self.assertAlmostEqual(probs_triple[1], 1.0 / 3.0)
        self.assertAlmostEqual(probs_triple[2], 1.0 / 3.0)

    def test_softmax_policy_temperature_scale(self) -> None:
        """High temperature should flatten the policy, low temperature should sharpen it."""
        visits = [10, 0]
        # Low temp -> highly sharpened
        sharp = softmax_policy(visits, temperature=0.1)
        self.assertAlmostEqual(sharp[0], 1.0, places=5)
        self.assertAlmostEqual(sharp[1], 0.0, places=5)

        # High temp -> flattened
        flat = softmax_policy(visits, temperature=100.0)
        self.assertAlmostEqual(flat[0], 0.524979, places=5)
        self.assertAlmostEqual(flat[1], 0.475020, places=5)

    def test_kl_divergence_identical_distributions(self) -> None:
        """Identically distributed policies must have exactly 0.0 KL divergence."""
        p = [0.6, 0.4]
        q = [0.6, 0.4]
        self.assertAlmostEqual(kl_divergence(p, q), 0.0)

    def test_kl_divergence_non_zero_drift(self) -> None:
        """Drifting from an early policy to a late policy yields positive KL divergence."""
        # p_early is concentrated on the first option, p_late becomes more distributed
        p_early = softmax_policy([10, 2])
        p_late = softmax_policy([6, 6])
        kl = kl_divergence(p_early, p_late)
        self.assertGreater(kl, 0.0)
        
        # Test asymmetry: D_KL(P || Q) != D_KL(Q || P)
        kl_rev = kl_divergence(p_late, p_early)
        self.assertNotAlmostEqual(kl, kl_rev)

    def test_policy_drift_satisfies_n_min_threshold(self) -> None:
        """If total early visits are below n_min, drift measurement should return None."""
        early_visits = [1, 1]
        late_visits = [5, 3]
        
        # Case A: total early visits = 2, n_min = 5 -> None
        measurement = compute_policy_drift(node_id=42, early_visits=early_visits, late_visits=late_visits, n_min=5)
        self.assertIsNone(measurement)

        # Case B: total early visits = 2, n_min = 2 -> Measurement successful
        measurement2 = compute_policy_drift(node_id=42, early_visits=early_visits, late_visits=late_visits, n_min=2)
        self.assertIsNotNone(measurement2)
        assert measurement2 is not None
        self.assertEqual(measurement2.node_id, 42)
        self.assertEqual(measurement2.early_visits, [1, 1])
        self.assertEqual(measurement2.late_visits, [5, 3])
        self.assertGreater(measurement2.kl_divergence, 0.0)

    def test_single_child_or_leaf_is_ignored(self) -> None:
        """Leaf nodes or nodes with a single child have trivial/undefined policies and return None."""
        # Leaf node (0 children)
        self.assertIsNone(compute_policy_drift(node_id=1, early_visits=[], late_visits=[]))
        # Single child
        self.assertIsNone(compute_policy_drift(node_id=2, early_visits=[10], late_visits=[20]))

    def test_max_expansions_limit_boundary(self) -> None:
        """Confirm that a simulated trace up to the maximum 96 expansions is valid."""
        # Generate visits for 96 expansions.
        # k (early) is at t=10, k+n (late) is at t=96.
        # We model 3 children where early visits sum to 10, and late visits sum to 96.
        early_visits = [6, 3, 1]  # total = 10
        late_visits = [60, 26, 10]  # total = 96 (max_t)
        
        measurement = compute_policy_drift(node_id=0, early_visits=early_visits, late_visits=late_visits, n_min=5)
        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(sum(measurement.late_visits), 96)
        self.assertGreater(measurement.kl_divergence, 0.0)

    def test_record_serialization_round_trip_with_policy_drift(self) -> None:
        """Verify that policy_drift round-trips correctly through RawPretrainExampleRecord and target scaling."""
        from cts.data.preprocess_gnn.teacher_targets import PretrainExample, RawPretrainExampleRecord, scaled_teacher_node_matrix
        from cts.core.tree import SearchTree, ExpansionChild
        import torch

        # Create a simple tree
        tree = SearchTree(root_fen="root", root_scalar_features={"value": 0.0})
        tree.add_children(
            0,
            [
                ExpansionChild("e2e4", "e2e4", {"value": 0.0}),
                ExpansionChild("d2d4", "d2d4", {"value": 0.0}),
            ],
        )

        example = PretrainExample(
            tree=tree,
            node_target_values=[0.0] * tree.num_nodes(),
            value_gap=[10.0, float("nan"), float("nan")],
            policy_drift=[0.25, float("nan"), float("nan")],
        )

        record = RawPretrainExampleRecord.from_example(example)
        payload = record.to_payload()
        self.assertIn("policy_drift", payload)
        
        # Load from payload
        reconstructed = RawPretrainExampleRecord.from_payload(payload)
        self.assertTrue(torch.allclose(
            reconstructed.policy_drift, 
            torch.tensor([0.25, float("nan"), float("nan")], dtype=torch.float32), 
            equal_nan=True
        ))

        # Test to_pretrain_example
        rehydrated = reconstructed.to_pretrain_example()
        self.assertAlmostEqual(rehydrated.policy_drift[0], 0.25)
        self.assertTrue(math.isnan(rehydrated.policy_drift[1]))

        # Test scaled_teacher_node_matrix
        matrix = scaled_teacher_node_matrix(reconstructed)
        self.assertEqual(tuple(matrix.shape), (3, 1))
        self.assertAlmostEqual(float(matrix[0, 0].item()), 0.25)


if __name__ == "__main__":
    unittest.main()
