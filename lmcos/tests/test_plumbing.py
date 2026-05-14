import unittest

import torch

from cts.core.schema import NodeFeatureSchema
from cts.core.tensorizer import (
    collate_tensorized_observations,
    tensorize_forest,
    tensorize_tree,
    tensorize_tree_observation,
)
from cts.core.tree import ExpansionChild, SearchTree


def build_sample_tree() -> SearchTree:
    tree = SearchTree(
        root_fen="root-fen",
        root_scalar_features={"value": 0.0, "prior": 1.0},
        root_metadata={"name": "root"},
    )
    root_id = tree.root_id
    tree.add_children(
        root_id,
        [
            ExpansionChild(
                move_uci="e2e4",
                fen="child-1",
                scalar_features={"value": 0.5, "prior": 0.7},
            ),
            ExpansionChild(
                move_uci="d2d4",
                fen="child-2",
                scalar_features={"value": 0.8},
            ),
        ],
    )
    tree.add_children(
        1,
        [
            ExpansionChild(
                move_uci="e7e5",
                fen="grandchild-1",
                scalar_features={"value": -0.2, "prior": 0.3},
            ),
            ExpansionChild(
                move_uci="c7c5",
                fen="grandchild-2",
                scalar_features={"value": 0.1},
                is_terminal=True,
            ),
        ],
    )
    return tree


class SearchTreeTests(unittest.TestCase):
    def test_tree_construction_and_validation(self) -> None:
        tree = build_sample_tree()

        self.assertEqual(tree.root_id, 0)
        self.assertEqual(tree.num_nodes(), 5)
        self.assertEqual(tree.num_edges(), 4)
        self.assertEqual(tree.children(0), [1, 2])
        self.assertEqual(tree.children(1), [3, 4])
        self.assertEqual(tree.get_node(4).depth, 2)
        self.assertTrue(tree.get_node(0).is_expanded)
        self.assertTrue(tree.get_node(1).is_expanded)
        tree.validate()

    def test_reexpanding_a_node_raises(self) -> None:
        tree = build_sample_tree()

        with self.assertRaises(ValueError):
            tree.add_children(
                0,
                [
                    ExpansionChild(
                        move_uci="g1f3",
                        fen="extra-child",
                        scalar_features={"value": 0.1},
                    )
                ],
            )

    def test_clone_preserves_structure_without_sharing_state(self) -> None:
        tree = build_sample_tree()

        cloned = tree.clone()

        self.assertIsNot(cloned, tree)
        self.assertEqual(cloned.root_id, tree.root_id)
        self.assertEqual(cloned.num_nodes(), tree.num_nodes())
        self.assertEqual(cloned.children(0), [1, 2])
        self.assertEqual(cloned.children(1), [3, 4])
        self.assertEqual(cloned.get_node(1).scalar_features, tree.get_node(1).scalar_features)

        cloned.get_node(1).scalar_features["value"] = 9.0
        self.assertEqual(tree.get_node(1).scalar_features["value"], 0.5)

    def test_clone_expansion_prefix_rebuilds_prefix_with_stable_ids(self) -> None:
        tree = build_sample_tree()

        truncated = tree.clone_expansion_prefix(1)

        self.assertEqual(truncated.num_nodes(), 3)
        self.assertEqual(truncated.root_id, 0)
        self.assertEqual(truncated.children(0), [1, 2])
        self.assertEqual(truncated.get_node(1).incoming_move_uci, "e2e4")
        self.assertFalse(truncated.get_node(1).is_expanded)
        truncated.validate()


class SchemaTests(unittest.TestCase):
    def test_schema_vectorizes_with_defaults(self) -> None:
        schema = NodeFeatureSchema.from_ordered_features(
            ["value", "prior", "bonus"],
            defaults={"prior": 0.25, "bonus": -1.0},
        )

        vector = schema.vectorize({"value": 0.5, "prior": 0.75})

        self.assertEqual(vector, (0.5, 0.75, -1.0))
        self.assertEqual(schema.index("bonus"), 2)

    def test_schema_vectorizes_to_tensor_explicitly(self) -> None:
        schema = NodeFeatureSchema.from_ordered_features(
            ["value", "prior"],
            defaults={"prior": 0.25},
        )

        vector = schema.vectorize_tensor({"value": 0.5})

        self.assertTrue(torch.equal(vector, torch.tensor([0.5, 0.25], dtype=torch.float32)))


class TensorizerTests(unittest.TestCase):
    def test_single_tree_tensorization(self) -> None:
        tree = build_sample_tree()
        schema = NodeFeatureSchema.from_ordered_features(
            ["value", "prior"],
            defaults={"prior": 0.0},
        )
        batch = tensorize_tree(tree, schema=schema)

        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.num_nodes, 5)
        self.assertEqual(batch.num_edges, 4)
        self.assertEqual(batch.feature_names, ("value", "prior"))
        self.assertTrue(
            torch.equal(batch.parent_index.cpu(), torch.tensor([-1, 0, 0, 1, 1], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.root_index.cpu(), torch.tensor([0], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.edge_parent.cpu(), torch.tensor([0, 0, 1, 1], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.edge_child.cpu(), torch.tensor([2, 1, 4, 3], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.edge_slot.cpu(), torch.tensor([0, 1, 0, 1], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.child_ptr.cpu(), torch.tensor([0, 2, 4, 4, 4, 4], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.depth.cpu(), torch.tensor([0, 1, 1, 2, 2], dtype=torch.long))
        )
        self.assertAlmostEqual(batch.node_features[2, 1].item(), 0.0)

    def test_forest_tensorization_preserves_boundaries(self) -> None:
        tree_a = build_sample_tree()
        tree_b = SearchTree("root-b", {"value": 0.0})
        root_id = tree_b.root_id
        tree_b.add_children(
            root_id,
            [
                ExpansionChild("g1f3", "b-child-1", {"value": 0.2}),
                ExpansionChild("c2c4", "b-child-2", {"value": 0.3}),
                ExpansionChild("b1c3", "b-child-3", {"value": 0.1}),
            ],
        )

        schema = NodeFeatureSchema.from_ordered_features(["value"], defaults={"value": 0.0})
        batch = tensorize_forest([tree_a, tree_b], schema=schema)

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.num_nodes, 9)
        self.assertEqual(batch.num_edges, 7)
        self.assertTrue(
            torch.equal(batch.root_index.cpu(), torch.tensor([0, 5], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.tree_index.cpu(), torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.parent_index.cpu(), torch.tensor([-1, 0, 0, 1, 1, -1, 5, 5, 5], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(batch.child_ptr.cpu(), torch.tensor([0, 2, 4, 4, 4, 4, 7, 7, 7, 7], dtype=torch.long))
        )

    def test_tensorized_observation_collation_preserves_boundaries(self) -> None:
        tree_a = build_sample_tree()
        tree_b = SearchTree("root-b", {"value": 0.0})
        root_id = tree_b.root_id
        tree_b.add_children(
            root_id,
            [
                ExpansionChild("g1f3", "b-child-1", {"value": 0.2}),
                ExpansionChild("c2c4", "b-child-2", {"value": 0.3}),
            ],
        )

        schema = NodeFeatureSchema.from_ordered_features(["value"], defaults={"value": 0.0})
        batch = collate_tensorized_observations(
            [
                tensorize_tree_observation(tree_a, schema=schema),
                tensorize_tree_observation(tree_b, schema=schema),
            ]
        )

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.num_nodes, 8)
        self.assertEqual(batch.num_edges, 6)
        self.assertTrue(torch.equal(batch.root_index.cpu(), torch.tensor([0, 5], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
