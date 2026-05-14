import unittest

from cts.core.schema import NodeFeatureSchema
from cts.core.tensorizer import tensorize_tree
from cts.core.tree import ExpansionChild, SearchTree
from cts.models.gnn import TreeEncoder


def make_tree(value_scale=1.0):
    tree = SearchTree(
        root_fen="root",
        root_scalar_features={"value": 0.0 * value_scale, "prior": 1.0},
    )
    root_id = tree.root_id
    child_ids = tree.add_children(
        root_id,
        [
            ExpansionChild(
                move_uci="e2e4",
                fen="child-1",
                scalar_features={"value": 0.4 * value_scale, "prior": 0.6},
            ),
            ExpansionChild(
                move_uci="d2d4",
                fen="child-2",
                scalar_features={"value": 0.2 * value_scale, "prior": 0.4},
            ),
        ],
    )
    tree.add_children(
        child_ids[0],
        [
            ExpansionChild(
                move_uci="e7e5",
                fen="grandchild-1",
                scalar_features={"value": -0.1 * value_scale, "prior": 0.3},
            )
        ],
    )
    return tree


class TreeModelTests(unittest.TestCase):
    def setUp(self):
        schema = NodeFeatureSchema.from_ordered_features(
            ["value", "prior"],
            defaults={"prior": 0.0},
        )
        self.single_batch = tensorize_tree(make_tree(), schema=schema)

    def test_tree_encoder_shapes(self):
        model = TreeEncoder(
            k=2,
            node_feat=2,
            device="cpu",
            node_embed_hidden=16,
            d_embed=12,
            d_message=10,
            n_heads=2,
            d_att=4,
        )

        encoded = model(self.single_batch)

        self.assertEqual(tuple(encoded.node_states.shape), (4, 12))
        self.assertEqual(tuple(encoded.root_states.shape), (1, 12))


if __name__ == "__main__":
    unittest.main()
