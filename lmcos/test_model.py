import unittest

import torch

from GNN import HaltController, TreeNN, TreeSearchModel
from schema import NodeFeatureSchema
from tensorizer import TreeTensorizer
from tree import ExpansionChild, SearchTree


def make_tree(value_scale=1.0):
    tree = SearchTree()
    root_id = tree.create_root(
        fen="root",
        scalar_features={"value": 0.0 * value_scale, "prior": 1.0},
    )
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
        tensorizer = TreeTensorizer(schema)
        self.single_batch = tensorizer.tensorize_tree(make_tree())
        self.forest_batch = tensorizer.tensorize_forest([make_tree(1.0), make_tree(2.0)])

    def test_tree_encoder_shapes(self):
        model = TreeNN(
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

    def test_tree_search_model_outputs(self):
        model = TreeSearchModel(
            k=3,
            node_feat=2,
            device="cpu",
            node_embed_hidden=16,
            d_embed=12,
            d_message=10,
            n_heads=2,
            d_att=4,
            controller_hidden=8,
        )

        output = model(self.forest_batch)

        self.assertEqual(tuple(output.node_states.shape), (8, 12))
        self.assertEqual(tuple(output.root_states.shape), (2, 12))
        self.assertEqual(tuple(output.halt_logits.shape), (2,))
        self.assertEqual(tuple(output.halt_prob.shape), (2,))
        self.assertTrue(torch.all(output.halt_prob >= 0.0))
        self.assertTrue(torch.all(output.halt_prob <= 1.0))

    def test_model_backward_pass(self):
        model = TreeSearchModel(
            k=1,
            node_feat=2,
            device="cpu",
            node_embed_hidden=8,
            d_embed=10,
            d_message=6,
            n_heads=1,
            d_att=5,
            controller_hidden=8,
        )

        output = model(self.forest_batch)
        loss = output.halt_logits.sum() + output.node_states.sum()
        loss.backward()

        grad_found = False
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_found = True
                break
        self.assertTrue(grad_found)

    def test_halt_controller_accepts_root_states(self):
        controller = HaltController(d_embed=12, hidden_dim=8, device="cpu")
        root_states = torch.randn(3, 12)

        logits, prob = controller(root_states)

        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(prob.shape), (3,))


if __name__ == "__main__":
    unittest.main()
