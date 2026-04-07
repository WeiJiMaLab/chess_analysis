import unittest

import torch

from schema import NodeFeatureSchema
from scripts.train_fitted_q_controller import (
    FittedQCollator,
    FittedQEpisode,
    QValueTreeSearchModel,
    _q_loss_components,
    _predict_stop_step,
    bellman_q_targets,
)
from tensorizer import TreeTensorizer
from tree import SearchTree


def _root_tree(value: float) -> SearchTree:
    tree = SearchTree()
    tree.create_root("root", {"value": value, "prior": 1.0})
    return tree


class TrainFittedQControllerTests(unittest.TestCase):
    def test_bellman_q_targets_use_action_order_continue_then_halt(self):
        q_targets, oracle_stop_step, oracle_value = bellman_q_targets(
            [0.0, 0.2, 0.5],
            continue_cost=0.1,
        )

        self.assertEqual(oracle_stop_step, 2)
        self.assertAlmostEqual(oracle_value, 0.3)
        self.assertTrue(
            torch.allclose(
                q_targets,
                torch.tensor(
                    [
                        [0.3, 0.0],
                        [0.4, 0.2],
                        [0.4, 0.5],
                    ]
                ),
            )
        )

    def test_fitted_q_collator_flattens_episode_snapshots(self):
        schema = NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})
        tensorizer = TreeTensorizer(schema, device="cpu")
        episode = FittedQEpisode(
            path="example.pt",
            snapshots=[_root_tree(0.1), _root_tree(0.2)],
            halt_rewards=[0.0, 0.5],
            q_targets=torch.tensor([[0.4, 0.0], [0.4, 0.5]], dtype=torch.float32),
            oracle_stop_step=1,
            oracle_value=0.4,
        )

        batch = FittedQCollator(tensorizer)([episode, None])

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.tree_batch.batch_size, 2)
        self.assertEqual(batch.paths, ["example.pt"])
        self.assertEqual(batch.skipped, 1)
        self.assertTrue(torch.equal(batch.q_targets, episode.q_targets))

    def test_q_model_outputs_continue_value_from_advantage_parameterization(self):
        schema = NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = QValueTreeSearchModel(
            k=1,
            node_feat=2,
            device="cpu",
            node_embed_hidden=4,
            d_embed=4,
            d_message=4,
            n_heads=1,
            d_att=2,
            q_hidden=4,
        )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.q_head[-1].bias.copy_(torch.tensor([0.5, 0.2]))

        q_values = model(tensorizer.tensorize_tree(_root_tree(0.0)))

        self.assertTrue(torch.allclose(q_values, torch.tensor([[0.7, 0.2]])))

    def test_q_loss_components_train_halt_value_and_continue_advantage(self):
        q_values = torch.tensor([[0.7, 0.2]])
        q_targets = torch.tensor([[0.3, 0.1]])

        q_mse, halt_value_mse, continue_advantage_mse = _q_loss_components(q_values, q_targets)

        self.assertAlmostEqual(float(q_mse.item()), 0.085)
        self.assertAlmostEqual(float(halt_value_mse.item()), 0.01)
        self.assertAlmostEqual(float(continue_advantage_mse.item()), 0.09)

    def test_predict_stop_step_uses_greedy_q_argmax(self):
        schema = NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})
        tensorizer = TreeTensorizer(schema, device="cpu")
        episode = FittedQEpisode(
            path="example.pt",
            snapshots=[_root_tree(0.0), _root_tree(1.0)],
            halt_rewards=[0.0, 1.0],
            q_targets=torch.tensor([[0.9, 0.0], [0.9, 1.0]], dtype=torch.float32),
            oracle_stop_step=1,
            oracle_value=0.9,
        )
        model = QValueTreeSearchModel(
            k=1,
            node_feat=2,
            device="cpu",
            node_embed_hidden=4,
            d_embed=4,
            d_message=4,
            n_heads=1,
            d_att=2,
            q_hidden=4,
        )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.q_head[-1].bias.copy_(torch.tensor([-1.0, 1.0]))

        self.assertEqual(_predict_stop_step(model, tensorizer, episode), 0)

        with torch.no_grad():
            model.q_head[-1].bias.copy_(torch.tensor([1.0, 0.0]))

        self.assertEqual(_predict_stop_step(model, tensorizer, episode), 1)


if __name__ == "__main__":
    unittest.main()
