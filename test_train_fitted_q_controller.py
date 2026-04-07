import unittest

import torch

from schema import NodeFeatureSchema
from scripts.train_fitted_q_controller import (
    ComputeAdvantageTreeSearchModel,
    FittedQCollator,
    FittedQEpisode,
    _advantage_loss_components,
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

    def test_model_outputs_compute_advantage(self):
        schema = NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = ComputeAdvantageTreeSearchModel(
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
            model.advantage_head[-1].bias.fill_(0.5)

        advantages = model(tensorizer.tensorize_tree(_root_tree(0.0)))

        self.assertTrue(torch.allclose(advantages, torch.tensor([0.5])))

    def test_advantage_loss_components_train_compute_advantage(self):
        predicted_advantages = torch.tensor([0.5])
        q_targets = torch.tensor([[0.3, 0.1]])

        advantage_mse, mean_abs_advantage_error, sign_accuracy = _advantage_loss_components(
            predicted_advantages,
            q_targets,
        )

        self.assertAlmostEqual(float(advantage_mse.item()), 0.09)
        self.assertAlmostEqual(float(mean_abs_advantage_error.item()), 0.3)
        self.assertAlmostEqual(float(sign_accuracy.item()), 1.0)

    def test_predict_stop_step_uses_positive_compute_advantage(self):
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
        model = ComputeAdvantageTreeSearchModel(
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
            model.advantage_head[-1].bias.fill_(-1.0)

        self.assertEqual(_predict_stop_step(model, tensorizer, episode), 0)

        with torch.no_grad():
            model.advantage_head[-1].bias.fill_(1.0)

        self.assertEqual(_predict_stop_step(model, tensorizer, episode), 1)


if __name__ == "__main__":
    unittest.main()
