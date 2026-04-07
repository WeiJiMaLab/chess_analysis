import unittest
from unittest.mock import patch

import torch

from schema import NodeFeatureSchema
from scripts.train_fitted_q_controller import (
    ComputeAdvantageTreeSearchModel,
    FittedQCollator,
    FittedQEpisode,
    FittedQEpisodeDataset,
    LinearComputeAdvantageModel,
    _advantage_loss_components,
    _oracle_action_now_schema,
    _oracle_action_now_tree,
    _oracle_actions_from_targets,
    _predict_stop_step,
    bellman_q_targets,
)
from supervised_branch import TeacherSearchConfig
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

    def test_oracle_action_now_tree_encodes_target_advantage_sign(self):
        q_targets = torch.tensor([[0.3, 0.1], [0.2, 0.2], [0.1, 0.4]])

        actions = _oracle_actions_from_targets(q_targets)

        self.assertEqual(actions, [0, 1, 1])
        tree = _oracle_action_now_tree(actions[0])
        root_features = tree.get_node(tree.root_id).scalar_features
        self.assertEqual(root_features["action_0"], 1.0)
        self.assertEqual(root_features["action_1"], 0.0)

    def test_oracle_action_now_dataset_replaces_snapshots_with_one_hot_features(self):
        dataset = FittedQEpisodeDataset(
            paths=["example.pt"],
            quality_config=TeacherSearchConfig(max_depth=1, search_budget=1),
            continue_cost=0.001,
            reward_scale=1.0,
            representation="oracle-action-now",
            min_decision_margin=0.0,
        )

        raw_episode = type("RawEpisode", (), {"snapshots": [_root_tree(0.0), _root_tree(0.2)]})()
        with (
            patch("scripts.train_fitted_q_controller.torch.load", return_value={"raw": True}),
            patch(
                "scripts.train_fitted_q_controller.build_trimmed_decision_episode_with_halt_rewards",
                return_value=(raw_episode, [0.0, 0.2]),
            ),
        ):
            episode = dataset[0]

        self.assertIsNotNone(episode)
        assert episode is not None
        first_features = episode.snapshots[0].get_node(episode.snapshots[0].root_id).scalar_features
        second_features = episode.snapshots[1].get_node(episode.snapshots[1].root_id).scalar_features
        self.assertEqual(first_features["action_0"], 1.0)
        self.assertEqual(first_features["action_1"], 0.0)
        self.assertEqual(second_features["action_0"], 0.0)
        self.assertEqual(second_features["action_1"], 1.0)

    def test_linear_compute_advantage_model_reads_fixed_features(self):
        schema = _oracle_action_now_schema()
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = LinearComputeAdvantageModel(node_feat=2, device="cpu")

        with torch.no_grad():
            model.advantage_head.weight.copy_(torch.tensor([[2.0, -3.0]]))
            model.advantage_head.bias.fill_(0.5)

        advantages = model(
            tensorizer.tensorize_forest(
                [_oracle_action_now_tree(0), _oracle_action_now_tree(1)],
                validate=False,
            )
        )

        self.assertTrue(torch.allclose(advantages, torch.tensor([2.5, -2.5])))

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
