import json
import os
import tempfile
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
    OracleActionNowEpisode,
    PackedControllerCollator,
    PackedControllerEpisodeDataset,
    _advantage_loss_components,
    _oracle_action_now_schema,
    _oracle_action_now_tensor_dataset,
    _oracle_action_now_tree,
    _oracle_actions_from_targets,
    _predict_stop_step,
    bellman_q_targets,
)
from supervised_branch import TeacherSearchConfig
from tensorizer import TreeBatch, TreeTensorizer
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

    def test_oracle_action_now_tensor_dataset_materializes_features_and_targets(self):
        episode = OracleActionNowEpisode(
            path="example.pt",
            features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            target_advantages=torch.tensor([0.2, -0.3]),
            halt_rewards=[0.0, 0.1],
            oracle_stop_step=1,
            oracle_value=0.2,
        )

        dataset = _oracle_action_now_tensor_dataset([episode])

        self.assertEqual(len(dataset), 2)
        self.assertTrue(torch.equal(dataset.tensors[0], episode.features))
        self.assertTrue(torch.equal(dataset.tensors[1], episode.target_advantages))

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


    def test_packed_controller_episode_round_trip(self):
        """Build a shard from hand-crafted tree data and verify the dataset+collator recovers a valid TreeBatch."""
        schema = NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})
        tensorizer = TreeTensorizer(schema, device="cpu")

        tree_a = SearchTree()
        tree_a.create_root("r_a", {"value": 0.1, "prior": 1.0})
        tree_b = SearchTree()
        tree_b.create_root("r_b", {"value": 0.2, "prior": 1.0})
        tree_c = SearchTree()
        tree_c.create_root("r_c", {"value": 0.5, "prior": 1.0})

        batch_a = tensorizer.tensorize_tree(tree_a, validate=False)
        batch_b = tensorizer.tensorize_tree(tree_b, validate=False)
        batch_c = tensorizer.tensorize_tree(tree_c, validate=False)

        episode_step_ptr = torch.tensor([0, 2, 3], dtype=torch.long)
        step_node_ptr = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        step_edge_ptr = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        node_features = torch.cat([batch_a.node_features, batch_b.node_features, batch_c.node_features], dim=0)
        parent_index = torch.cat([batch_a.parent_index, batch_b.parent_index, batch_c.parent_index], dim=0)
        depth = torch.cat([batch_a.depth, batch_b.depth, batch_c.depth], dim=0)

        halt_rewards = torch.tensor([0.0, 0.2, 0.5], dtype=torch.float32)
        q_targets = torch.tensor([[0.1, 0.0], [0.3, 0.2], [0.4, 0.5]], dtype=torch.float32)
        target_advantages = q_targets[:, 0] - q_targets[:, 1]
        oracle_stop_steps = torch.tensor([1, 2], dtype=torch.long)
        oracle_values = torch.tensor([0.1, 0.4], dtype=torch.float32)

        payload = {
            "format": "cts_controller_episode_shard_v1",
            "num_episodes": 2,
            "feature_names": list(schema.feature_names),
            "continue_cost": 0.001,
            "reward_scale": 1.0,
            "episode_step_ptr": episode_step_ptr,
            "step_node_ptr": step_node_ptr,
            "step_edge_ptr": step_edge_ptr,
            "node_features": node_features,
            "parent_index": parent_index,
            "edge_parent": torch.empty(0, dtype=torch.long),
            "edge_child": torch.empty(0, dtype=torch.long),
            "edge_slot": torch.empty(0, dtype=torch.long),
            "depth": depth,
            "halt_rewards": halt_rewards,
            "q_targets": q_targets,
            "target_advantages": target_advantages,
            "oracle_stop_steps": oracle_stop_steps,
            "oracle_values": oracle_values,
            "source_paths": ["ep_0.pt", "ep_1.pt"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = os.path.join(tmpdir, "shard_00000.pt")
            torch.save(payload, shard_path)

            manifest = {
                "format": "cts_controller_episode_manifest_v1",
                "split": "test",
                "total_episodes": 2,
                "total_skipped": 0,
                "continue_cost": 0.001,
                "reward_scale": 1.0,
                "min_decision_margin": 0.0,
                "entries": [{"path": shard_path, "num_episodes": 2, "shard_index": 0}],
            }
            manifest_path = os.path.join(tmpdir, "test_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            dataset = PackedControllerEpisodeDataset(manifest_path)
            self.assertEqual(len(dataset), 2)

            ep0 = dataset[0]
            self.assertEqual(len(ep0.step_node_features), 2)
            self.assertEqual(ep0.oracle_stop_step, 1)
            self.assertEqual(ep0.path, "ep_0.pt")
            self.assertAlmostEqual(ep0.halt_rewards[0], 0.0)
            self.assertAlmostEqual(ep0.halt_rewards[1], 0.2)

            ep1 = dataset[1]
            self.assertEqual(len(ep1.step_node_features), 1)
            self.assertEqual(ep1.oracle_stop_step, 2)
            self.assertEqual(ep1.path, "ep_1.pt")

            collator = PackedControllerCollator()
            batch = collator([ep0, ep1])
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertIsInstance(batch.tree_batch, TreeBatch)
            self.assertEqual(batch.tree_batch.batch_size, 3)
            self.assertEqual(batch.tree_batch.num_nodes, 3)
            self.assertEqual(batch.q_targets.shape, (3, 2))
            self.assertEqual(batch.path_lengths, [2, 1])
            self.assertEqual(batch.paths, ["ep_0.pt", "ep_1.pt"])
            self.assertTrue(torch.allclose(batch.tree_batch.node_features[0], batch_a.node_features[0]))
            self.assertTrue(torch.allclose(batch.tree_batch.node_features[2], batch_c.node_features[0]))


if __name__ == "__main__":
    unittest.main()
