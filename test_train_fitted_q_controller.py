import json
import os
import tempfile
import unittest

import torch

from budgeted_controller_oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    deterministic_starting_budgets,
    return_for_stop_step,
)
from cts_pretrain import PretrainExample
from schema import NodeFeatureSchema
from scripts.analyze_tree_stratification import _summarize_mode
from scripts.pack_controller_episodes import (
    _compute_tree_stratified_keep_probabilities_from_rows,
    _tree_budget_sensitivity_keep_probability,
    _tree_budget_sensitivity_score,
    _tree_budget_action_variance,
    _source_top_level_root_churn_category,
    _tree_stop_depth_excess,
)
from scripts.train_fitted_q_controller import (
    ComputeAdvantageTreeSearchModel,
    MaterializedAdvantageEpisode,
    PackedControllerCollator,
    PackedControllerEpisode,
    PackedControllerEpisodeDataset,
    _materialized_tensor_dataset,
    _predict_stop_step,
)
from tensorizer import TreeTensorizer
from tree import ExpansionChild, SearchTree


def _root_tree(value: float) -> SearchTree:
    tree = SearchTree()
    tree.create_root("root", {"value": value, "prior": 1.0})
    return tree


class BudgetedControllerOracleTests(unittest.TestCase):
    def test_root_churn_category_identifies_xaba(self):
        example = PretrainExample(
            tree=_root_tree(0.0),
            node_target_values=[0.0],
            oracle_best_move_trace=["a", "b", "a"],
            oracle_root_moves=["a", "b"],
            oracle_trace_expansion_counts=[1, 2, 3],
            oracle_root_q_trace=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            oracle_final_root_q_values={"a": 0.0, "b": 0.0},
        )
        self.assertEqual(_source_top_level_root_churn_category(example), "X*AB*A")

    def test_large_budget_modest_tree_favors_continue(self):
        policy = compute_budgeted_oracle([0.0, 0.6], [5, 5], 20, BudgetedOracleConfig())
        self.assertEqual(policy.optimal_stop_step, 1)
        self.assertGreater(policy.target_advantages[0], 0.0)

    def test_low_budget_flips_to_halt(self):
        policy = compute_budgeted_oracle([0.0, 0.6], [5, 5], 1, BudgetedOracleConfig())
        self.assertEqual(policy.optimal_stop_step, 0)
        self.assertLess(policy.target_advantages[0], 0.0)

    def test_large_tree_flips_to_halt_via_maintenance_cost(self):
        policy = compute_budgeted_oracle([0.0, 0.6], [3000, 3000], 20, BudgetedOracleConfig())
        self.assertEqual(policy.optimal_stop_step, 0)
        self.assertLess(policy.target_advantages[0], 0.0)

    def test_timeout_boundary_at_t1_uses_absorbing_loss(self):
        policy = compute_budgeted_oracle([0.2, 0.9], [5, 5], 1, BudgetedOracleConfig())
        self.assertEqual(policy.time_budgets, [1])
        self.assertAlmostEqual(policy.continue_values[0], policy.target_advantages[0] + policy.halt_rewards[0], places=6)
        self.assertLess(policy.continue_values[0], 0.0)
        self.assertEqual(policy.optimal_stop_step, 0)

    def test_return_for_stop_step_matches_oracle_value(self):
        config = BudgetedOracleConfig()
        policy = compute_budgeted_oracle([0.0, 0.6], [5, 5], 20, config)
        realized = return_for_stop_step(
            policy.halt_rewards,
            policy.tree_sizes,
            policy.time_budgets,
            policy.optimal_stop_step,
            config,
        )
        self.assertAlmostEqual(realized, policy.oracle_value, places=6)

    def test_deterministic_budget_sampling_is_stable(self):
        config = BudgetedOracleConfig(seed=17)
        first = deterministic_starting_budgets("example.pt", config)
        second = deterministic_starting_budgets("example.pt", config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(config.budget_buckets) * config.samples_per_bucket)

    def test_tree_budget_sensitivity_score_and_probability(self):
        config = BudgetedOracleConfig(maintenance_scale=0.0, time_lambda=0.0)
        tree_sizes = [1, 1]
        flat = [
            {"oracle_value": 0.1, "oracle_stop_step": 0, "starting_budget": 2, "halt_rewards": [0.1, 0.1]},
            {"oracle_value": 0.1, "oracle_stop_step": 0, "starting_budget": 2, "halt_rewards": [0.1, 0.1]},
        ]
        varied = [
            {"oracle_value": 0.5, "oracle_stop_step": 0, "starting_budget": 2, "halt_rewards": [0.5, 0.1]},
            {"oracle_value": 0.5, "oracle_stop_step": 1, "starting_budget": 2, "halt_rewards": [0.1, 0.5]},
        ]
        flat_score = _tree_budget_sensitivity_score(flat, tree_sizes, config)
        varied_score = _tree_budget_sensitivity_score(varied, tree_sizes, config)
        self.assertEqual(flat_score, 0.0)
        self.assertAlmostEqual(varied_score, 0.2, places=6)
        self.assertGreater(varied_score, flat_score)
        self.assertEqual(_tree_budget_sensitivity_keep_probability(0.0, varied_score), 0.25)
        self.assertEqual(_tree_budget_sensitivity_keep_probability(varied_score, varied_score), 1.0)

    def test_tree_stratification_summaries_and_probability(self):
        trivial = [
            {"oracle_stop_step": 0, "target_advantages": [0.0]},
            {"oracle_stop_step": 1, "target_advantages": [-0.1, -0.1]},
        ]
        budget_sensitive = [
            {"oracle_stop_step": 0, "target_advantages": [-0.2, -0.2, -0.2]},
            {"oracle_stop_step": 3, "target_advantages": [0.2, 0.2, -0.2]},
        ]
        self.assertEqual(_tree_stop_depth_excess(trivial), 0.0)
        self.assertEqual(_tree_budget_action_variance(trivial), 0.0)
        self.assertAlmostEqual(_tree_stop_depth_excess(budget_sensitive), 1.0, places=6)
        self.assertAlmostEqual(_tree_budget_action_variance(budget_sensitive), 1.0 / 6.0, places=6)
        rows = (
            [{"source_path": f"d0_{idx}", "stop_depth_excess": 0.0, "budget_action_variance": float(idx), "num_episodes": 10} for idx in range(9)]
            + [{"source_path": f"d1_{idx}", "stop_depth_excess": 2.0, "budget_action_variance": float(idx), "num_episodes": 10} for idx in range(3)]
        )
        keep_probabilities, metadata = _compute_tree_stratified_keep_probabilities_from_rows(rows, "dj")
        self.assertEqual(metadata["tree_stratification_mode"], "dj")
        self.assertGreater(keep_probabilities["d0_8"], keep_probabilities["d0_0"])
        self.assertEqual(keep_probabilities["d1_0"], 1.0)

    def test_tree_stratification_analyzer_summary_smoke(self):
        rows = (
            [{"source_path": f"d0_{idx}", "stop_depth_excess": 0.0, "budget_action_variance": float(idx), "num_episodes": 10} for idx in range(9)]
            + [{"source_path": f"d1_{idx}", "stop_depth_excess": 2.0, "budget_action_variance": float(idx), "num_episodes": 10} for idx in range(3)]
        )
        for mode in ("d", "j", "dj"):
            summary = _summarize_mode(rows, mode, seed=0)
            self.assertEqual(summary["mode"], mode)
            self.assertEqual(summary["num_accepted_trees"], len(rows))
            self.assertIn("strata", summary)
            self.assertGreater(len(summary["strata"]), 0)


class TrainFittedQControllerTests(unittest.TestCase):
    def test_model_builds_multi_layer_advantage_head(self):
        model = ComputeAdvantageTreeSearchModel(
            k=1,
            node_feat=2,
            device="cpu",
            node_embed_hidden=4,
            d_embed=4,
            d_message=4,
            n_heads=1,
            d_att=2,
            q_hidden=8,
            q_hidden_layers=3,
        )
        linear_layers = [module for module in model.advantage_head if isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(linear_layers[0].in_features, model.encoder.d_embed + 2)
        self.assertEqual(linear_layers[0].out_features, 8)
        self.assertEqual(linear_layers[-1].out_features, 1)

    def test_model_appends_literal_tree_size_and_budget_features(self):
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
            q_hidden_layers=1,
        )
        features = model.encode_with_state_features(
            tensorizer.tensorize_tree(_root_tree(0.0), validate=False),
            torch.tensor([7], dtype=torch.long),
            torch.tensor([11], dtype=torch.long),
        )
        self.assertEqual(features.shape, (1, model.encoder.d_embed + 2))
        self.assertTrue(torch.equal(features[:, -2:], torch.tensor([[7.0, 11.0]])))

    def test_predict_stop_step_uses_positive_compute_advantage(self):
        episode = PackedControllerEpisode(
            path="episode",
            source_path="source.pt",
            step_node_features=[torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 1.0]])],
            step_parent_index=[torch.tensor([-1]), torch.tensor([-1])],
            step_edge_parent=[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
            step_edge_child=[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
            step_edge_slot=[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
            step_depth=[torch.tensor([0]), torch.tensor([0])],
            halt_rewards=[0.0, 1.0],
            target_advantages=torch.tensor([0.9, -0.1], dtype=torch.float32),
            tree_sizes=torch.tensor([1, 1], dtype=torch.long),
            time_budgets=torch.tensor([10, 9], dtype=torch.long),
            oracle_stop_step=1,
            oracle_value=0.9,
            starting_budget=10,
            budget_bucket_name="large",
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
            q_hidden_layers=1,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.advantage_head[-1].bias.fill_(-1.0)
        stop, advantages = _predict_stop_step(model, episode)
        self.assertEqual(stop, 0)
        self.assertEqual(len(advantages), 2)

        with torch.no_grad():
            model.advantage_head[-1].bias.fill_(1.0)
        stop, advantages = _predict_stop_step(model, episode)
        self.assertEqual(stop, 1)
        self.assertEqual(len(advantages), 2)

    def test_materialized_tensor_dataset_round_trips(self):
        episode = MaterializedAdvantageEpisode(
            path="episode",
            source_path="source.pt",
            features=torch.tensor([[0.1, 0.2, 5.0, 10.0], [0.3, 0.4, 6.0, 9.0]]),
            target_advantages=torch.tensor([0.5, -0.25]),
            halt_rewards=[0.0, 0.3],
            tree_sizes=[5, 6],
            time_budgets=[10, 9],
            oracle_stop_step=1,
            oracle_value=0.5,
            starting_budget=10,
            budget_bucket_name="large",
        )
        dataset = _materialized_tensor_dataset([episode])
        self.assertEqual(len(dataset), 2)
        self.assertTrue(torch.equal(dataset.tensors[0], episode.features))
        self.assertTrue(torch.equal(dataset.tensors[1], episode.target_advantages))

    def test_packed_controller_episode_round_trip(self):
        schema = NodeFeatureSchema.from_ordered_features(["value", "prior"], defaults={"prior": 0.0})
        tensorizer = TreeTensorizer(schema, device="cpu")

        tree = SearchTree()
        root_id = tree.create_root("root", {"value": 0.1, "prior": 1.0})
        child_id = tree.add_children(
            root_id,
            [ExpansionChild("m1", "root ||moves|| m1", {"value": 0.2, "prior": 1.0})],
        )[0]
        tree.add_children(
            child_id,
            [ExpansionChild("m2", "root ||moves|| m1 m2", {"value": 0.5, "prior": 1.0})],
        )
        batch = tensorizer.tensorize_tree(tree, validate=False)

        payload = {
            "format": "cts_budgeted_controller_episode_shard_v4",
            "num_trajectories": 1,
            "num_episodes": 2,
            "feature_names": list(schema.feature_names),
            "oracle_type": "budgeted_controller_v1",
            "maintenance_scale": 0.01,
            "maintenance_ref_nodes": 30.0,
            "maintenance_exponent": 1.1,
            "time_lambda": 18.537,
            "time_p": 2.8,
            "time_tau": 2.5,
            "time_delta": 1,
            "timeout_value": -1.0,
            "samples_per_bucket": 2,
            "budget_seed": 0,
            "budget_buckets": [
                {"name": "scramble", "min_time": 1, "max_time": 3},
                {"name": "medium-small", "min_time": 4, "max_time": 10},
                {"name": "medium-large", "min_time": 11, "max_time": 25},
                {"name": "large", "min_time": 26, "max_time": 60},
                {"name": "very-large", "min_time": 61, "max_time": 120},
            ],
            "trajectory_node_ptr": torch.tensor([0, 3], dtype=torch.long),
            "trajectory_edge_ptr": torch.tensor([0, 2], dtype=torch.long),
            "trajectory_child_ptr_ptr": torch.tensor([0, 4], dtype=torch.long),
            "trajectory_expansion_parent_ptr": torch.tensor([0, 2], dtype=torch.long),
            "trajectory_step_ptr": torch.tensor([0, 2], dtype=torch.long),
            "episode_step_ptr": torch.tensor([0, 2, 3], dtype=torch.long),
            "episode_trajectory_index": torch.tensor([0, 0], dtype=torch.long),
            "node_features": batch.node_features,
            "parent_index": batch.parent_index,
            "edge_child": batch.edge_child,
            "edge_slot": batch.edge_slot,
            "depth": batch.depth,
            "child_ptr": torch.tensor([0, 1, 2, 2], dtype=torch.long),
            "expansion_parent_ids": torch.tensor([0, 1], dtype=torch.long),
            "step_node_cutoffs": torch.tensor([2, 3], dtype=torch.long),
            "trajectory_halt_rewards": torch.tensor([0.0, 0.2], dtype=torch.float32),
            "first_decision_expansion_counts": torch.tensor([1], dtype=torch.long),
            "target_advantages": torch.tensor([0.1, -0.1, -0.2], dtype=torch.float32),
            "oracle_stop_steps": torch.tensor([1, 0], dtype=torch.long),
            "oracle_values": torch.tensor([0.1, 0.5], dtype=torch.float32),
            "starting_budgets": torch.tensor([10, 3], dtype=torch.long),
            "budget_bucket_indices": torch.tensor([3, 0], dtype=torch.long),
            "budget_bucket_names": ["large", "scramble"],
            "episode_keys": ["ep_0", "ep_1"],
            "trajectory_source_paths": ["raw_0.pt"],
        }

        manifest = {
            "format": "cts_budgeted_controller_episode_manifest_v4",
            "split": "test",
            "total_episodes": 2,
            "reward_scale": 1.0,
            "oracle_type": "budgeted_controller_v1",
            "maintenance_scale": 0.01,
            "maintenance_ref_nodes": 30.0,
            "maintenance_exponent": 1.1,
            "time_lambda": 18.537,
            "time_p": 2.8,
            "time_tau": 2.5,
            "time_delta": 1,
            "timeout_value": -1.0,
            "samples_per_bucket": 2,
            "budget_seed": 0,
            "budget_buckets": payload["budget_buckets"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = os.path.join(tmpdir, "shard_00000.pt")
            torch.save(payload, shard_path)
            manifest["entries"] = [{"path": shard_path, "num_episodes": 2, "shard_index": 0}]
            manifest_path = os.path.join(tmpdir, "test_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

            dataset = PackedControllerEpisodeDataset(manifest_path)
            self.assertEqual(len(dataset), 2)

            episode0 = dataset[0]
            self.assertEqual(episode0.path, "ep_0")
            self.assertEqual(episode0.source_path, "raw_0.pt")
            self.assertEqual(episode0.starting_budget, 10)
            self.assertEqual(episode0.time_budgets.tolist(), [10, 9])
            self.assertEqual(episode0.tree_sizes.tolist(), [2, 3])

            episode1 = dataset[1]
            self.assertEqual(episode1.path, "ep_1")
            self.assertEqual(episode1.budget_bucket_name, "scramble")
            self.assertEqual(episode1.source_path, "raw_0.pt")
            self.assertEqual(len(episode1.step_node_features), 1)
            self.assertEqual(episode1.tree_sizes.tolist(), [2])

            batch = PackedControllerCollator()([episode0, episode1])
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.tree_batch.batch_size, 3)
            self.assertEqual(batch.tree_batch.num_nodes, 7)
            self.assertEqual(batch.target_advantages.shape, (3,))
            self.assertEqual(batch.tree_sizes.tolist(), [2, 3, 2])
            self.assertEqual(batch.time_budgets.tolist(), [10, 9, 3])
            self.assertEqual(batch.paths, ["ep_0", "ep_1"])
            self.assertEqual(batch.source_paths, ["raw_0.pt", "raw_0.pt"])


if __name__ == "__main__":
    unittest.main()
