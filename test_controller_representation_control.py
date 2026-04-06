import unittest
from types import SimpleNamespace

import torch

from scripts.controller_representation_control import (
    ControlEpisode,
    RepresentationControlEnv,
    _build_schema,
    _make_observation_tree,
    _optimal_stop_step,
    evaluate_oracle_agreement,
)
from supervised_branch import evaluate_controller
from tensorizer import TreeTensorizer


class _AlwaysHaltModel:
    def __init__(self) -> None:
        self.encoder = SimpleNamespace(device=torch.device("cpu"))

    def eval(self):
        return self

    def __call__(self, tree_batch):
        batch_size = int(tree_batch.root_index.shape[0])
        return SimpleNamespace(
            halt_logits=torch.full((batch_size,), 10.0),
            state_value=torch.zeros(batch_size),
        )


class ControllerRepresentationControlTests(unittest.TestCase):
    def test_optimal_stop_step_matches_simple_cases(self):
        self.assertEqual(_optimal_stop_step([0.5, 0.4, 0.3], 0.05), 0)
        self.assertEqual(_optimal_stop_step([0.1, 0.3, 0.31], 0.05), 1)
        self.assertEqual(_optimal_stop_step([0.1, 0.3, 0.6], 0.05), 2)

    def test_observation_tree_encodes_step_and_label_one_hot(self):
        tree = _make_observation_tree(step_index=1, label_index=2, max_steps=3, num_labels=4)
        root = tree.get_node(tree.root_id)
        schema = _build_schema(max_steps=3, num_labels=4)
        vector = schema.vectorize(root.scalar_features)
        self.assertEqual(len(vector), 7)
        self.assertEqual(list(vector), [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_evaluate_controller_reuses_env_across_deterministic_episode_sequence(self):
        episodes = [
            ControlEpisode(example_path="a", halt_rewards=[1.0], oracle_stop_step=0, label_index=0),
            ControlEpisode(example_path="b", halt_rewards=[3.0], oracle_stop_step=0, label_index=1),
        ]
        schema = _build_schema(max_steps=1, num_labels=2)
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = _AlwaysHaltModel()

        def _make_env():
            return RepresentationControlEnv(
                episodes,
                continue_cost=0.1,
                max_steps=1,
                num_labels=2,
                seed=0,
                shuffle=False,
            )

        metrics = evaluate_controller(model, tensorizer, _make_env, num_episodes=2)
        self.assertAlmostEqual(metrics.average_return, 2.0)
        self.assertEqual(metrics.halt_step_histogram, {0: 2})

    def test_evaluate_oracle_agreement_reports_exact_matches(self):
        episodes = [
            ControlEpisode(example_path="a", halt_rewards=[1.0], oracle_stop_step=0, label_index=0),
            ControlEpisode(example_path="b", halt_rewards=[2.0], oracle_stop_step=0, label_index=1),
        ]
        schema = _build_schema(max_steps=1, num_labels=2)
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = _AlwaysHaltModel()

        metrics = evaluate_oracle_agreement(
            model,
            tensorizer,
            episodes,
            continue_cost=0.1,
            max_steps=1,
            num_labels=2,
        )
        self.assertAlmostEqual(metrics.exact_stop_step_accuracy, 1.0)
        self.assertAlmostEqual(metrics.first_action_accuracy, 1.0)
        self.assertEqual(metrics.confusion_matrix, {0: {0: 2}})


if __name__ == "__main__":
    unittest.main()
