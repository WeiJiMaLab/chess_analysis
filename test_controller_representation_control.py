import unittest
from types import SimpleNamespace

import torch

from scripts.controller_representation_control import (
    ControlEpisode,
    LinearPolicyValueControlModel,
    OracleActionBanditCase,
    OracleActionBanditEnv,
    RepresentationControlEnv,
    _build_schema,
    _build_balanced_oracle_action_bandit_cases,
    _make_observation_tree,
    _oracle_action_from_observation,
    _optimal_stop_step,
    evaluate_bandit_agreement,
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


class _OracleActionModel:
    def __init__(self) -> None:
        self.encoder = SimpleNamespace(device=torch.device("cpu"))

    def eval(self):
        return self

    def __call__(self, tree_batch):
        root_features = tree_batch.node_features[tree_batch.root_index]
        halt_logits = 10.0 * (root_features[:, 1] - root_features[:, 0])
        return SimpleNamespace(
            halt_logits=halt_logits,
            state_value=torch.zeros_like(halt_logits),
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

    def test_oracle_action_now_observation_tree_encodes_only_action_one_hot(self):
        tree = _make_observation_tree(
            step_index=3,
            label_index=7,
            max_steps=5,
            num_labels=11,
            representation="oracle-action-now",
            oracle_action=1,
        )
        root = tree.get_node(tree.root_id)
        schema = _build_schema(max_steps=1, num_labels=2, representation="oracle-action-now")
        vector = schema.vectorize(root.scalar_features)
        self.assertEqual(len(vector), 2)
        self.assertEqual(list(vector), [0.0, 1.0])
        self.assertEqual(set(root.scalar_features.keys()), {"action_0", "action_1"})
        self.assertEqual(_oracle_action_from_observation(tree), 1)

        continue_tree = _make_observation_tree(
            step_index=3,
            label_index=7,
            max_steps=5,
            num_labels=11,
            representation="oracle-action-now",
            oracle_action=0,
        )
        self.assertEqual(_oracle_action_from_observation(continue_tree), 0)

    def test_control_model_uses_linear_halt_readout(self):
        model = LinearPolicyValueControlModel(node_feat=2, device="cpu")
        self.assertIsInstance(model.halt_controller, torch.nn.Linear)
        self.assertEqual(tuple(model.halt_controller.weight.shape), (1, 2))
        self.assertEqual(tuple(model.value_head.weight.shape), (1, 2))

    def test_evaluate_controller_reuses_env_across_deterministic_episode_sequence(self):
        episodes = [
            ControlEpisode(example_path="a", halt_rewards=[1.0], oracle_stop_step=0, label_index=0, oracle_actions=[1]),
            ControlEpisode(example_path="b", halt_rewards=[3.0], oracle_stop_step=0, label_index=1, oracle_actions=[1]),
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
            ControlEpisode(example_path="a", halt_rewards=[1.0], oracle_stop_step=0, label_index=0, oracle_actions=[1]),
            ControlEpisode(example_path="b", halt_rewards=[2.0], oracle_stop_step=0, label_index=1, oracle_actions=[1]),
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
            representation="oracle-stop-step",
        )
        self.assertAlmostEqual(metrics.exact_stop_step_accuracy, 1.0)
        self.assertAlmostEqual(metrics.first_action_accuracy, 1.0)
        self.assertEqual(metrics.confusion_matrix, {0: {0: 2}})

    def test_oracle_action_now_env_observation_tracks_current_oracle_action_only(self):
        episode = ControlEpisode(
            example_path="a",
            halt_rewards=[-0.1, 0.3],
            oracle_stop_step=1,
            label_index=0,
            oracle_actions=[0, 1],
        )
        env = RepresentationControlEnv(
            [episode],
            continue_cost=0.05,
            max_steps=1,
            num_labels=2,
            representation="oracle-action-now",
            seed=0,
            shuffle=False,
        )

        first_tree = env.reset()
        first_root = first_tree.get_node(first_tree.root_id)
        self.assertEqual(first_root.scalar_features, {"action_0": 1.0, "action_1": 0.0})

        step_result = env.step(0)
        second_root = step_result.next_tree.get_node(step_result.next_tree.root_id)
        self.assertEqual(second_root.scalar_features, {"action_0": 0.0, "action_1": 1.0})

    def test_evaluate_oracle_agreement_uses_oracle_action_now_representation(self):
        episodes = [
            ControlEpisode(
                example_path="a",
                halt_rewards=[-0.1, 0.3],
                oracle_stop_step=1,
                label_index=0,
                oracle_actions=[0, 1],
            ),
            ControlEpisode(
                example_path="b",
                halt_rewards=[0.2],
                oracle_stop_step=0,
                label_index=0,
                oracle_actions=[1],
            ),
        ]
        schema = _build_schema(max_steps=1, num_labels=2, representation="oracle-action-now")
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = _OracleActionModel()

        metrics = evaluate_oracle_agreement(
            model,
            tensorizer,
            episodes,
            continue_cost=0.05,
            max_steps=1,
            num_labels=2,
            representation="oracle-action-now",
        )
        self.assertAlmostEqual(metrics.exact_stop_step_accuracy, 1.0)
        self.assertAlmostEqual(metrics.first_action_accuracy, 1.0)
        self.assertEqual(metrics.confusion_matrix, {0: {0: 1}, 1: {1: 1}})

    def test_balanced_oracle_action_bandit_cases_balance_state_labels(self):
        episodes = [
            ControlEpisode(
                example_path="a",
                halt_rewards=[0.0, 0.1, 0.2],
                oracle_stop_step=2,
                label_index=0,
                oracle_actions=[0, 0, 1],
            ),
            ControlEpisode(
                example_path="b",
                halt_rewards=[0.3],
                oracle_stop_step=0,
                label_index=0,
                oracle_actions=[1],
            ),
        ]

        cases = _build_balanced_oracle_action_bandit_cases(episodes, seed=0)

        self.assertEqual(len(cases), 4)
        self.assertEqual(
            {action: sum(1 for case in cases if case.oracle_action == action) for action in (0, 1)},
            {0: 2, 1: 2},
        )

    def test_oracle_action_bandit_env_is_one_step_reward_diagnostic(self):
        cases = [OracleActionBanditCase(example_path="a", step_index=0, oracle_action=0)]
        env = OracleActionBanditEnv(cases, max_steps=1, num_labels=2, seed=0, shuffle=False)

        tree = env.reset()
        self.assertEqual(_oracle_action_from_observation(tree), 0)
        correct = env.step(0)
        self.assertTrue(correct.done)
        self.assertEqual(correct.reward, 1.0)

        tree = env.reset()
        self.assertEqual(_oracle_action_from_observation(tree), 0)
        incorrect = env.step(1)
        self.assertTrue(incorrect.done)
        self.assertEqual(incorrect.reward, -1.0)

    def test_evaluate_bandit_agreement_reports_action_accuracy(self):
        cases = [
            OracleActionBanditCase(example_path="a", step_index=0, oracle_action=0),
            OracleActionBanditCase(example_path="b", step_index=0, oracle_action=1),
        ]
        schema = _build_schema(max_steps=1, num_labels=2, representation="oracle-action-now")
        tensorizer = TreeTensorizer(schema, device="cpu")
        model = _OracleActionModel()

        metrics = evaluate_bandit_agreement(
            model,
            tensorizer,
            cases,
            max_steps=1,
            num_labels=2,
        )

        self.assertAlmostEqual(metrics.action_accuracy, 1.0)
        self.assertEqual(metrics.confusion_matrix, {0: {0: 1}, 1: {1: 1}})


if __name__ == "__main__":
    unittest.main()
