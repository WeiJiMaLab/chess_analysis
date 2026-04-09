import copy
import math
import os
import random
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import torch

from GNN import NodeValueModel, PolicyValueTreeSearchModel
from schema import tree_encoder_feature_schema
from supervised_branch import (
    EdgeStats,
    FrozenEncoderControllerTrainer,
    GeneratedTree,
    GeneratedTreeHaltEnv,
    NodeBudgetDistribution,
    PPOConfig,
    PretrainExample,
    PretrainExampleDirectoryDataset,
    ReinforceConfig,
    ReinforceControllerTrainer,
    ReinforceEpisodeTransition,
    RolloutTransition,
    TreeObservationSnapshot,
    SnapshotEpisodeCache,
    build_snapshot_episode,
    build_snapshot_episode_metadata,
    build_trimmed_decision_episode,
    build_trimmed_decision_episode_with_halt_rewards,
    trim_episode_to_first_root_decision,
    load_pretrain_example_dataset,
    load_raw_pretrain_example_paths,
    SupervisedPretrainConfig,
    SupervisedPretrainer,
    TeacherSearchConfig,
    ToyHaltEnv,
    TreeExpansionProvider,
    build_pretrain_example,
    build_tree_from_provider,
    consolidate_generated_tree,
    compute_teacher_targets,
    evaluate_controller,
    generate_partial_tree_from_provider,
    load_encoder_checkpoint,
    normalize_prior_scores,
    _backup_target_from_child_q,
)
from tensorizer import TreeTensorizer
from tree import ExpansionChild, SearchTree


class DummyProvider(TreeExpansionProvider):
    def root_features(self, fen):
        if fen == "root_alt":
            return {"value": 0.1, "prior": 1.0}
        return {"value": 0.0, "prior": 1.0}

    def expand_node(self, fen, depth, max_children=None):
        mapping = {
            "root": [
                ExpansionChild("a", "a", {"value": 0.4, "prior": 0.7}),
                ExpansionChild("b", "b", {"value": -0.2, "prior": 0.3}),
            ],
            "root_alt": [
                ExpansionChild("c", "c", {"value": 0.2, "prior": 0.6}),
                ExpansionChild("d", "d", {"value": 0.1, "prior": 0.4}),
            ],
            "a": [
                ExpansionChild("a1", "a1", {"value": -0.1, "prior": 0.6}),
                ExpansionChild("a2", "a2", {"value": 0.2, "prior": 0.4}),
            ],
            "b": [ExpansionChild("b1", "b1", {"value": 0.3, "prior": 1.0})],
            "c": [ExpansionChild("c1", "c1", {"value": 0.5, "prior": 1.0})],
            "d": [ExpansionChild("d1", "d1", {"value": -0.3, "prior": 1.0})],
        }
        children = mapping.get(fen, [])
        if max_children is None:
            return children
        return children[:max_children]

    def provider_metadata(self):
        return {"provider": "dummy"}


def make_schema():
    return tree_encoder_feature_schema()


def make_config():
    return TeacherSearchConfig(
        max_depth=2,
        search_budget=8,
        c_puct=1.0,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id="dummy-config",
    )


def make_episode_snapshot(best_value):
    tree = SearchTree()
    root_id = tree.create_root("episode-root", {"value": 0.0, "prior": 1.0})
    tree.add_children(
        root_id,
        [
            ExpansionChild("best", f"best-{best_value}", {"value": best_value, "prior": 0.7}),
            ExpansionChild("other", f"other-{best_value}", {"value": 0.1, "prior": 0.3}),
        ],
    )
    return tree


def make_toy_env():
    snapshots = [
        make_episode_snapshot(0.2),
        make_episode_snapshot(0.5),
        make_episode_snapshot(0.9),
    ]
    return ToyHaltEnv(snapshots, continue_cost=0.05)


class SupervisedBranchTests(unittest.TestCase):
    def setUp(self):
        self.provider = DummyProvider()
        self.config = make_config()
        self.schema = make_schema()
        self.node_feat = len(self.schema.feature_names)
        self.tensorizer = TreeTensorizer(self.schema)
        self.node_budget_distribution = NodeBudgetDistribution(min_nodes=3, max_nodes=5)

    def test_prior_score_normalization(self):
        normalized = normalize_prior_scores([2.0, 1.0, 1.0])
        self.assertAlmostEqual(sum(normalized), 1.0)
        self.assertGreater(normalized[0], normalized[1])

    def test_log_uniform_budget_generation_and_consolidation(self):
        rng = random.Random(7)
        generated = generate_partial_tree_from_provider(
            "root",
            self.provider,
            self.config,
            self.node_budget_distribution,
            rng=rng,
        )
        teacher_result = consolidate_generated_tree(generated, self.config)

        self.assertGreaterEqual(generated.sampled_node_budget, 3)
        self.assertLessEqual(generated.sampled_node_budget, 5)
        self.assertLessEqual(generated.num_expansions, generated.sampled_node_budget)
        self.assertEqual(len(teacher_result.node_target_values), generated.tree.num_nodes())
        self.assertEqual(
            [generated.tree.get_node(child_id).incoming_move_uci for child_id in generated.tree.children(0)],
            ["a", "b"],
        )

        for node in generated.tree.iter_nodes():
            child_ids = generated.tree.children(node.node_id)
            if not child_ids:
                self.assertAlmostEqual(
                    teacher_result.node_target_values[node.node_id],
                    node.scalar_features["value"],
                )
            else:
                prior_sum = sum(generated.tree.get_node(child_id).scalar_features["prior"] for child_id in child_ids)
                self.assertAlmostEqual(prior_sum, 1.0)

    def test_budgeted_pretrain_example_uses_partial_tree_generation(self):
        rng = random.Random(11)
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=self.node_budget_distribution,
            rng=rng,
        )

        self.assertEqual(len(example.node_target_values), example.tree.num_nodes())
        self.assertEqual(
            [example.tree.get_node(child_id).incoming_move_uci for child_id in example.tree.children(0)],
            ["a", "b"],
        )

    def test_teacher_targets_are_deterministic_and_leaf_targets_match_static_values(self):
        tree = build_tree_from_provider("root", self.provider, self.config)
        result_a = compute_teacher_targets(tree, self.config)
        result_b = compute_teacher_targets(tree, self.config)

        self.assertEqual(result_a.node_target_values, result_b.node_target_values)

        for node in tree.iter_nodes():
            if not tree.children(node.node_id):
                self.assertAlmostEqual(
                    result_a.node_target_values[node.node_id],
                    node.scalar_features["value"],
                )

        root_children = tree.children(0)
        total_visits = sum(result_a.edge_stats[(0, child_id)].visit_count for child_id in root_children)
        expected_root = sum(
            result_a.edge_stats[(0, child_id)].visit_count * result_a.edge_stats[(0, child_id)].q_value
            for child_id in root_children
        ) / total_visits
        self.assertAlmostEqual(result_a.node_target_values[0], expected_root)

    def test_backup_target_uses_child_q_statistics_not_selection_score(self):
        tree = SearchTree()
        root_id = tree.create_root("root", {"value": 0.0, "prior": 1.0})
        child_ids = tree.add_children(
            root_id,
            [
                ExpansionChild("a", "a", {"value": 0.2, "prior": 0.9}),
                ExpansionChild("b", "b", {"value": -0.4, "prior": 0.1}),
            ],
        )

        edge_stats = {
            (root_id, child_ids[0]): EdgeStats(visit_count=1, total_value=-0.8, q_value=-0.8),
            (root_id, child_ids[1]): EdgeStats(visit_count=4, total_value=1.2, q_value=0.3),
        }
        expected = (1 * -0.8 + 4 * 0.3) / 5

        target = _backup_target_from_child_q(tree, root_id, edge_stats, "value")

        self.assertAlmostEqual(target, expected)

    def test_pretrain_example_target_order_matches_tensorized_node_order(self):
        example = build_pretrain_example("root", self.provider, self.config, root_position_id="p0")
        tree_batch = self.tensorizer.tensorize_tree(example.tree)

        self.assertEqual(len(example.node_target_values), tree_batch.num_nodes)
        self.assertEqual(example.metadata["root_position_id"], "p0")
        self.assertEqual(example.metadata["search_config_id"], self.config.search_config_id)
        self.assertEqual(example.metadata["provider_metadata"]["provider"], "dummy")

    def test_pretrain_example_directory_dataset_loads_examples_lazily(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, example in enumerate(examples):
                torch.save(example, os.path.join(tmpdir, f"{index:06d}.pt"))

            with patch("supervised_branch.torch.load", wraps=torch.load) as mocked_load:
                dataset = PretrainExampleDirectoryDataset(tmpdir)
                self.assertEqual(len(dataset), 2)
                self.assertEqual(mocked_load.call_count, 0)

                loaded = dataset[1]
                self.assertEqual(mocked_load.call_count, 1)
                self.assertEqual(loaded.metadata["root_position_id"], "p1")

    def test_load_raw_pretrain_example_paths_rejects_packed_json_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "train_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write('{"format":"cts_tensorized_pretrain_manifest_v1","entries":[]}')

            with self.assertRaises(ValueError):
                load_raw_pretrain_example_paths(manifest_path)

    def test_pretrain_example_manifest_dataset_loads_examples_lazily(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index, example in enumerate(examples):
                path = os.path.join(tmpdir, f"{index:06d}.pt")
                torch.save(example, path)
                paths.append(path)

            manifest_path = os.path.join(tmpdir, "train_manifest.txt")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for path in paths:
                    handle.write(f"{path}\n")

            with patch("supervised_branch.torch.load", wraps=torch.load) as mocked_load:
                dataset = load_pretrain_example_dataset(manifest_path)
                self.assertEqual(len(dataset), 2)
                self.assertEqual(mocked_load.call_count, 0)

                loaded = dataset[0]
                self.assertEqual(mocked_load.call_count, 1)
                self.assertEqual(loaded.metadata["root_position_id"], "p0")

    def test_packed_pretrain_manifest_loads_examples_lazily(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = os.path.join(tmpdir, "shard_00000.pt")
            torch.save(
                {
                    "format": "cts_pretrain_packed_shard_v1",
                    "num_examples": len(examples),
                    "examples": examples,
                },
                shard_path,
            )
            manifest_path = os.path.join(tmpdir, "train_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(
                    '{"format":"cts_pretrain_packed_manifest_v1","entries":[{"path":"%s","num_examples":2}]}' % shard_path
                )

            with patch("supervised_branch.torch.load", wraps=torch.load) as mocked_load:
                dataset = load_pretrain_example_dataset(manifest_path)
                self.assertEqual(len(dataset), 2)
                self.assertEqual(mocked_load.call_count, 0)

                loaded = dataset[1]
                self.assertEqual(mocked_load.call_count, 1)
                self.assertEqual(loaded.metadata["root_position_id"], "p1")

    def test_supervised_pretraining_reduces_validation_loss_and_checkpoint_restores_encoder(self):
        train_examples = [
            build_pretrain_example("root", self.provider, self.config),
            build_pretrain_example("root_alt", self.provider, self.config),
        ]
        model = NodeValueModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=16,
            d_embed=12,
            d_message=8,
            n_heads=1,
            d_att=4,
            value_hidden=8,
        )
        trainer = SupervisedPretrainer(
            model=model,
            tensorizer=self.tensorizer,
            train_examples=train_examples,
            validation_examples=train_examples,
            config=SupervisedPretrainConfig(
                batch_size=2,
                learning_rate=0.05,
                root_loss_weight=2.0,
                epochs=8,
                shuffle=False,
            ),
        )

        initial_validation = trainer.validate().total_loss
        history = trainer.fit()
        final_validation = trainer.validate().total_loss

        self.assertTrue(history)
        self.assertLess(final_validation, initial_validation)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "encoder.pt")
            trainer.save_best_encoder(checkpoint_path, metadata={"tag": "pretrain"})

            loaded_model = NodeValueModel(
                k=1,
                node_feat=self.node_feat,
                device="cpu",
                node_embed_hidden=16,
                d_embed=12,
                d_message=8,
                n_heads=1,
                d_att=4,
                value_hidden=8,
            )
            metadata = load_encoder_checkpoint(checkpoint_path, loaded_model.encoder)
            self.assertEqual(metadata["tag"], "pretrain")

            for key, value in trainer.model.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, loaded_model.encoder.state_dict()[key]))

    def test_frozen_encoder_rl_update_keeps_encoder_fixed_and_updates_heads(self):
        example = build_pretrain_example("root", self.provider, self.config)
        pretrain_model = NodeValueModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=16,
            d_embed=10,
            d_message=6,
            n_heads=1,
            d_att=4,
            value_hidden=8,
        )
        pretrainer = SupervisedPretrainer(
            model=pretrain_model,
            tensorizer=self.tensorizer,
            train_examples=[example],
            validation_examples=[example],
            config=SupervisedPretrainConfig(
                batch_size=1,
                learning_rate=0.05,
                root_loss_weight=1.0,
                epochs=2,
                shuffle=False,
            ),
        )
        pretrainer.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "encoder.pt")
            pretrainer.save_best_encoder(checkpoint_path)

            rl_model = PolicyValueTreeSearchModel(
                k=1,
                node_feat=self.node_feat,
                device="cpu",
                node_embed_hidden=16,
                d_embed=10,
                d_message=6,
                n_heads=1,
                d_att=4,
                controller_hidden=8,
                value_hidden=8,
            )
            trainer = FrozenEncoderControllerTrainer(
                model=rl_model,
                tensorizer=self.tensorizer,
                envs=[make_toy_env()],
                config=PPOConfig(
                    rollout_steps=12,
                    learning_rate=0.05,
                    ppo_epochs=2,
                    minibatch_size=4,
                ),
                encoder_checkpoint_path=checkpoint_path,
            )

            encoder_before = {
                key: value.detach().clone()
                for key, value in rl_model.encoder.state_dict().items()
            }
            trainable_before = {
                name: parameter.detach().clone()
                for name, parameter in rl_model.named_parameters()
                if parameter.requires_grad
            }

            metrics = trainer.train_update()

            self.assertTrue(math.isfinite(metrics.policy_loss))
            self.assertTrue(math.isfinite(metrics.value_loss))
            self.assertTrue(math.isfinite(metrics.entropy))

            for key, value in rl_model.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, encoder_before[key]))

            changed = False
            for name, parameter in rl_model.named_parameters():
                if parameter.requires_grad and not torch.equal(parameter.detach(), trainable_before[name]):
                    changed = True
                    break
            self.assertTrue(changed)

    def test_collect_rollout_steps_all_envs_each_rollout_step(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        envs = [make_toy_env(), make_toy_env()]
        trainer = FrozenEncoderControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=envs,
            config=PPOConfig(
                rollout_steps=4,
                learning_rate=0.01,
                ppo_epochs=1,
                minibatch_size=4,
            ),
        )

        rollout, _ = trainer._collect_rollout()

        self.assertEqual(len(rollout), 8)
        counts = Counter(transition.env_id for transition in rollout)
        self.assertEqual(counts, Counter({0: 4, 1: 4}))

    def test_collect_rollout_uses_action_override_for_diagnostic_rollouts(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        trainer = FrozenEncoderControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=[make_toy_env(), make_toy_env()],
            config=PPOConfig(
                rollout_steps=4,
                learning_rate=0.01,
                ppo_epochs=1,
                minibatch_size=4,
            ),
            rollout_action_override=lambda observation, env_id, rollout_step: 1,
        )

        rollout, completed = trainer._collect_rollout()

        self.assertEqual(len(rollout), 8)
        self.assertTrue(all(transition.action == 1 for transition in rollout))
        self.assertTrue(all(transition.done for transition in rollout))
        self.assertEqual(len(completed), 8)

    def test_collect_rollout_snapshots_are_isolated_from_live_env_state(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        trainer = FrozenEncoderControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=[make_toy_env()],
            config=PPOConfig(
                rollout_steps=1,
                learning_rate=0.01,
                ppo_epochs=1,
                minibatch_size=1,
            ),
        )

        rollout, _ = trainer._collect_rollout()

        self.assertIsInstance(rollout[0].observation, TreeObservationSnapshot)
        self.assertIsNotNone(rollout[0].observation.tensorized)
        observed_tree = rollout[0].observation.tree
        observed_value = observed_tree.get_node(observed_tree.root_id).scalar_features["value"]
        trainer.current_trees[0].get_node(trainer.current_trees[0].root_id).scalar_features["value"] = 7.0

        self.assertEqual(observed_tree.get_node(observed_tree.root_id).scalar_features["value"], observed_value)

    def test_ppo_update_increases_halt_log_prob_for_consistent_positive_signal(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        trainer = FrozenEncoderControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=[make_toy_env()],
            config=PPOConfig(
                rollout_steps=1,
                learning_rate=0.05,
                ppo_epochs=1,
                minibatch_size=2,
                entropy_coef=0.0,
                value_loss_coef=0.0,
            ),
            freeze_encoder=False,
        )

        observation = make_toy_env().reset()
        before_distribution, _ = trainer._predict_single(observation)
        before_halt_log_prob = float(before_distribution.log_prob(torch.tensor(1.0)).item())

        rollout = [
            RolloutTransition(
                observation=copy.deepcopy(observation),
                env_id=0,
                action=1,
                reward=0.0,
                done=True,
                old_log_prob=before_halt_log_prob,
                value=0.0,
                advantage=1.0,
                return_value=0.0,
            ),
            RolloutTransition(
                observation=copy.deepcopy(observation),
                env_id=0,
                action=0,
                reward=0.0,
                done=True,
                old_log_prob=float(before_distribution.log_prob(torch.tensor(0.0)).item()),
                value=0.0,
                advantage=-1.0,
                return_value=0.0,
            ),
        ]

        trainer._update_policy(rollout)

        after_distribution, _ = trainer._predict_single(observation)
        after_halt_log_prob = float(after_distribution.log_prob(torch.tensor(1.0)).item())
        self.assertGreater(after_halt_log_prob, before_halt_log_prob)

    def test_ppo_update_reaches_encoder_when_unfrozen(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        trainer = FrozenEncoderControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=[make_toy_env()],
            config=PPOConfig(
                rollout_steps=1,
                learning_rate=0.05,
                ppo_epochs=1,
                minibatch_size=2,
                entropy_coef=0.0,
                value_loss_coef=0.0,
            ),
            freeze_encoder=False,
        )

        observation = make_toy_env().reset()
        distribution, _ = trainer._predict_single(observation)
        encoder_before = {
            key: value.detach().clone()
            for key, value in rl_model.encoder.state_dict().items()
        }
        rollout = [
            RolloutTransition(
                observation=copy.deepcopy(observation),
                env_id=0,
                action=1,
                reward=0.0,
                done=True,
                old_log_prob=float(distribution.log_prob(torch.tensor(1.0)).item()),
                value=0.0,
                advantage=1.0,
                return_value=0.0,
            ),
            RolloutTransition(
                observation=copy.deepcopy(observation),
                env_id=0,
                action=0,
                reward=0.0,
                done=True,
                old_log_prob=float(distribution.log_prob(torch.tensor(0.0)).item()),
                value=0.0,
                advantage=-1.0,
                return_value=0.0,
            ),
        ]

        trainer._update_policy(rollout)

        changed = any(
            not torch.equal(parameter.detach(), encoder_before[name])
            for name, parameter in rl_model.encoder.state_dict().items()
        )
        self.assertTrue(changed)
        self.assertTrue(any(parameter.grad is not None for parameter in rl_model.encoder.parameters()))

    def test_reinforce_positive_return_increases_halt_log_prob(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        trainer = ReinforceControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=[make_toy_env()],
            config=ReinforceConfig(
                learning_rate=0.05,
                max_grad_norm=1.0,
                batch_episodes=1,
                entropy_coef=0.0,
                use_return_normalization=False,
            ),
            freeze_encoder=False,
        )

        observation = make_toy_env().reset()
        before_distribution = trainer._predict_single(observation)
        before_halt_log_prob = float(before_distribution.log_prob(torch.tensor(1.0)).item())

        trainer._update_policy(
            [[ReinforceEpisodeTransition(observation=copy.deepcopy(observation), action=1, reward=1.0, done=True)]]
        )

        after_distribution = trainer._predict_single(observation)
        after_halt_log_prob = float(after_distribution.log_prob(torch.tensor(1.0)).item())
        self.assertGreater(after_halt_log_prob, before_halt_log_prob)

    def test_reinforce_update_reaches_encoder_but_not_value_head(self):
        rl_model = PolicyValueTreeSearchModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            controller_hidden=8,
            value_hidden=8,
        )
        trainer = ReinforceControllerTrainer(
            model=rl_model,
            tensorizer=self.tensorizer,
            envs=[make_toy_env()],
            config=ReinforceConfig(
                learning_rate=0.05,
                max_grad_norm=1.0,
                batch_episodes=1,
                entropy_coef=0.0,
                use_return_normalization=False,
            ),
            freeze_encoder=False,
        )

        observation = make_toy_env().reset()
        encoder_before = {
            key: value.detach().clone()
            for key, value in rl_model.encoder.state_dict().items()
        }
        value_head_before = {
            key: value.detach().clone()
            for key, value in rl_model.value_head.state_dict().items()
        }

        trainer._update_policy(
            [[ReinforceEpisodeTransition(observation=copy.deepcopy(observation), action=1, reward=1.0, done=True)]]
        )

        encoder_changed = any(
            not torch.equal(parameter.detach(), encoder_before[name])
            for name, parameter in rl_model.encoder.state_dict().items()
        )
        value_head_changed = any(
            not torch.equal(parameter.detach(), value_head_before[name])
            for name, parameter in rl_model.value_head.state_dict().items()
        )
        self.assertTrue(encoder_changed)
        self.assertFalse(value_head_changed)
        self.assertTrue(any(parameter.grad is not None for parameter in rl_model.encoder.parameters()))

    def test_end_to_end_smoke_path_produces_evaluation_metrics(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config),
            build_pretrain_example("root_alt", self.provider, self.config),
        ]
        pretrain_model = NodeValueModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=12,
            d_embed=8,
            d_message=6,
            n_heads=1,
            d_att=4,
            value_hidden=8,
        )
        pretrainer = SupervisedPretrainer(
            model=pretrain_model,
            tensorizer=self.tensorizer,
            train_examples=examples,
            validation_examples=examples,
            config=SupervisedPretrainConfig(
                batch_size=2,
                learning_rate=0.03,
                root_loss_weight=1.0,
                epochs=3,
                shuffle=False,
            ),
        )
        pretrainer.fit()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "encoder.pt")
            pretrainer.save_best_encoder(checkpoint_path)

            rl_model = PolicyValueTreeSearchModel(
                k=1,
                node_feat=self.node_feat,
                device="cpu",
                node_embed_hidden=12,
                d_embed=8,
                d_message=6,
                n_heads=1,
                d_att=4,
                controller_hidden=8,
                value_hidden=8,
            )
            trainer = FrozenEncoderControllerTrainer(
                model=rl_model,
                tensorizer=self.tensorizer,
                envs=[make_toy_env(), make_toy_env()],
                config=PPOConfig(
                    rollout_steps=16,
                    learning_rate=0.03,
                    ppo_epochs=2,
                    minibatch_size=4,
                ),
                encoder_checkpoint_path=checkpoint_path,
            )
            trainer.train(num_updates=2)

            metrics = evaluate_controller(
                model=rl_model,
                tensorizer=self.tensorizer,
                env_factory=make_toy_env,
                num_episodes=4,
            )

            self.assertTrue(math.isfinite(metrics.average_return))
            self.assertTrue(math.isfinite(metrics.average_expansions))
            self.assertTrue(math.isfinite(metrics.average_terminal_quality))
            self.assertTrue(metrics.halt_step_histogram)
            self.assertTrue(metrics.quality_by_expansions)

    def test_generated_tree_halt_env_builds_snapshot_sequence(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=3, max_nodes=3),
            rng=random.Random(3),
        )
        snapshots, qualities = build_snapshot_episode(example, self.config)
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertEqual(len(snapshots), len(qualities))
        self.assertEqual(snapshots[0].num_nodes(), 1)
        self.assertTrue(all(math.isfinite(value) for value in qualities))

        trimmed_snapshots, trimmed_qualities = trim_episode_to_first_root_decision(snapshots, qualities)
        self.assertTrue(trimmed_snapshots[0].root_children())
        self.assertEqual(len(trimmed_snapshots), len(trimmed_qualities))

        with tempfile.TemporaryDirectory() as tmpdir:
            example_path = os.path.join(tmpdir, "example.pt")
            torch.save(example, example_path)

            env = GeneratedTreeHaltEnv(
                example_paths=[example_path],
                quality_config=self.config,
                continue_cost=0.05,
                seed=0,
                shuffle=False,
                max_cache_size=2,
            )
            tree = env.reset()
            self.assertTrue(tree.root_children())

            done = False
            while not done:
                result = env.step(0)
                done = result.done

            self.assertIn("episode_return", result.info)
            self.assertIn("terminal_quality", result.info)
            self.assertEqual(result.info["example_path"], example_path)

    def test_generated_tree_halt_env_shared_cache_reuses_loaded_episode(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=3, max_nodes=3),
            rng=random.Random(3),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            example_path = os.path.join(tmpdir, "example.pt")
            torch.save(example, example_path)

            shared_cache = SnapshotEpisodeCache(max_size=2)
            env_a = GeneratedTreeHaltEnv(
                example_paths=[example_path],
                quality_config=self.config,
                continue_cost=0.05,
                seed=0,
                shuffle=False,
                max_cache_size=2,
                episode_cache=shared_cache,
            )
            env_b = GeneratedTreeHaltEnv(
                example_paths=[example_path],
                quality_config=self.config,
                continue_cost=0.05,
                seed=1,
                shuffle=False,
                max_cache_size=2,
                episode_cache=shared_cache,
            )

            with patch("cts_episode_envs.torch.load", wraps=torch.load) as mocked_load:
                env_a.reset()
                env_b.reset()

            self.assertEqual(mocked_load.call_count, 1)

    def test_trimmed_decision_episode_helper_extracts_halt_rewards(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=3, max_nodes=3),
            rng=random.Random(3),
        )

        episode = build_trimmed_decision_episode(example, self.config)
        episode_with_rewards, halt_rewards = build_trimmed_decision_episode_with_halt_rewards(example, self.config)

        self.assertEqual(episode.snapshots[0].root_children(), episode_with_rewards.snapshots[0].root_children())
        self.assertEqual(episode.best_moves, episode_with_rewards.best_moves)
        self.assertEqual(halt_rewards, [episode.final_root_q_values[move] for move in episode.best_moves])
        self.assertTrue(all(snapshot.root_children() for snapshot in episode.snapshots))

    def test_snapshot_reconstruction_handles_nontrivial_expansion_order(self):
        tree = SearchTree()
        root_id = tree.create_root("root", {"value": 0.0, "prior": 1.0})
        root_children = tree.add_children(
            root_id,
            [
                ExpansionChild("a", "a", {"value": 0.1, "prior": 0.5}),
                ExpansionChild("b", "b", {"value": 0.2, "prior": 0.5}),
            ],
        )
        b_children = tree.add_children(
            root_children[1],
            [
                ExpansionChild("b1", "b1", {"value": 0.3, "prior": 1.0}),
            ],
        )
        tree.add_children(
            b_children[0],
            [
                ExpansionChild("b1a", "b1a", {"value": -0.4, "prior": 1.0}, is_terminal=True),
            ],
        )
        tree.add_children(
            root_children[0],
            [
                ExpansionChild("a1", "a1", {"value": 0.4, "prior": 1.0}, is_terminal=True),
            ],
        )

        example = PretrainExample(tree=tree, node_target_values=[0.0] * tree.num_nodes())
        snapshots, qualities, best_moves, final_root_q_values, final_best_quality = build_snapshot_episode_metadata(
            example,
            self.config,
        )

        self.assertEqual(len(snapshots), 5)
        self.assertEqual(len(qualities), 5)
        self.assertEqual(len(best_moves), 5)
        self.assertTrue(final_root_q_values)
        self.assertTrue(math.isfinite(final_best_quality))
        for snapshot in snapshots:
            snapshot.validate()


if __name__ == "__main__":
    unittest.main()
