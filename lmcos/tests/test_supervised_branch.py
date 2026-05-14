import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from cts.core.providers.base import TreeExpansionProvider
from cts.core.schema import tree_encoder_feature_schema
from cts.core.tensorizer import tensorize_tree, tensorize_tree_with_targets
from cts.train import gnn_pretrain as gnn_pretrain_module
from cts.core.tree import ExpansionChild, SearchTree
from cts.data.episode_envs import (
    build_snapshot_episode,
    build_snapshot_episode_metadata,
    build_trimmed_decision_episode,
    build_trimmed_decision_episode_with_halt_rewards,
    trim_episode_to_first_root_decision,
)
from cts.data.preprocess_gnn.teacher_targets import (
    EdgeStats,
    GeneratedTree,
    NodeBudgetDistribution,
    PretrainExample,
    PretrainExampleDirectoryDataset,
    RawPretrainExampleRecord,
    TeacherSearchConfig,
    _backup_target_from_child_q,
    _backup_target_from_child_wdl,
    build_pretrain_example,
    build_tree_from_provider,
    compute_teacher_targets,
    consolidate_generated_tree,
    derive_prefix_pretrain_example,
    generate_partial_tree_from_provider,
    load_pretrain_example,
    load_pretrain_example_dataset,
    load_raw_pretrain_example_paths,
    normalize_prior_scores,
    prefix_expansion_count_schedule,
    save_pretrain_example,
)
from cts.models.gnn import ChildWdlModel
from cts.train.gnn_pretrain import (
    ChildWdlPretrainConfig,
    ChildWdlPretrainer,
    load_encoder_checkpoint,
)


class DummyProvider(TreeExpansionProvider):
    def root_features(self, fen):
        if fen == "root_alt":
            return make_wdl_features(0.4, 0.3, 0.3)
        return make_wdl_features(0.35, 0.3, 0.35)

    def expand_node(self, fen, depth, max_children=None):
        mapping = {
            "root": [
                ExpansionChild("a", "a", make_wdl_features(0.6, 0.2, 0.2, prior=0.7)),
                ExpansionChild("b", "b", make_wdl_features(0.2, 0.4, 0.4, prior=0.3)),
            ],
            "root_alt": [
                ExpansionChild("c", "c", make_wdl_features(0.5, 0.3, 0.2, prior=0.6)),
                ExpansionChild("d", "d", make_wdl_features(0.45, 0.2, 0.35, prior=0.4)),
            ],
            "a": [
                ExpansionChild("a1", "a1", make_wdl_features(0.35, 0.3, 0.35, prior=0.6)),
                ExpansionChild("a2", "a2", make_wdl_features(0.55, 0.2, 0.25, prior=0.4)),
            ],
            "b": [ExpansionChild("b1", "b1", make_wdl_features(0.6, 0.1, 0.3, prior=1.0))],
            "c": [ExpansionChild("c1", "c1", make_wdl_features(0.65, 0.2, 0.15, prior=1.0))],
            "d": [ExpansionChild("d1", "d1", make_wdl_features(0.25, 0.2, 0.55, prior=1.0))],
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


def make_wdl_features(win: float, draw: float, loss: float, prior: float = 1.0) -> dict[str, float]:
    total = win + draw + loss
    p_win = win / total
    p_draw = draw / total
    p_loss = loss / total
    value = p_win - p_loss
    variance = (p_win + p_loss) - value * value
    return {
        "value": value,
        "prior": prior,
        "wdl_win": p_win,
        "wdl_draw": p_draw,
        "wdl_loss": p_loss,
        "wdl_var": variance,
    }


def build_slot_test_tree() -> SearchTree:
    tree = SearchTree("root", make_wdl_features(0.5, 0.3, 0.2))
    root_id = tree.root_id
    tree.add_children(
        root_id,
        [
            ExpansionChild("g1f3", "child_b", make_wdl_features(0.2, 0.3, 0.5, prior=0.4)),
            ExpansionChild("e2e4", "child_a", make_wdl_features(0.7, 0.2, 0.1, prior=0.6)),
        ],
    )
    return tree


class SupervisedBranchTests(unittest.TestCase):
    def setUp(self):
        self.provider = DummyProvider()
        self.config = make_config()
        self.schema = make_schema()
        self.node_feat = len(self.schema.feature_names)
        self.device = "cpu"
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
        self.assertEqual(result_a.edge_target_wdls, result_b.edge_target_wdls)

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
        for child_id in root_children:
            edge_key = (0, child_id)
            self.assertEqual(result_a.edge_target_wdls[edge_key], result_a.edge_stats[edge_key].mean_wdl)

    def test_backup_target_uses_child_q_statistics_not_selection_score(self):
        tree = SearchTree("root", {"value": 0.0, "prior": 1.0})
        root_id = tree.root_id
        child_ids = tree.add_children(
            root_id,
            [
                ExpansionChild("a", "a", {"value": 0.2, "prior": 0.9}),
                ExpansionChild("b", "b", {"value": -0.4, "prior": 0.1}),
            ],
        )

        edge_stats = {
            (root_id, child_ids[0]): EdgeStats(visit_count=1, total_value=-0.8),
            (root_id, child_ids[1]): EdgeStats(visit_count=4, total_value=1.2),
        }
        expected = (1 * -0.8 + 4 * 0.3) / 5

        target = _backup_target_from_child_q(tree, root_id, edge_stats, "value")

        self.assertAlmostEqual(target, expected)

    def test_backup_target_uses_visit_weighted_child_wdls(self):
        tree = SearchTree("root", make_wdl_features(0.4, 0.2, 0.4))
        root_id = tree.root_id
        child_ids = tree.add_children(
            root_id,
            [
                ExpansionChild("a", "a", make_wdl_features(0.7, 0.2, 0.1, prior=0.5)),
                ExpansionChild("b", "b", make_wdl_features(0.2, 0.5, 0.3, prior=0.5)),
            ],
        )

        edge_stats = {
            (root_id, child_ids[0]): EdgeStats(visit_count=2, total_wdl=(1.6, 0.2, 0.2)),
            (root_id, child_ids[1]): EdgeStats(visit_count=3, total_wdl=(0.3, 1.8, 0.9)),
        }

        target = _backup_target_from_child_wdl(tree, root_id, edge_stats)
        expected = (
            (2 * 0.8 + 3 * 0.1) / 5,
            (2 * 0.1 + 3 * 0.6) / 5,
            (2 * 0.1 + 3 * 0.3) / 5,
        )

        for actual, target_component in zip(target, expected):
            self.assertAlmostEqual(actual, target_component)
        self.assertAlmostEqual(sum(target), 1.0)

    def test_pretrain_example_target_order_matches_tensorized_node_order(self):
        example = build_pretrain_example("root", self.provider, self.config, root_position_id="p0")
        tree_batch = tensorize_tree(example.tree, schema=self.schema, device=self.device)

        self.assertEqual(len(example.node_target_values), tree_batch.num_nodes)
        self.assertEqual(len(example.edge_wdl_targets), tree_batch.num_edges)
        self.assertEqual(example.metadata["root_position_id"], "p0")
        self.assertEqual(example.metadata["search_config_id"], self.config.search_config_id)
        self.assertEqual(example.metadata["provider_metadata"]["provider"], "dummy")
        self.assertEqual(example.metadata["edge_wdl_target_generation_version"], "search_consolidated_edge_wdl_v1")

    def test_prefix_expansion_count_schedule_tracks_root_prefix_sizes(self):
        tree = build_tree_from_provider("root", self.provider, self.config)
        counts = prefix_expansion_count_schedule(tree)

        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[-1], len(tree.ordered_expansion_parent_ids()))
        self.assertTrue(all(left < right for left, right in zip(counts, counts[1:])))

    def test_derive_prefix_pretrain_example_uses_root_prefix_and_recomputes_targets(self):
        source_example = build_pretrain_example("root", self.provider, self.config, root_position_id="p0")
        prefix_example = derive_prefix_pretrain_example(
            source_example,
            config=self.config,
            min_nodes=3,
            max_nodes=5,
            rng=random.Random(0),
        )

        self.assertEqual(prefix_example.tree.root_id, 0)
        self.assertEqual(prefix_example.metadata["source_root_position_id"], "p0")
        self.assertGreaterEqual(prefix_example.metadata["prefix_expansion_count"], 3)
        self.assertLessEqual(prefix_example.metadata["prefix_expansion_count"], 5)
        self.assertEqual(prefix_example.metadata["prefix_expansion_count"], len(prefix_example.tree.ordered_expansion_parent_ids()))
        self.assertEqual(prefix_example.metadata["prefix_total_node_count"], prefix_example.tree.num_nodes())
        self.assertEqual(len(prefix_example.edge_wdl_targets), prefix_example.tree.num_edges())

        recomputed = compute_teacher_targets(prefix_example.tree, self.config)
        self.assertEqual(prefix_example.node_target_values, recomputed.node_target_values)
        self.assertEqual(set(prefix_example.edge_wdl_targets), set(recomputed.edge_target_wdls))
        for edge_key, target in prefix_example.edge_wdl_targets.items():
            recomputed_target = recomputed.edge_target_wdls[edge_key]
            for actual, expected in zip(target, recomputed_target):
                self.assertAlmostEqual(actual, expected)

    def test_derive_prefix_pretrain_example_handles_large_total_tree_under_expanded_budget(self):
        tree = SearchTree("wide-root", make_wdl_features(0.4, 0.3, 0.3))
        root_id = tree.root_id
        children = [
            ExpansionChild(f"m{i:02d}", f"child-{i}", make_wdl_features(0.5, 0.25, 0.25, prior=1.0 / 60.0))
            for i in range(60)
        ]
        child_ids = tree.add_children(root_id, children)
        for child_id in child_ids[:59]:
            child_fen = tree.get_node(child_id).fen
            tree.add_children(
                child_id,
                [ExpansionChild(f"{child_fen}-next", f"{child_fen}-next", make_wdl_features(0.55, 0.2, 0.25, prior=1.0))],
            )
        source_example = PretrainExample(
            tree=tree,
            node_target_values={node.node_id: 0.0 for node in tree.iter_nodes()},
            edge_wdl_targets={},
            metadata={"root_position_id": "wide-root"},
        )

        prefix_example = derive_prefix_pretrain_example(
            source_example,
            config=self.config,
            min_nodes=8,
            max_nodes=48,
            rng=random.Random(0),
        )

        self.assertGreater(prefix_example.tree.num_nodes(), 48)
        self.assertEqual(prefix_example.tree.root_id, 0)
        self.assertGreaterEqual(prefix_example.metadata["prefix_expansion_count"], 8)
        self.assertLessEqual(prefix_example.metadata["prefix_expansion_count"], 48)
        self.assertEqual(prefix_example.metadata["prefix_expansion_count"], len(prefix_example.tree.ordered_expansion_parent_ids()))
        self.assertEqual(prefix_example.metadata["prefix_total_node_count"], prefix_example.tree.num_nodes())
        self.assertEqual(len(prefix_example.edge_wdl_targets), prefix_example.tree.num_edges())

    def test_derive_pretrain_prefixes_script_supports_multiprocessing(self):
        repo_root = Path(__file__).resolve().parents[1]
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]

        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as config_dir:
            for index, example in enumerate(examples):
                save_pretrain_example(os.path.join(input_dir, f"{index:06d}.pt"), example)

            config_path = Path(config_dir) / "derive.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "input_data": input_dir,
                        "output_dir": output_dir,
                        "min_nodes": 3,
                        "max_nodes": 5,
                        "search_budget": 8,
                        "c_puct": 1.0,
                        "max_depth": 2,
                        "seed": 0,
                        "log_interval": 1,
                        "num_workers": 2,
                    }
                ),
                encoding="utf-8",
            )

            env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cts.data.preprocess_gnn.derive_prefixes",
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("base_examples_per_s=", result.stdout)
            output_paths = sorted(Path(output_dir).glob("*.pt"))
            self.assertEqual(len(output_paths), 2)

    def test_pack_pretrain_examples_script_supports_multiprocessing(self):
        repo_root = Path(__file__).resolve().parents[1]
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]

        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as split_root, tempfile.TemporaryDirectory() as output_root, tempfile.TemporaryDirectory() as config_dir:
            raw_paths = []
            for index, example in enumerate(examples):
                raw_path = os.path.join(raw_dir, f"{index:06d}.pt")
                save_pretrain_example(raw_path, example)
                raw_paths.append(raw_path)

            train_manifest = Path(split_root) / "train_manifest.txt"
            validation_manifest = Path(split_root) / "validation_manifest.txt"
            train_manifest.write_text(f"{raw_paths[0]}\n", encoding="utf-8")
            validation_manifest.write_text(f"{raw_paths[1]}\n", encoding="utf-8")

            config_path = Path(config_dir) / "pack.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "split_root": split_root,
                        "output_root": output_root,
                        "shard_size": 1,
                        "num_workers": 2,
                        "log_interval": 1,
                        "clear": True,
                    }
                ),
                encoding="utf-8",
            )

            env = dict(os.environ, PYTHONPATH=str(repo_root / "src"))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cts.data.preprocess_gnn.pack",
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("base_examples_per_s=", result.stdout)
            self.assertTrue((Path(output_root) / "train_manifest.json").exists())
            self.assertTrue((Path(output_root) / "validation_manifest.json").exists())

    def test_pretrain_example_directory_dataset_loads_examples_lazily(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            for index, example in enumerate(examples):
                save_pretrain_example(os.path.join(tmpdir, f"{index:06d}.pt"), example)

            with patch("cts.data.preprocess_gnn.teacher_targets.torch.load", wraps=torch.load) as mocked_load:
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

    def test_load_raw_pretrain_example_paths_recurses_through_shard_directories(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_a = os.path.join(tmpdir, "shard_00000")
            shard_b = os.path.join(tmpdir, "shard_00001")
            os.makedirs(shard_a, exist_ok=True)
            os.makedirs(shard_b, exist_ok=True)
            save_pretrain_example(os.path.join(shard_a, "000000_root.pt"), examples[0])
            save_pretrain_example(os.path.join(shard_b, "000001_root.pt"), examples[1])

            paths = load_raw_pretrain_example_paths(tmpdir)

            self.assertEqual(len(paths), 2)
            self.assertTrue(paths[0].endswith("shard_00000/000000_root.pt"))
            self.assertTrue(paths[1].endswith("shard_00001/000001_root.pt"))

    def test_pretrain_example_manifest_dataset_loads_examples_lazily(self):
        examples = [
            build_pretrain_example("root", self.provider, self.config, root_position_id="p0"),
            build_pretrain_example("root_alt", self.provider, self.config, root_position_id="p1"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index, example in enumerate(examples):
                path = os.path.join(tmpdir, f"{index:06d}.pt")
                save_pretrain_example(path, example)
                paths.append(path)

            manifest_path = os.path.join(tmpdir, "train_manifest.txt")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                for path in paths:
                    handle.write(f"{path}\n")

            with patch("cts.data.preprocess_gnn.teacher_targets.torch.load", wraps=torch.load) as mocked_load:
                dataset = load_pretrain_example_dataset(manifest_path)
                self.assertEqual(len(dataset), 2)
                self.assertEqual(mocked_load.call_count, 0)

                loaded = dataset[0]
                self.assertEqual(mocked_load.call_count, 1)
                self.assertEqual(loaded.metadata["root_position_id"], "p0")

    def test_load_pretrain_example_dataset_rejects_obsolete_packed_raw_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "train_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write('{"format":"cts_pretrain_packed_manifest_v1","entries":[]}')

            with self.assertRaises(ValueError):
                load_pretrain_example_dataset(manifest_path)

    def test_child_wdl_pretrainer_uses_precomputed_edge_targets_from_packed_tensorized_data(self):
        tree = build_slot_test_tree()
        edge_wdl_targets = {
            (0, 1): (0.2, 0.5, 0.3),
            (0, 2): (0.6, 0.2, 0.2),
        }
        tensorized = tensorize_tree_with_targets(
            tree,
            [0.0] * tree.num_nodes(),
            schema=self.schema,
            edge_wdl_targets=edge_wdl_targets,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = os.path.join(tmpdir, "shard_00000.pt")
            torch.save(
                {
                    "format": "cts_tensorized_pretrain_shard_v1",
                    "num_examples": 1,
                    "feature_names": list(self.schema.feature_names),
                    "node_ptr": torch.tensor([0, tensorized.node_features.shape[0]], dtype=torch.long),
                    "edge_ptr": torch.tensor([0, tensorized.edge_parent.shape[0]], dtype=torch.long),
                    "node_features": tensorized.node_features,
                    "parent_index": tensorized.parent_index,
                    "edge_parent": tensorized.edge_parent,
                    "edge_child": tensorized.edge_child,
                    "edge_slot": tensorized.edge_slot,
                    "edge_wdl_targets": tensorized.edge_wdl_targets,
                    "depth": tensorized.depth,
                    "node_targets": tensorized.node_targets,
                },
                shard_path,
            )
            manifest_path = os.path.join(tmpdir, "train_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "format": "cts_tensorized_pretrain_manifest_v1",
                        "entries": [{"path": shard_path, "num_examples": 1}],
                    },
                    handle,
                )

            dataset = load_pretrain_example_dataset(manifest_path)
            model = ChildWdlModel(
                k=1,
                node_feat=self.node_feat,
                device="cpu",
                node_embed_hidden=16,
                d_embed=12,
                d_message=8,
                n_heads=1,
                d_att=4,
                decoder_hidden=8,
            )
            trainer = ChildWdlPretrainer(
                model=model,
                schema=self.schema,
                device=self.device,
                train_examples=dataset,
                validation_examples=dataset,
                config=ChildWdlPretrainConfig(
                    batch_size=1,
                    learning_rate=0.01,
                    epochs=1,
                    shuffle=False,
                ),
            )

            with patch.object(gnn_pretrain_module, "edge_wdl_target_tensor", side_effect=AssertionError("recomputed")):
                metrics = trainer.validate()

            self.assertTrue(math.isfinite(metrics.total_loss))
            self.assertEqual(metrics.num_supervised_edges, 2)

    def test_child_wdl_pretraining_reduces_validation_loss_and_checkpoint_restores_encoder(self):
        train_examples = [
            build_pretrain_example("root", self.provider, self.config),
            build_pretrain_example("root_alt", self.provider, self.config),
        ]
        model = ChildWdlModel(
            k=1,
            node_feat=self.node_feat,
            device="cpu",
            node_embed_hidden=16,
            d_embed=12,
            d_message=8,
            n_heads=1,
            d_att=4,
            decoder_hidden=8,
        )
        trainer = ChildWdlPretrainer(
            model=model,
            schema=self.schema,
            device=self.device,
            train_examples=train_examples,
            validation_examples=train_examples,
            config=ChildWdlPretrainConfig(
                batch_size=2,
                learning_rate=0.05,
                epochs=6,
                shuffle=False,
            ),
        )

        initial_validation = trainer.validate().total_loss
        history = trainer.fit()
        final_validation = trainer.validate().total_loss

        self.assertTrue(history)
        self.assertLess(final_validation, initial_validation)
        self.assertGreaterEqual(history[-1]["validation"].target_entropy, 0.0)
        self.assertGreaterEqual(history[-1]["validation"].loss_gap, 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "child_wdl_encoder.pt")
            trainer.save_best_encoder(
                checkpoint_path,
                metadata={"pretrain_objective": "search_consolidated_edge_wdl_v1"},
            )

            loaded_model = ChildWdlModel(
                k=1,
                node_feat=self.node_feat,
                device="cpu",
                node_embed_hidden=16,
                d_embed=12,
                d_message=8,
                n_heads=1,
                d_att=4,
                decoder_hidden=8,
            )
            metadata = load_encoder_checkpoint(checkpoint_path, loaded_model.encoder)
            self.assertEqual(metadata["pretrain_objective"], "search_consolidated_edge_wdl_v1")

            for key, value in trainer.model.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, loaded_model.encoder.state_dict()[key]))

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

    def test_generated_examples_store_oracle_root_trace(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=3, max_nodes=3),
            rng=random.Random(5),
        )

        self.assertEqual(
            example.oracle_trace_expansion_counts,
            list(range(1, len(example.tree.ordered_expansion_parent_ids()) + 1)),
        )
        self.assertEqual(len(example.oracle_root_q_trace), len(example.oracle_trace_expansion_counts))
        self.assertEqual(len(example.oracle_best_move_trace), len(example.oracle_trace_expansion_counts))
        self.assertTrue(example.oracle_root_moves)
        self.assertEqual(
            example.oracle_final_root_q_values,
            {
                move: q_value
                for move, q_value in zip(example.oracle_root_moves, example.oracle_root_q_trace[-1])
            },
        )

    def test_snapshot_episode_metadata_uses_stored_oracle_trace_without_research(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=3, max_nodes=3),
            rng=random.Random(9),
        )

        with patch("cts.data.episode_envs.compute_teacher_targets", side_effect=AssertionError("should not re-search")):
            snapshots, qualities, best_moves, final_root_q_values, final_best_quality = build_snapshot_episode_metadata(
                example,
                self.config,
            )

        self.assertEqual(len(snapshots), len(example.tree.ordered_expansion_parent_ids()) + 1)
        self.assertEqual(best_moves[1:], example.oracle_best_move_trace)
        self.assertEqual(final_root_q_values, example.oracle_final_root_q_values)
        self.assertAlmostEqual(final_best_quality, max(example.oracle_final_root_q_values.values()))
        self.assertEqual(
            qualities[1:],
            [max(row) for row in example.oracle_root_q_trace],
        )

    def test_snapshot_reconstruction_handles_nontrivial_expansion_order(self):
        tree = SearchTree("root", {"value": 0.0, "prior": 1.0})
        root_id = tree.root_id
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

    def test_pretrain_example_compact_serialization_round_trips(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=4, max_nodes=4),
            rng=random.Random(11),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "example.pt")
            save_pretrain_example(path, example)
            loaded = load_pretrain_example(path)
            record = RawPretrainExampleRecord.load(path)

        self.assertEqual(len(loaded.node_target_values), len(example.node_target_values))
        for loaded_value, expected_value in zip(loaded.node_target_values, example.node_target_values):
            self.assertAlmostEqual(loaded_value, expected_value)
        self.assertEqual(set(loaded.edge_wdl_targets), set(example.edge_wdl_targets))
        for edge_key, target in example.edge_wdl_targets.items():
            loaded_target = loaded.edge_wdl_targets[edge_key]
            for loaded_value, expected_value in zip(loaded_target, target):
                self.assertAlmostEqual(loaded_value, expected_value)
        self.assertEqual(loaded.metadata, example.metadata)
        self.assertEqual(loaded.oracle_trace_expansion_counts, example.oracle_trace_expansion_counts)
        self.assertEqual(loaded.oracle_root_moves, example.oracle_root_moves)
        self.assertEqual(len(loaded.oracle_root_q_trace), len(example.oracle_root_q_trace))
        for loaded_row, expected_row in zip(loaded.oracle_root_q_trace, example.oracle_root_q_trace):
            self.assertEqual(len(loaded_row), len(expected_row))
            for loaded_value, expected_value in zip(loaded_row, expected_row):
                self.assertAlmostEqual(loaded_value, expected_value)
        self.assertEqual(loaded.oracle_best_move_trace, example.oracle_best_move_trace)
        self.assertEqual(set(loaded.oracle_final_root_q_values), set(example.oracle_final_root_q_values))
        for move, expected_value in example.oracle_final_root_q_values.items():
            self.assertAlmostEqual(loaded.oracle_final_root_q_values[move], expected_value)
        self.assertEqual(loaded.tree.root_id, example.tree.root_id)
        self.assertEqual(loaded.tree.num_nodes(), example.tree.num_nodes())
        self.assertEqual(loaded.tree.num_edges(), example.tree.num_edges())
        for loaded_node, original_node in zip(loaded.tree.iter_nodes(), example.tree.iter_nodes()):
            self.assertEqual(loaded_node.parent_id, original_node.parent_id)
            self.assertEqual(loaded_node.incoming_move_uci, original_node.incoming_move_uci)
            self.assertEqual(loaded_node.fen, original_node.fen)
            self.assertEqual(loaded_node.depth, original_node.depth)
            self.assertEqual(loaded_node.is_terminal, original_node.is_terminal)
            self.assertEqual(loaded_node.is_expanded, original_node.is_expanded)
            self.assertEqual(set(loaded_node.scalar_features), set(original_node.scalar_features))
            for feature_name, expected_value in original_node.scalar_features.items():
                self.assertAlmostEqual(loaded_node.scalar_features[feature_name], expected_value)
            self.assertEqual(loaded_node.metadata, original_node.metadata)
        loaded.tree.validate()
        reconstructed = record.to_pretrain_example().oracle_final_root_q_values
        self.assertEqual(set(reconstructed), set(example.oracle_final_root_q_values))
        for move, expected_value in example.oracle_final_root_q_values.items():
            self.assertAlmostEqual(reconstructed[move], expected_value)

    def test_raw_pretrain_record_direct_tensorization_matches_tree_tensorization(self):
        example = build_pretrain_example(
            "root",
            self.provider,
            self.config,
            node_budget_distribution=NodeBudgetDistribution(min_nodes=4, max_nodes=4),
            rng=random.Random(19),
        )

        record = RawPretrainExampleRecord.from_example(example)
        direct = record.to_tensorized_tree_example(self.schema)
        tensorized = tensorize_tree_with_targets(
            example.tree,
            example.node_target_values,
            schema=self.schema,
            edge_wdl_targets=example.edge_wdl_targets,
        )

        self.assertEqual(direct.feature_names, tensorized.feature_names)
        self.assertTrue(torch.equal(direct.node_features, tensorized.node_features))
        self.assertTrue(torch.equal(direct.parent_index, tensorized.parent_index))
        self.assertTrue(torch.equal(direct.edge_parent, tensorized.edge_parent))
        self.assertTrue(torch.equal(direct.edge_child, tensorized.edge_child))
        self.assertTrue(torch.equal(direct.edge_slot, tensorized.edge_slot))
        self.assertTrue(torch.equal(direct.depth, tensorized.depth))
        self.assertTrue(torch.equal(direct.node_targets, tensorized.node_targets))
        self.assertTrue(torch.equal(direct.edge_wdl_targets, tensorized.edge_wdl_targets))

    def test_pretrain_example_compact_serialization_is_smaller_than_legacy_object_graph(self):
        tree = SearchTree("root-position-spec", make_wdl_features(0.4, 0.2, 0.4, prior=1.0))
        root_id = tree.root_id
        root_child_ids = tree.add_children(
            root_id,
            [
                ExpansionChild(
                    f"m{child_index:02d}",
                    f"root-position-spec moves m{child_index:02d}",
                    make_wdl_features(0.55, 0.15, 0.30, prior=1.0 / 32.0),
                )
                for child_index in range(32)
            ],
        )
        for child_index, child_id in enumerate(root_child_ids):
            grandchildren = [
                ExpansionChild(
                    f"m{child_index:02d}g{grandchild_index:02d}",
                    f"root-position-spec moves m{child_index:02d} m{child_index:02d}g{grandchild_index:02d}",
                    make_wdl_features(
                        0.2 + 0.01 * (grandchild_index % 10),
                        0.3,
                        0.5 - 0.01 * (grandchild_index % 10),
                        prior=1.0 / 8.0,
                    ),
                    is_terminal=(grandchild_index % 3 == 0),
                )
                for grandchild_index in range(8)
            ]
            tree.add_children(child_id, grandchildren)

        edge_wdl_targets = {}
        for parent_id in range(tree.num_nodes()):
            for child_id in tree.child_ids(parent_id):
                edge_wdl_targets[(parent_id, child_id)] = (0.25, 0.35, 0.40)

        oracle_root_moves = [
            tree.get_node(child_id).incoming_move_uci
            for child_id in tree.root_children()
        ]
        example = PretrainExample(
            tree=tree,
            node_target_values=[float(node.depth) / 10.0 for node in tree.iter_nodes()],
            edge_wdl_targets=edge_wdl_targets,
            metadata={"root_position_id": "synthetic-large"},
            oracle_trace_expansion_counts=list(range(1, 1 + len(tree.ordered_expansion_parent_ids()))),
            oracle_root_moves=oracle_root_moves,
            oracle_root_q_trace=[
                [0.1 + 0.001 * idx for idx in range(len(oracle_root_moves))]
                for _ in range(len(tree.ordered_expansion_parent_ids()))
            ],
            oracle_best_move_trace=[oracle_root_moves[-1]] * len(tree.ordered_expansion_parent_ids()),
            oracle_final_root_q_values={
                move: 0.1 + 0.001 * idx for idx, move in enumerate(oracle_root_moves)
            },
        )

        legacy_tree_state = {
            "root_id": example.tree.root_id,
            "_nodes": [copy.deepcopy(node) for node in example.tree.iter_nodes()],
            "_children": {node_id: list(child_ids) for node_id, child_ids in example.tree._children.items()},
        }
        legacy_example_state = {
            "tree": legacy_tree_state,
            "node_target_values": list(example.node_target_values),
            "edge_wdl_targets": dict(example.edge_wdl_targets),
            "metadata": dict(example.metadata),
            "oracle_trace_expansion_counts": list(example.oracle_trace_expansion_counts),
            "oracle_root_moves": list(example.oracle_root_moves),
            "oracle_root_q_trace": [list(row) for row in example.oracle_root_q_trace],
            "oracle_best_move_trace": list(example.oracle_best_move_trace),
            "oracle_final_root_q_values": dict(example.oracle_final_root_q_values),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            compact_path = os.path.join(tmpdir, "compact.pt")
            legacy_path = os.path.join(tmpdir, "legacy.pt")
            save_pretrain_example(compact_path, example)
            torch.save(legacy_example_state, legacy_path)
            compact_size = os.path.getsize(compact_path)
            legacy_size = os.path.getsize(legacy_path)

        self.assertLess(compact_size, legacy_size)


if __name__ == "__main__":
    unittest.main()
