import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.probe_controller_representation import (
    ProbeTensorDataset,
    _backfill_advantages_from_paths,
    _load_cached_dataset,
)


class ProbeControllerRepresentationTests(unittest.TestCase):
    def test_legacy_cache_without_advantages_remains_tensor_dataset_compatible(self):
        metadata = {"split": "train"}
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "legacy_probe_cache.pt"
            torch.save(
                {
                    "embeddings": torch.zeros((2, 3), dtype=torch.float32),
                    "labels": torch.tensor([0, 1], dtype=torch.int64),
                    "phase_indices": torch.tensor([0, 1], dtype=torch.int64),
                    "oracle_stop_steps": torch.tensor([1, 1], dtype=torch.int64),
                    "metadata": metadata,
                    "processed_examples": 2,
                    "total_examples": 2,
                    "complete": True,
                },
                cache_path,
            )

            dataset = _load_cached_dataset(cache_path, metadata)

        self.assertIsNotNone(dataset)
        assert dataset is not None
        self.assertEqual(tuple(dataset.advantages.shape), (2,))
        self.assertTrue(torch.isnan(dataset.advantages).all())
        self.assertEqual(len(dataset.as_tensor_dataset()), 2)

    def test_backfill_advantages_replaces_legacy_nan_placeholders(self):
        dataset = ProbeTensorDataset(
            embeddings=torch.zeros((2, 3), dtype=torch.float32),
            labels=torch.tensor([0, 1], dtype=torch.int64),
            advantages=torch.full((2,), float("nan"), dtype=torch.float32),
            phase_indices=torch.tensor([0, 1], dtype=torch.int64),
            oracle_stop_steps=torch.tensor([1, 1], dtype=torch.int64),
            metadata={"split": "train"},
            processed_examples=2,
            total_examples=2,
            complete=True,
        )
        samples = [
            SimpleNamespace(target_advantage=0.25, target_action=0, phase_index=0, oracle_stop_step=1),
            SimpleNamespace(target_advantage=-0.1, target_action=1, phase_index=1, oracle_stop_step=1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "probe_cache.pt"
            with (
                patch("scripts.probe_controller_representation._select_paths", return_value=["a.pt", "b.pt"]),
                patch("scripts.probe_controller_representation._load_samples_from_paths", return_value=samples),
            ):
                updated = _backfill_advantages_from_paths(
                    dataset,
                    split_name="train",
                    data="manifest.txt",
                    sample_size=0,
                    seed=0,
                    cache_path=cache_path,
                    args=SimpleNamespace(continue_cost=0.001, load_log_interval=1),
                    quality_config=SimpleNamespace(),
                )

        self.assertTrue(torch.allclose(updated.advantages, torch.tensor([0.25, -0.1])))
        self.assertFalse(torch.isnan(updated.advantages).any())


if __name__ == "__main__":
    unittest.main()
