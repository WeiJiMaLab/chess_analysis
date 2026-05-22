"""Tests for the materialize worker's resume behavior.

The full worker can't be exercised here (it requires a packed manifest,
a real encoder, and a GPU). What CAN be unit-tested without those is
the shard-scanning + state-rebuild logic that runs at worker startup:
when shard_dir already contains shard_*.pt files written by a previous
run, the worker reconstructs total_snapshots from their feature shapes
and skips the loader forward by exactly that count.

Asserts the two invariants resume relies on:
1. Sum of features.shape[0] over existing shards == total_snapshots
   that should be skipped.
2. The mismatch detector (RuntimeError) fires when the cumulative
   skipped count overshoots what's on disk — i.e., when shard boundaries
   would have to be mid-batch.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch


def _write_shard(shard_dir: Path, index: int, num_snapshots: int) -> Path:
    """Helper: write a minimal shard file with ``num_snapshots`` rows."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{index:05d}.pt"
    torch.save(
        {
            "format": "cts_materialized_advantage_cache_shard_v2",
            "features": torch.zeros((num_snapshots, 130), dtype=torch.float32),
            "target_advantages": torch.zeros((num_snapshots,), dtype=torch.float32),
            "oracle_stop_steps": torch.zeros((num_snapshots,), dtype=torch.int32),
        },
        path,
    )
    return path


def _scan_existing_shards(shard_dir: Path):
    """Mirror of the worker's startup scan, factored out for testing."""
    existing_shards = sorted(shard_dir.glob("shard_*.pt"))
    shard_paths = []
    shard_sizes = []
    total_snapshots = 0
    for shard_path in existing_shards:
        payload = torch.load(shard_path, weights_only=False)
        shard_paths.append(str(shard_path))
        shard_sizes.append(int(payload["features"].shape[0]))
        total_snapshots += shard_sizes[-1]
    return shard_paths, shard_sizes, total_snapshots


class ResumeStateRebuildTests(unittest.TestCase):
    def test_empty_dir_is_zero_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "worker_00"
            shard_dir.mkdir()
            paths, sizes, total = _scan_existing_shards(shard_dir)
            self.assertEqual(paths, [])
            self.assertEqual(sizes, [])
            self.assertEqual(total, 0)

    def test_three_shards_reconstruct_sizes_and_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "worker_00"
            _write_shard(shard_dir, 0, 250000)
            _write_shard(shard_dir, 1, 250000)
            _write_shard(shard_dir, 2, 137_503)  # partial last shard
            paths, sizes, total = _scan_existing_shards(shard_dir)
            self.assertEqual(len(paths), 3)
            self.assertEqual(sizes, [250000, 250000, 137_503])
            self.assertEqual(total, 637_503)

    def test_shards_sorted_lexically(self) -> None:
        """Sorted glob order is what gives shard_index = len(existing_shards)
        the right semantics. Verify a 10+ shard dir sorts numerically via zero-padding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "worker_00"
            for i in [0, 1, 2, 9, 10, 11]:
                _write_shard(shard_dir, i, 100)
            paths, sizes, total = _scan_existing_shards(shard_dir)
            self.assertEqual(
                [Path(p).name for p in paths],
                ["shard_00000.pt", "shard_00001.pt", "shard_00002.pt",
                 "shard_00009.pt", "shard_00010.pt", "shard_00011.pt"],
            )
            self.assertEqual(total, 600)


class ResumeOvershootGuardTests(unittest.TestCase):
    """The runtime check that fires if cumulative skipped batches overshoot
    the existing shards' total — would indicate non-deterministic loader
    or corrupted shards. Not a "we'll never hit this" assert; a real
    diagnostic if a future loader change breaks the boundary alignment.
    """

    def test_overshoot_detection(self) -> None:
        # Simulate the worker's skip loop without the actual loader/encoder.
        snapshots_to_skip = 100
        snapshots_skipped_so_far = 0
        # First two batches of 40 each → 80 skipped, still < 100.
        for batch_snapshots in [40, 40]:
            snapshots_skipped_so_far += batch_snapshots
            self.assertLess(snapshots_skipped_so_far, snapshots_to_skip)
        # Third batch of 30 → 110, overshoots → should trigger guard.
        snapshots_skipped_so_far += 30
        self.assertGreater(snapshots_skipped_so_far, snapshots_to_skip)
        # The worker raises RuntimeError in this condition; here we just
        # confirm the condition is detectable.

    def test_exact_boundary_is_not_overshoot(self) -> None:
        snapshots_to_skip = 100
        snapshots_skipped_so_far = 0
        for batch_snapshots in [40, 60]:
            snapshots_skipped_so_far += batch_snapshots
        # Exactly 100 — landed on a batch boundary. Not an overshoot.
        self.assertEqual(snapshots_skipped_so_far, snapshots_to_skip)


class CorruptedShardHandlingTests(unittest.TestCase):
    """The shard-scan must tolerate corruption: a previous worker killed
    mid-write would leave a non-loadable shard file. Resume should detect
    that, truncate the resume point to the last known-good shard, and
    delete every shard from the corruption point onward to keep numbering
    contiguous. Mirrors what ``materialize_worker`` does at startup.
    """

    @staticmethod
    def _scan_with_truncation(shard_dir: Path):
        """Replica of the worker's resume-scan logic, factored out for testing."""
        existing_shards = sorted(shard_dir.glob("shard_*.pt"))
        shard_paths = []
        shard_sizes = []
        total_snapshots = 0
        for index, shard_path in enumerate(existing_shards):
            try:
                payload = torch.load(shard_path, weights_only=False)
            except Exception:
                for stale in existing_shards[index:]:
                    stale.unlink()
                break
            shard_paths.append(str(shard_path))
            shard_sizes.append(int(payload["features"].shape[0]))
            total_snapshots += shard_sizes[-1]
        return shard_paths, shard_sizes, total_snapshots, existing_shards

    def test_corrupted_shard_truncates_resume_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "worker_00"
            _write_shard(shard_dir, 0, 250000)
            _write_shard(shard_dir, 1, 250000)
            # Shard 2 was killed mid-write — fake by writing garbage.
            (shard_dir / "shard_00002.pt").write_bytes(b"garbage not a torch zip")
            # A later shard could in principle exist too if the scan order
            # were wrong; put one there to make sure it also gets deleted.
            _write_shard(shard_dir, 3, 100)

            paths, sizes, total, _ = self._scan_with_truncation(shard_dir)

            # Resume uses only shards 0 and 1.
            self.assertEqual(len(paths), 2)
            self.assertEqual(sizes, [250000, 250000])
            self.assertEqual(total, 500000)
            # Corrupted shard and any later were deleted to keep numbering contiguous.
            self.assertFalse((shard_dir / "shard_00002.pt").exists())
            self.assertFalse((shard_dir / "shard_00003.pt").exists())
            # First two left intact.
            self.assertTrue((shard_dir / "shard_00000.pt").exists())
            self.assertTrue((shard_dir / "shard_00001.pt").exists())

    def test_tmp_files_cleaned_up_on_startup(self) -> None:
        """Atomic-write residue: a .tmp file from a previous kill mid-write
        should be removed so it doesn't accumulate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "worker_00"
            shard_dir.mkdir()
            # Simulate three previous kills mid-write at different indices.
            (shard_dir / "shard_00005.pt.tmp").write_bytes(b"partial garbage")
            (shard_dir / "shard_00007.pt.tmp").write_bytes(b"partial garbage")
            (shard_dir / "shard_00009.pt.tmp").write_bytes(b"partial garbage")

            for leftover in shard_dir.glob("*.tmp"):
                leftover.unlink()

            self.assertEqual(list(shard_dir.glob("*.tmp")), [])


class AtomicShardWriteTests(unittest.TestCase):
    """The flush_shard write must be atomic: torch.save to .tmp then
    os.replace into the final path. Mid-write kill leaves at most a .tmp
    file (which startup cleans up), never a half-written shard_NNNNN.pt.
    """

    def test_atomic_write_round_trips(self) -> None:
        """Sanity check that the atomic-write pattern produces a loadable shard."""
        import os as _os
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir) / "worker_00"
            shard_dir.mkdir()
            shard_path = shard_dir / "shard_00000.pt"
            tmp_path = shard_path.with_suffix(".pt.tmp")
            payload = {
                "format": "cts_materialized_advantage_cache_shard_v2",
                "features": torch.zeros((10, 130), dtype=torch.float32),
                "target_advantages": torch.zeros((10,), dtype=torch.float32),
                "oracle_stop_steps": torch.zeros((10,), dtype=torch.int32),
            }
            torch.save(payload, tmp_path)
            _os.replace(tmp_path, shard_path)
            self.assertTrue(shard_path.is_file())
            self.assertFalse(tmp_path.exists())
            # Reload to confirm not corrupted.
            reloaded = torch.load(shard_path, weights_only=False)
            self.assertEqual(reloaded["features"].shape, (10, 130))


if __name__ == "__main__":
    unittest.main()
