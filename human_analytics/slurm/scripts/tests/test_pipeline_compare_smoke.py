"""
Regression guard: legacy vs new pipeline agree on ``processed_moves_nonzero`` cohort.

Requires cluster data paths and is **skipped** unless ``RUN_PIPELINE_COMPARE=1``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest


@unittest.skipUnless(
    os.environ.get("RUN_PIPELINE_COMPARE") == "1",
    "set RUN_PIPELINE_COMPARE=1 to run (uses lichess.db + raw parquets; ~minutes)",
)
class TestPipelineCompareSmoke(unittest.TestCase):
    def test_legacy_new_nonzero_games_match_short_window(self) -> None:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        scratch = os.environ.get(
            "PIPELINE_COMPARE_SCRATCH",
            "/scratch/gpfs/GRIFFITHS/hl4291/tmp/pipeline_smoke_unittest",
        )
        cmd = [
            sys.executable,
            os.path.join(repo, "src", "slurm", "scripts", "tests", "compare_legacy_new_pipeline_smoke.py"),
            "--clean",
            "--scratch-base",
            scratch,
            "--start-date",
            "2023-10-01",
            "--end-date",
            "2023-10-05",
            "--threads",
            "4",
            "--memory-limit",
            "12GB",
        ]
        env = {**os.environ, "PYTHONPATH": os.path.join(repo, "src")}
        p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=7200)
        self.assertEqual(p.returncode, 0, msg=p.stdout + "\n" + p.stderr)
        self.assertIn("delta: +0", p.stdout)
        self.assertIn("Gids in NEW nonzero only (not in legacy): 0", p.stdout)


if __name__ == "__main__":
    unittest.main()
