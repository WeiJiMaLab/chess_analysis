"""
Tests for converged_expansions (first step at which oracle_best_move_index agrees
with the final verdict and never deviates).

Run from lmcos/: pytest tests/test_converged_expansions.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.converged_expansions_analysis import converged_expansions


class TestConvergedExpansions:
    """
    converged_expansions(t) = first step t >= 1 at which best_idx[t] == final_best
                        AND best_idx[s] == final_best for all s > t.

    Step 0 is excluded because oracle_root_q_trace[0] is all-zero (no
    evaluations yet), making argmax = 0 an artifact of initialisation.
    """

    def _t(self, lst: list[int]) -> torch.Tensor:
        return torch.tensor(lst, dtype=torch.int32)

    # -------------------------------------------------------------------
    # Basic correctness
    # -------------------------------------------------------------------

    def test_never_changes_returns_one(self):
        """Best move is correct from step 1 and never deviates → converged_expansions = 1."""
        # step 0 = artifact (0), step 1..4 = final best (5)
        idx = self._t([0, 5, 5, 5, 5])
        assert converged_expansions(idx) == 1

    def test_single_late_flip(self):
        """Oracle settles at step 3, stays there → converged_expansions = 3."""
        # steps 1-2 wrong, step 3 onwards correct (final=5)
        idx = self._t([0, 3, 3, 5, 5, 5])
        assert converged_expansions(idx) == 3

    def test_oscillation_then_settle(self):
        """Oracle flip-flops before settling at step 4."""
        # final = 5; steps 1,2=5, step 3 reverts to 3, step 4+ = 5
        idx = self._t([0, 5, 5, 3, 5, 5, 5])
        assert converged_expansions(idx) == 4

    def test_settles_at_last_step(self):
        """Oracle changes its mind until the very last step."""
        # final = 5; wrong at all steps except the last
        idx = self._t([0, 3, 3, 3, 3, 5])
        assert converged_expansions(idx) == 5

    def test_only_one_step_after_init(self):
        """Budget = 1: only step 0 (artifact) and step 1 (first real eval)."""
        idx = self._t([0, 7])
        assert converged_expansions(idx) == 1

    # -------------------------------------------------------------------
    # Step-0 artifact handling
    # -------------------------------------------------------------------

    def test_step0_artifact_ignored_when_matches_final(self):
        """Even if step-0 index happens to equal final best, we count from step 1."""
        # step 0 = 5 (coincidence — argmax of zeros could be 5 in some impl),
        # but final = 5 from step 1 anyway
        idx = self._t([5, 5, 5, 5])
        assert converged_expansions(idx) == 1

    def test_step0_artifact_ignored_when_differs_from_final(self):
        """Step 0 = 0 (typical artifact); final = 7; settle at step 2."""
        idx = self._t([0, 3, 7, 7, 7])
        assert converged_expansions(idx) == 2

    # -------------------------------------------------------------------
    # Range / properties
    # -------------------------------------------------------------------

    def test_result_in_valid_range(self):
        """converged_expansions must be >= 1 and <= T-1 (step-1 to last step)."""
        for seed in range(20):
            torch.manual_seed(seed)
            T = torch.randint(3, 20, (1,)).item()
            idx = torch.randint(0, 5, (T,), dtype=torch.int32)
            result = converged_expansions(idx)
            assert 1 <= result <= T - 1, f"Out of range: {result} for T={T}"

    def test_deterministic(self):
        """Same input → same output."""
        idx = self._t([0, 3, 5, 3, 5, 5])
        assert converged_expansions(idx) == converged_expansions(idx)

    # -------------------------------------------------------------------
    # Failure mode: wrong definition
    # -------------------------------------------------------------------

    def test_not_first_time_seen(self):
        """
        converged_expansions is NOT the first time we see the final best move —
        it's the LAST time we see any other move, + 1.
        Oracle sees 5 at step 1 but reverts to 3 at step 3 before settling at 5.
        """
        idx = self._t([0, 5, 5, 3, 5, 5])
        # First time we see 5: step 1. But it reverts at step 3.
        # Correct answer: step 4 (first stable convergence)
        result = converged_expansions(idx)
        assert result == 4, f"Should be 4, got {result}"
        assert result != 1, "Must not return first-seen; that ignores subsequent deviations"
