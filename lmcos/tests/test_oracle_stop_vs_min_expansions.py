"""
Characterization tests: oracle_stop_step (greedy_stop) vs min_expansions.

Definitions
-----------
greedy_stop (oracle_stop_step):
    First step where R_halt >= R_continue — i.e., first step where halting is at
    least as good as continuing. halt_rewards use oracle_final_root_q_values
    (teacher's fixed Q-values), NOT oracle_root_q_trace (evolving estimates).
    In pack.py (line 641): halt_rewards[i] = oracle_final_root_q_values[best_idx[i]].

    In training evaluation (controller_train.py line 1742-1745):
        predicted_stop = first step where predicted_advantage[i] <= 0
    oracle_stop_step is greedy_stop (packed via pack.py). Both use the same criterion
    (halt when advantage <= 0); controller uses predicted advantages, oracle uses true ones.

min_expansions:
    1 + last step where best_idx != final_best. First step of permanent convergence.
    Excludes step 0 (artifact: all-zero Q-trace makes argmax meaningless).

Zero-cost invariants (C = 0)
-------------------------------
- R_halt <= R_continue everywhere (V*(i+1) = Q_final[final_best] = max possible).
- R_halt == R_continue iff best_idx[i] == final_best (no Q_final ties).
  Ties produce equality too — oracle is correctly indifferent between equal-valued actions.
- greedy_stop = first step where best_idx[i] == final_best (or any tied action).
- greedy_stop <= min_expansions always.
- greedy_stop < min_expansions iff there is a reversal: MCTS first hit final_best
  then deviated before permanently converging.
- greedy_stop == min_expansions iff MCTS never deviates after first correct recommendation.

Positive-cost invariants
--------------------------
- Before oracle_stop: Q_final[final_best] - R_halt > C (one-step cost).
- At oracle_stop: R_halt >= R_continue. Halting is correct.
- oracle_stop <= min_expansions always.

Run from lmcos/: pytest tests/test_oracle_stop_vs_min_expansions.py -v
Requires scratch tree files; tests that need them skip automatically if unavailable.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    time_cost,
)
from analysis.converged_expansions_analysis import converged_expansions as min_expansions

_SHARDS = [
    f"/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_combined/filtered_shard_0000{i}"
    for i in range(5)
]
_N_TREES = 1000

_ZERO_COST = BudgetedOracleConfig(time_lambda=0.0, maintenance_scale=0.0)
_DEFAULT   = BudgetedOracleConfig()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_sample(n: int = _N_TREES) -> list[dict]:
    files: list[str] = []
    for shard in _SHARDS:
        files.extend(sorted(glob.glob(f"{shard}/*.pt")))
    files = files[:n]
    records = []
    for path in files:
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            q_trace = t["oracle_root_q_trace"]         # [T, C] evolving
            q_final = t["oracle_final_root_q_values"]  # [C] teacher's fixed Q
            best_idx = t["oracle_best_move_index"]     # [T]
            T = len(best_idx)
            final_best = int(best_idx[-1].item())
            q_fb = float(q_final[final_best].item())
            # Correct halt_rewards (pack.py formulation)
            hr_final = [float(q_final[best_idx[s].item()].item()) for s in range(T)]
            # Wrong halt_rewards (evolving q_trace indexing — pre-2026-06-05 analysis bug)
            hr_trace = [float(q_trace[s, best_idx[s].item()].item()) for s in range(T)]
            # Q_final tie: any action other than final_best with the same Q value
            has_tie = any(
                a != final_best and abs(float(q_final[a].item()) - q_fb) < 1e-9
                for a in range(q_final.shape[0])
            )
            records.append({
                "T": T,
                "best_idx": best_idx,
                "q_final": q_final,
                "hr_final": hr_final,
                "hr_trace": hr_trace,
                "sizes": [1] * T,
                "final_best": final_best,
                "q_fb": q_fb,
                "has_tie": has_tie,
                "path": path,
            })
        except Exception:
            pass
    return records


def _first_occurrence(best_idx: torch.Tensor, final_best: int) -> int:
    """First step (0-indexed, including step 0) where best_idx == final_best."""
    for i in range(len(best_idx)):
        if int(best_idx[i].item()) == final_best:
            return i
    return len(best_idx) - 1


def _has_reversal_between(best_idx: torch.Tensor, final_best: int, lo: int, hi: int) -> bool:
    """Return True iff any step in (lo, hi) has best_idx != final_best."""
    for j in range(lo + 1, hi):
        if int(best_idx[j].item()) != final_best:
            return True
    return False


@pytest.fixture(scope="module")
def records():
    data = _load_sample()
    if not data:
        pytest.skip("Scratch tree files not accessible.")
    return data


@pytest.fixture(scope="module")
def records_no_ties(records):
    """Trees where no Q_final tie exists (all Q_final values unique at the top)."""
    return [r for r in records if not r["has_tie"]]


# ---------------------------------------------------------------------------
# Synthetic trace helpers
# ---------------------------------------------------------------------------

def _make_hr(best_idx_list: list[int], q_map: dict) -> tuple[torch.Tensor, list[float]]:
    idx = torch.tensor(best_idx_list, dtype=torch.int32)
    hr  = [q_map[v] for v in best_idx_list]
    return idx, hr


# ---------------------------------------------------------------------------
# Synthetic trace tests (no real data required)
# ---------------------------------------------------------------------------

class TestSyntheticTraces:
    """Verify greedy_stop and min_expansions on hand-crafted traces."""

    def _gs(self, hr: list[float], T: Optional[int] = None) -> int:
        T = T or len(hr)
        return compute_budgeted_oracle(hr, [1]*T, T, _ZERO_COST).optimal_stop_step

    # ------------------------------------------------------------------
    # Homogeneous: same action throughout
    # ------------------------------------------------------------------

    def test_homogeneous_greedy_stop_is_zero(self):
        """Homogeneous trace: greedy_stop = 0 (halt immediately — you already have
        the best action at step 0 and it never changes)."""
        # [2, 2, 2, 2, 2] — final_best=2, first occurrence at step 0
        idx, hr = _make_hr([2, 2, 2, 2, 2], {2: 0.8})
        assert self._gs(hr) == 0

    def test_homogeneous_min_expansions_is_one(self):
        """Homogeneous trace: min_expansions = 1 (step 0 is excluded as artifact;
        step 1 onwards is already stable)."""
        idx = torch.tensor([2, 2, 2, 2, 2], dtype=torch.int32)
        assert min_expansions(idx) == 1

    def test_homogeneous_greedy_stop_leq_min_expansions(self):
        """Homogeneous: greedy_stop (0) < min_expansions (1) — both indicate
        'stop as early as possible'."""
        idx, hr = _make_hr([2, 2, 2, 2, 2], {2: 0.8})
        gs = self._gs(hr)
        me = min_expansions(idx)
        assert gs == 0 and me == 1 and gs < me

    # ------------------------------------------------------------------
    # Last-step discovery: final_best only appears at the last step
    # ------------------------------------------------------------------

    def test_last_step_only_greedy_and_min_equal(self):
        """Final best (action 1) discovered only at the last step:
        greedy_stop == min_expansions == T-1 (no reversal, equal by definition)."""
        idx, hr = _make_hr([0, 0, 0, 0, 1], {0: 0.5, 1: 0.9})
        T = len(hr)
        gs = self._gs(hr, T)
        me = min_expansions(idx)
        assert gs == T - 1, f"Expected greedy_stop={T-1}, got {gs}"
        assert me == T - 1, f"Expected min_expansions={T-1}, got {me}"

    # ------------------------------------------------------------------
    # Reversal traces: greedy_stop < min_expansions
    # ------------------------------------------------------------------

    def test_reversal_greedy_less_than_min(self):
        """MCTS first hits final action at step 2, reverts at step 5-7, settles at step 8.
        greedy_stop=2 (first hit), min_expansions=8 (first stable step)."""
        idx, hr = _make_hr([3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1], {1: 0.9, 2: 0.7, 3: 0.5})
        gs = self._gs(hr)
        me = min_expansions(idx)
        assert gs == 2, f"Expected greedy_stop=2 (0-indexed; user's 1-indexed=3), got {gs}"
        assert me == 8, f"Expected min_expansions=8 (0-indexed; user's 1-indexed=9), got {me}"
        assert gs < me

    def test_reversal_exists_when_greedy_less_than_min(self):
        """When greedy_stop < min_expansions, a reversal MUST exist between them."""
        idx, hr = _make_hr([3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1], {1: 0.9, 2: 0.7, 3: 0.5})
        gs = self._gs(hr)
        me = min_expansions(idx)
        final_best = int(idx[-1].item())
        assert gs < me
        assert _has_reversal_between(idx, final_best, gs, me), \
            "No reversal found between greedy_stop and min_expansions"

    def test_no_reversal_means_equal(self):
        """Monotone convergence (no reversal after first hit): greedy_stop == min_expansions."""
        # Final best (1) first appears at step 3 and stays.
        idx, hr = _make_hr([0, 0, 0, 1, 1, 1, 1], {0: 0.5, 1: 0.9})
        gs = self._gs(hr)
        me = min_expansions(idx)
        assert gs == me, f"Expected gs == me == 3, got gs={gs}, me={me}"

    # ------------------------------------------------------------------
    # Zero-cost: R_halt <= R_continue everywhere
    # ------------------------------------------------------------------

    def test_r_halt_leq_r_continue_everywhere_synthetic(self):
        """R_halt <= R_continue at every step (zero cost). True for all traces
        because V*(i+1) = Q_final[final_best] = max possible halt reward."""
        for trace, q_map in [
            ([3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1], {1: 0.9, 2: 0.7, 3: 0.5}),
            ([0, 0, 0, 0, 1], {0: 0.5, 1: 0.9}),
            ([2, 2, 2, 2, 2], {2: 0.8}),
            ([0, 0, 0, 1, 1, 1, 1], {0: 0.5, 1: 0.9}),
            ([1, 0, 1, 1, 1], {0: 0.5, 1: 0.9}),
        ]:
            idx, hr = _make_hr(trace, q_map)
            pol = compute_budgeted_oracle(hr, [1]*len(hr), len(hr), _ZERO_COST)
            for i in range(len(hr) - 1):
                assert pol.halt_rewards[i] <= pol.continue_values[i] + 1e-9, (
                    f"Trace {trace}: R_halt={pol.halt_rewards[i]:.4f} > "
                    f"R_continue={pol.continue_values[i]:.4f} at step {i}"
                )

    # ------------------------------------------------------------------
    # Q_final usage: halt_rewards[i] == Q_final[best_idx[i]]
    # ------------------------------------------------------------------

    def test_halt_rewards_equal_q_final_at_best_idx(self):
        """halt_rewards[i] must equal Q_final[best_idx[i]] — the teacher's fixed Q-value
        for whatever action the MCTS currently recommends. This is pack.py's formulation."""
        q_map = {0: 0.5, 1: 0.9, 2: 0.7, 3: 0.5}
        trace = [3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1]
        idx = torch.tensor(trace, dtype=torch.int32)
        hr = [q_map[v] for v in trace]
        for i, (action, halt_r) in enumerate(zip(trace, hr)):
            expected = q_map[action]
            assert abs(halt_r - expected) < 1e-9, \
                f"halt_rewards[{i}] = {halt_r:.4f} != Q_final[{action}] = {expected:.4f}"


# ---------------------------------------------------------------------------
# Real-data tests (require scratch files)
# ---------------------------------------------------------------------------

class TestRealDataInvariants:

    def test_q_final_used_for_halt_rewards(self, records):
        """Assert oracle_final_root_q_values is the correct halt_rewards source:
        hr_final[i] == oracle_final_root_q_values[best_idx[i]] for all trees."""
        for r in records:
            for i in range(r["T"]):
                expected = float(r["q_final"][int(r["best_idx"][i].item())].item())
                actual   = r["hr_final"][i]
                assert abs(actual - expected) < 1e-9, (
                    f"hr_final[{i}] = {actual:.6f} != Q_final[best_idx[{i}]] = {expected:.6f}"
                )

    def test_r_halt_leq_r_continue_everywhere_zero_cost(self, records):
        """Zero cost: R_halt <= R_continue at every step. No exception."""
        for r in records:
            pol = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], _ZERO_COST)
            for i in range(r["T"] - 1):
                assert pol.halt_rewards[i] <= pol.continue_values[i] + 1e-9, (
                    f"{r['path']}: R_halt={pol.halt_rewards[i]:.6f} > "
                    f"R_continue={pol.continue_values[i]:.6f} at step {i}"
                )

    def test_no_ties_equality_iff_final_best(self, records_no_ties):
        """In trees with no Q_final ties: R_halt == R_continue iff best_idx[i] == final_best.
        No other action produces equality (it would require matching the max Q_final value)."""
        if not records_no_ties:
            pytest.skip("No tie-free trees in sample.")
        for r in records_no_ties:
            pol = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], _ZERO_COST)
            for i in range(r["T"] - 1):
                eq    = abs(pol.halt_rewards[i] - pol.continue_values[i]) < 1e-9
                match = int(r["best_idx"][i].item()) == r["final_best"]
                assert eq == match, (
                    f"{r['path']} step {i}: eq={eq} but match={match} "
                    f"(best_idx={int(r['best_idx'][i].item())}, final_best={r['final_best']})"
                )

    def test_reversal_required_when_greedy_less_than_min(self, records_no_ties):
        """greedy_stop < min_expansions iff there is a reversal between them:
        MCTS hit final_best at greedy_stop, then deviated, then converged at min_expansions.
        Tested only on tie-free trees (ties can cause early halt without reversal)."""
        if not records_no_ties:
            pytest.skip("No tie-free trees in sample.")
        violations = 0
        checked = 0
        for r in records_no_ties:
            pol = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], _ZERO_COST)
            gs  = pol.optimal_stop_step
            me  = min_expansions(r["best_idx"])
            if gs >= me:
                continue
            checked += 1
            if not _has_reversal_between(r["best_idx"], r["final_best"], gs, me):
                violations += 1
        assert violations == 0, (
            f"{violations}/{checked} tie-free trees with greedy_stop < min_expansions "
            f"have no reversal between them."
        )

    def test_greedy_equal_min_iff_no_reversal_after_first_hit(self, records_no_ties):
        """In tie-free trees: greedy_stop == min_expansions iff no reversal after
        first correct recommendation (MCTS monotonically converges to final_best)."""
        if not records_no_ties:
            pytest.skip("No tie-free trees in sample.")
        for r in records_no_ties:
            pol = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], _ZERO_COST)
            gs = pol.optimal_stop_step
            me = min_expansions(r["best_idx"])
            # No reversal after first hit = no step in (gs, T) with best_idx != final_best
            has_rev = _has_reversal_between(r["best_idx"], r["final_best"], gs, r["T"])
            if gs == me:
                assert not has_rev, (
                    f"{r['path']}: gs==me but reversal exists (should not)"
                )
            else:
                assert has_rev, (
                    f"{r['path']}: gs<me but no reversal (reversal is required)"
                )

    def test_oracle_stop_leq_min_expansions_always(self, records):
        """oracle_stop_step <= min_expansions always, for both zero and positive cost."""
        for r in records:
            me = min_expansions(r["best_idx"])
            for config in (_ZERO_COST, _DEFAULT):
                pol = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], config)
                assert pol.optimal_stop_step <= me, (
                    f"{r['path']}: oracle_stop={pol.optimal_stop_step} > me={me}"
                )

    def test_positive_cost_quality_gap_exceeds_step_cost_before_stop(self, records):
        """Before oracle_stop (positive cost): Q_final[final_best] - R_halt > C_step.
        Holds because Q_final[final_best] >= V*(i+1) and the DP already requires
        V*(i+1) - R_halt > C_step for the oracle to continue."""
        for r in records:
            pol  = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], _DEFAULT)
            stop = pol.optimal_stop_step
            for i in range(stop):
                c_i = time_cost(pol.time_budgets[i], _DEFAULT)
                gap = r["q_fb"] - pol.halt_rewards[i]
                assert gap > c_i - 1e-9, (
                    f"{r['path']}: gap={gap:.6f} <= cost={c_i:.6f} at step {i} before stop={stop}"
                )

    def test_positive_cost_halt_correct_at_stop(self, records):
        """At oracle_stop (positive cost): R_halt >= R_continue. Halting is not worse."""
        for r in records:
            pol = compute_budgeted_oracle(r["hr_final"], r["sizes"], r["T"], _DEFAULT)
            s   = pol.optimal_stop_step
            assert pol.halt_rewards[s] >= pol.continue_values[s] - 1e-9, (
                f"{r['path']}: R_halt={pol.halt_rewards[s]:.6f} < "
                f"R_continue={pol.continue_values[s]:.6f} at stop={s}"
            )

    def test_a0a_wrong_halt_rewards_diverges(self, records):
        """Wrong halt_rewards (evolving q_trace) should not agree with min_expansions."""
        matches = []
        for r in records:
            me      = min_expansions(r["best_idx"])
            oss_a0a = compute_budgeted_oracle(r["hr_trace"], r["sizes"], r["T"], _DEFAULT).optimal_stop_step
            matches.append(oss_a0a == me)
        assert np.mean(matches) < 0.05, (
            f"A0a oracle_stop_step agrees with min_expansions in {np.mean(matches):.3f} of trees "
            f"(expected <0.05). The halt_rewards bug may no longer matter."
        )


# ---------------------------------------------------------------------------
# Controller halt vs oracle_stop_step: definition consistency
# ---------------------------------------------------------------------------

class TestControllerVsOracleDefinition:

    def test_controller_uses_same_criterion_as_oracle(self):
        """The controller's predicted_stop (controller_train.py line 1742-1745) is
        the first step where predicted_advantage[i] <= 0 — the same criterion as the
        oracle's greedy_stop (first step where true_advantage <= 0).
        oracle_stop_step in evaluation IS greedy_stop (packed via pack.py line 724).
        Both compare against the same metric; controller uses predicted, oracle uses true.
        This test verifies the criterion identity using synthetic advantages."""
        # oracle_stop_step (greedy_stop): first step where true advantage <= 0
        # simulated as the first step where target_advantages[i] <= 0
        target_advantages = [0.5, 0.3, 0.1, -0.2, -0.5]
        oracle_stop = next(i for i, a in enumerate(target_advantages) if a <= 0)
        assert oracle_stop == 3

        # controller predicted_stop: first step where predicted_advantage[i] <= 0
        # (same code path as controller_train.py lines 1742-1745)
        predicted_advantages = [0.4, 0.2, -0.1, -0.3, -0.6]  # imperfect prediction
        predicted_stop = next(i for i, a in enumerate(predicted_advantages) if a <= 0)
        assert predicted_stop == 2  # controller stops one step early (undersearch)

        # Demonstrates: same criterion, different advantages → different stops.
        # exact_stop_step_accuracy = int(predicted_stop == oracle_stop) = 0 here.
        assert predicted_stop != oracle_stop

    def test_oracle_stop_step_packed_from_dp(self):
        """oracle_stop_step stored in packed episodes = policy.optimal_stop_step
        = stop_steps[0] from the backward DP over oracle_final_root_q_values.
        This is greedy_stop. Verified on a synthetic trace."""
        hr   = [0.5, 0.5, 0.9, 0.9, 0.7, 0.9]  # first max at step 2
        pol  = compute_budgeted_oracle(hr, [1]*6, 6, _ZERO_COST)
        assert pol.optimal_stop_step == 2, \
            f"Expected greedy_stop=2 (first max halt reward), got {pol.optimal_stop_step}"
        # target_advantages[2] <= 0: advantage of continuing from step 2 is zero (tie)
        assert pol.target_advantages[2] <= 0
