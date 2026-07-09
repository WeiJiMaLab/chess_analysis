"""TDD-lite regression test for plan.md Phase 2, Agent 2 (our_trees_continued).

Guards against a real bug caught during this investigation: ``pg_checkpoint_stops`` (formerly
misnamed ``pg_checkpoint_regret``) returns GREEDY STOP STEPS (small integers like 9-10), not
regret. An earlier version of ``evaluate_pg_checkpoint`` used that array directly as "regret"
(skipping the ``_regret_at(ev, stops)`` conversion), which produced a checkpoint "regret" of
~9.8 -- literally the mean stop step -- instead of the correct ~0.1-0.2 scale every other regret
number in this investigation lives at. A live SLURM smoke run (job 10842067) caught this via the
paired-diff CIs being absurdly large ([-9.8, -9.5]); this test catches the same class of bug
without needing a cluster job.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analysis.evaluate import _regret_at
from analysis.evaluate_our_trees_continued import _pg_features, pg_checkpoint_stops
from cts.data.preprocess_mc.oracle import predicted_stop_from_advantages


class _ConstantAdvantageModel:
    """Fake model whose ``predict_from_features`` always returns a FIXED advantage trace
    ``[+1, +1, ..., -1, -1, ...]`` (positive for the first ``stop_at`` steps, then negative) --
    just enough of ``MetaController``'s interface (``predict_from_features``) for
    ``pg_checkpoint_stops`` to call, without needing a real encoder/packed data."""

    def __init__(self, stop_at: int):
        self.stop_at = stop_at

    def predict_from_features(self, features: torch.Tensor):
        n = features.shape[0]
        adv = torch.where(torch.arange(n) < self.stop_at, torch.tensor(1.0), torch.tensor(-1.0))
        return adv, adv


def test_pg_checkpoint_stops_returns_step_indices_not_regret():
    """``pg_checkpoint_stops`` must return small integer STEP indices (bounded by episode
    length), never float regret-scale values -- the exact confusion the real bug hinged on."""
    d_embed = 4
    z_ep = np.zeros((10, d_embed), dtype=np.float32)
    episodes = [{"halt_rewards": [0.0] * 10}]
    model = _ConstantAdvantageModel(stop_at=3)
    stops = pg_checkpoint_stops(model, episodes, {0: z_ep}, [0], d_embed)
    assert stops.dtype.kind in "iu"
    assert stops.tolist() == [3]  # first index where the fake advantage trace goes <= 0


def test_regret_at_with_checkpoint_stops_stays_in_regret_scale_not_step_scale():
    """The composition ``_regret_at(ev, pg_checkpoint_stops(...))`` -- the fix -- must produce
    values on the SAME scale as the return curve (bounded by max(curve) - min(curve)), never the
    raw stop-step integers themselves (the bug: using stops directly AS "regret" silently produced
    numbers like 9.8, an order of magnitude off every other regret number in this investigation)."""
    d_embed = 4
    n_steps = 12
    z_ep = np.zeros((n_steps, d_embed), dtype=np.float32)
    curve = np.linspace(0.0, 1.0, n_steps)  # monotone increasing -> best stop is the LAST step
    model = _ConstantAdvantageModel(stop_at=n_steps)  # never crosses zero -> stops at last step
    stops = pg_checkpoint_stops(model, [{"halt_rewards": [0.0] * n_steps}], {0: z_ep}, [0], d_embed)
    regret = _regret_at([curve], stops)
    assert regret.shape == (1,)
    # Stopping at the last step on a monotone-increasing curve is OPTIMAL -> regret ~ 0, and in
    # particular nowhere near the raw stop-step value (n_steps - 1 = 11, the value the bug produced).
    assert regret[0] == pytest_approx(0.0)
    assert abs(regret[0]) < 1.0, "regret should be curve-scale (<1), not step-index-scale (~11)"


def pytest_approx(x, abs=1e-9):
    import pytest
    return pytest.approx(x, abs=abs)
