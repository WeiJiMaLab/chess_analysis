"""Tests for the D0 ``Readout`` family and the regret-direct selection.

Cover, on synthetic per-step features (no generated trees needed):

  - the shared decision rule (continue iff A>0; STOP at first A<=0, else last);
  - AlwaysStop / NeverStop stop at step 0 / the last step regardless of input;
  - FractionStop(theta) stops at the first step with N_t >= theta*B;
  - StatsReadout and GnnMetaController share an IDENTICAL head architecture
    (same parameter shapes), so a 4-vs-5 comparison isolates representation;
  - the regret-direct fit (``_fit_fraction`` / ``_fit_stats_readout``) selects
    on regret, and a nested model (Fraction) never beats the oracle.

Run from lmcos/: pytest tests/test_readout.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.readout import (
    AlwaysStop,
    FractionStop,
    GnnMetaController,
    NeverStop,
    StatsReadout,
    stop_step_from_advantages,
)
from src.data.preprocess_mc.oracle import BudgetedOracleConfig
from analysis._budgeted.alt_models_eval import (
    _derive_step_stats,
    _fit_fraction,
    _fit_stats_readout,
    _mean_regret,
    _stats_rule,
    _fraction_rule,
)


# --- shared decision rule ---------------------------------------------------
def test_stop_rule_first_nonpositive():
    assert stop_step_from_advantages(torch.tensor([1.0, 0.5, -0.1, 0.9])) == 2


def test_stop_rule_zero_counts_as_stop():
    assert stop_step_from_advantages(torch.tensor([1.0, 0.0, 1.0])) == 1


def test_stop_rule_never_crosses_returns_last():
    assert stop_step_from_advantages(torch.tensor([1.0, 2.0, 3.0])) == 2


# --- trivial readouts -------------------------------------------------------
def test_always_stop_is_zero():
    feats = torch.zeros(5, 3)
    assert AlwaysStop().stop_step(feats, starting_budget=10) == 0


def test_never_stop_is_last():
    feats = torch.zeros(5, 3)
    assert NeverStop().stop_step(feats, starting_budget=10) == 4


# --- FractionStop -----------------------------------------------------------
def test_fraction_stop_threshold():
    # n_nodes per step in column 0; theta*B = 0.5 * 10 = 5 -> stop when N_t >= 5.
    feats = torch.tensor([[1.0], [3.0], [5.0], [8.0]])
    model = FractionStop(theta=0.5, n_nodes_col=0)
    # advantages = 5 - N_t = [4, 2, 0, -3] -> first <=0 at index 2.
    assert model.stop_step(feats, starting_budget=10) == 2


def test_fraction_stop_continues_when_below_threshold():
    feats = torch.tensor([[1.0], [2.0], [3.0]])
    model = FractionStop(theta=0.9, n_nodes_col=0)  # theta*B=9, never reached
    assert model.stop_step(feats, starting_budget=10) == 2


# --- StatsReadout / GnnMetaController: identical head architecture -----------
def test_stats_and_gnn_share_head_architecture():
    stats = StatsReadout(hidden_dim=64, hidden_layers=2)
    # GNN input dim = stats input dim (3 stats + B == d_embed + B) => identical head.
    gnn = GnnMetaController(d_embed=StatsReadout.NUM_STATS, hidden_dim=64, hidden_layers=2)
    stats_shapes = [tuple(p.shape) for p in stats.head.parameters()]
    gnn_shapes = [tuple(p.shape) for p in gnn.head.parameters()]
    assert stats_shapes == gnn_shapes


def test_stats_readout_runs():
    feats = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 6.0]])
    adv = StatsReadout().advantages(feats, starting_budget=8)
    assert adv.shape == (2,)


def test_gnn_readout_runs():
    feats = torch.randn(4, 16)
    adv = GnnMetaController(d_embed=16).advantages(feats, starting_budget=12)
    assert adv.shape == (4,)


# --- height/width derivation ------------------------------------------------
def test_derive_step_stats():
    # A 3-node prefix then a 5-node prefix; depths root=0, then children at 1, 2.
    traj_depth = torch.tensor([0, 1, 1, 2, 2])
    heights, widths = _derive_step_stats(traj_depth, node_cutoffs=[3, 5])
    assert heights == [1, 2]      # max depth among first 3 nodes is 1; among 5 is 2
    assert widths == [2, 2]       # two nodes at depth 1; two at depth 2


# --- regret-direct selection ------------------------------------------------
def _synthetic_episodes() -> list[dict]:
    """Episodes whose halt_rewards peak mid-trajectory so stopping has a clear optimum."""
    episodes = []
    for budget in (6, 8, 10):
        n = budget
        # halt reward rises then is flat; optimum is to stop where it plateaus.
        halt = [min(0.1 * s, 0.5) for s in range(n)]
        episodes.append({
            "halt_rewards": halt,
            "tree_sizes": list(range(1, n + 1)),
            "heights": [1 + s // 3 for s in range(n)],
            "widths": [1 + s for s in range(n)],
            "time_budgets": list(range(budget, budget - n, -1)),
            "oracle_stop_step": 5,
            "oracle_value": 0.5,
            "starting_budget": budget,
        })
    return episodes


def test_fit_fraction_selects_on_regret():
    config = BudgetedOracleConfig()
    episodes = _synthetic_episodes()
    best_f, curve = _fit_fraction(episodes, config)
    # best_f minimizes mean regret over the grid -> no other grid point is lower.
    best_regret = _mean_regret(episodes, config, _fraction_rule(best_f))
    assert all(best_regret <= r + 1e-9 for _, r in curve)


def test_fitted_fraction_never_beats_oracle():
    config = BudgetedOracleConfig()
    episodes = _synthetic_episodes()
    best_f, _ = _fit_fraction(episodes, config)
    assert _mean_regret(episodes, config, _fraction_rule(best_f)) >= -1e-9


def test_fit_stats_readout_threshold_tuned_on_regret():
    config = BudgetedOracleConfig()
    episodes = _synthetic_episodes()
    model, tau = _fit_stats_readout(episodes, config, epochs=5)
    # The deployable rule produces valid in-range stop steps and non-negative regret.
    rule = _stats_rule(model)
    for ep in episodes:
        stop = rule(ep)
        assert 0 <= stop <= len(ep["halt_rewards"]) - 1
    assert _mean_regret(episodes, config, rule) >= -1e-9
