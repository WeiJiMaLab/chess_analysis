"""Unified stop-policy ``Readout`` family for the meta-controller D0 comparison.

Every readout shares ONE decision rule (``Readout.stop_step``): walk the
episode's per-step snapshots, ask the subclass for a predicted advantage
``A`` at each step, and STOP at the first step where ``A <= 0`` (continue iff
``A > 0``), falling back to the last step if ``A`` never crosses zero. This is
the same greedy rule the controller eval uses
(:func:`cts.data.preprocess_mc.oracle.predicted_stop_from_advantages`), so all
five D0 tiers are compared on identical machinery and only the *advantage
signal* differs between them.

The D0 ladder (all trained/selected on **regret directly**, never the
surrogate MSE+sign-BCE — see ``mc_pipeline.md`` §10b):

  1. :class:`AlwaysStop`        -- A<=0 everywhere       (stop@0, no parameters)
  2. :class:`NeverStop`         -- A>0 everywhere        (full budget, no parameters)
  3. :class:`FractionStop`      -- A>0 iff N_t < theta*B (one scalar theta)
  4. :class:`StatsReadout`      -- MLP on [height, width, n_nodes, B]
  5. :class:`GnnMetaController` -- MLP on [z, B]

Tiers 4 and 5 share an IDENTICAL head architecture (:func:`build_advantage_head`)
so the 4-vs-5 gap isolates *representation* (hand-crafted stats vs learned GNN
embedding), not capacity.

Per-step features are supplied as a ``[num_steps, feature_dim]`` tensor by the
caller (the eval harness derives ``[height, width, n_nodes]`` from the packed
tree payload and ``z`` from the materialized encoder cache). Parameter fitting
on regret lives in the eval harness; this module is the model zoo + the shared
decision rule only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_advantage_head(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    """Shared advantage-MLP factory: ``hidden_layers`` ReLU layers + scalar output.

    Both ``StatsReadout`` (tier 4) and ``GnnMetaController`` (tier 5) build their
    head through this single function so a 4-vs-5 comparison isolates the input
    representation rather than head capacity. Mirrors the architecture of
    ``cts.models.mc._build_advantage_head``.
    """
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


def stop_step_from_advantages(advantages: torch.Tensor) -> int:
    """First step with ``advantage <= 0`` (continue iff ``A > 0``), else the last step.

    The single decision rule shared by every ``Readout``. Identical to
    ``cts.data.preprocess_mc.oracle.predicted_stop_from_advantages`` but operates
    on a tensor.
    """
    num_steps = int(advantages.shape[0])
    for step_index in range(num_steps):
        if float(advantages[step_index]) <= 0.0:
            return step_index
    return num_steps - 1


class Readout(nn.Module):
    """Base class: a per-step advantage predictor + the shared stop rule.

    Subclasses implement :meth:`advantages`, returning a ``[num_steps]`` tensor
    of predicted advantages for one episode. ``features`` is a
    ``[num_steps, feature_dim]`` per-step feature matrix whose column layout is
    fixed by the eval harness (see module docstring); subclasses that need no
    features ignore it. ``stop_step`` applies the universal continue-iff-``A>0``
    rule to those advantages.
    """

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        """Return predicted per-step advantages ``[num_steps]`` for one episode."""
        raise NotImplementedError

    def stop_step(self, features: torch.Tensor, starting_budget: int) -> int:
        """Apply the shared continue-iff-``A>0`` rule to the predicted advantages."""
        return stop_step_from_advantages(self.advantages(features, starting_budget))


class AlwaysStop(Readout):
    """Tier 1: halt at step 0 (advantage <= 0 everywhere). No parameters."""

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        return torch.full((num_steps,), -1.0)


class NeverStop(Readout):
    """Tier 2: search to the last step (advantage > 0 everywhere). No parameters."""

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        return torch.full((num_steps,), 1.0)


class FractionStop(Readout):
    """Tier 3: continue iff ``N_t < theta * B`` — the one-scalar budget-only baseline.

    Encodes the Fraction-of-Budget rule as an advantage: ``A = theta*B - N_t``,
    so ``A > 0`` (continue) exactly while the tree size ``N_t`` is below the
    fraction ``theta`` of the starting budget ``B``. ``theta`` is the single
    learnable parameter, fit on regret directly (grid search or GD).

    The feature layout for this readout exposes ``n_nodes`` (``N_t``) at column
    ``n_nodes_col``.
    """

    def __init__(self, theta: float = 0.5, n_nodes_col: int = 0) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.n_nodes_col = int(n_nodes_col)

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        n_nodes = features[:, self.n_nodes_col]
        return self.theta * float(starting_budget) - n_nodes


class StatsReadout(Readout):
    """Tier 4: MLP advantage head on hand-crafted ``[height, width, n_nodes, B]``.

    The feature matrix passed in is ``[num_steps, 3]`` = ``[height, width,
    n_nodes]`` per step; ``B`` (the scalar starting budget) is broadcast and
    concatenated here so the head sees the same budget signal the
    ``FractionStop`` baseline keys off. Uses the shared
    :func:`build_advantage_head` so the architecture matches tier 5.
    """

    NUM_STATS = 3  # [height, width, n_nodes]

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2) -> None:
        super().__init__()
        self.head = build_advantage_head(self.NUM_STATS + 1, hidden_dim, hidden_layers)

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        budget_col = torch.full((num_steps, 1), float(starting_budget), dtype=features.dtype)
        head_input = torch.cat([features[:, : self.NUM_STATS], budget_col], dim=-1)
        return self.head(head_input).squeeze(-1)


class ValueGainReadout(Readout):
    """Train-once / sweep-many head: MLP regressing the COST-FREE value gain.

    Unlike the cost-baked tiers above, this head predicts ``value_gain(t)`` (the
    expected move-quality gain from continuing, in pure reward units, >= 0) from
    the same hand-crafted ``[height, width, n_nodes, B]`` features as
    :class:`StatsReadout`.  Because the target carries NO cost, a single trained
    instance is reused across every cost regime: the decision-time cost is
    subtracted by the eval harness
    (``cts.data.preprocess_mc.value_gain.swept_advantage_trace``), NOT baked into
    the head.

    ``advantages`` returns the raw cost-free predicted gain (clamped >= 0, since
    the target is non-negative); the harness turns it into a regime advantage by
    subtracting the analytic cost before applying the shared stop rule.  Calling
    ``stop_step`` directly therefore corresponds to the zero-cost / search-
    forever regime (gain > 0 => keep going), a useful sanity limit.
    """

    NUM_STATS = 3  # [height, width, n_nodes]

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2) -> None:
        super().__init__()
        self.head = build_advantage_head(self.NUM_STATS + 1, hidden_dim, hidden_layers)

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        budget_col = torch.full((num_steps, 1), float(starting_budget), dtype=features.dtype)
        head_input = torch.cat([features[:, : self.NUM_STATS], budget_col], dim=-1)
        return self.head(head_input).squeeze(-1).clamp_min(0.0)


class HaltCurveReadout(Readout):
    """Train-once / sweep-many head: MLP regressing the COST-FREE per-step halt curve.

    The production trajectory-prediction head.  It predicts ``halt_reward(t)`` at
    each step from the same ``[height, width, n_nodes, B]`` features, producing a
    per-step VALUE CURVE rather than a single collapsed gain scalar.  At eval the
    harness runs the EXACT oracle backward-induction over this predicted curve
    with the regime's analytic per-step cost
    (``cts.data.preprocess_mc.value_gain.dp_stop_over_curve``), recovering the
    true OSS for any lambda from one trained model -- not greedy, not horizon-
    mismatched.

    ``advantages`` here returns the raw predicted halt-reward level (the curve);
    it is NOT an advantage and is not meant for the shared greedy ``stop_step``
    -- the harness consumes the curve via the DP instead.
    """

    NUM_STATS = 3  # [height, width, n_nodes]

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 2) -> None:
        super().__init__()
        self.head = build_advantage_head(self.NUM_STATS + 1, hidden_dim, hidden_layers)

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        budget_col = torch.full((num_steps, 1), float(starting_budget), dtype=features.dtype)
        head_input = torch.cat([features[:, : self.NUM_STATS], budget_col], dim=-1)
        return self.head(head_input).squeeze(-1)


class GnnMetaController(Readout):
    """Tier 5: MLP advantage head on the learned GNN root embedding ``[z, B]``.

    The feature matrix is ``[num_steps, d_embed]`` = the per-step root embedding
    ``z``; ``B`` is broadcast and concatenated, mirroring ``StatsReadout``. Built
    via the SAME :func:`build_advantage_head` so a 4-vs-5 comparison isolates the
    representation (learned ``z`` vs hand-crafted stats), not head capacity.
    """

    def __init__(self, d_embed: int, hidden_dim: int = 64, hidden_layers: int = 2) -> None:
        super().__init__()
        self.d_embed = int(d_embed)
        self.head = build_advantage_head(self.d_embed + 1, hidden_dim, hidden_layers)

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        budget_col = torch.full((num_steps, 1), float(starting_budget), dtype=features.dtype)
        head_input = torch.cat([features[:, : self.d_embed], budget_col], dim=-1)
        return self.head(head_input).squeeze(-1)
