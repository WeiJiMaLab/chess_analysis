"""Every readout shares one decision rule (``Readout.stop_step``): stop at the
first step where predicted advantage ``A <= 0``, falling back to the last step
if ``A`` never crosses zero."""

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


class StatsReadout(Readout):
    """Tier 3: MLP advantage head on hand-crafted ``[height, width, n_nodes, B]``.

    The feature matrix passed in is ``[num_steps, 3]`` = ``[height, width,
    n_nodes]`` per step; ``B`` (the scalar starting budget) is broadcast and
    concatenated here. Uses the shared :func:`build_advantage_head` so the
    architecture matches tier 4.
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
    """Tier 4: MLP advantage head on the learned GNN root embedding ``[z, B]``.

    The feature matrix is ``[num_steps, d_embed]`` = the per-step root embedding
    ``z``; ``B`` is broadcast and concatenated, mirroring ``StatsReadout``. Built
    via the SAME :func:`build_advantage_head` so a 3-vs-4 comparison isolates the
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
