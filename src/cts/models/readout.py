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


class AlwaysStop(Readout):
    """Tier 1: halt at step 0 (advantage <= 0 everywhere). No parameters."""

    def advantages(self, features: torch.Tensor, starting_budget: int) -> torch.Tensor:
        num_steps = int(features.shape[0])
        return torch.full((num_steps,), -1.0)


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
