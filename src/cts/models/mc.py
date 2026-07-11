"""The canonical on-disk feature layout ``[z_t, N_t, T_t]`` is unchanged; the model
slices to the ``controller_inputs``-configured subset at the boundary."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from cts.core.tensorizer import TreeBatch
from cts.models.gnn import TreeEncoderOutput, TreeEncoder


# Canonical input names in the canonical feature layout (and the only allowed
# values for ``controller_inputs``). Order matters: it defines the column
# layout of the cached feature tensor (``z_t`` first, then ``N_t``, then
# ``T_t``).
CONTROLLER_INPUT_NAMES: Tuple[str, ...] = ("z_t", "N_t", "T_t")



def validate_controller_inputs(controller_inputs: Sequence[str]) -> Tuple[str, ...]:
    """Normalize + validate ``controller_inputs``; reject unknown names, empty, dups."""
    names = tuple(controller_inputs)
    if not names:
        raise ValueError("controller_inputs must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"controller_inputs must be unique; got {names!r}.")
    for name in names:
        if name not in CONTROLLER_INPUT_NAMES:
            raise ValueError(
                f"controller_inputs entry {name!r} not in {CONTROLLER_INPUT_NAMES!r}."
            )
    return names


def _build_advantage_head(
    *,
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    device: torch.device,
) -> nn.Sequential:
    """Build the single-head advantage MLP: ``hidden_layers`` ReLU layers + scalar output."""
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim, device=device))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1, device=device))
    return nn.Sequential(*layers)


class MetaController(nn.Module):
    """Encoder + advantage MLP head over a configurable subset of ``[z_t, N_t, T_t]``.

    Wraps a ``TreeEncoder`` and a small ReLU MLP. The MLP's ``input_dim`` is
    determined by ``controller_inputs``: which of the canonical features
    (``z_t`` = encoder root embedding, ``N_t`` = current tree size, ``T_t`` =
    remaining time budget) the head actually consumes. The canonical feature
    layout passed in (concatenation of all three in that order) is unchanged;
    the model slices to the configured subset at the head boundary, so caching
    + storage of the full feature tensor doesn't depend on the controller
    configuration.

    Optionally exposes a second linear head on the same backbone for the
    auxiliary sign BCE loss (``--separate-sign-head``); by default the same
    logit is reused for both targets.
    """

    def __init__(
        self,
        *,
        k: int,
        node_feat: int,
        device: str,
        node_embed_hidden: int,
        d_embed: int,
        d_message: int,
        n_heads: int,
        d_att: int,
        hidden_dim: int,
        hidden_layers: int,
        separate_sign_head: bool = False,
        controller_inputs: Sequence[str] = CONTROLLER_INPUT_NAMES,
    ) -> None:
        """
        Args:
            k, node_feat, device, node_embed_hidden, d_embed, d_message,
            n_heads, d_att: forwarded to ``TreeEncoder``.
            hidden_dim: hidden width of the advantage MLP.
            hidden_layers: number of hidden ReLU layers in the advantage MLP.
            separate_sign_head: if True, split the final layer into separate
                advantage / sign heads on a shared backbone instead of reusing
                a single logit for both targets.
            controller_inputs: subset of ``CONTROLLER_INPUT_NAMES`` that the
                advantage head consumes. The head's ``input_dim`` is derived
                from this; the canonical caching layout (``[z_t, N_t, T_t]``)
                is unchanged.
        """
        super().__init__()
        self.controller_inputs: Tuple[str, ...] = validate_controller_inputs(controller_inputs)
        self.encoder = TreeEncoder(
            k=k,
            node_feat=node_feat,
            device=device,
            node_embed_hidden=node_embed_hidden,
            d_embed=d_embed,
            d_message=d_message,
            n_heads=n_heads,
            d_att=d_att,
        )
        resolved_device = self.encoder.device
        # Head input dim is determined by which canonical features the
        # controller is configured to consume.
        input_dim = self._head_input_dim(self.encoder.d_embed, self.controller_inputs)
        self.sign_head: nn.Linear | None = None
        if separate_sign_head:
            # Shared backbone (ReLU MLP) that feeds two separate linear heads:
            # one for the scalar advantage, one for the sign logit. Lets the
            # sign loss shape the representation without coupling its scale
            # to the regression target.
            backbone_layers: list[nn.Module] = []
            current_dim = input_dim
            for _ in range(hidden_layers):
                backbone_layers.append(nn.Linear(current_dim, hidden_dim, device=resolved_device))
                backbone_layers.append(nn.ReLU())
                current_dim = hidden_dim
            self.advantage_backbone = nn.Sequential(*backbone_layers)
            self.advantage_proj = nn.Linear(hidden_dim, 1, device=resolved_device)
            self.sign_head = nn.Linear(hidden_dim, 1, device=resolved_device)
            self.advantage_head = None
        else:
            # Single MLP whose output is reused for both regression and sign
            # losses (the predicted advantage doubles as the sign logit).
            self.advantage_head = _build_advantage_head(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                device=resolved_device,
            )

    def freeze_encoder(self) -> None:
        """Disable gradients on the encoder; the default path during fitted-Q training."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _head_input_dim(d_embed: int, controller_inputs: Sequence[str]) -> int:
        """Dimensionality of the head's input given which canonical features it uses."""
        dim = 0
        if "z_t" in controller_inputs:
            dim += int(d_embed)
        if "N_t" in controller_inputs:
            dim += 1
        if "T_t" in controller_inputs:
            dim += 1
        return dim

    def _select_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project a canonical ``[..., d_embed + 2]`` feature tensor onto the configured subset.

        Caller passes the full ``[z_t, N_t, T_t]`` concatenation (this is the
        cache layout). We slice the columns the head is configured to consume
        and concatenate them in canonical order. The output's last-dim size
        equals ``self._head_input_dim``.
        """
        d_embed = int(self.encoder.d_embed)
        parts = []
        if "z_t" in self.controller_inputs:
            parts.append(features[..., :d_embed])
        if "N_t" in self.controller_inputs:
            parts.append(features[..., d_embed : d_embed + 1])
        if "T_t" in self.controller_inputs:
            parts.append(features[..., d_embed + 1 : d_embed + 2])
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def predict_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (advantage, sign_logit). Identical when sign_head is None.

        ``features`` is the canonical ``[..., d_embed + 2]`` cache layout
        ``[z_t, N_t, T_t]``; the model slices to the configured
        ``controller_inputs`` before applying the head, so the caller never
        has to know which subset is in use.
        """
        head_input = self._select_features(features)
        if self.sign_head is not None:
            backbone_out = self.advantage_backbone(head_input)
            advantage = self.advantage_proj(backbone_out).squeeze(-1)
            sign_logit = self.sign_head(backbone_out).squeeze(-1)
            return advantage, sign_logit
        # Single-head path: the predicted advantage is also the sign logit.
        advantage = self.advantage_head(head_input).squeeze(-1)
        return advantage, advantage

    def encode_with_state_features(
        self,
        tree_batch: TreeBatch,
        tree_sizes: torch.Tensor,
        time_budgets: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the batch and concat ``(N_t, T_t)`` onto each root embedding.

        Args:
            tree_batch: packed batch consumed by the encoder.
            tree_sizes: [B] N_t per snapshot, cast to float and stacked in.
            time_budgets: [B] T_t per snapshot, cast to float and stacked in.

        Returns:
            [B, d_embed + 2] feature matrix ready for ``predict_from_features``.
        """
        encoded: TreeEncoderOutput = self.encoder(tree_batch)
        scalars = torch.stack(
            [
                tree_sizes.to(self.encoder.device, dtype=torch.float32),
                time_budgets.to(self.encoder.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        return torch.cat([encoded.root_states, scalars], dim=-1)

    def forward(
        self,
        tree_batch: TreeBatch,
        tree_sizes: torch.Tensor,
        time_budgets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder + advantage head end-to-end; returns (advantage, sign_logit)."""
        features = self.encode_with_state_features(tree_batch, tree_sizes, time_budgets)
        return self.predict_from_features(features)
