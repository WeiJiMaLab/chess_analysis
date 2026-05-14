"""Meta-controller model: encoder + advantage head over ``[z_root, N_t, T_t]``.

The training loop in :mod:`cts.train.controller_train` wraps this model with
the data plumbing, loss, and diagnostics, but the model itself only depends
on the encoder (``cts.models.gnn``) and the packed tensor format
(``cts.core.tensorizer``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cts.core.tensorizer import TreeBatch
from cts.models.gnn import TreeEncoderOutput, TreeEncoder


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
    """Encoder + advantage MLP head over ``[z_root, N_t, T_t]``.

    Wraps a ``TreeEncoder`` encoder and a small ReLU MLP that maps the encoder's
    root embedding (concatenated with the two scalar state features) to a
    scalar advantage. Optionally exposes a second linear head on the same
    backbone for the auxiliary sign BCE loss (``--separate-sign-head``); by
    default the same logit is reused for both targets.
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
        """
        super().__init__()
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
        # +2 for the (N_t, T_t) scalars concatenated onto the root embedding.
        input_dim = self.encoder.d_embed + 2
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

    def predict_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (advantage, sign_logit). Identical when sign_head is None."""
        if self.sign_head is not None:
            backbone_out = self.advantage_backbone(features)
            advantage = self.advantage_proj(backbone_out).squeeze(-1)
            sign_logit = self.sign_head(backbone_out).squeeze(-1)
            return advantage, sign_logit
        # Single-head path: the predicted advantage is also the sign logit.
        advantage = self.advantage_head(features).squeeze(-1)
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
