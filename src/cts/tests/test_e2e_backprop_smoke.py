"""Feasibility smoke test: does an unfrozen ``MetaController`` (encoder + advantage
head trained jointly) get a live gradient into the encoder on a real batch?

Requires the real validation manifest at ``$MCP/validation_manifest.json`` (source
``slurm/helpers/setup_env.sh`` first) -- skipped if that env/data isn't available.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

# This login node has 192 cores; torch's default (num_threads == cpu_count()) causes
# severe threading-oversubscription overhead for the small-tensor, many-sequential-ops
# workload below (observed: a single 6-episode batch effectively hung past 100s wall
# with default threads, vs ~19s forward alone at 4 threads). Bound it before any tensor
# op runs. This is a CPU-perf workaround only; it doesn't change the encoder math.
torch.set_num_threads(4)

from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_mc.oracle import (
    budgeted_oracle_config_from_metadata,
    return_for_stop_step,
)
from cts.models.mc import MetaController
from cts.train.controller_train import (
    ControllerEpisodeDataset,
    collate_controller_episodes,
)
from cts.train.pg_controller_train import expected_regret_batched

_MCP = os.environ.get("MCP")
_VALIDATION_MANIFEST = os.path.join(_MCP, "validation_manifest.json") if _MCP else None
_SKIP_REASON = (
    "requires $MCP/validation_manifest.json -- "
    "source slurm/helpers/setup_env.sh (gpu) first"
)


def _requires_real_manifest() -> str:
    if not _MCP:
        return _SKIP_REASON
    if not _VALIDATION_MANIFEST or not os.path.exists(_VALIDATION_MANIFEST):
        return _SKIP_REASON
    return ""


def _build_tiny_real_batch(num_episodes: int = 6):
    """Load ``num_episodes`` real episodes from the real validation manifest and collate them."""
    dataset = ControllerEpisodeDataset(_VALIDATION_MANIFEST)
    n = min(num_episodes, len(dataset))
    episodes = [dataset[i] for i in range(n)]
    batch = collate_controller_episodes(episodes)
    assert batch is not None
    return dataset, episodes, batch


def _real_regret_loss(episodes, advantages: torch.Tensor) -> torch.Tensor:
    """Build (adv, g, mask, lengths, oracle_values) for ``expected_regret_batched`` from
    real per-episode oracle data -- the exact machinery ``pg_controller_train.py`` uses
    (``return_for_stop_step`` + the manifest's own ``budgeted_oracle_config_from_metadata``),
    not a synthetic/fake target.
    """
    with open(_VALIDATION_MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    oracle_config = budgeted_oracle_config_from_metadata(manifest)
    assert oracle_config is not None, "validation manifest missing budgeted-oracle metadata"

    lengths = torch.tensor([len(ep.step_node_features) for ep in episodes], dtype=torch.long)
    tmax = int(lengths.max())
    b = len(episodes)

    g_pad = torch.zeros(b, tmax)
    mask = torch.zeros(b, tmax)
    oracle_values = torch.zeros(b)
    for i, ep in enumerate(episodes):
        t = len(ep.step_node_features)
        g = [
            return_for_stop_step(ep.halt_rewards, ep.tree_sizes.tolist(), ep.time_budgets.tolist(), s, oracle_config)
            for s in range(t)
        ]
        g_pad[i, :t] = torch.tensor(g, dtype=torch.float32)
        mask[i, :t] = 1.0
        oracle_values[i] = max(g)

    # advantages come in flat (sum(lengths),) order matching collate_controller_episodes'
    # per-episode-then-per-step ordering; scatter into the same [B, Tmax] padding as g/mask.
    adv_pad = torch.zeros(b, tmax)
    offset = 0
    for i, ep in enumerate(episodes):
        t = len(ep.step_node_features)
        adv_pad[i, :t] = advantages[offset : offset + t]
        offset += t

    return expected_regret_batched(adv_pad, g_pad, mask, lengths, oracle_values)


@pytest.mark.skipif(_requires_real_manifest() != "", reason=_requires_real_manifest())
def test_backprop_reaches_encoder():
    """The core Z0/Z1 smoke test: unfrozen encoder + real batch + real loss -> real grad."""
    torch.manual_seed(0)
    # 4 episodes (384 steps) keeps this test in the tens-of-seconds range on a
    # contended CPU-only login node -- see Z0's separate timing script for the
    # realistic episode_batch_size=8 measurement.
    dataset, episodes, batch = _build_tiny_real_batch(num_episodes=4)

    schema = tree_encoder_feature_schema()
    model = MetaController(
        k=1,  # matches config_minply15_maxply75.yaml's encoder.k
        node_feat=len(schema.feature_names),
        device="cpu",
        node_embed_hidden=16,
        d_embed=16,
        d_message=16,
        n_heads=2,
        d_att=16,
        hidden_dim=16,
        hidden_layers=0,
    )
    # Deliberately do NOT call model.freeze_encoder() -- that's the whole point.
    assert all(p.requires_grad for p in model.encoder.parameters()), (
        "encoder parameters must start requires_grad=True (freeze_encoder() must not "
        "have been called) for this to be a meaningful test"
    )

    advantage, _sign_logit = model.forward(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
    assert advantage.shape == (batch.tree_sizes.shape[0],)

    loss = _real_regret_loss(episodes, advantage)
    assert torch.isfinite(loss)

    model.zero_grad(set_to_none=True)
    loss.backward()

    encoder_params = list(model.encoder.parameters())
    assert encoder_params, "encoder has no parameters at all -- can't test grad flow"

    grad_norms = [
        float(p.grad.detach().abs().sum())
        for p in encoder_params
        if p.grad is not None
    ]
    num_with_grad = sum(1 for p in encoder_params if p.grad is not None)

    assert num_with_grad > 0, (
        "no encoder parameter received ANY gradient (.grad is None for all of them) -- "
        "backprop is not reaching the encoder at all"
    )
    assert any(g > 0.0 for g in grad_norms), (
        "encoder parameters received .grad tensors, but they are all-zero -- "
        "gradient reaches the encoder module but carries no signal"
    )


@pytest.mark.skipif(_requires_real_manifest() != "", reason=_requires_real_manifest())
def test_optimizer_steps_decrease_loss():
    """Stronger correctness check than the grad-flow smoke test above: a real
    Adam optimizer, stepped a handful of times on a tiny REAL batch, must
    actually DECREASE the expected-regret loss -- not merely produce nonzero
    gradients (which ``test_backprop_reaches_encoder``
    already covers). This is the cheap pre-flight check plan.md's Z2 section
    asks for before trusting a long (hours) e2e SLURM training run: if a few
    manual optimizer steps on real data don't move the loss down at all, the
    long run isn't worth submitting.

    Reuses ``cts.train.e2e_controller_train``'s own ``forward_batch_loss``
    helper (collate -> MetaController.forward -> expected_regret_batched) so
    this test exercises the EXACT function the real training loop calls, not
    a re-derived parallel implementation.
    """
    from cts.train.e2e_controller_train import build_regret_targets, forward_batch_loss  # noqa: F401 (build_regret_targets kept for parity/documentation)

    torch.manual_seed(0)
    dataset, episodes, _batch = _build_tiny_real_batch(num_episodes=6)

    with open(_VALIDATION_MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    oracle_config = budgeted_oracle_config_from_metadata(manifest)
    assert oracle_config is not None

    schema = tree_encoder_feature_schema()
    model = MetaController(
        k=1,  # matches config_minply15_maxply75.yaml's encoder.k
        node_feat=len(schema.feature_names),
        device="cpu",
        node_embed_hidden=16,
        d_embed=16,
        d_message=16,
        n_heads=2,
        d_att=16,
        hidden_dim=16,
        hidden_layers=0,
    )
    # Deliberately do NOT call model.freeze_encoder() -- joint training is the point.
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    def _loss() -> float:
        return float(forward_batch_loss(model, episodes, oracle_config, "cpu").detach())

    initial_loss = _loss()
    assert torch.isfinite(torch.tensor(initial_loss))

    losses = [initial_loss]
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_batch_loss(model, episodes, oracle_config, "cpu")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    final_loss = _loss()
    assert final_loss < initial_loss - 1e-4, (
        f"loss did not meaningfully decrease over 6 real optimizer steps: "
        f"initial={initial_loss:.6f} final={final_loss:.6f} trace={losses}"
    )
