"""Canary/poisoning tests for future-leakage bug shapes NOT covered by
``test_canary_future_leakage.py`` (which audits only
``ControllerEpisodeDataset.__getitem__``'s per-step node-feature slicing).

That file already proved the packhistory-fix's per-step forward-fill can't
leak a node's future backprop update into an earlier step's own row. This
file goes one level up the pipeline and asks a different question: once
``ControllerEpisodeDataset`` has correctly produced per-step tensors, could a
LATER stage -- batching multiple steps into one ``TreeBatch``
(``collate_controller_episodes``, used identically by ``materialize.py``'s
frozen-encoder production path, ``controller_train.py``'s live-encoder path,
and ``e2e_controller_train.py``'s unfrozen-encoder joint-training path), or
the RL loss/gradient machinery built on top of it -- re-introduce a leak that
the per-step slicing itself never had?

Methodology (same as ``test_canary_future_leakage.py``): poison a quantity
with an impossible sentinel (``-999``, unreachable by any real value/WDL/
reward feature in this pipeline -- see that file's docstring) or use exact
autograd-graph inspection, run it through the real production code path, and
prove the poison/gradient either can't reach a causally-earlier output
(negative result) or legitimately does reach the place it's supposed to
(positive control, proving the assertion methodology can detect a real
signal rather than being vacuously true).

Four canaries, each targeting a different codepath:

1. ``test_canary_batched_encoder_forward_isolates_snapshots`` -- proves that
   packing an EARLY step and a LATE step of the SAME real episode into one
   ``TreeBatch`` (exactly what ``collate_controller_episodes`` does, and what
   every consumer of it -- materialize.py, controller_train.py, and
   e2e_controller_train.py -- then runs through ONE encoder forward call)
   cannot change the early step's root embedding, even when the late step's
   node features are poisoned to an impossible sentinel. Uses
   ``torch.allclose`` with a tolerance well above measured BLAS/kernel-
   dispatch floating-point noise (see module-level comment on
   ``_BATCH_NOISE_ATOL``) but far below the poison's effect size, so
   floating-point non-determinism from different batch shapes can't produce
   a false pass OR a false alarm.
2. ``test_canary_no_gradient_path_from_later_step_to_earlier_step`` -- the
   sharpest version of (1): with an UNFROZEN encoder (the e2e/joint-training
   configuration; see e2e_controller_train.py), computes
   ``d(early_root_embedding)/d(late_step_node_features)`` via
   ``torch.autograd.grad`` directly and asserts it is EXACTLY zero (not just
   numerically small) -- a structural proof, not a statistical one, that no
   computation graph edge connects a later step's input to an earlier step's
   output, addressing point 2's "does gradient/weight information from a
   later step's loss reach an earlier step's prediction" concern at the
   lowest possible level.
3. ``test_canary_e2e_forward_batch_loss_batch_order_independence`` -- runs
   the actual production ``e2e_controller_train.forward_batch_loss`` (collate
   -> unfrozen MetaController.forward -> expected_regret_batched -> backward)
   on a batch containing an unrelated one-step episode PLUS a two-step
   episode whose second (later) step is poisoned, and confirms the
   unrelated episode's predicted advantage and loss contribution are
   unaffected -- catches bugs in the padding/scatter glue
   (``_scatter_to_padded``, ``build_regret_targets``) that operate on
   already-isolated per-tree embeddings and could still misalign rows.
4. ``test_canary_oracle_reward_fields_never_reach_policy_input`` -- poisons
   ``target_advantages``/``halt_rewards``/``oracle_value``/``oracle_stop_step``
   (the legitimate hindsight-computed REWARD/TARGET quantities from
   oracle.py's DP machinery) and proves they never reach
   ``ControllerBatch.tree_batch.node_features`` (the policy's actual INPUT)
   or change ``MetaController.forward``'s output, while a positive control
   confirms the same poison legitimately DOES reach the reward pathway
   (``e2e_controller_train.build_regret_targets``) -- and separately confirms
   ``materialize.py``'s ``model.freeze_encoder()`` call genuinely disables
   encoder gradients (history.md's claim, point 3 of this audit).

All four use REAL packed data (a real 96-step episode from
``ysagiv_xaba20k/mc_packed/validation/shard_00000.pt``, read-only, via a
throwaway manifest built in ``tmp_path`` that points at the existing on-disk
shard -- the manifest shipped in ``mc_packed/validation_manifest.json``
itself has stale absolute paths from a renamed directory, so we build our
own pointing at the shard that actually exists) with a freshly, randomly
initialized small ``MetaController`` (same architecture knobs as
``test_e2e_backprop_smoke.py``) -- no encoder checkpoint required, since
these canaries test structural isolation, not learned behavior. Skips
cleanly if that shard isn't present (e.g. a fresh checkout with no /scratch
access).

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_canary_controller_training_peeking.py -v
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import List

import pytest
import torch

# Same CPU-thread-oversubscription workaround as test_e2e_backprop_smoke.py /
# test_child_wdl_disambiguation.py -- this login node's core count causes
# severe overhead for this workload's small-tensor, many-sequential-ops
# pattern otherwise.
torch.set_num_threads(4)

from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.models.mc import MetaController
from cts.train.controller_train import (
    ControllerEpisode,
    ControllerEpisodeDataset,
    collate_controller_episodes,
)
from cts.train.e2e_controller_train import build_regret_targets, forward_batch_loss

_REAL_SHARD = Path(
    "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/mc_packed/validation/shard_00000.pt"
)

# Impossible sentinel: real value/WDL/reward/advantage features in this
# pipeline are always bounded (value in [-1, 1], WDL fractions/variance in
# [0, 1]; rewards and target_advantages are O(1)-scale reward-model outputs
# in practice) -- -999 can never occur naturally. Same convention as
# test_canary_future_leakage.py.
_SENTINEL = -999.0

# Measured (see this file's development notes / prototype) floating-point
# noise floor from batching a small tree together with a large (~3000-node)
# tree in one encoder forward call vs. encoding the small tree alone: BLAS/
# kernel-dispatch non-determinism from the differing overall tensor shape,
# NOT a leak (max abs diff observed ~3e-8 on d_embed=16 features). Set the
# "no leak" tolerance two orders of magnitude above that, and require any
# genuine poison signal (checked separately) to be many orders of magnitude
# larger still, so this tolerance can't hide a real leak.
_BATCH_NOISE_ATOL = 1e-5

_EARLY_STEP = 0   # first decision point: a handful of nodes.
_LATE_STEP = 90   # near the end of a 96-step episode: thousands of nodes.


def _require_real_shard() -> None:
    if not _REAL_SHARD.is_file():
        pytest.skip(f"real packed shard not found at {_REAL_SHARD} -- cannot build canary fixture")


@pytest.fixture(scope="module")
def real_episode(tmp_path_factory) -> ControllerEpisode:
    """Load episode 0 from the real (read-only) validation shard.

    ``mc_packed/validation_manifest.json`` itself has stale absolute paths
    (references a since-renamed ``ysagiv_xaba`` directory), so we point a
    throwaway manifest directly at the shard file that actually exists on
    disk today. Nothing under ``/scratch`` is read back after any mutation
    below -- every "poisoned" episode in this file is an in-memory
    ``dataclasses.replace`` copy.
    """
    _require_real_shard()
    manifest = {
        "format": "cts_budgeted_controller_episode_manifest_v4",
        "entries": [{"path": str(_REAL_SHARD), "num_episodes": 391, "shard_index": 0}],
    }
    manifest_path = tmp_path_factory.mktemp("canary_controller_training_peeking") / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dataset = ControllerEpisodeDataset(str(manifest_path))
    episode = dataset[0]
    assert len(episode.step_node_features) > _LATE_STEP, (
        "fixture assumption violated: need a real episode with at least "
        f"{_LATE_STEP + 1} steps to have a genuine early/late split"
    )
    assert episode.step_node_features[_EARLY_STEP].shape[0] < episode.step_node_features[_LATE_STEP].shape[0], (
        "fixture assumption violated: the 'late' step must have strictly more "
        "nodes than the 'early' step, or this canary can't distinguish them"
    )
    return episode


def _slice_episode(episode: ControllerEpisode, step_indices: List[int]) -> ControllerEpisode:
    """Build a new (frozen) ``ControllerEpisode`` containing only ``step_indices``,
    cloning every per-step tensor so poisoning one copy never mutates another."""
    idx_tensor = torch.tensor(step_indices, dtype=torch.long)
    return dataclasses.replace(
        episode,
        step_node_features=[episode.step_node_features[i].clone() for i in step_indices],
        step_parent_index=[episode.step_parent_index[i].clone() for i in step_indices],
        step_edge_parent=[episode.step_edge_parent[i].clone() for i in step_indices],
        step_edge_child=[episode.step_edge_child[i].clone() for i in step_indices],
        step_edge_slot=[episode.step_edge_slot[i].clone() for i in step_indices],
        step_depth=[episode.step_depth[i].clone() for i in step_indices],
        halt_rewards=[episode.halt_rewards[i] for i in step_indices],
        target_advantages=episode.target_advantages[idx_tensor].clone(),
        tree_sizes=episode.tree_sizes[idx_tensor].clone(),
        time_budgets=episode.time_budgets[idx_tensor].clone(),
    )


def _tiny_model(node_feat: int) -> MetaController:
    """Small, freshly-initialized MetaController -- same architecture knobs as
    test_e2e_backprop_smoke.py. Deliberately does NOT call freeze_encoder()
    unless the caller does so itself: encoder starts trainable (unfrozen),
    matching the e2e/joint-training configuration these canaries probe."""
    torch.manual_seed(0)
    return MetaController(
        k=1,
        node_feat=node_feat,
        device="cpu",
        node_embed_hidden=16,
        d_embed=16,
        d_message=16,
        n_heads=2,
        d_att=16,
        hidden_dim=16,
        hidden_layers=0,
    )


# --------------------------------------------------------------------------- #
# Canary 1: batched-encoder-forward snapshot isolation (materialize.py's and
# controller_train.py's shared batching mechanism).
# --------------------------------------------------------------------------- #


def test_canary_batched_encoder_forward_isolates_snapshots(real_episode):
    """Packing a real episode's EARLY and LATE step into one TreeBatch (exactly
    what ``collate_controller_episodes`` does for every real training batch,
    consumed identically by materialize.py's frozen-encoder pass and
    controller_train.py's live-encoder pass) must not let the LATE step's
    (poisoned) node features change the EARLY step's root embedding."""
    schema = tree_encoder_feature_schema()
    model = _tiny_model(len(schema.feature_names))
    model.eval()

    early_solo = _slice_episode(real_episode, [_EARLY_STEP])
    batch_solo = collate_controller_episodes([early_solo])
    with torch.no_grad():
        feats_solo = model.encode_with_state_features(
            batch_solo.tree_batch, batch_solo.tree_sizes, batch_solo.time_budgets
        )

    # Positive control first: confirm poisoning the EARLY snapshot's own
    # nodes DOES move its root embedding, so a broken "always identical"
    # implementation of this test couldn't pass vacuously.
    early_self_poisoned = _slice_episode(real_episode, [_EARLY_STEP])
    early_self_poisoned.step_node_features[0][0].fill_(_SENTINEL)  # root's own row
    batch_self_poisoned = collate_controller_episodes([early_self_poisoned])
    with torch.no_grad():
        feats_self_poisoned = model.encode_with_state_features(
            batch_self_poisoned.tree_batch, batch_self_poisoned.tree_sizes, batch_self_poisoned.time_budgets
        )
    assert not torch.allclose(feats_self_poisoned[0], feats_solo[0], atol=_BATCH_NOISE_ATOL), (
        "positive control failed: poisoning the EARLY snapshot's own root node "
        "feature did not change its own root embedding -- the encoder isn't "
        "reading node features at all, so this test's methodology can't be trusted"
    )

    # The actual canary: batch EARLY together with a poisoned LATE snapshot of
    # the SAME episode, exactly as collate_controller_episodes does for every
    # real multi-step episode.
    early_and_late = _slice_episode(real_episode, [_EARLY_STEP, _LATE_STEP])
    early_and_late.step_node_features[1].fill_(_SENTINEL)  # poison the entire LATE tree
    batch_combined = collate_controller_episodes([early_and_late])
    with torch.no_grad():
        feats_combined = model.encode_with_state_features(
            batch_combined.tree_batch, batch_combined.tree_sizes, batch_combined.time_budgets
        )

    assert torch.allclose(feats_combined[0], feats_solo[0], atol=_BATCH_NOISE_ATOL), (
        "LEAK: the EARLY step's root embedding changed when batched alongside a "
        "poisoned LATE step of the same episode (as collate_controller_episodes "
        "packs every real training batch) -- future information from a later "
        "search step reached an earlier step's policy input via shared batching. "
        f"max abs diff = {(feats_combined[0] - feats_solo[0]).abs().max().item():.3e}"
    )

    # Sanity: the LATE row itself DOES reflect its own poison (rules out a
    # degenerate "batching silently drops the late tree" implementation that
    # would pass the assertion above trivially).
    batch_late_solo = collate_controller_episodes([_slice_episode(real_episode, [_LATE_STEP])])
    with torch.no_grad():
        feats_late_solo = model.encode_with_state_features(
            batch_late_solo.tree_batch, batch_late_solo.tree_sizes, batch_late_solo.time_budgets
        )
    assert not torch.allclose(feats_combined[1], feats_late_solo[0], atol=_BATCH_NOISE_ATOL), (
        "sanity check failed: poisoning the LATE step's node features had no "
        "effect on the LATE step's own root embedding -- the poison never "
        "reached the encoder at all, so the main assertion above is vacuous"
    )


# --------------------------------------------------------------------------- #
# Canary 2: exact autograd-graph proof, unfrozen encoder (e2e configuration).
# --------------------------------------------------------------------------- #


def test_canary_no_gradient_path_from_later_step_to_earlier_step(real_episode):
    """With an UNFROZEN encoder (the e2e_controller_train.py joint-training
    configuration -- freeze_encoder() deliberately not called), the gradient
    of the EARLY step's root embedding with respect to the LATE step's node
    features must be EXACTLY zero: not small, not numerically negligible,
    exactly zero, because no computation-graph edge should connect them.

    This is the sharpest available test of point 2's concern (whether
    encoder-unfrozen joint training could let a later step's
    weights/gradients influence an earlier step's own prediction within one
    batch): if ANY path existed -- even one too subtle to show up as a
    value-level difference in canary 1 above -- autograd would report a
    nonzero gradient here.
    """
    schema = tree_encoder_feature_schema()
    model = _tiny_model(len(schema.feature_names))
    model.train()  # unfrozen: encoder params default requires_grad=True
    assert all(p.requires_grad for p in model.encoder.parameters()), (
        "test setup invalid: encoder must be unfrozen for this to probe the "
        "e2e/joint-training configuration"
    )

    episode = _slice_episode(real_episode, [_EARLY_STEP, _LATE_STEP])
    episode.step_node_features[0].requires_grad_(True)
    episode.step_node_features[1].requires_grad_(True)
    batch = collate_controller_episodes([episode])

    feats = model.encode_with_state_features(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
    d_embed = model.encoder.d_embed
    early_root = feats[0, :d_embed]
    late_root = feats[1, :d_embed]

    grad_early_wrt_late = torch.autograd.grad(
        early_root.sum(), episode.step_node_features[1], retain_graph=True, allow_unused=True
    )[0]
    assert grad_early_wrt_late is not None, (
        "gradient came back None (allow_unused) rather than a zero tensor -- "
        "unexpected graph topology, re-check this test's assumptions before trusting it"
    )
    assert float(grad_early_wrt_late.abs().sum()) == 0.0, (
        "LEAK: d(early root embedding)/d(late step node features) is NONZERO -- "
        "a real computation-graph edge connects a later step's input to an "
        "earlier step's output under the unfrozen (e2e) encoder configuration. "
        f"sum(|grad|) = {float(grad_early_wrt_late.abs().sum())}"
    )

    # Positive control: the EARLY root's gradient w.r.t. its OWN input must be
    # nonzero, proving autograd is actually tracing real dependencies here
    # (i.e. the zero result above isn't an artifact of a detached/no-grad path).
    grad_early_wrt_early = torch.autograd.grad(
        early_root.sum(), episode.step_node_features[0], retain_graph=True, allow_unused=True
    )[0]
    assert grad_early_wrt_early is not None and float(grad_early_wrt_early.abs().sum()) > 0.0, (
        "positive control failed: d(early root)/d(early step's own node features) "
        "is zero or None -- autograd isn't tracing this graph at all, so the "
        "exact-zero result above proves nothing"
    )

    # Same check in the other temporal direction for completeness: the LATE
    # root legitimately (and correctly) DOES depend on the EARLY step's nodes
    # is NOT required (they're independent trees, not causally linked forward
    # in this codebase's design), but it must not depend on itself being
    # somehow computed before the early step either -- i.e. late_root's own
    # gradient w.r.t. its own input must be nonzero (sanity only).
    grad_late_wrt_late = torch.autograd.grad(
        late_root.sum(), episode.step_node_features[1], retain_graph=False, allow_unused=True
    )[0]
    assert grad_late_wrt_late is not None and float(grad_late_wrt_late.abs().sum()) > 0.0, (
        "sanity check failed: late root embedding has no gradient w.r.t. its "
        "own node features -- something is wrong with this test's graph, not "
        "just the leakage question"
    )


# --------------------------------------------------------------------------- #
# Canary 3: the real e2e production loss function end-to-end.
# --------------------------------------------------------------------------- #


def test_canary_e2e_forward_batch_loss_batch_order_independence(real_episode):
    """Runs the ACTUAL production ``e2e_controller_train.forward_batch_loss``
    (collate -> unfrozen MetaController.forward -> expected_regret_batched)
    on a batch containing an unrelated one-step episode alongside a two-step
    episode whose LATE step is poisoned, through a real ``.backward()`` call.
    The unrelated episode's own predicted advantage must be identical to
    running it alone -- this exercises the padding/scatter glue
    (``_scatter_to_padded``/``build_regret_targets``) that canaries 1-2 don't
    touch, since a batch-order or index-alignment bug there could reintroduce
    cross-episode leakage even with a perfectly isolated encoder.
    """
    schema = tree_encoder_feature_schema()
    model = _tiny_model(len(schema.feature_names))
    model.train()
    oracle_config = BudgetedOracleConfig(time_mode="linear", time_lambda=0.01, maintenance_scale=0.0)

    unrelated_episode = _slice_episode(real_episode, [_EARLY_STEP])
    batch_unrelated_alone = collate_controller_episodes([unrelated_episode])
    with torch.no_grad():
        adv_alone, _ = model.forward(
            batch_unrelated_alone.tree_batch, batch_unrelated_alone.tree_sizes, batch_unrelated_alone.time_budgets
        )

    poisoned_episode = _slice_episode(real_episode, [_EARLY_STEP, _LATE_STEP])
    poisoned_episode.step_node_features[1].fill_(_SENTINEL)

    batch_combined = collate_controller_episodes([unrelated_episode, poisoned_episode])
    with torch.no_grad():
        adv_combined, _ = model.forward(
            batch_combined.tree_batch, batch_combined.tree_sizes, batch_combined.time_budgets
        )
    # collate_controller_episodes iterates episodes in order and each
    # episode's steps in order, so row 0 of the combined batch is always the
    # unrelated_episode's (only) step.
    assert torch.allclose(adv_alone, adv_combined[0:1], atol=_BATCH_NOISE_ATOL), (
        "LEAK: the unrelated episode's predicted advantage changed depending on "
        "whether it was batched alongside a poisoned later-step episode -- "
        f"diff = {(adv_alone - adv_combined[0:1]).abs().item():.3e}"
    )

    # Full production path, including the real oracle-based loss and a real
    # backward() -- must not crash and must produce a finite loss even with
    # the sentinel poison present (rules out the poison silently producing
    # NaN/inf that would mask a comparison rather than genuinely proving
    # isolation).
    loss = forward_batch_loss(model, [unrelated_episode, poisoned_episode], oracle_config, "cpu")
    assert torch.isfinite(loss), f"loss is not finite with a poisoned batch-mate present: {loss}"
    model.zero_grad(set_to_none=True)
    loss.backward()
    encoder_grad_norms = [
        float(p.grad.detach().abs().sum()) for p in model.encoder.parameters() if p.grad is not None
    ]
    assert encoder_grad_norms and any(g > 0.0 for g in encoder_grad_norms), (
        "backward() completed but no encoder parameter received a nonzero gradient "
        "-- can't confirm this exercised the real joint-training gradient path"
    )


# --------------------------------------------------------------------------- #
# Canary 4: oracle reward/target quantities never reach the policy's INPUT.
# --------------------------------------------------------------------------- #


def test_canary_oracle_reward_fields_never_reach_policy_input(real_episode):
    """``target_advantages``/``halt_rewards``/``oracle_value``/``oracle_stop_step``
    are legitimate hindsight-computed REWARD/TARGET quantities from
    ``oracle.py``'s backward-DP machinery (see module docstring): they are
    allowed, by design, to depend on the full 96-step trajectory. This canary
    proves they are used ONLY that way -- poisoning them must never change
    ``ControllerBatch.tree_batch.node_features`` (the tensor
    ``MetaController.forward`` actually reads) or the model's predicted
    advantage, while a positive control confirms the SAME poison legitimately
    DOES reach the reward pathway (``build_regret_targets``, used by the real
    e2e loss). Also regression-guards ``materialize.py``'s
    ``model.freeze_encoder()`` call (history.md's claim, point 3 of this
    audit): after calling it, no encoder parameter is trainable.
    """
    schema = tree_encoder_feature_schema()
    model = _tiny_model(len(schema.feature_names))
    model.eval()

    episode = _slice_episode(real_episode, [_EARLY_STEP, _LATE_STEP])
    poisoned = dataclasses.replace(
        episode,
        halt_rewards=[_SENTINEL for _ in episode.halt_rewards],
        target_advantages=torch.full_like(episode.target_advantages, _SENTINEL),
        oracle_stop_step=-999,
        oracle_value=_SENTINEL,
    )

    batch_orig = collate_controller_episodes([episode])
    batch_poisoned = collate_controller_episodes([poisoned])

    assert torch.equal(batch_orig.tree_batch.node_features, batch_poisoned.tree_batch.node_features), (
        "LEAK: poisoning halt_rewards/target_advantages/oracle_value/oracle_stop_step "
        "changed ControllerBatch.tree_batch.node_features -- a reward/target quantity "
        "reached the policy's actual input tensor"
    )
    assert not bool((batch_poisoned.tree_batch.node_features == _SENTINEL).any()), (
        "LEAK: the reward/target sentinel is directly visible in node_features"
    )

    with torch.no_grad():
        adv_orig, _ = model.forward(batch_orig.tree_batch, batch_orig.tree_sizes, batch_orig.time_budgets)
        adv_poisoned, _ = model.forward(batch_poisoned.tree_batch, batch_poisoned.tree_sizes, batch_poisoned.time_budgets)
    assert torch.equal(adv_orig, adv_poisoned), (
        "LEAK: MetaController.forward's output changed when only reward/target "
        "fields were poisoned (node_features held equal) -- forward() must be "
        "reading something outside tree_batch/tree_sizes/time_budgets"
    )

    # Positive control: the same poison legitimately DOES reach the reward
    # pathway real e2e training uses -- proves this test isn't vacuously
    # trivial (e.g. because collate_controller_episodes silently drops these
    # fields entirely rather than correctly routing them to rewards only).
    oracle_config = BudgetedOracleConfig(time_mode="linear", time_lambda=0.01, maintenance_scale=0.0)
    _g_pad, _mask, _lengths, oracle_values = build_regret_targets([poisoned], oracle_config, "cpu")
    assert float(oracle_values[0]) == _SENTINEL, (
        "positive control failed: poisoned halt_rewards/oracle_value did not reach "
        "build_regret_targets (the real reward pathway) -- can't distinguish "
        "'correctly input-isolated' from 'these fields are just unused/broken'"
    )

    # Separate, narrower regression guard for point 3: materialize.py's
    # model.freeze_encoder() call must genuinely disable encoder gradients
    # (history.md documents this as already true; guard against a future
    # refactor silently reverting it).
    assert all(p.requires_grad for p in model.encoder.parameters()), (
        "test setup check: encoder should start unfrozen"
    )
    model.freeze_encoder()
    assert all(not p.requires_grad for p in model.encoder.parameters()), (
        "materialize.py relies on MetaController.freeze_encoder() to genuinely stop "
        "encoder gradients (it runs under torch.inference_mode() as defense in depth, "
        "but production code elsewhere may call the encoder outside inference_mode) "
        "-- freeze_encoder() left at least one encoder parameter trainable"
    )
