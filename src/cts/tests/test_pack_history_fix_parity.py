"""Regression suite for the 2026-07-11 fix to ``build_snapshot_pair_example``'s
never-visited-leaf filter (``pack_history.py``).

Background (see ``pack_history.py``'s own updated docstrings on ``_never_visited``
and ``build_snapshot_pair_example``, and ``test_canary_pack_history_structural.py``'s
module docstring points 2/3 for the original characterization of the bug):
``build_snapshot_pair_example`` used to drop any edge whose child had zero
update-log entries anywhere across the whole 96-step trajectory ("never visited")
from T_n's STRUCTURE itself -- not just from the loss target. Because
``TreeAttMsgLayer.forward`` (``models/tree_mha.py``) computes each parent's
per-child attention via a segmented softmax over exactly the children present in
``edge_parent``/``edge_child``, dropping a never-visited sibling changed the
softmax denominator -- and therefore the prediction -- for every OTHER (kept)
child of that parent. Production inference (``ControllerEpisodeDataset.__getitem__``
in ``controller_train.py``, and ``materialize.py``) always sees the FULL, unfiltered
edge set, so training-time graphs (sparse) and production-time graphs (full) had
systematically different topology. The fix removes the filter: T_n's structural
edge set returned by ``build_snapshot_pair_example`` is now always exactly
``_build_tree_n(trajectory, n)``'s own edge set; Delta-visits=0 edges (including
what used to be filtered) are kept and get a per-edge loss WEIGHT of 0 instead of
being removed from the graph.

This file was written and finalized AFTER the fix landed (confirmed via
``git diff`` on ``pack_history.py`` showing the filter removed, ``_never_visited``
kept-but-unused) -- every test below asserts the CORRECTED, permanent invariant,
not the old buggy behavior. It does not need to run before/after the fix the way
the task briefing anticipated, since the fix was already present by the time this
file was written.

Four things are covered, matching the four requirements this suite was
commissioned for:

1. ``test_core_invariant_snapshot_pair_edges_match_build_tree_n`` -- the single
   strongest regression guard: across many real (trajectory, n) pairs,
   ``build_snapshot_pair_example``'s returned edge set must be IDENTICAL (as an
   order-independent set of (parent, child, slot) triples) to
   ``_build_tree_n(trajectory, n)``'s own raw edge set. This is the exact
   invariant the original bug violated.

2. ``test_cross_pipeline_parity_controller_dataset_matches_pack_history`` -- goes
   further and compares against the REAL production inference path
   (``ControllerEpisodeDataset.__getitem__``), re-deriving the
   local_step <-> n index mapping from both functions' own source rather than
   assuming it.

3. ``test_loss_neutrality_of_previously_filtered_edges`` -- confirms, with real
   numbers (not just "should be negligible"), that (a) a previously-filtered
   (never-visited, weight=0) edge's own target contributes EXACTLY zero to the
   weighted loss ``gnn_pretrain.py`` actually computes, via poisoning that edge's
   target with an adversarial distribution and confirming the loss doesn't move,
   and (b) the segmented-softmax topology coupling documented above is real and
   measurable: including the previously-filtered edges in the graph (as the fix
   now does) measurably changes the KEPT edges' own loss relative to the old,
   filtered topology -- reported as an actual number, not asserted away.

4. ``test_core_invariant_is_not_vacuous_against_reintroduced_old_filter`` --
   mutation test: reintroduces a standalone reimplementation of the OLD
   (pre-fix) keep_mask filter (independent of ``_never_visited``, in case that
   helper is ever removed) and confirms this file's own core-invariant
   assertion helper correctly FAILS against it, proving test 1 is not vacuous.

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_pack_history_fix_parity.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import pytest
import torch

from cts.core.schema import tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeExample, collate_tensorized_examples
from cts.data.preprocess_gnn.pack_history import (
    HistoryTrajectory,
    _build_tree_n,
    _delta_visits,
    _forward_filled_wdl_at_step,
    build_snapshot_pair_example,
    iter_history_trajectories,
)
from cts.data.preprocess_mc.pack import (
    PackControllerEpisodesConfig,
    _oracle_config,
    _pack_split,
    _quality_config,
)
from cts.models.gnn import ChildWdlModel
from cts.train.controller_train import ControllerEpisodeDataset
from cts.train.gnn_pretrain import ChildWdlPretrainConfig, ChildWdlPretrainer

_SPLIT_MANIFEST = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/split/train_manifest.txt")
_N_FIXTURE_TREES = 20
_LOOKAHEAD_K = 12


def _require_real_data():
    if not _SPLIT_MANIFEST.is_file():
        pytest.skip(f"real split manifest not found at {_SPLIT_MANIFEST} -- cannot build canary fixture")


# --------------------------------------------------------------------------- #
# Fixture: build one small, REAL pack_history shard via the actual production
# packing code (preprocess_mc.pack._pack_split), on real xaba20k trees. Written
# entirely under pytest's tmp_path -- never touches /scratch's real
# pack_history/mc_packed output. Same construction as
# test_canary_future_leakage.py's `real_fixture`, duplicated here (rather than
# imported cross-file) per this test directory's own convention of each file
# being self-contained.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_history_fixture(tmp_path_factory) -> Dict[str, Any]:
    _require_real_data()
    out_dir = tmp_path_factory.mktemp("pack_history_fix_parity")

    with _SPLIT_MANIFEST.open() as handle:
        tree_paths = [line.strip() for line in handle if line.strip()][:_N_FIXTURE_TREES]
    if len(tree_paths) < _N_FIXTURE_TREES:
        pytest.skip("split manifest has fewer real trees than requested for the fixture")

    tiny_manifest = out_dir / "train_manifest.txt"
    tiny_manifest.write_text("\n".join(tree_paths) + "\n")

    config = PackControllerEpisodesConfig(
        split_root=str(out_dir),
        output_root=str(out_dir),
        shard_size=50,
        num_workers=0,
        reward_scale=1.0,
        min_halt_reward_range=0.0,
        min_decision_margin=0.0,
        exclude_xaba=False,
        search_budget=96,
        max_depth=36,
        maintenance_scale=0.0,
        maintenance_ref_nodes=30.0,
        maintenance_exponent=1.0,
        time_lambda=0.01,
        time_p=2.8,
        time_tau=2.5,
        time_delta=1,
        timeout_value=-1.0,
        time_mode="linear",
        samples_per_bucket=2,
        fixed_budget=96,
        seed=0,
        clear=True,
    )
    quality_config = _quality_config(config)
    oracle_config = _oracle_config(config)

    manifest_path, total_episodes, total_skipped, _sf, _bf = _pack_split(
        tiny_manifest,
        out_dir,
        quality_config,
        oracle_config,
        reward_scale=config.reward_scale,
        min_halt_reward_range=config.min_halt_reward_range,
        min_decision_margin=config.min_decision_margin,
        exclude_xaba=config.exclude_xaba,
        sample_trees_by_tree_strata=False,
        tree_stratification_mode="dj",
        sample_trees_by_budget_score=False,
        shard_size=config.shard_size,
        num_workers=0,
        log_interval=50,
    )
    if total_episodes == 0:
        pytest.skip("fixture packing produced zero episodes -- cannot run these tests")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "cts_budgeted_controller_episode_manifest_v4"
    shard_path = Path(manifest["entries"][0]["path"])
    shard_payload = torch.load(shard_path, weights_only=False)
    assert shard_payload["format"] == "cts_packhistory_trees_shard_v1", (
        "fixture did not produce the packhistory_trees shard format -- "
        "these tests require the update log to be present"
    )

    trajectories = list(iter_history_trajectories(shard_payload))
    assert trajectories, "fixture produced zero trajectories"

    return {
        "out_dir": out_dir,
        "manifest_path": manifest_path,
        "shard_path": shard_path,
        "payload": shard_payload,
        "trajectories": trajectories,
        "total_episodes": total_episodes,
    }


def _sampled_ns(num_steps: int, lookahead_k: int, count: int = 8) -> List[int]:
    """A handful of deterministic n's spanning [1, num_steps], not just one."""
    if num_steps < 1:
        return []
    step = max(1, num_steps // count)
    candidates = list(range(1, num_steps + 1, step))
    if num_steps not in candidates:
        candidates.append(num_steps)
    # Also include the largest n for which T_n -> T_{n+k} is a valid pair, since
    # that's the range deterministic_snapshot_steps actually samples from in production.
    high = num_steps - lookahead_k
    if high >= 1 and high not in candidates:
        candidates.append(high)
    return sorted(set(candidates))


def _edge_triples(parent: torch.Tensor, child: torch.Tensor, slot: torch.Tensor) -> Set[Tuple[int, int, int]]:
    return set(zip(parent.tolist(), child.tolist(), slot.tolist()))


# --------------------------------------------------------------------------- #
# 1. Core invariant: build_snapshot_pair_example's edge set == _build_tree_n's
# --------------------------------------------------------------------------- #


def _assert_edge_sets_match(actual: Set[Tuple[int, int, int]], expected: Set[Tuple[int, int, int]], context: str) -> None:
    """Shared assertion helper -- reused verbatim by the real test (below) and by
    the mutation test, so the mutation test is provably exercising the exact same
    logic as the real regression guard, not a look-alike copy of it."""
    assert actual == expected, (
        f"{context}: build_snapshot_pair_example's edge set differs from "
        f"_build_tree_n's raw edge set. Extra (present in build_snapshot_pair_example, "
        f"not in _build_tree_n): {actual - expected}. Missing (present in _build_tree_n, "
        f"filtered out of build_snapshot_pair_example): {expected - actual}. This is "
        f"exactly the original never-visited-leaf-filter bug shape if any edges are "
        f"missing -- training-time graph topology no longer matches "
        f"production-time (_build_tree_n / ControllerEpisodeDataset) topology."
    )


def test_core_invariant_snapshot_pair_edges_match_build_tree_n(real_history_fixture):
    """The single strongest regression guard for the original bug: for a real
    sample of (trajectory, n) pairs -- many trees, several n's per tree, not one
    hand-picked example -- ``build_snapshot_pair_example``'s returned
    (edge_parent, edge_child, edge_slot) must be IDENTICAL, as an order-independent
    set of (parent, child, slot) triples, to ``_build_tree_n(trajectory, n)``'s own
    raw edge set for the same inputs.

    Before the fix: this would fail on almost every (tree, n) with at least one
    never-visited child under some parent (history.md measured 88.8-97.7% of
    nodes as typically never-visited, so this is the overwhelming majority of
    real (tree, n) pairs, not an edge case).
    """
    trajectories = real_history_fixture["trajectories"]
    schema = tree_encoder_feature_schema()

    checked_pairs = 0
    checked_edges = 0
    trees_with_dropped_children = 0
    for traj in trajectories:
        num_steps = int(traj.step_node_cutoffs.shape[0])
        for n in _sampled_ns(num_steps, _LOOKAHEAD_K):
            snap = _build_tree_n(traj, n)
            if snap is None:
                continue
            expected = _edge_triples(snap.edge_parent, snap.edge_child, snap.edge_slot)

            example = build_snapshot_pair_example(traj, n, _LOOKAHEAD_K, schema)
            if example is None:
                assert not expected, (
                    f"{traj.source_path} n={n}: build_snapshot_pair_example returned None but "
                    f"_build_tree_n has {len(expected)} real edges -- a genuine (non-vacuous) "
                    f"filtering regression"
                )
                continue
            actual = _edge_triples(example.edge_parent, example.edge_child, example.edge_slot)
            _assert_edge_sets_match(actual, expected, f"{traj.source_path} n={n}")
            checked_pairs += 1
            checked_edges += len(expected)

            # Track whether this fixture actually exercises children with zero
            # update-log entries anywhere (the never-visited-leaf case the old bug
            # filtered) -- if it never does, the test above would be vacuously easy.
            never_visited_children = [
                int(c) for c in snap.edge_child.tolist()
                if int(traj.node_update_ptr[c].item()) >= int(traj.node_update_ptr[c + 1].item())
            ]
            if never_visited_children:
                trees_with_dropped_children += 1

    assert checked_pairs > 30, f"sanity check on the check itself -- only tested {checked_pairs} (tree, n) pairs"
    assert checked_edges > 500, f"sanity check on the check itself -- only compared {checked_edges} total edges"
    assert trees_with_dropped_children > 0, (
        "fixture never exercised a (tree, n) with a never-visited child edge -- this test would be "
        "vacuously easy to pass even with the old bug reintroduced (nothing to filter)"
    )


# --------------------------------------------------------------------------- #
# 2. Cross-pipeline parity: production inference path (ControllerEpisodeDataset)
# --------------------------------------------------------------------------- #


def test_cross_pipeline_parity_controller_dataset_matches_pack_history(real_history_fixture):
    """Compares against the ACTUAL production inference path, not just two
    functions inside pack_history.py.

    Index mapping, re-derived from source (not assumed): ``ControllerEpisodeDataset.
    __getitem__`` (``controller_train.py:889-950``) loops ``for local_step in
    range(num_steps)`` and computes ``expansion_count = first_decision_expansion_count
    + local_step`` -- IDENTICAL formula to ``_build_tree_n``'s ``expansion_count =
    trajectory.first_decision_expansion_count + local_step`` where
    ``local_step = n - 1``. Both index into the SAME trajectory-level
    ``step_node_cutoffs``/``expansion_parent_ids``/``child_ptr``/``edge_child``/
    ``edge_slot`` arrays starting from position 0 regardless of an episode's own
    ``starting_budget`` (that only affects the ``time_budgets`` feature, not which
    steps are built). So: ControllerEpisodeDataset's per-episode local_step ``i``
    corresponds to pack_history.py's ``n = i + 1``.

    For every real trajectory covered by at least one packed episode, and every
    local_step of that episode, the edge set ControllerEpisodeDataset actually
    built for production inference must equal both _build_tree_n's and
    build_snapshot_pair_example's edge set at the corresponding n.
    """
    manifest_path = real_history_fixture["manifest_path"]
    trajectories = real_history_fixture["trajectories"]
    schema = tree_encoder_feature_schema()
    traj_by_source_path = {traj.source_path: traj for traj in trajectories}

    ds = ControllerEpisodeDataset(str(manifest_path))
    assert len(ds) == real_history_fixture["total_episodes"]

    seen_source_paths: Set[str] = set()
    checked_steps = 0
    checked_build_snapshot_pairs = 0
    for episode_index in range(len(ds)):
        episode = ds[episode_index]
        if episode.source_path in seen_source_paths:
            continue  # one episode per trajectory is enough -- edges don't depend on starting_budget
        traj = traj_by_source_path.get(episode.source_path)
        if traj is None:
            continue
        seen_source_paths.add(episode.source_path)

        num_steps = len(episode.step_edge_parent)
        for local_step in range(num_steps):
            n = local_step + 1
            production_triples = _edge_triples(
                episode.step_edge_parent[local_step],
                episode.step_edge_child[local_step],
                episode.step_edge_slot[local_step],
            )

            snap = _build_tree_n(traj, n)
            assert snap is not None, (
                f"{episode.source_path} local_step={local_step} (n={n}): ControllerEpisodeDataset "
                f"built a step that _build_tree_n refuses to build at all -- index mapping is wrong"
            )
            pack_history_triples = _edge_triples(snap.edge_parent, snap.edge_child, snap.edge_slot)

            assert production_triples == pack_history_triples, (
                f"{episode.source_path} local_step={local_step} (n={n}): ControllerEpisodeDataset's "
                f"real production edge set differs from _build_tree_n's. Extra in production: "
                f"{production_triples - pack_history_triples}. Extra in pack_history: "
                f"{pack_history_triples - production_triples}."
            )
            checked_steps += 1

            # Also check build_snapshot_pair_example directly, wherever n is in its
            # valid (n, n+k) range -- the full three-way parity: production ==
            # _build_tree_n == build_snapshot_pair_example.
            if n + _LOOKAHEAD_K <= num_steps:
                example = build_snapshot_pair_example(traj, n, _LOOKAHEAD_K, schema)
                assert example is not None
                example_triples = _edge_triples(example.edge_parent, example.edge_child, example.edge_slot)
                assert example_triples == production_triples, (
                    f"{episode.source_path} local_step={local_step} (n={n}): "
                    f"build_snapshot_pair_example's edge set differs from the real production "
                    f"(ControllerEpisodeDataset) edge set -- train/inference topology mismatch."
                )
                checked_build_snapshot_pairs += 1

        if len(seen_source_paths) >= 15:  # plenty of distinct trees; keep runtime reasonable
            break

    assert checked_steps > 200, f"sanity check on the check itself -- only compared {checked_steps} steps"
    assert checked_build_snapshot_pairs > 20, (
        f"sanity check on the check itself -- only exercised build_snapshot_pair_example "
        f"{checked_build_snapshot_pairs} times in this cross-pipeline comparison"
    )


# --------------------------------------------------------------------------- #
# 3. Loss neutrality: filtered edges contribute ~0, kept edges' loss is affected
#    by topology (measured, not assumed)
# --------------------------------------------------------------------------- #


class _InMemoryTensorizedDataset(Sequence[TensorizedTreeExample]):
    """Minimal in-memory stand-in for PackedFutureWdlShardDataset (same collate_fn
    convention), so ChildWdlPretrainer's REAL loss-weighting code path
    (_iterate_supervised_edge_batches) can be exercised directly on hand-built
    examples without writing/reading shard files."""

    def __init__(self, examples: List[TensorizedTreeExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> TensorizedTreeExample:
        return self._examples[index]

    @staticmethod
    def collate_fn(batch):
        return collate_tensorized_examples(batch)


def _validate_single_example(model: ChildWdlModel, schema, example: TensorizedTreeExample):
    """Run gnn_pretrain.py's REAL validate() path (no grad, no optimizer step) over
    exactly one example, returning the ChildWdlMetrics it actually computes."""
    config = ChildWdlPretrainConfig(batch_size=1, epochs=1, num_workers=0, shuffle=False)
    pretrainer = ChildWdlPretrainer(
        model=model,
        schema=schema,
        device="cpu",
        train_examples=_InMemoryTensorizedDataset([example]),
        validation_examples=_InMemoryTensorizedDataset([example]),
        config=config,
    )
    return pretrainer.validate()


def _first_real_snapshot_pair_example(trajectories: List[HistoryTrajectory], schema, lookahead_k: int):
    """Find a real (trajectory, n) whose T_n has both a never-visited child edge
    (weight will be 0) and at least one genuinely-visited child edge under the
    SAME parent (so the segmented-softmax coupling test below is not vacuous)."""
    for traj in trajectories:
        num_steps = int(traj.step_node_cutoffs.shape[0])
        for n in _sampled_ns(num_steps, lookahead_k, count=12):
            if n + lookahead_k > num_steps:
                continue
            snap = _build_tree_n(traj, n)
            if snap is None or snap.edge_child.numel() < 2:
                continue
            never_visited_mask = [
                int(traj.node_update_ptr[c].item()) >= int(traj.node_update_ptr[c + 1].item())
                for c in snap.edge_child.tolist()
            ]
            if not any(never_visited_mask) or all(never_visited_mask):
                continue
            # Need a parent with BOTH a never-visited and a visited child, to
            # isolate the segmented-softmax coupling effect from unrelated parents.
            by_parent: Dict[int, List[int]] = {}
            for idx, parent_id in enumerate(snap.edge_parent.tolist()):
                by_parent.setdefault(parent_id, []).append(idx)
            for parent_id, idxs in by_parent.items():
                flags = [never_visited_mask[i] for i in idxs]
                if any(flags) and not all(flags):
                    example = build_snapshot_pair_example(traj, n, lookahead_k, schema)
                    if example is not None:
                        return traj, n, example
    return None


def test_loss_neutrality_of_previously_filtered_edges(real_history_fixture):
    """Confirms, with real numbers, the reasoning build_snapshot_pair_example's
    docstring relies on: "gnn_pretrain.py's Delta-visits weighting branch already
    treats weight=0 as contributes ~nothing to the loss".

    Two separate, independently-meaningful checks:

    (a) POISON CHECK: a previously-filtered (never-visited, weight=0) edge's own
        target contributes EXACTLY zero to the weighted loss. Poison that edge's
        edge_wdl_targets with an adversarial one-hot distribution and confirm the
        computed total_loss/target_entropy from ChildWdlPretrainer's real
        validate() path are bit-identical to the unpoisoned run -- if weight=0
        truly zeroes out the contribution, an arbitrarily wrong target for that
        edge cannot move the aggregate loss at all.

    (b) COUPLING MAGNITUDE: including the previously-filtered edges in the
        topology (as the fix now does) versus physically removing them (the old
        keep_mask behavior) DOES measurably change the loss attributable to the
        KEPT (real, nonzero-weight) edges, because TreeAttMsgLayer's segmented
        softmax attention couples siblings under the same parent. This is NOT
        asserted to be zero -- it is measured and reported. A nonzero delta here
        is expected and correct: it is exactly the train/inference topology
        alignment the fix is for, not a residual bug.
    """
    trajectories = real_history_fixture["trajectories"]
    schema = tree_encoder_feature_schema()

    found = _first_real_snapshot_pair_example(trajectories, schema, _LOOKAHEAD_K)
    assert found is not None, (
        "no real (tree, n, parent) triple in the fixture has both a never-visited and a "
        "genuinely-visited child under the same parent -- cannot isolate the coupling effect"
    )
    traj, n, full_example = found

    never_visited_mask = torch.tensor(
        [
            int(traj.node_update_ptr[c].item()) >= int(traj.node_update_ptr[c + 1].item())
            for c in full_example.edge_child.tolist()
        ],
        dtype=torch.bool,
    )
    assert full_example.edge_visit_weights is not None
    # Sanity: every never-visited child edge really does have weight exactly 0
    # (delta_visits over any window is provably 0 for a node with zero update-log
    # rows at all) -- if this fails, the premise of the poison check below is void.
    assert torch.all(full_example.edge_visit_weights[never_visited_mask] == 0.0), (
        "a never-visited child edge has nonzero edge_visit_weights -- delta_visits "
        "should be provably 0 for such a node regardless of (n, k)"
    )
    assert bool((~never_visited_mask).any()), "no kept (nonzero-weight-eligible) edges in this example"

    torch.manual_seed(0)
    model = ChildWdlModel(
        k=1, node_feat=len(schema.feature_names), device="cpu",
        node_embed_hidden=16, d_embed=16, d_message=16, n_heads=2, d_att=8, decoder_hidden=16,
    )
    model.eval()

    # --- (a) Poison check ---
    poisoned_example = TensorizedTreeExample(
        node_features=full_example.node_features.clone(),
        parent_index=full_example.parent_index.clone(),
        edge_parent=full_example.edge_parent.clone(),
        edge_child=full_example.edge_child.clone(),
        edge_slot=full_example.edge_slot.clone(),
        depth=full_example.depth.clone(),
        node_targets=full_example.node_targets.clone(),
        feature_names=full_example.feature_names,
        edge_wdl_targets=full_example.edge_wdl_targets.clone(),
        edge_visit_weights=full_example.edge_visit_weights.clone(),
    )
    # Adversarial one-hot target, deliberately as far as possible from a plausible
    # real WDL distribution, for every zero-weight (never-visited) edge only.
    poisoned_example.edge_wdl_targets[never_visited_mask] = torch.tensor([0.0, 0.0, 1.0])

    metrics_clean = _validate_single_example(model, schema, full_example)
    metrics_poisoned = _validate_single_example(model, schema, poisoned_example)

    loss_delta = abs(metrics_clean.total_loss - metrics_poisoned.total_loss)
    entropy_delta = abs(metrics_clean.target_entropy - metrics_poisoned.target_entropy)
    assert loss_delta < 1e-9, (
        f"poisoning a weight=0 edge's target changed total_loss by {loss_delta:.3e} "
        f"(clean={metrics_clean.total_loss!r}, poisoned={metrics_poisoned.total_loss!r}) -- "
        f"weight=0 is supposed to zero out this edge's contribution entirely"
    )
    assert entropy_delta < 1e-9, (
        f"poisoning a weight=0 edge's target changed target_entropy by {entropy_delta:.3e} -- "
        f"same expectation as total_loss above"
    )

    # --- (b) Coupling magnitude: full (fixed) topology vs. old-filtered topology ---
    kept_mask = ~never_visited_mask
    filtered_example = TensorizedTreeExample(
        node_features=full_example.node_features.clone(),
        parent_index=full_example.parent_index.clone(),
        edge_parent=full_example.edge_parent[kept_mask].clone(),
        edge_child=full_example.edge_child[kept_mask].clone(),
        edge_slot=full_example.edge_slot[kept_mask].clone(),
        depth=full_example.depth.clone(),
        node_targets=full_example.node_targets.clone(),
        feature_names=full_example.feature_names,
        edge_wdl_targets=full_example.edge_wdl_targets[kept_mask].clone(),
        edge_visit_weights=full_example.edge_visit_weights[kept_mask].clone(),
    )
    assert filtered_example.edge_parent.numel() == int(kept_mask.sum().item())
    # Note: kept_mask is "not never-visited-ANYWHERE" (the old filter's condition),
    # a strictly stronger/coarser test than "nonzero weight for THIS (n, k) window"
    # -- a kept edge can still legitimately have edge_visit_weights==0 if it was
    # visited outside the (n-1, n_plus_k-1] window, exactly as
    # build_snapshot_pair_example's own docstring notes ("exactly like any other
    # kept edge that happens to have zero Delta-visits for a given (n, k) already
    # did before this fix"). So no per-edge weight>0 invariant is asserted here.

    metrics_full_topology = _validate_single_example(model, schema, full_example)
    metrics_filtered_topology = _validate_single_example(model, schema, filtered_example)

    coupling_delta = abs(metrics_full_topology.total_loss - metrics_filtered_topology.total_loss)
    relative_delta = coupling_delta / max(abs(metrics_filtered_topology.total_loss), 1e-8)

    print(
        f"\n[loss-neutrality] weight=0 poison delta: loss={loss_delta:.3e} entropy={entropy_delta:.3e} "
        f"(exactly zero, as expected)\n"
        f"[loss-neutrality] topology coupling: full_topology_loss={metrics_full_topology.total_loss:.6f} "
        f"filtered_topology_loss={metrics_filtered_topology.total_loss:.6f} "
        f"abs_delta={coupling_delta:.6f} relative_delta={relative_delta:.4%} "
        f"(num_kept_edges={int(kept_mask.sum().item())}, num_filtered_edges={int(never_visited_mask.sum().item())})"
    )

    # The coupling effect is real (this is WHY the fix matters) -- confirm it's
    # actually nonzero here, i.e. this fixture is not a degenerate case where
    # attention happens not to move. If this assertion fails (delta == 0), it
    # would mean either the fixture's parent has too few real heads/message
    # dimensions to express any coupling, or TreeAttMsgLayer's attention is not
    # actually sensitive to sibling count here -- worth a second look, not silently
    # treated as "good news" without checking why.
    assert coupling_delta > 0.0, (
        "including previously-filtered edges in the topology produced NO measurable change in the "
        "kept edges' loss relative to the old-filtered topology -- either this fixture doesn't "
        "exercise the segmented-softmax coupling mechanism, or that mechanism is not live; either way "
        "this contradicts test_canary_pack_history_structural.py's "
        "test_never_visited_filter_couples_sibling_predictions_via_attention, which found it IS live"
    )


# --------------------------------------------------------------------------- #
# 4. Mutation test: the core invariant test must fail against the OLD filter
# --------------------------------------------------------------------------- #


def _pre_fix_filtered_edge_triples(traj: HistoryTrajectory, n: int) -> Set[Tuple[int, int, int]]:
    """Standalone reimplementation of the OLD (pre-fix) keep_mask filter, for
    mutation-testing purposes ONLY -- never used by any real assertion above.
    Deliberately does NOT call ``_never_visited`` (even though that helper still
    exists post-fix, see pack_history.py's own note on it) so this mutation stays
    meaningful even if a future change removes that helper entirely.
    """
    snap = _build_tree_n(traj, n)
    if snap is None:
        return set()
    kept = []
    for parent_id, child_id, slot in zip(snap.edge_parent.tolist(), snap.edge_child.tolist(), snap.edge_slot.tolist()):
        lo = int(traj.node_update_ptr[child_id].item())
        hi = int(traj.node_update_ptr[child_id + 1].item())
        never_visited = hi <= lo
        if not never_visited:
            kept.append((parent_id, child_id, slot))
    return set(kept)


def test_core_invariant_is_not_vacuous_against_reintroduced_old_filter(real_history_fixture):
    """Mutation test: proves test_core_invariant_snapshot_pair_edges_match_build_tree_n
    (and the shared _assert_edge_sets_match helper it uses) is not vacuous, by
    applying that SAME assertion helper to a deliberately reintroduced old
    (pre-fix, buggy) filtering reimplementation.

    Two-part proof:
      1. Against the reintroduced OLD filter -- the assertion must FAIL (raise
         AssertionError), because the old filter really does drop edges that
         _build_tree_n's real edge set contains.
      2. Against the REAL, current build_snapshot_pair_example (post-fix) -- the
         same assertion must PASS, on the exact same (trajectory, n) case.
    """
    trajectories = real_history_fixture["trajectories"]
    schema = tree_encoder_feature_schema()

    discriminating_case = None
    for traj in trajectories:
        num_steps = int(traj.step_node_cutoffs.shape[0])
        for n in _sampled_ns(num_steps, _LOOKAHEAD_K, count=12):
            snap = _build_tree_n(traj, n)
            if snap is None or snap.edge_child.numel() == 0:
                continue
            full_triples = _edge_triples(snap.edge_parent, snap.edge_child, snap.edge_slot)
            buggy_triples = _pre_fix_filtered_edge_triples(traj, n)
            if buggy_triples != full_triples:
                discriminating_case = (traj, n, full_triples, buggy_triples)
                break
        if discriminating_case is not None:
            break

    assert discriminating_case is not None, (
        "no (tree, n) in the fixture has any never-visited child edge -- the mutation can't be "
        "distinguished from correct behavior anywhere in this fixture"
    )
    traj, n, full_triples, buggy_triples = discriminating_case
    assert len(buggy_triples) < len(full_triples), "mutation helper did not actually drop any edges"

    # Part 1: the shared assertion helper must reject the reintroduced old filter.
    with pytest.raises(AssertionError):
        _assert_edge_sets_match(buggy_triples, full_triples, context=f"MUTATION {traj.source_path} n={n}")

    # Part 2: the same assertion helper must accept the real, current (fixed) code.
    example = build_snapshot_pair_example(traj, n, _LOOKAHEAD_K, schema)
    assert example is not None
    real_triples = _edge_triples(example.edge_parent, example.edge_child, example.edge_slot)
    _assert_edge_sets_match(real_triples, full_triples, context=f"REAL {traj.source_path} n={n}")
