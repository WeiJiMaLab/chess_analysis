"""Poisons the on-disk "final value/WDL" baseline array with an impossible sentinel
and runs it through every downstream per-step consumer: if the sentinel is
reachable from any per-step output, that consumer is leaking the future.

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_canary_future_leakage.py -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from cts.core.providers.parsers import value_features_from_wdl
from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_gnn.pack_history import (
    HistoryTrajectory,
    _build_tree_n,
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
from cts.train.controller_train import ControllerEpisodeDataset, collate_controller_episodes

_SPLIT_MANIFEST = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/split/train_manifest.txt")
_ENCODER_CHECKPOINT = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/packed/encoder_latest.pt")

# Impossible sentinel: real value/WDL features are always bounded in [-1, 1] (value)
# or [0, 1] (WDL fractions/variance) -- see value_features_from_wdl. -999 can never
# occur naturally anywhere in this pipeline.
_SENTINEL = -999.0
_N_FIXTURE_TREES = 16


def _require_real_data():
    if not _SPLIT_MANIFEST.is_file():
        pytest.skip(f"real split manifest not found at {_SPLIT_MANIFEST} -- cannot build canary fixture")


# --------------------------------------------------------------------------- #
# Fixture: build one small, REAL pack_history shard once per test session,
# via the actual production packing code (preprocess_mc.pack._pack_split), on
# real trees from xaba20k's own split. Written entirely under pytest's tmp_path
# tree -- never touches /scratch's real pack_history/mc_packed output.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_fixture(tmp_path_factory) -> Dict[str, Any]:
    _require_real_data()
    out_dir = tmp_path_factory.mktemp("canary_pack_history")

    with _SPLIT_MANIFEST.open() as handle:
        tree_paths = [line.strip() for line in handle if line.strip()][:_N_FIXTURE_TREES]
    if len(tree_paths) < _N_FIXTURE_TREES:
        pytest.skip("split manifest has fewer real trees than requested for the canary fixture")

    tiny_manifest = out_dir / "train_manifest.txt"
    tiny_manifest.write_text("\n".join(tree_paths) + "\n")

    # Mirrors config_ysagiv_xaba20k_history.yaml's mc_pack: section (see history.md),
    # except min_halt_reward_range/min_decision_margin relaxed to 0 (keep every one
    # of our small handful of trees -- we want real update logs, not a filtered
    # subset) and num_workers=0 (serial; simpler/deterministic for a test fixture).
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
        pytest.skip("fixture packing produced zero episodes -- cannot run canary tests")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "cts_budgeted_controller_episode_manifest_v4"
    shard_path = Path(manifest["entries"][0]["path"])
    shard_payload = torch.load(shard_path, weights_only=False)
    assert shard_payload["format"] == "cts_packhistory_trees_shard_v1", (
        "fixture did not produce the packhistory_trees shard format -- "
        "canary tests require the update log to be present"
    )

    return {
        "out_dir": out_dir,
        "manifest_path": manifest_path,
        "shard_path": shard_path,
        "total_episodes": total_episodes,
    }


@pytest.fixture(scope="module")
def payload(real_fixture) -> Dict[str, Any]:
    return torch.load(real_fixture["shard_path"], weights_only=False)


@pytest.fixture(scope="module")
def trajectories(payload) -> List[HistoryTrajectory]:
    return list(iter_history_trajectories(payload))


# --------------------------------------------------------------------------- #
# Poisoning helpers
# --------------------------------------------------------------------------- #


def _clone_trajectory(traj: HistoryTrajectory) -> HistoryTrajectory:
    """Deep-copy a HistoryTrajectory's tensors so mutations never bleed into the
    shared module-scoped `trajectories` fixture."""
    kwargs = {}
    for field in traj.__dataclass_fields__:
        value = getattr(traj, field)
        kwargs[field] = value.clone() if torch.is_tensor(value) else value
    return HistoryTrajectory(**kwargs)


def _poison_entire_baseline(payload: Dict[str, Any], sentinel: float = _SENTINEL) -> Dict[str, Any]:
    """Shallow-copy the shard payload dict, replacing ``node_features`` (the
    final_value/final_wdl baseline array -- see this file's module docstring) with
    an all-sentinel tensor of the same shape/dtype. Every other field is the SAME
    tensor object as the input (read-only for these tests, never mutated)."""
    poisoned = dict(payload)
    poisoned["node_features"] = torch.full_like(payload["node_features"], sentinel)
    return poisoned


def _write_shard_variant(out_dir: Path, name: str, manifest_path: Path, poisoned_payload: Dict[str, Any]) -> Path:
    """Write ``poisoned_payload`` as a new shard file and a manifest pointing at it
    (same structure as the original manifest, only the shard path replaced)."""
    manifest = json.loads(manifest_path.read_text())
    shard_out = out_dir / f"{name}.pt"
    torch.save(poisoned_payload, shard_out)
    manifest = dict(manifest)
    manifest["entries"] = [dict(entry, path=str(shard_out)) for entry in manifest["entries"]]
    manifest_out = out_dir / f"{name}_manifest.json"
    manifest_out.write_text(json.dumps(manifest))
    return manifest_out


def _global_update_log_index(payload: Dict[str, Any], traj_idx: int, node_id: int, step_index: int) -> int:
    """Map (trajectory index, local node id, step_index) to the row index into the
    shard's flat, concatenated ``update_log_*`` arrays -- the inverse of what
    ``iter_history_trajectories`` does when it slices those arrays per trajectory."""
    ulog_begin = int(payload["trajectory_update_log_ptr"][traj_idx].item())
    nup_ptr_begin = int(payload["trajectory_node_update_ptr_ptr"][traj_idx].item())
    node_update_ptr_full = payload["node_update_ptr"]
    lo_local = int(node_update_ptr_full[nup_ptr_begin + node_id].item())
    hi_local = int(node_update_ptr_full[nup_ptr_begin + node_id + 1].item())
    steps_local = payload["update_log_step_index"][ulog_begin + lo_local : ulog_begin + hi_local].tolist()
    pos = steps_local.index(step_index)
    return ulog_begin + lo_local + pos


def _find_node_with_step_gap(
    trajectories: List[HistoryTrajectory],
    min_gap: int,
    exclude: Tuple[Tuple[int, int], ...] = (),
) -> Optional[Tuple[int, int, int, int]]:
    """Find (trajectory_index, node_id, step_a, step_b) for some real node with two
    update-log entries at least ``min_gap`` steps apart. Deterministic given a fixed
    fixture (no randomness), so re-running this file against the same 16 trees
    always finds the same candidate."""
    for traj_idx, traj in enumerate(trajectories):
        ptr = traj.node_update_ptr.tolist()
        num_nodes = traj.node_features.shape[0]
        for node_id in range(1, num_nodes):
            if (traj_idx, node_id) in exclude:
                continue
            lo, hi = ptr[node_id], ptr[node_id + 1]
            if hi - lo < 2:
                continue
            steps = traj.update_step_index[lo:hi].tolist()
            for i in range(len(steps) - 1):
                if steps[i + 1] - steps[i] >= min_gap:
                    return traj_idx, node_id, steps[i], steps[i + 1]
    return None


# --------------------------------------------------------------------------- #
# Canary 1: baseline poisoning through ControllerEpisodeDataset
# --------------------------------------------------------------------------- #


def test_canary1_dataset_baseline_leak(real_fixture, payload):
    """Poison the ENTIRE final_value/final_wdl baseline array (node_features) with
    an impossible sentinel and confirm ControllerEpisodeDataset.__getitem__'s output
    is bit-for-bit IDENTICAL to the unpoisoned run, for every step of every episode.

    This is the exact bug fixed in controller_train.py 2026-07-10: a node not yet
    individually backed-up as of a queried step was falling back to this same
    baseline array (its FINAL, end-of-search value) instead of reading 0. If that
    regressed, this poison would appear in the dataset's output and this test would
    fail; today, with the fix in place, the baseline is provably irrelevant.
    """
    poisoned_payload = _poison_entire_baseline(payload)
    poisoned_manifest_path = _write_shard_variant(
        real_fixture["out_dir"], "canary1_poisoned", real_fixture["manifest_path"], poisoned_payload
    )

    ds_clean = ControllerEpisodeDataset(str(real_fixture["manifest_path"]))
    ds_poisoned = ControllerEpisodeDataset(str(poisoned_manifest_path))
    assert len(ds_clean) == len(ds_poisoned) == real_fixture["total_episodes"]

    any_nonzero = False
    for index in range(len(ds_clean)):
        episode_clean = ds_clean[index]
        episode_poisoned = ds_poisoned[index]
        assert episode_clean.source_path == episode_poisoned.source_path
        assert len(episode_clean.step_node_features) == len(episode_poisoned.step_node_features)
        for step_index, (clean_nf, poisoned_nf) in enumerate(
            zip(episode_clean.step_node_features, episode_poisoned.step_node_features)
        ):
            assert torch.equal(clean_nf, poisoned_nf), (
                f"future leakage regression: episode {index} ({episode_clean.source_path}) step "
                f"{step_index} differs between the clean run and the baseline-poisoned run -- "
                f"the fallback in ControllerEpisodeDataset.__getitem__ has regressed to reading "
                f"the poisoned final_value/final_wdl baseline."
            )
            # Literal sentinel-absence check, as specified: never -999 anywhere.
            assert not torch.any(torch.isclose(poisoned_nf, torch.tensor(_SENTINEL), atol=1.0)), (
                f"sentinel leaked directly into episode {index} step {step_index}: {poisoned_nf}"
            )
            if torch.any(poisoned_nf != 0):
                any_nonzero = True

    # Sanity: confirm this isn't a vacuous pass from a degenerate always-zero dataset.
    assert any_nonzero, "canary produced only all-zero rows everywhere -- test fixture too degenerate to be meaningful"


# --------------------------------------------------------------------------- #
# Canary 2: baseline poisoning through pack_history.py
# --------------------------------------------------------------------------- #


def test_canary2_pack_history_baseline_leak(payload, trajectories):
    """Same poison as canary 1 (the whole node_features baseline array), same
    bit-identity proof, for the SECOND, independently-fixed consumer:
    pack_history.py's _build_tree_n / _forward_filled_wdl_at_step (the GNN
    pretraining input builder). A regression here is a DIFFERENT bug (different
    file, different fix commit) that canary 1 would not catch.
    """
    poisoned_payload = _poison_entire_baseline(payload)
    poisoned_trajectories = list(iter_history_trajectories(poisoned_payload))
    assert len(poisoned_trajectories) == len(trajectories)

    checked = 0
    any_nonzero = False
    for traj_clean, traj_poisoned in zip(trajectories, poisoned_trajectories):
        num_steps = int(traj_clean.step_node_cutoffs.shape[0])
        # Every 5th step is plenty (this mirrors test_no_future_leakage.py's own
        # sampling density for its exhaustive-episode check) -- still hundreds of
        # (tree, step) comparisons across the whole fixture.
        for n in range(1, num_steps + 1, 5):
            snap_clean = _build_tree_n(traj_clean, n)
            snap_poisoned = _build_tree_n(traj_poisoned, n)
            assert snap_clean is not None and snap_poisoned is not None
            checked += 1
            assert torch.equal(snap_clean.node_features, snap_poisoned.node_features), (
                f"future leakage regression: tree {traj_clean.source_path} n={n} differs between "
                f"the clean run and the baseline-poisoned run -- _build_tree_n has regressed to "
                f"reading the poisoned final_value/final_wdl baseline."
            )
            assert not torch.any(torch.isclose(snap_poisoned.node_features, torch.tensor(_SENTINEL), atol=1.0)), (
                f"sentinel leaked directly into tree {traj_clean.source_path} n={n}"
            )
            if torch.any(snap_poisoned.node_features != 0):
                any_nonzero = True

    assert checked > 100, f"sanity check on the check itself -- only tested {checked} snapshots"
    assert any_nonzero, "canary produced only all-zero rows everywhere -- test fixture too degenerate to be meaningful"


# --------------------------------------------------------------------------- #
# Canary 3: target-vs-input separation for the T_n -> T_{n+k} GNN pretraining pair
# --------------------------------------------------------------------------- #


def test_canary3_target_input_separation(trajectories):
    """Poison ONLY the update-log entry that falls in the "future" window (n, n+k]
    used as build_snapshot_pair_example's WDL TARGET, and confirm the sentinel
    appears ONLY in edge_wdl_targets, never in T_n's own INPUT node_features for the
    same example.

    This is a different bug shape than canaries 1/2: not "did the final-value
    baseline leak" but "does the future-relative training TARGET ever accidentally
    get copied into the model's INPUT" (e.g. a copy-paste / shared-tensor-reference
    bug between the two halves of build_snapshot_pair_example).
    """
    candidate = _find_node_with_step_gap(trajectories, min_gap=20)
    assert candidate is not None, "no real node in the canary fixture has update-log entries >=20 steps apart"
    traj_idx, node_id, step_a, step_b = candidate

    traj = _clone_trajectory(trajectories[traj_idx])
    lo = int(traj.node_update_ptr[node_id].item())
    hi = int(traj.node_update_ptr[node_id + 1].item())
    steps_local = traj.update_step_index[lo:hi].tolist()
    pos_b = steps_local.index(step_b)
    real_wdl_b = traj.update_wdl[lo + pos_b].clone()
    traj.update_wdl[lo + pos_b] = torch.tensor([_SENTINEL, _SENTINEL, _SENTINEL])

    # n chosen so local_step = n-1 == step_a (T_n's input forward-fill only sees
    # entries with step_index <= step_a, i.e. NOT the poisoned step_b entry).
    # k chosen so n+k-1 == step_b (the target forward-fill DOES land on it).
    n = step_a + 1
    lookahead_k = step_b - n + 1
    assert n - 1 == step_a and n + lookahead_k - 1 == step_b

    schema = tree_encoder_feature_schema()

    # --- Input side: must be completely clean. ---
    snapshot = _build_tree_n(traj, n)
    assert snapshot is not None and node_id < snapshot.node_features.shape[0]
    assert snapshot.node_features.abs().max().item() < 100.0, (
        f"future leakage: the poisoned future-window update (step {step_b}) leaked into "
        f"T_n's own INPUT node_features (n={n}): {snapshot.node_features[node_id].tolist()}"
    )

    # --- Target side, raw (pre-normalization): must show the poison (positive control). ---
    raw_target = _forward_filled_wdl_at_step(traj, torch.tensor([node_id]), n + lookahead_k - 1)
    assert torch.allclose(raw_target[0], torch.tensor([_SENTINEL, _SENTINEL, _SENTINEL])), (
        "positive control failed: the target window's forward-fill query did not pick up the "
        "poisoned entry -- this test isn't exercising the intended code path"
    )

    # --- Full production function, end to end. ---
    example = build_snapshot_pair_example(traj, n, lookahead_k, schema)
    assert example is not None
    assert example.node_features.abs().max().item() < 100.0, (
        "future leakage: build_snapshot_pair_example's INPUT node_features reflect the poisoned "
        "future-window target"
    )
    edge_positions = (example.edge_child == node_id).nonzero(as_tuple=True)[0]
    assert edge_positions.numel() > 0, "poisoned node's edge was filtered out of the example -- pick a different candidate"
    # clamp_min(1e-8) on a negative-sum sentinel triple blows the normalized target up
    # to roughly sentinel / 1e-8 -- unmistakably outside any real WDL fraction's range.
    assert example.edge_wdl_targets[edge_positions].abs().max().item() > 100.0, (
        "positive control failed: edge_wdl_targets for the poisoned edge doesn't show the expected "
        "poison signature -- this test isn't exercising the intended code path"
    )


# --------------------------------------------------------------------------- #
# Canary 4: forward-fill respects step order, not array/insertion order
# --------------------------------------------------------------------------- #


def test_canary4_step_order_respected(trajectories):
    """Poison one LATER update-log entry for a real node and confirm forward-filled
    queries at every step strictly before it never see the sentinel (proving the
    lookup keys off step_index via a genuine ordered search, not e.g. "the last
    entry by array position" or an off-by-one that includes one extra step) -- then
    confirm a query at/after that step DOES see it (positive control, proving the
    poison is actually reachable and this isn't a vacuous pass).
    """
    candidate = _find_node_with_step_gap(trajectories, min_gap=8)
    assert candidate is not None, "no real node in the canary fixture has update-log entries >=8 steps apart"
    traj_idx, node_id, _step_a, step_b = candidate

    traj = _clone_trajectory(trajectories[traj_idx])
    lo = int(traj.node_update_ptr[node_id].item())
    hi = int(traj.node_update_ptr[node_id + 1].item())
    steps_local = traj.update_step_index[lo:hi].tolist()
    pos_b = steps_local.index(step_b)
    traj.update_wdl[lo + pos_b] = torch.tensor([_SENTINEL, _SENTINEL, _SENTINEL])

    for query_step in range(0, step_b):
        out = _forward_filled_wdl_at_step(traj, torch.tensor([node_id]), query_step)
        assert not torch.any(torch.isclose(out, torch.tensor(_SENTINEL), atol=1.0)), (
            f"future leakage regression: forward-fill query at step={query_step} (strictly before "
            f"the poisoned entry's step={step_b}) returned the poisoned value {out.tolist()} -- "
            f"the lookup is not correctly respecting step order."
        )

    # Positive control: at and after step_b, the poison must be visible.
    for query_step in (step_b, step_b + 1):
        if query_step >= 96:
            continue
        out = _forward_filled_wdl_at_step(traj, torch.tensor([node_id]), query_step)
        assert torch.allclose(out[0], torch.tensor([_SENTINEL, _SENTINEL, _SENTINEL])), (
            f"positive control failed: forward-fill query at step={query_step} (>= the poisoned "
            f"entry's step={step_b}) did not return the poisoned value -- this test isn't "
            f"exercising the intended code path"
        )


# --------------------------------------------------------------------------- #
# Canary 5 (Part 3): positive/sensitivity check -- the RIGHT value must get IN
# --------------------------------------------------------------------------- #


def test_canary5_perturbation_propagates(real_fixture, payload, trajectories):
    """The mirror-image of canaries 1-4: those all check "does the wrong (future or
    poisoned) value stay OUT". This checks "does the RIGHT (current, real) value
    actually get IN" -- a pipeline that always emitted 0 (or any other constant)
    would pass every leakage test above trivially while being completely broken in
    this other direction.

    Takes a real node's real WDL at a real update step, replaces it with a
    PERTURBED-but-valid distribution, and confirms all four derived per-step
    quantities (value, wdl_win/draw/loss, wdl_var) reflect the perturbation for
    exactly the steps it covers -- not before its own step, and not after the
    node's NEXT real update (i.e. a correctly SCOPED forward-fill, not a global
    overwrite).
    """
    candidate = _find_node_with_step_gap(trajectories, min_gap=15)
    assert candidate is not None
    traj_idx, node_id, step_a, step_b = candidate
    traj = trajectories[traj_idx]

    lo = int(traj.node_update_ptr[node_id].item())
    real_wdl_a = traj.update_wdl[lo + traj.update_step_index[lo : int(traj.node_update_ptr[node_id + 1].item())].tolist().index(step_a)]

    perturbed = torch.tensor([0.7, 0.1, 0.2])
    assert not torch.allclose(perturbed, real_wdl_a, atol=1e-2), "perturbation collided with the real value -- pick a different one"
    p_win, p_draw, p_loss = (float(x) for x in perturbed)
    expected_value = p_win - p_loss
    expected_features = value_features_from_wdl(p_win, p_draw, p_loss)

    poisoned_payload = dict(payload)
    poisoned_payload["update_log_wdl"] = payload["update_log_wdl"].clone()
    poisoned_payload["update_log_q_value"] = payload["update_log_q_value"].clone()
    global_index = _global_update_log_index(payload, traj_idx, node_id, step_a)
    assert int(payload["update_log_node_id"][global_index].item()) == node_id
    poisoned_payload["update_log_wdl"][global_index] = perturbed
    poisoned_payload["update_log_q_value"][global_index] = expected_value

    poisoned_manifest_path = _write_shard_variant(
        real_fixture["out_dir"], "canary5_perturbed", real_fixture["manifest_path"], poisoned_payload
    )

    # --- ControllerEpisodeDataset ---
    dataset = ControllerEpisodeDataset(str(poisoned_manifest_path))
    episode = None
    for index in range(len(dataset)):
        candidate_episode = dataset[index]
        if candidate_episode.source_path == traj.source_path:
            episode = candidate_episode
            break
    assert episode is not None, "could not find the poisoned trajectory's episode in the rebuilt dataset"

    first_decision = traj.first_decision_expansion_count
    checked_active = checked_before = checked_after = 0
    for local_step, node_features in enumerate(episode.step_node_features):
        as_of_step = first_decision - 1 + local_step
        if node_id >= node_features.shape[0]:
            continue
        value, wdl_win, wdl_draw, wdl_loss, wdl_var = (float(x) for x in node_features[node_id])
        if as_of_step < step_a:
            checked_before += 1
            assert value == pytest.approx(0.0, abs=1e-9) and wdl_win == pytest.approx(0.0, abs=1e-9), (
                f"local_step={local_step} (before the perturbed step {step_a}) is not zero: "
                f"{node_features[node_id].tolist()}"
            )
        elif step_a <= as_of_step < step_b:
            checked_active += 1
            assert value == pytest.approx(expected_value, abs=1e-4), f"local_step={local_step}: value mismatch"
            assert wdl_win == pytest.approx(p_win, abs=1e-4), f"local_step={local_step}: wdl_win mismatch"
            assert wdl_draw == pytest.approx(p_draw, abs=1e-4), f"local_step={local_step}: wdl_draw mismatch"
            assert wdl_loss == pytest.approx(p_loss, abs=1e-4), f"local_step={local_step}: wdl_loss mismatch"
            assert wdl_var == pytest.approx(expected_features["wdl_var"], abs=1e-4), (
                f"local_step={local_step}: wdl_var did not recompute from the perturbed wdl triple "
                f"(this is exactly the second, independently-found leak fixed 2026-07-10)"
            )
        else:
            checked_after += 1
            # Must NOT still show the perturbation (would mean it bled forward past
            # the node's next real update at step_b).
            assert not (wdl_win == pytest.approx(p_win, abs=1e-4) and wdl_loss == pytest.approx(p_loss, abs=1e-4)), (
                f"local_step={local_step} (at/after the node's NEXT real update, step {step_b}) still "
                f"shows the perturbed value -- forward-fill is not correctly scoped, it bled forward."
            )

    assert checked_before > 0 and checked_active > 0 and checked_after > 0, (
        "fixture didn't exercise all three regions (before/active/after) -- pick a different candidate"
    )

    # --- pack_history.py: same three-region check via _build_tree_n. ---
    poisoned_trajectories = list(iter_history_trajectories(poisoned_payload))
    poisoned_traj = poisoned_trajectories[traj_idx]
    num_steps = int(poisoned_traj.step_node_cutoffs.shape[0])
    checked_active_ph = 0
    for n in range(1, num_steps + 1):
        as_of_step = n - 1
        if not (step_a <= as_of_step < step_b):
            continue
        snapshot = _build_tree_n(poisoned_traj, n)
        if snapshot is None or node_id >= snapshot.node_features.shape[0]:
            continue
        checked_active_ph += 1
        value, wdl_win = float(snapshot.node_features[node_id, 0]), float(snapshot.node_features[node_id, 1])
        assert value == pytest.approx(expected_value, abs=1e-4), f"pack_history n={n}: value mismatch"
        assert wdl_win == pytest.approx(p_win, abs=1e-4), f"pack_history n={n}: wdl_win mismatch"
    assert checked_active_ph > 0, "pack_history side of the check never ran"


# --------------------------------------------------------------------------- #
# Canaries 6/7 (best-effort): the same two checks through a REAL trained encoder
# forward pass, not just the raw feature tensors.
# --------------------------------------------------------------------------- #


def _load_real_encoder():
    from cts.models.gnn import TreeEncoder
    from cts.train.gnn_pretrain import load_encoder_architecture, load_encoder_checkpoint

    if not _ENCODER_CHECKPOINT.is_file():
        pytest.skip(f"no trained encoder checkpoint at {_ENCODER_CHECKPOINT} -- skipping encoder-level canary")
    architecture = load_encoder_architecture(str(_ENCODER_CHECKPOINT))
    encoder = TreeEncoder(device="cpu", **architecture)
    load_encoder_checkpoint(str(_ENCODER_CHECKPOINT), encoder)
    encoder.eval()
    return encoder


def test_canary6_encoder_baseline_leak(real_fixture, payload):
    """Extends canary 1 through materialize.py's actual encode path
    (MetaController.encode_with_state_features / TreeEncoder.forward): poison the
    whole final_value/final_wdl baseline, run a real trained encoder over both the
    clean and poisoned episodes, and confirm the resulting root embedding (z_root)
    is bit-identical -- i.e. the poison doesn't reach what the RL controller
    actually sees, not just the intermediate feature tensor nobody reads downstream.
    """
    encoder = _load_real_encoder()

    poisoned_payload = _poison_entire_baseline(payload)
    poisoned_manifest_path = _write_shard_variant(
        real_fixture["out_dir"], "canary6_poisoned", real_fixture["manifest_path"], poisoned_payload
    )

    ds_clean = ControllerEpisodeDataset(str(real_fixture["manifest_path"]))
    ds_poisoned = ControllerEpisodeDataset(str(poisoned_manifest_path))

    episode_clean = ds_clean[0]
    episode_poisoned = ds_poisoned[0]
    batch_clean = collate_controller_episodes([episode_clean])
    batch_poisoned = collate_controller_episodes([episode_poisoned])
    assert batch_clean is not None and batch_poisoned is not None

    with torch.inference_mode():
        out_clean = encoder(batch_clean.tree_batch)
        out_poisoned = encoder(batch_poisoned.tree_batch)

    assert torch.equal(out_clean.root_states, out_poisoned.root_states), (
        "future leakage regression: the trained encoder's z_root differs between the clean and "
        "baseline-poisoned runs -- the poison reached the encoder's input despite canary 1 passing "
        "at the raw-feature level (would indicate a leak introduced downstream of "
        "ControllerEpisodeDataset, e.g. in collate_controller_episodes)."
    )
    assert torch.equal(out_clean.node_states, out_poisoned.node_states)


def test_canary7_encoder_perturbation_propagates(real_fixture, payload, trajectories):
    """Extends canary 5 through the same real encoder forward pass: a genuine,
    valid perturbation to one node's WDL must make z_root for the perturbed episode
    differ measurably from the unperturbed one -- confirming per-step value changes
    actually propagate all the way to what the controller sees, not just sit
    correctly in an intermediate tensor.
    """
    encoder = _load_real_encoder()

    candidate = _find_node_with_step_gap(trajectories, min_gap=15)
    assert candidate is not None
    traj_idx, node_id, step_a, _step_b = candidate
    traj = trajectories[traj_idx]

    lo = int(traj.node_update_ptr[node_id].item())
    real_wdl_a = traj.update_wdl[lo + traj.update_step_index[lo : int(traj.node_update_ptr[node_id + 1].item())].tolist().index(step_a)]
    perturbed = torch.tensor([0.7, 0.1, 0.2])
    assert not torch.allclose(perturbed, real_wdl_a, atol=1e-2)
    expected_value = float(perturbed[0] - perturbed[2])

    perturbed_payload = dict(payload)
    perturbed_payload["update_log_wdl"] = payload["update_log_wdl"].clone()
    perturbed_payload["update_log_q_value"] = payload["update_log_q_value"].clone()
    global_index = _global_update_log_index(payload, traj_idx, node_id, step_a)
    perturbed_payload["update_log_wdl"][global_index] = perturbed
    perturbed_payload["update_log_q_value"][global_index] = expected_value

    perturbed_manifest_path = _write_shard_variant(
        real_fixture["out_dir"], "canary7_perturbed", real_fixture["manifest_path"], perturbed_payload
    )

    ds_clean = ControllerEpisodeDataset(str(real_fixture["manifest_path"]))
    ds_perturbed = ControllerEpisodeDataset(str(perturbed_manifest_path))

    def _find_episode(dataset):
        for index in range(len(dataset)):
            episode = dataset[index]
            if episode.source_path == traj.source_path:
                return episode
        raise AssertionError("trajectory not found in rebuilt dataset")

    episode_clean = _find_episode(ds_clean)
    episode_perturbed = _find_episode(ds_perturbed)
    batch_clean = collate_controller_episodes([episode_clean])
    batch_perturbed = collate_controller_episodes([episode_perturbed])

    with torch.inference_mode():
        out_clean = encoder(batch_clean.tree_batch)
        out_perturbed = encoder(batch_perturbed.tree_batch)

    assert not torch.equal(out_clean.root_states, out_perturbed.root_states), (
        "perturbing one real node's WDL at one real step did not change ANY root embedding across "
        "the whole episode -- either the perturbation never reached the encoder's input, or the "
        "encoder is insensitive to it end to end (both would hide a real leak-direction bug)."
    )
