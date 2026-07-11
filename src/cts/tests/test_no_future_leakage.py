"""Regression test: nodes not yet individually backed up as of a queried step must
read 0.0 ("no signal yet"), not their final end-of-search value -- checked in
``ControllerEpisodeDataset`` and ``pack_history._forward_filled_wdl_at_step`` on
real data.

Fixtures (episode index, node id, first-update step, etc.) are DISCOVERED
DYNAMICALLY against whatever real trees live under ``_PACK_HISTORY_ROOT`` --
they are not hardcoded to any specific tree. This was repointed 2026-07-11
from ``ysagiv_xaba20k`` (whose ``pack_history/`` output was purged in an
earlier cleanup and is not being regenerated) to
``ysagiv_xaba100k_minply15_maxply75``, which has both ``pack_history/`` and
``mc_packed/`` on disk, is a fresher run than xaba20k's ever was, and
postdates every correctness fix landed so far. Since xaba100k's trees/split
differ from xaba20k's, any specific (episode_index, node_id, step) triple
that held for the old corpus does not generally hold for the new one --
hence the dynamic search below, mirroring the search pattern already used by
``test_pack_history_no_leak_before_first_visit``.

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_no_future_leakage.py -v
"""
from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch

# Repointed from ysagiv_xaba20k/pack_history (purged) -- see module docstring.
_PACK_HISTORY_ROOT = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba100k_minply15_maxply75/pack_history")
_TRAIN_MANIFEST = _PACK_HISTORY_ROOT / "train_manifest.json"

# Search budgets for dynamic fixture discovery (see module docstring for why
# these are searched for rather than hardcoded).
_EPISODE_SEARCH_BUDGET = 50
_MIN_PRE_UPDATE_GAP = 5  # require the node to sit structurally-present-but-zero
                          # for at least this many steps before its first update --
                          # i.e. a REAL "created well before first visit" gap, not
                          # a node that happens to update the instant it's created.
_MIN_UPDATE_LOG_ENTRIES = 2  # a node visited exactly once trivially has
                              # row_first == row_last (forward-fill of a single
                              # entry is constant from that step to the end by
                              # construction) -- not a leak, just a fixture that
                              # can't demonstrate "the value keeps evolving".
                              # Confirmed empirically 2026-07-11: the first
                              # dynamically-discovered fixture (episode 0, node
                              # 40) had exactly one update-log entry (single
                              # step=66, q_value=0.2529296875), which is why
                              # test_no_leak_before_first_visit's "keeps
                              # evolving" assertion failed on a bit-exact match
                              # -- a benign fixture-selection gap, not a real
                              # leak. See _node_update_log_entry_count below.


def _require_real_pack_history():
    if not _TRAIN_MANIFEST.is_file():
        pytest.skip(f"real pack_history/ output not found at {_PACK_HISTORY_ROOT} -- run packhistory_trees first")


def _node_update_log_entry_count(ds, episode_index: int, node_id: int) -> int:
    """Count how many times ``node_id`` appears in the REAL update log of the
    packed pack_history/ shard backing ``episode_index`` -- i.e. how many
    times it was actually backed up during search, independent of anything
    ``ControllerEpisodeDataset`` derives from that log.

    Replicates ``ControllerEpisodeDataset.__getitem__``'s own shard/trajectory
    mapping (``controller_train.py``) rather than importing anything private
    from it, since that mapping isn't otherwise exposed. Used to require a
    fixture node have been visited more than once -- see
    ``_MIN_UPDATE_LOG_ENTRIES``.
    """
    shard_index = bisect_right(ds.cumulative_sizes, episode_index)
    shard_start = 0 if shard_index == 0 else ds.cumulative_sizes[shard_index - 1]
    episode_offset = episode_index - shard_start
    payload = ds._load_shard(shard_index)

    episode_trajectory_index = payload["episode_trajectory_index"]
    trajectory_index = int(episode_trajectory_index[episode_offset].item())

    update_log_node_id_all = payload.get("update_log_node_id")
    if update_log_node_id_all is None:
        return 0
    trajectory_update_log_ptr = payload["trajectory_update_log_ptr"]
    log_begin = int(trajectory_update_log_ptr[trajectory_index].item())
    log_end = int(trajectory_update_log_ptr[trajectory_index + 1].item())
    log_node_id = update_log_node_id_all[log_begin:log_end]
    return int((log_node_id == node_id).sum().item())


def _find_leak_fixture(ds, episode_budget: int = _EPISODE_SEARCH_BUDGET) -> Optional[Tuple[int, int, int]]:
    """Search real episodes for a (episode_index, node_id, first_update_local_step)
    triple where node_id genuinely exists (all-zero row) for several steps before
    its first real (non-zero) update -- i.e. a real instance of the "no leak
    before first visit" scenario, not a node that updates the instant it's
    created (which would make the pre-update loop vacuous) -- AND has been
    backed up more than once (see ``_MIN_UPDATE_LOG_ENTRIES``), so that
    "the value keeps evolving after first update" is a claim that can
    meaningfully be true or false, rather than trivially false for a
    visited-exactly-once node.

    Mirrors the search pattern already used by
    ``test_pack_history_no_leak_before_first_visit`` in this same file, just
    against ``ControllerEpisodeDataset`` instead of ``HistoryTrajectory``.
    """
    for ep_idx in range(min(episode_budget, len(ds))):
        episode = ds[ep_idx]
        step_node_features = episode.step_node_features
        num_steps = len(step_node_features)
        if num_steps == 0:
            continue
        n_nodes_final = step_node_features[-1].shape[0]
        for node_id in range(1, n_nodes_final):  # node 0 is the root -- structurally always zero, not this scenario
            zero_streak = 0
            first_update = None
            for local_step in range(num_steps):
                nf = step_node_features[local_step]
                if node_id >= nf.shape[0]:
                    continue  # not yet structurally created
                row = nf[node_id]
                if torch.allclose(row, torch.zeros_like(row), atol=1e-9):
                    zero_streak += 1
                    continue
                first_update = local_step
                break
            if first_update is None or zero_streak < _MIN_PRE_UPDATE_GAP:
                continue
            if _node_update_log_entry_count(ds, ep_idx, node_id) < _MIN_UPDATE_LOG_ENTRIES:
                continue  # visited exactly once -- can't demonstrate "keeps evolving", keep searching
            return ep_idx, node_id, first_update
    return None


def _find_root_multi_child_episode(ds, episode_budget: int = 10) -> Optional[int]:
    """Search a handful of real episodes for one whose root has accumulated
    more than one child by the final step -- the precondition
    ``test_root_always_zero`` needs to be a non-vacuous check of "not a leaf
    but still zero", rather than assuming a specific episode index.
    """
    for ep_idx in range(min(episode_budget, len(ds))):
        episode = ds[ep_idx]
        step_parent_index = episode.step_parent_index
        if len(step_parent_index) == 0:
            continue
        last_pi = step_parent_index[-1]
        if int((last_pi == 0).sum()) > 1:
            return ep_idx
    return None


def test_no_leak_before_first_visit():
    """ControllerEpisodeDataset.__getitem__ (controller_train.py): a node's fed
    value must be exactly 0.0 for every step before its own first update-log
    entry, and must match the real forward-filled value from that step onward --
    never the node's end-of-search final value while still "unvisited so far"."""
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    fixture = _find_leak_fixture(ds)
    if fixture is None:
        pytest.skip(
            f"could not find an (episode, node) pair with a >= {_MIN_PRE_UPDATE_GAP}-step "
            f"pre-update gap in the first {_EPISODE_SEARCH_BUDGET} episodes of {_TRAIN_MANIFEST}"
        )
    episode_index, node_id, first_update_local_step = fixture

    episode = ds[episode_index]
    step_node_features = episode.step_node_features
    assert node_id < step_node_features[-1].shape[0], "fixture node id out of range -- pick a different (episode, node)"

    # Before the node's own first backprop update: must read as "no signal yet" (0),
    # not its eventual final value. This is the exact leak that was fixed.
    for local_step in range(0, first_update_local_step):
        nf = step_node_features[local_step]
        if node_id >= nf.shape[0]:
            continue  # node not yet structurally created at this step -- not the case under test
        row = nf[node_id]
        assert torch.allclose(row, torch.zeros_like(row), atol=1e-9), (
            f"future leakage regression: node {node_id} (episode {episode_index}) at "
            f"local_step={local_step} (before its first real update at "
            f"{first_update_local_step}) shows row={row.tolist()}, expected all-zero "
            f"(no info yet, including wdl_var). If this fails, the fallback in "
            f"ControllerEpisodeDataset.__getitem__ has regressed to reading the "
            f"final-value baseline again."
        )

    # From the node's first real update onward, it must be genuinely non-zero and
    # must evolve (not itself be frozen) -- confirms this isn't just "always 0".
    nf_first = step_node_features[first_update_local_step]
    nf_last = step_node_features[-1]
    value_first = float(nf_first[node_id, 0])
    value_last = float(nf_last[node_id, 0])
    assert value_first != pytest.approx(0.0, abs=1e-9), "expected a real value at the node's first update step"
    assert value_first != pytest.approx(value_last, abs=1e-6), (
        "expected the value to keep evolving after first update, not jump straight to (and freeze at) "
        "the final value -- would indicate a different leak"
    )


def test_wdl_var_tracks_current_step():
    """Second, independently-found leak (2026-07-10, same day as the first): wdl_var
    (column 4) is a pure function of the wdl triple (value_features_from_wdl:
    internally, value := p_win - p_loss, variance := (p_win+p_loss) - value**2)
    and must be recomputed from the CURRENT per-step wdl, not left frozen at the
    node's original pre-search variance while wdl_win/wdl_draw/wdl_loss correctly
    evolve.

    IMPORTANT (found 2026-07-11 while validating this dynamically-discovered
    fixture): the recomputation below must use ``p_win - p_loss`` -- the same
    quantity ``value_features_from_wdl`` itself derives internally to compute
    wdl_var -- NOT the row's own ``value`` column. The ``value`` column is
    populated from ``update_log_q_value``, an independently-tracked running
    Q-value average that is closely correlated with but not algebraically
    identical to ``p_win - p_loss`` (both are ultimately float16-quantized
    somewhere upstream, per ``RawPretrainExampleRecord.__post_init__`` in
    teacher_targets.py, so they can differ by a few ~1e-4 float16 ulps even
    when everything is working correctly). Using the ``value`` column here
    produced a spurious ~1.2e-4 "mismatch" against a real fixture (episode 0,
    node 42) whose wdl_var was independently confirmed to be genuinely
    tracking the current step (changing at each of the node's 3 real
    updates, not frozen) -- i.e. a mismatched-comparison-basis bug in the
    test, not a production regression. Recomputing from p_win - p_loss makes
    this an exact reproduction of production's own formula (up to float32
    rounding), so the tolerance below is tightened accordingly rather than
    loosened to paper over the mismatch.
    """
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    fixture = _find_leak_fixture(ds)
    if fixture is None:
        pytest.skip(
            f"could not find an (episode, node) pair with a >= {_MIN_PRE_UPDATE_GAP}-step "
            f"pre-update gap in the first {_EPISODE_SEARCH_BUDGET} episodes of {_TRAIN_MANIFEST}"
        )
    episode_index, node_id, first_update_local_step = fixture

    episode = ds[episode_index]
    step_node_features = episode.step_node_features
    num_steps = len(step_node_features)

    # Sample the first-update step plus a spread of later steps (roughly 1/4,
    # 1/2 way to the end, and the final step) -- same spirit as the original
    # fixed-offset spot-checks, just computed relative to the dynamically
    # discovered first_update_local_step instead of hardcoded absolute steps.
    candidate_steps = sorted(
        {
            first_update_local_step,
            first_update_local_step + (num_steps - first_update_local_step) // 4,
            first_update_local_step + (num_steps - first_update_local_step) // 2,
            num_steps - 1,
        }
    )

    checked_any = False
    for local_step in candidate_steps:
        if local_step < 0 or local_step >= num_steps:
            continue
        nf = step_node_features[local_step]
        if node_id >= nf.shape[0]:
            continue
        row = nf[node_id]
        _value_col, p_win, p_draw, p_loss, wdl_var = (float(x) for x in row)
        # Deliberately NOT using _value_col (== update_log_q_value, a separate
        # running statistic) -- see the docstring above. wdl_value/expected_var
        # mirror value_features_from_wdl's own internal derivation exactly.
        wdl_value = p_win - p_loss
        expected_var = (p_win + p_loss) - wdl_value * wdl_value
        assert wdl_var == pytest.approx(expected_var, abs=1e-6), (
            f"wdl_var regression: node {node_id} (episode {episode_index}) at "
            f"local_step={local_step} has wdl_var={wdl_var:.6f} but "
            f"(p_win+p_loss)-(p_win-p_loss)**2={expected_var:.6f} from the SAME "
            f"step's wdl triple -- wdl_var is not tracking the current wdl, it "
            f"has regressed to a frozen baseline again."
        )
        checked_any = True
    assert checked_any, "fixture node never got a real update in this episode -- pick a different (episode, node)"


def test_unvisited_leaves_all_zero():
    """Strongest, most general form of the check: at EVERY step, for EVERY node
    that is still a leaf (has no children yet in the tree's structure), the fed
    feature row must be all-zero -- not just one hand-picked example above. In
    PUCT-style search, expansion and a node's own first backprop update happen
    together, so "still a leaf" and "no update-log entry yet" coincide for every
    non-root node; the root is a standing exception (see the sibling test below)
    since it has no incoming edge to encode a value for at all, at any step,
    leaf or not."""
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    total_checked = 0
    violations = []
    for ep_idx in range(20):
        episode = ds[ep_idx]
        step_node_features = episode.step_node_features
        step_parent_index = episode.step_parent_index
        num_steps = len(step_node_features)
        for local_step in range(0, num_steps, 5):  # sample every 5th step -- exhaustive-per-episode is slow, this is still 1000s of (episode, step) points
            row_now = step_node_features[local_step]
            pi_now = step_parent_index[local_step]
            n_nodes = row_now.shape[0]
            has_child = torch.zeros(n_nodes, dtype=torch.bool)
            if pi_now.numel():
                valid_parents = pi_now[pi_now >= 0]
                if valid_parents.numel():
                    has_child[valid_parents.unique()] = True
            leaves = (~has_child).nonzero(as_tuple=True)[0]
            for node_id in leaves.tolist():
                total_checked += 1
                if not torch.allclose(row_now[node_id], torch.zeros(5), atol=1e-9):
                    violations.append((ep_idx, local_step, node_id, row_now[node_id].tolist()))

    assert total_checked > 1000, f"sanity check on the check itself -- only tested {total_checked} triples, fixture too small"
    assert not violations, (
        f"future leakage regression: {len(violations)}/{total_checked} unvisited-leaf checks failed, "
        f"first few: {violations[:5]}"
    )


def test_root_always_zero():
    """The one structural exception to "leaf implies zero, non-leaf implies real
    value": the root has no incoming edge at all (nothing backprops INTO the
    root itself, only into its children), so its row must stay all-zero at
    every step regardless of how many children it accumulates -- confirming
    this is a deliberate, understood exception, not an unnoticed second leak."""
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    episode_index = _find_root_multi_child_episode(ds)
    if episode_index is None:
        pytest.skip("could not find an episode whose root accumulates >1 child by the final step in the first 10 episodes")

    episode = ds[episode_index]
    step_node_features = episode.step_node_features
    step_parent_index = episode.step_parent_index
    root_id = 0

    # Confirm the root genuinely accumulates children (i.e. this is a real test of
    # the "not a leaf but still zero" case, not a vacuous check on a node that
    # happens to stay a leaf the whole trajectory).
    last_pi = step_parent_index[-1]
    assert int((last_pi == root_id).sum()) > 1, "expected the root to have accumulated more than one child by the final step"

    num_steps = len(step_node_features)
    sample_steps = sorted({0, 1, min(5, num_steps - 1), num_steps // 2, num_steps - 1})
    for local_step in sample_steps:
        row = step_node_features[local_step][root_id]
        assert torch.allclose(row, torch.zeros(5), atol=1e-9), (
            f"root row regression: episode {episode_index}, expected all-zero at "
            f"local_step={local_step}, got {row.tolist()}"
        )


def test_pack_history_no_leak_before_first_visit():
    """pack_history.py's _build_tree_n / _forward_filled_wdl_at_step (the GNN
    pretraining input builder): same invariant, independently re-checked against
    the second, separate consumer that had the same bug."""
    _require_real_pack_history()
    from cts.data.preprocess_gnn.pack_history import _build_tree_n, iter_history_trajectories

    payload = torch.load(_PACK_HISTORY_ROOT / "train" / "shard_00000.pt", weights_only=False)
    trajectories = list(iter_history_trajectories(payload))
    # Locate a trajectory containing a node with the expected pre-first-visit gap;
    # search across nodes (not just a hardcoded node id), since the specific ids
    # in this corpus differ from the old xaba20k corpus.
    target = None
    target_node_id = None
    for traj in trajectories:
        n_nodes = traj.node_update_ptr.shape[0] - 1
        for node_id in range(1, n_nodes):  # skip root
            lo = int(traj.node_update_ptr[node_id].item())
            hi = int(traj.node_update_ptr[node_id + 1].item())
            if hi > lo:
                first_step = int(traj.update_step_index[lo].item())
                if first_step > 5:  # a real "created well before first visit" gap
                    target = traj
                    target_node_id = node_id
                    break
        if target is not None:
            break
    if target is None:
        pytest.skip("could not locate a trajectory in shard_00000.pt with a node visited well after creation")
    traj = target
    node_id = target_node_id
    lo = int(traj.node_update_ptr[node_id].item())
    first_step = int(traj.update_step_index[lo].item())

    for n in (1, first_step - 4, first_step):
        snap = _build_tree_n(traj, n)
        if snap is None or node_id >= snap.node_features.shape[0]:
            continue
        value = float(snap.node_features[node_id, 0])
        wdl = snap.node_features[node_id, 1:4]
        if n - 1 < first_step:  # local_step = n-1 (see _build_tree_n's own convention)
            assert value == pytest.approx(0.0, abs=1e-9), (
                f"future leakage regression in pack_history.py: node {node_id} at n={n} "
                f"(before its first update at step {first_step}) shows value={value}, expected 0.0"
            )
            assert torch.allclose(wdl, torch.zeros_like(wdl), atol=1e-9), (
                f"future leakage regression in pack_history.py: node {node_id} at n={n} wdl={wdl.tolist()}, expected zeros"
            )
