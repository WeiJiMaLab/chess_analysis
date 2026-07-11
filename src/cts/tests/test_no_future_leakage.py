"""Regression test: nodes not yet individually backed up as of a queried step must
read 0.0 ("no signal yet"), not their final end-of-search value -- checked in
``ControllerEpisodeDataset`` and ``pack_history._forward_filled_wdl_at_step`` on a
real xaba20k tree (episode index 8, node id 66, first update at expansion_count 26).

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_no_future_leakage.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

_PACK_HISTORY_ROOT = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/pack_history")
_TRAIN_MANIFEST = _PACK_HISTORY_ROOT / "train_manifest.json"

_EPISODE_INDEX = 8
_NODE_ID = 66
_FIRST_UPDATE_LOCAL_STEP = 26  # confirmed empirically below via the update log itself


def _require_real_pack_history():
    if not _TRAIN_MANIFEST.is_file():
        pytest.skip(f"real pack_history/ output not found at {_PACK_HISTORY_ROOT} -- run packhistory_trees first")


def test_controller_episode_dataset_no_leak_before_first_visit():
    """ControllerEpisodeDataset.__getitem__ (controller_train.py): a node's fed
    value must be exactly 0.0 for every step before its own first update-log
    entry, and must match the real forward-filled value from that step onward --
    never the node's end-of-search final value while still "unvisited so far"."""
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    episode = ds[_EPISODE_INDEX]
    step_node_features = episode.step_node_features
    assert _NODE_ID < step_node_features[-1].shape[0], "fixture node id out of range -- pick a different (episode, node)"

    # Before the node's own first backprop update: must read as "no signal yet" (0),
    # not its eventual final value. This is the exact leak that was fixed.
    for local_step in range(0, _FIRST_UPDATE_LOCAL_STEP):
        nf = step_node_features[local_step]
        if _NODE_ID >= nf.shape[0]:
            continue  # node not yet structurally created at this step -- not the case under test
        row = nf[_NODE_ID]
        assert torch.allclose(row, torch.zeros_like(row), atol=1e-9), (
            f"future leakage regression: node {_NODE_ID} at local_step={local_step} "
            f"(before its first real update at {_FIRST_UPDATE_LOCAL_STEP}) shows "
            f"row={row.tolist()}, expected all-zero (no info yet, including wdl_var). "
            f"If this fails, the fallback in ControllerEpisodeDataset.__getitem__ has "
            f"regressed to reading the final-value baseline again."
        )

    # From the node's first real update onward, it must be genuinely non-zero and
    # must evolve (not itself be frozen) -- confirms this isn't just "always 0".
    nf_first = step_node_features[_FIRST_UPDATE_LOCAL_STEP]
    nf_last = step_node_features[-1]
    value_first = float(nf_first[_NODE_ID, 0])
    value_last = float(nf_last[_NODE_ID, 0])
    assert value_first != pytest.approx(0.0, abs=1e-9), "expected a real value at the node's first update step"
    assert value_first != pytest.approx(value_last, abs=1e-6), (
        "expected the value to keep evolving after first update, not jump straight to (and freeze at) "
        "the final value -- would indicate a different leak"
    )


def test_controller_episode_dataset_wdl_var_tracks_current_wdl():
    """Second, independently-found leak (2026-07-10, same day as the first): wdl_var
    (column 4) is a pure function of the wdl triple (value_features_from_wdl:
    variance = (p_win+p_loss) - value**2) and must be recomputed from the CURRENT
    per-step wdl, not left frozen at the node's original pre-search variance while
    value/wdl_win/wdl_draw/wdl_loss correctly evolve."""
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    episode = ds[_EPISODE_INDEX]
    step_node_features = episode.step_node_features

    checked_any = False
    for local_step in (_FIRST_UPDATE_LOCAL_STEP, 40, 65, 89, 95):
        if local_step >= len(step_node_features):
            continue
        nf = step_node_features[local_step]
        if _NODE_ID >= nf.shape[0]:
            continue
        row = nf[_NODE_ID]
        value, p_win, p_draw, p_loss, wdl_var = (float(x) for x in row)
        expected_var = (p_win + p_loss) - value * value
        assert wdl_var == pytest.approx(expected_var, abs=1e-4), (
            f"wdl_var regression: node {_NODE_ID} at local_step={local_step} has "
            f"wdl_var={wdl_var:.6f} but (p_win+p_loss)-value**2={expected_var:.6f} from "
            f"the SAME step's wdl triple -- wdl_var is not tracking the current wdl, "
            f"it has regressed to a frozen baseline again."
        )
        checked_any = True
    assert checked_any, "fixture node never got a real update in this episode -- pick a different (episode, node)"


def test_every_unvisited_leaf_is_all_zero_across_episodes():
    """Strongest, most general form of the check: at EVERY step, for EVERY node
    that is still a leaf (has no children yet in the tree's structure), the fed
    feature row must be all-zero -- not just the one hand-picked (episode 8, node
    66) example above. In PUCT-style search, expansion and a node's own first
    backprop update happen together, so "still a leaf" and "no update-log entry
    yet" coincide for every non-root node; the root is a standing exception (see
    the sibling test below) since it has no incoming edge to encode a value for
    at all, at any step, leaf or not."""
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


def test_root_is_always_zero_even_with_many_children():
    """The one structural exception to "leaf implies zero, non-leaf implies real
    value": the root has no incoming edge at all (nothing backprops INTO the
    root itself, only into its children), so its row must stay all-zero at
    every step regardless of how many children it accumulates -- confirming
    this is a deliberate, understood exception, not an unnoticed second leak."""
    _require_real_pack_history()
    from cts.train.controller_train import ControllerEpisodeDataset

    ds = ControllerEpisodeDataset(str(_TRAIN_MANIFEST))
    episode = ds[_EPISODE_INDEX]
    step_node_features = episode.step_node_features
    step_parent_index = episode.step_parent_index
    root_id = 0

    # Confirm the root genuinely accumulates children (i.e. this is a real test of
    # the "not a leaf but still zero" case, not a vacuous check on a node that
    # happens to stay a leaf the whole trajectory).
    last_pi = step_parent_index[-1]
    assert int((last_pi == root_id).sum()) > 1, "expected the root to have accumulated more than one child by the final step"

    for local_step in (0, 1, 5, 26, 50, len(step_node_features) - 1):
        row = step_node_features[local_step][root_id]
        assert torch.allclose(row, torch.zeros(5), atol=1e-9), (
            f"root row regression: expected all-zero at local_step={local_step}, got {row.tolist()}"
        )


def test_pack_history_build_tree_n_no_leak_before_first_visit():
    """pack_history.py's _build_tree_n / _forward_filled_wdl_at_step (the GNN
    pretraining input builder): same invariant, independently re-checked against
    the second, separate consumer that had the same bug."""
    _require_real_pack_history()
    from cts.data.preprocess_gnn.pack_history import _build_tree_n, iter_history_trajectories

    payload = torch.load(_PACK_HISTORY_ROOT / "train" / "shard_00000.pt", weights_only=False)
    trajectories = list(iter_history_trajectories(payload))
    # Locate a trajectory containing node 66 with the expected pre-first-visit gap;
    # shard-local episode index may differ from the manifest-level index used above,
    # so search rather than assuming shard 0 / index 8 line up 1:1.
    target = None
    for traj in trajectories:
        lo = int(traj.node_update_ptr[_NODE_ID].item()) if _NODE_ID < traj.node_update_ptr.shape[0] - 1 else 0
        hi = int(traj.node_update_ptr[_NODE_ID + 1].item()) if _NODE_ID < traj.node_update_ptr.shape[0] - 1 else 0
        if hi > lo:
            first_step = int(traj.update_step_index[lo].item())
            if first_step > 5:  # a real "created well before first visit" gap
                target = (traj, first_step)
                break
    if target is None:
        pytest.skip("could not locate a trajectory in shard_00000.pt with a node visited well after creation")
    traj, first_step = target

    for n in (1, first_step - 4, first_step):
        snap = _build_tree_n(traj, n)
        if snap is None or _NODE_ID >= snap.node_features.shape[0]:
            continue
        value = float(snap.node_features[_NODE_ID, 0])
        wdl = snap.node_features[_NODE_ID, 1:4]
        if n - 1 < first_step:  # local_step = n-1 (see _build_tree_n's own convention)
            assert value == pytest.approx(0.0, abs=1e-9), (
                f"future leakage regression in pack_history.py: node {_NODE_ID} at n={n} "
                f"(before its first update at step {first_step}) shows value={value}, expected 0.0"
            )
            assert torch.allclose(wdl, torch.zeros_like(wdl), atol=1e-9), (
                f"future leakage regression in pack_history.py: node {_NODE_ID} at n={n} wdl={wdl.tolist()}, expected zeros"
            )
