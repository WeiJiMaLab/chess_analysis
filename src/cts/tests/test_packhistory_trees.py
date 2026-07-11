"""Test suite for the ``packhistory_trees`` stage: replays each tree's own
backprop history so every node's value/WDL is correct as of the queried step,
instead of frozen at the tree's final value.

**pytest.ini does NOT cover this directory** (testpaths = src/analysis/tests
only) -- a bare ``pytest`` run silently skips this file. Run it explicitly:

    pytest src/cts/tests/test_packhistory_trees.py -v

Each real-data test is paired with a ``*_logic_synthetic_fixture`` test that
validates this file's own lookup/comparison helpers against a hand-built
fixture, independent of whether the production code has landed. If real field
names differ from what's assumed, ``_extract_update_log`` is the one place to
patch (its candidate-name lists document exactly what it tried).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from cts.data.preprocess_mc.pack import build_compact_trajectory

# ---------------------------------------------------------------------------
# Real data locations (read-only source data -- see history.md "Directory
# layout" and "Existing pipeline" sections). Never write anything here.
# ---------------------------------------------------------------------------
_HUMAN_TREES_DIR = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_ORACLE96_DIR = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees_oracle96_trace_filtered"
_XABA20K_RUN_DIR = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k"
_OLD_MC_PACKED_DIR = os.path.join(_XABA20K_RUN_DIR, "mc_packed")
# Naming convention fixed by history.md's "Directory layout" section:
# ${run_dir}/pack_history (new dir mirroring mc_packed 1:1).
_NEW_PACK_HISTORY_DIR = os.path.join(_XABA20K_RUN_DIR, "pack_history")

_SCHEMA = tree_encoder_feature_schema()
_VALUE_COL = _SCHEMA.index("value")
_TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# Real-file sampling helpers
# ---------------------------------------------------------------------------
def _sample_human_tree_paths(n: int = 6) -> List[str]:
    """First ``n`` files by the ``{idx:06d}_root_{idx}.pt`` naming convention.

    Deliberately avoids listing the ~1.08M-entry directory (verified via
    ``find | wc -l`` during investigation) -- constructs candidate paths
    directly and checks existence, which is fast regardless of directory size.
    """
    paths = []
    idx = 0
    while len(paths) < n and idx < n * 50:
        candidate = os.path.join(_HUMAN_TREES_DIR, f"{idx:06d}_root_{idx}.pt")
        if os.path.exists(candidate):
            paths.append(candidate)
        idx += 1
    return paths


def _sample_oracle96_paths(n: int = 6) -> List[str]:
    """First ``n`` files in ``shard_00000`` (2016 entries -- cheap to list)."""
    shard_dir = os.path.join(_ORACLE96_DIR, "shard_00000")
    if not os.path.isdir(shard_dir):
        return []
    names = sorted(os.listdir(shard_dir))[:n]
    return [os.path.join(shard_dir, name) for name in names]


def _requires_real_source_trees() -> str:
    if not os.path.isdir(_HUMAN_TREES_DIR) and not os.path.isdir(_ORACLE96_DIR):
        return f"neither {_HUMAN_TREES_DIR!r} nor {_ORACLE96_DIR!r} is reachable from this host"
    return ""


def _load_sample_records(n_per_source: int = 6) -> List[Tuple[str, RawPretrainExampleRecord]]:
    """Load a small deterministic sample of real records from both sources."""
    out: List[Tuple[str, RawPretrainExampleRecord]] = []
    for path in _sample_human_tree_paths(n_per_source) + _sample_oracle96_paths(n_per_source):
        try:
            record = RawPretrainExampleRecord.load(path)
        except Exception:  # noqa: BLE001 -- a malformed/legacy file shouldn't kill the whole sample
            continue
        out.append((path, record))
    return out


# ---------------------------------------------------------------------------
# Update-log schema access (history.md "Update log schema" block):
#   step_index:  int32[M]
#   node_id:     int32[M]
#   visit_count: int32[M]
#   q_value:     float32[M]
#   wdl:         float32[M,3]
# sorted by (node_id, step_index) ascending, with node_update_ptr: int32[N+1]
# CSR-indexing each node's own contiguous slice.
# ---------------------------------------------------------------------------
_UPDATE_LOG_FIELD_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "step_index": ("step_index", "update_log_step_index"),
    "node_id": ("node_id", "update_log_node_id"),
    "visit_count": ("visit_count", "update_log_visit_count"),
    "q_value": ("q_value", "update_log_q_value"),
    "wdl": ("wdl", "update_log_wdl"),
    # "trajectory_update_log_ptr" is Task 5's independent guess for this same
    # field (controller_train.py:ControllerEpisodeDataset.__getitem__, per
    # history.md's Task 5 Progress Log) -- included here too so this file
    # doesn't need another edit if Task 1 lands using that name instead.
    "node_update_ptr": ("node_update_ptr", "trajectory_update_log_ptr"),
}


def _extract_update_log(trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Locate the packed update-log fields inside a ``build_compact_trajectory``
    result dict, per the schema names above. Tries both flat keys directly on
    the trajectory dict and a nested ``trajectory["update_log"]`` sub-dict, in
    case Task 1 groups them instead of flattening. Returns ``None`` (never
    raises) if the fields are absent -- callers use that to skip with a clear
    reason rather than crash, since Task 1 may not have landed yet.
    """
    containers = [trajectory]
    nested = trajectory.get("update_log")
    if isinstance(nested, dict):
        containers.append(nested)

    out: Dict[str, Any] = {}
    for canonical, candidates in _UPDATE_LOG_FIELD_CANDIDATES.items():
        found = None
        for container in containers:
            for candidate in candidates:
                if candidate in container:
                    found = container[candidate]
                    break
            if found is not None:
                break
        if found is None:
            return None
        out[canonical] = found
    return out


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _value_as_of_step(update_log: Dict[str, Any], node_id: int, step_t: int) -> Optional[float]:
    """CSR lookup: the last ``q_value`` entry for ``node_id`` with
    ``step_index <= step_t`` -- i.e. "what did backprop believe this node's
    value was, as of step t". Returns ``None`` if the node has no update-log
    entries at or before ``step_t`` (unvisited-by-then), mirroring
    ``EdgeStats``' own documented zero-init-unvisited convention
    (``q_value == 0`` before the first real backup -- see
    ``teacher_targets.py``'s ``_select_leaf_by_puct`` docstring).

    Relies on the schema's guarantee that each node's slice is sorted by
    ``step_index`` ascending, so a single forward scan (not even a binary
    search) suffices -- correctness first, the schema's own CSR ptr already
    gives us the O(log updates-for-this-node) *localization*, this is just
    the scan within that already-small slice.
    """
    ptr = _as_numpy(update_log["node_update_ptr"])
    start, end = int(ptr[node_id]), int(ptr[node_id + 1])
    steps = _as_numpy(update_log["step_index"])[start:end]
    qvals = _as_numpy(update_log["q_value"])[start:end]
    best: Optional[float] = None
    for step, qvalue in zip(steps.tolist(), qvals.tolist()):
        if step <= step_t:
            best = float(qvalue)
        else:
            break
    return best


def _root_children_ascending(record: RawPretrainExampleRecord) -> List[int]:
    """Root's children in ASCENDING NODE ID order.

    history.md's "Validated groundwork" section: root-child order must come
    from ascending node id, NEVER ``children_index[child_ptr[...]]`` CSR
    order -- CSR order is sorted by UCI move string in the ``v5`` raw format,
    not creation order, and using it silently mismatches labels against
    ``oracle_root_q_trace``'s move-indexed columns. Root is always node 0
    (``RawPretrainExampleRecord._validate`` enforces ``parent_index[0] == -1``
    and a topological ordering), so this is simply "every node whose parent
    is 0, sorted".
    """
    parent_index = record.parent_index.tolist()
    return sorted(node_id for node_id, parent_id in enumerate(parent_index) if parent_id == 0)


def _oracle_move_index(record: RawPretrainExampleRecord, node_id: int) -> int:
    """Map a root-child node id to its column in ``oracle_root_q_trace`` via
    its incoming UCI move -- a label-based join, robust to any ordering
    assumption (the whole point of not trusting positional/CSR order here).
    """
    move = record.incoming_moves[node_id]
    return record.oracle_root_moves.index(move)


def _top1_minus_top2(values: Sequence[float]) -> float:
    ordered = sorted(values, reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


# ---------------------------------------------------------------------------
# 1a. Replay-vs-oracle: SYNTHETIC logic check (always runs)
# ---------------------------------------------------------------------------
def test_replay_lookup_logic_synthetic_fixture():
    """Validate ``_value_as_of_step`` against a hand-built update log with
    known-by-construction expected values, independent of Task 1's real
    implementation -- this is the "test LOGIC validated independently"
    requirement from history.md for pieces whose upstream isn't ready yet.

    Fixture: 1 root (node 0) + 2 children (node 1, node 2). Node 1 gets
    updates at steps 2 and 5 (q_value 0.10 then 0.40); node 2 gets a single
    update at step 4 (q_value -0.20) and is never touched again.
    """
    # Sorted by (node_id, step_index) ascending, per schema.
    update_log = {
        "node_id": np.array([1, 1, 2], dtype=np.int32),
        "step_index": np.array([2, 5, 4], dtype=np.int32),
        "visit_count": np.array([1, 2, 1], dtype=np.int32),
        "q_value": np.array([0.10, 0.40, -0.20], dtype=np.float32),
        "wdl": np.zeros((3, 3), dtype=np.float32),
        # CSR: node 0 has no entries [0,0); node 1 has entries [0,2);
        # node 2 has entry [2,3).
        "node_update_ptr": np.array([0, 0, 2, 3], dtype=np.int32),
    }

    # Before node 1's first update: unvisited.
    assert _value_as_of_step(update_log, node_id=1, step_t=1) is None
    # Exactly at / after the first update, before the second.
    assert _value_as_of_step(update_log, node_id=1, step_t=2) == pytest.approx(0.10)
    assert _value_as_of_step(update_log, node_id=1, step_t=4) == pytest.approx(0.10)
    # At / after the second update.
    assert _value_as_of_step(update_log, node_id=1, step_t=5) == pytest.approx(0.40)
    assert _value_as_of_step(update_log, node_id=1, step_t=96) == pytest.approx(0.40)

    # Node 2's single update.
    assert _value_as_of_step(update_log, node_id=2, step_t=3) is None
    assert _value_as_of_step(update_log, node_id=2, step_t=4) == pytest.approx(-0.20)
    assert _value_as_of_step(update_log, node_id=2, step_t=96) == pytest.approx(-0.20)

    # Node 0 (root) has zero entries: always unvisited under this lookup
    # (root's own "value" isn't itself a root-child edge in this schema).
    assert _value_as_of_step(update_log, node_id=0, step_t=96) is None


# ---------------------------------------------------------------------------
# 1b. Replay-vs-oracle: REAL data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_requires_real_source_trees() != "", reason=_requires_real_source_trees())
def test_replay_matches_oracle_root_q_trace():
    """Packhistory_trees's own replayed root-child Q trace (read via the
    packed update log's CSR "value as of step t" lookup) must match the
    tree's stored ``oracle_root_q_trace`` within 1e-3, per history.md's
    "Stage: packhistory_trees" Test 1 and the ~1e-4 residual noted under
    "Deferred" (1e-3 is comfortably above that known residual).

    Uses ``build_compact_trajectory`` -- the exact function history.md names
    as Task 1's own modification target (step 5 of its "Steps" list) -- on
    real ``human_trees``/``oracle96`` records, so this runs against the real
    production packing path the moment Task 1 lands, no edits needed here.
    """
    samples = _load_sample_records(n_per_source=6)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees/oracle96")

    checked_any_tree = False
    for path, record in samples:
        trajectory = build_compact_trajectory(record, _SCHEMA, source_path=path)
        if trajectory is None:
            continue  # e.g. no root expansion -- build_compact_trajectory's own documented None case
        update_log = _extract_update_log(trajectory)
        if update_log is None:
            pytest.skip(
                "build_compact_trajectory's returned dict carries no update-log fields yet "
                f"(tried candidates {_UPDATE_LOG_FIELD_CANDIDATES}) -- Task 1 "
                "(packhistory_trees implementation, preprocess_mc/pack.py) has not landed. "
                "See history.md Task 1 Progress Log for status."
            )

        root_children = _root_children_ascending(record)
        num_oracle_steps = int(record.oracle_root_q_trace.shape[0])
        for node_id in root_children:
            move_idx = _oracle_move_index(record, node_id)
            for row in range(num_oracle_steps):
                step_t = row  # step_index is 0-indexed (position in expansion_parent_ids); oracle row k lines up with step_index == k, not k+1
                replayed = _value_as_of_step(update_log, node_id, step_t)
                oracle = float(record.oracle_root_q_trace[row, move_idx].item())
                if replayed is None:
                    # Not yet visited by this step under the replay -- the
                    # oracle trace, by construction, only records moves that
                    # exist as root children from the start, so a persistent
                    # None here (never visited across all 96 steps) would
                    # itself be suspicious, but a *early*-step None is
                    # expected and not a mismatch to check.
                    continue
                assert abs(replayed - oracle) < _TOLERANCE, (
                    f"{path}: node {node_id} (move {record.incoming_moves[node_id]!r}) "
                    f"step {step_t}: replayed={replayed:.6f} vs oracle={oracle:.6f}"
                )
                checked_any_tree = True

    assert checked_any_tree, "no (tree, node, step) triples were actually compared -- sample too sparse"


# ---------------------------------------------------------------------------
# 2a. Packed action-gap: SYNTHETIC logic check (always runs)
# ---------------------------------------------------------------------------
def test_action_gap_logic_synthetic_fixture():
    """Validate the top1-top2 action-gap comparison against a hand-built
    fixture where the answer is known by construction -- this is the check
    that, per history.md, "would have caught the original bug immediately":
    a *constant* packed gap across steps must be distinguishable from a
    *moving* one by this exact comparison.
    """
    # Two root children (node 1, node 2); their Q values swap rank between
    # step 3 and step 8 -- the sort-based action gap must track that.
    update_log = {
        "node_id": np.array([1, 1, 2, 2], dtype=np.int32),
        "step_index": np.array([2, 8, 3, 6], dtype=np.int32),
        "visit_count": np.array([1, 2, 1, 2], dtype=np.int32),
        "q_value": np.array([0.50, 0.10, 0.20, 0.70], dtype=np.float32),
        "wdl": np.zeros((4, 3), dtype=np.float32),
        "node_update_ptr": np.array([0, 0, 2, 4], dtype=np.int32),
    }
    root_children = [1, 2]

    def gap_at(step_t: int) -> float:
        values = [_value_as_of_step(update_log, nid, step_t) or 0.0 for nid in root_children]
        return _top1_minus_top2(values)

    # Step 3: node1=0.50 (as of step2), node2=0.20 (as of step3) -> node1 leads by 0.30.
    assert gap_at(3) == pytest.approx(0.30)
    # Step 6: node1=0.50 (still, next update at 8), node2=0.70 (as of step6) -> node2 leads by 0.20.
    assert gap_at(6) == pytest.approx(0.20)
    # Step 8: node1=0.10 (as of step8), node2=0.70 -> node2 leads by 0.60.
    assert gap_at(8) == pytest.approx(0.60)

    # A "frozen" pre-fix comparator (always reads the FINAL values regardless
    # of t) would report the SAME 0.60 gap at every step above, including
    # step 3 -- demonstrating exactly the bug this whole plan fixes and why
    # a per-step check (not just endpoints) is the meaningful one.
    frozen_gap = _top1_minus_top2([0.10, 0.70])
    assert frozen_gap == pytest.approx(0.60)
    assert frozen_gap != pytest.approx(gap_at(3))


# ---------------------------------------------------------------------------
# 2b. Packed action-gap: REAL data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_requires_real_source_trees() != "", reason=_requires_real_source_trees())
def test_packed_action_gap_matches_oracle_at_every_step():
    """The single strongest, most direct regression test for the original
    bug (history.md, "Stage: packhistory_trees" Test 2): for a sample of
    trees, at EVERY step t (not just endpoints), the packed root-children
    "value" (read straight off packhistory_trees's own packed update log,
    no encoder / materialize / GNNpretrain involved) must produce a
    top1-top2 action gap matching the oracle action gap
    (top1-top2 of oracle_root_q_trace[t]) within tolerance.

    Before the fix: fails everywhere (the packed gap is constant across all
    96 steps). After the fix: should pass at every step.
    """
    samples = _load_sample_records(n_per_source=6)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees/oracle96")

    checked_any_step = False
    for path, record in samples:
        trajectory = build_compact_trajectory(record, _SCHEMA, source_path=path)
        if trajectory is None:
            continue
        update_log = _extract_update_log(trajectory)
        if update_log is None:
            pytest.skip(
                "build_compact_trajectory's returned dict carries no update-log fields yet "
                "-- Task 1 (packhistory_trees implementation) has not landed."
            )

        root_children = _root_children_ascending(record)
        if len(root_children) != len(record.oracle_root_moves):
            pytest.fail(
                f"{path}: {len(root_children)} root children vs "
                f"{len(record.oracle_root_moves)} oracle_root_moves -- structural mismatch, "
                "not an action-gap issue but must be resolved before this test means anything."
            )

        num_oracle_steps = int(record.oracle_root_q_trace.shape[0])
        for row in range(num_oracle_steps):
            step_t = row  # step_index is 0-indexed (position in expansion_parent_ids); oracle row k lines up with step_index == k, not k+1
            packed_values = [
                _value_as_of_step(update_log, node_id, step_t) or 0.0
                for node_id in root_children
            ]
            packed_gap = _top1_minus_top2(packed_values)
            oracle_row = record.oracle_root_q_trace[row].tolist()
            oracle_gap = _top1_minus_top2(oracle_row)
            assert abs(packed_gap - oracle_gap) < _TOLERANCE, (
                f"{path}: step {step_t}: packed_gap={packed_gap:.6f} vs oracle_gap={oracle_gap:.6f}"
            )
            checked_any_step = True

    assert checked_any_step, "no (tree, step) pairs were actually compared -- sample too sparse"


# ---------------------------------------------------------------------------
# 3. Diff vs. old mc_packed/ (structure/root-values identical, features differ)
# ---------------------------------------------------------------------------
def _load_shard(path: str) -> Dict[str, Any]:
    return torch.load(path, weights_only=False)


def _shard_trajectory_index_by_source_path(shard: Dict[str, Any]) -> Dict[str, int]:
    return {path: idx for idx, path in enumerate(shard["trajectory_source_paths"])}


def _trajectory_structure_slice(shard: Dict[str, Any], traj_idx: int) -> Dict[str, torch.Tensor]:
    """Pull one trajectory's structural columns (parent_index, depth, child_ptr)
    out of a shard's globally-concatenated arrays, using the shard's own CSR
    ptr arrays (``trajectory_node_ptr``, ``trajectory_child_ptr_ptr``) --
    same slicing pattern the rest of this codebase already uses (e.g.
    ``ControllerEpisodeDataset``).
    """
    node_start, node_end = int(shard["trajectory_node_ptr"][traj_idx]), int(shard["trajectory_node_ptr"][traj_idx + 1])
    cptr_start, cptr_end = int(shard["trajectory_child_ptr_ptr"][traj_idx]), int(shard["trajectory_child_ptr_ptr"][traj_idx + 1])
    return {
        "parent_index": shard["parent_index"][node_start:node_end],
        "depth": shard["depth"][node_start:node_end],
        "child_ptr": shard["child_ptr"][cptr_start:cptr_end],
        "node_features": shard["node_features"][node_start:node_end],
    }


def _trajectory_root_oracle_values(shard: Dict[str, Any], traj_idx: int) -> Dict[str, Any]:
    """Per-episode root oracle values for one trajectory: oracle_stop_step,
    oracle_value, and the halt_rewards trace -- everything history.md's
    diff test says must be byte-identical between old and new.
    """
    episode_indices = [
        e for e, t in enumerate(shard["episode_trajectory_index"].tolist()) if t == traj_idx
    ]
    step_start, step_end = int(shard["trajectory_step_ptr"][traj_idx]), int(shard["trajectory_step_ptr"][traj_idx + 1])
    return {
        "oracle_stop_steps": shard["oracle_stop_steps"][episode_indices],
        "oracle_values": shard["oracle_values"][episode_indices],
        "halt_rewards": shard["trajectory_halt_rewards"][step_start:step_end],
    }


def diff_old_new_shards(
    old_shard: Dict[str, Any],
    new_shard: Dict[str, Any],
    *,
    require_features_differ: bool = True,
) -> None:
    """Core diff logic shared by the synthetic-fixture test and the real
    integration test, per history.md's diff-based regression test spec:
    structure + root oracle values byte-identical, per-step node features
    must now differ. Raises ``AssertionError`` on any violation.
    """
    old_index = _shard_trajectory_index_by_source_path(old_shard)
    new_index = _shard_trajectory_index_by_source_path(new_shard)
    common_paths = sorted(set(old_index) & set(new_index))
    assert common_paths, "no source_path overlap between old and new shards -- can't diff anything"

    any_feature_diff = False
    for source_path in common_paths:
        old_traj, new_traj = old_index[source_path], new_index[source_path]
        old_struct = _trajectory_structure_slice(old_shard, old_traj)
        new_struct = _trajectory_structure_slice(new_shard, new_traj)

        assert torch.equal(old_struct["parent_index"], new_struct["parent_index"]), (
            f"{source_path}: parent_index differs -- structure must be untouched by this fix"
        )
        assert torch.equal(old_struct["depth"], new_struct["depth"]), (
            f"{source_path}: depth differs -- structure must be untouched by this fix"
        )
        assert torch.equal(old_struct["child_ptr"], new_struct["child_ptr"]), (
            f"{source_path}: child_ptr differs -- structure must be untouched by this fix"
        )

        old_oracle = _trajectory_root_oracle_values(old_shard, old_traj)
        new_oracle = _trajectory_root_oracle_values(new_shard, new_traj)
        assert torch.equal(old_oracle["oracle_stop_steps"], new_oracle["oracle_stop_steps"]), (
            f"{source_path}: oracle_stop_step differs -- these were never wrong, must stay identical"
        )
        assert torch.equal(old_oracle["oracle_values"], new_oracle["oracle_values"]), (
            f"{source_path}: oracle_value differs -- these were never wrong, must stay identical"
        )
        assert torch.equal(old_oracle["halt_rewards"], new_oracle["halt_rewards"]), (
            f"{source_path}: halt_rewards differ -- these were never wrong, must stay identical"
        )

        if not torch.equal(old_struct["node_features"], new_struct["node_features"]):
            any_feature_diff = True

    if require_features_differ:
        assert any_feature_diff, (
            "per-step node features are byte-identical between old and new for every "
            "common tree -- the fix did not take (this is the exact failure mode the "
            "original bug produces)"
        )


def test_diff_logic_synthetic_fixture():
    """Validate ``diff_old_new_shards`` against hand-built old+new shard
    fixtures matching the real ``mc_packed`` shard format (verified against
    ``.../ysagiv_xaba20k/mc_packed/train/shard_00000.pt``, format tag
    ``cts_budgeted_controller_episode_shard_v4``) -- proves the diff logic
    itself is correct, independent of whether Task 1/Task 6 have produced
    real ``pack_history/`` output yet. Covers both the positive case (a
    correctly-fixed "new" shard passes) and negative cases (a structure
    regression, and a "fix" that didn't actually change features, both fail
    loudly).
    """

    def _make_shard(*, node_features_are_frozen: bool) -> Dict[str, Any]:
        # 1 tree: root (0) + 2 children (1, 2). 1 episode.
        parent_index = torch.tensor([-1, 0, 0], dtype=torch.int64)
        depth = torch.tensor([0, 1, 1], dtype=torch.int64)
        child_ptr = torch.tensor([0, 2, 2, 2], dtype=torch.int32)  # root has 2 children
        if node_features_are_frozen:
            node_features = torch.tensor([[0.0, 0, 0, 0, 0], [0.5, 0, 0, 0, 0], [0.5, 0, 0, 0, 0]])
        else:
            # "new" values differ from the frozen baseline -- simulating the fix.
            node_features = torch.tensor([[0.0, 0, 0, 0, 0], [0.31, 0, 0, 0, 0], [0.62, 0, 0, 0, 0]])
        return {
            "trajectory_source_paths": ["tree_a.pt"],
            "trajectory_node_ptr": torch.tensor([0, 3], dtype=torch.int64),
            "trajectory_child_ptr_ptr": torch.tensor([0, 4], dtype=torch.int64),
            "trajectory_step_ptr": torch.tensor([0, 5], dtype=torch.int64),
            "parent_index": parent_index,
            "depth": depth,
            "child_ptr": child_ptr,
            "node_features": node_features,
            "episode_trajectory_index": torch.tensor([0], dtype=torch.int64),
            "oracle_stop_steps": torch.tensor([3], dtype=torch.int64),
            "oracle_values": torch.tensor([0.42], dtype=torch.float32),
            "trajectory_halt_rewards": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.42], dtype=torch.float32),
        }

    old_shard = _make_shard(node_features_are_frozen=True)

    # Positive case: a correctly-fixed new shard (features differ, everything
    # else identical) must pass cleanly.
    new_shard_fixed = _make_shard(node_features_are_frozen=False)
    diff_old_new_shards(old_shard, new_shard_fixed)  # must not raise

    # Negative case 1: "new" shard whose features are STILL frozen/identical
    # to old -- must be caught (this is exactly the pre-fix bug signature).
    new_shard_unfixed = _make_shard(node_features_are_frozen=True)
    with pytest.raises(AssertionError, match="byte-identical"):
        diff_old_new_shards(old_shard, new_shard_unfixed)

    # Negative case 2: a structure regression (child_ptr corrupted) must
    # also be caught, independent of the features-differ check.
    new_shard_bad_structure = _make_shard(node_features_are_frozen=False)
    new_shard_bad_structure["child_ptr"] = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    with pytest.raises(AssertionError, match="child_ptr differs"):
        diff_old_new_shards(old_shard, new_shard_bad_structure)


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_OLD_MC_PACKED_DIR, "train_manifest.json")),
    reason=f"old mc_packed data not reachable at {_OLD_MC_PACKED_DIR!r}",
)
def test_old_shard_parseable_by_diff_helpers():
    """Partial real-data check available right now, independent of Task 1/6:
    the structural/oracle-value extraction helpers this diff test relies on
    must actually parse a REAL old ``mc_packed`` shard without error (not
    just the hand-built synthetic fixture). This is the part of the diff
    test's real-data integration that doesn't have to wait for anything.
    """
    shard_paths = sorted(glob.glob(os.path.join(_OLD_MC_PACKED_DIR, "train", "shard_*.pt")))
    if not shard_paths:
        pytest.skip(f"no shard files found under {_OLD_MC_PACKED_DIR}/train")
    shard = _load_shard(shard_paths[0])
    index = _shard_trajectory_index_by_source_path(shard)
    assert index, "shard has no trajectory_source_paths -- can't join by source_path at all"
    sample_path, sample_traj = next(iter(index.items()))
    struct = _trajectory_structure_slice(shard, sample_traj)
    assert struct["parent_index"].numel() > 0
    oracle = _trajectory_root_oracle_values(shard, sample_traj)
    assert oracle["oracle_stop_steps"].numel() > 0


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(_NEW_PACK_HISTORY_DIR, "train_manifest.json")),
    reason=(
        f"new pack_history/ output not present at {_NEW_PACK_HISTORY_DIR!r} -- "
        "pending Task 6 (config_ysagiv_xaba20k_history.yaml) and an actual "
        "packhistory_trees SLURM run on xaba20k (Wave 3 in history.md's DAG)"
    ),
)
def test_diff_vs_old_mc_packed():
    """Real integration version of the diff test, per history.md's
    "Diff-based regression test" section and Task 2 DoD's third checklist
    item. Runs the moment ``pack_history/`` exists for xaba20k; until then,
    skips with an explicit reason (this is expected to be the last of the
    three tests to go green -- it depends on Task 1 AND Task 6 AND an
    actual SLURM run finishing, not just code).
    """
    old_shard_paths = sorted(glob.glob(os.path.join(_OLD_MC_PACKED_DIR, "train", "shard_*.pt")))
    new_shard_paths = sorted(glob.glob(os.path.join(_NEW_PACK_HISTORY_DIR, "train", "shard_*.pt")))
    assert old_shard_paths, f"no old shards found under {_OLD_MC_PACKED_DIR}/train"
    assert new_shard_paths, f"no new shards found under {_NEW_PACK_HISTORY_DIR}/train"

    # Merge all shards' trajectory index (source_path is the join key across
    # shard boundaries too, since shard assignment isn't guaranteed stable
    # between old and new packing runs -- history.md is explicit that
    # source_path, not shard index, is the correct join key).
    old_shard_by_path: Dict[str, Dict[str, Any]] = {}
    for shard_path in old_shard_paths:
        shard = _load_shard(shard_path)
        for source_path in shard["trajectory_source_paths"]:
            old_shard_by_path[source_path] = shard
    new_shard_by_path: Dict[str, Dict[str, Any]] = {}
    for shard_path in new_shard_paths:
        shard = _load_shard(shard_path)
        for source_path in shard["trajectory_source_paths"]:
            new_shard_by_path[source_path] = shard

    common = sorted(set(old_shard_by_path) & set(new_shard_by_path))
    assert common, "no source_path overlap between old mc_packed/ and new pack_history/"

    # Diff shard-pair by shard-pair over the common set (diff_old_new_shards
    # itself only needs the two shards containing a given tree, not the
    # whole corpus at once).
    checked_pairs = set()
    for source_path in common:
        old_shard = old_shard_by_path[source_path]
        new_shard = new_shard_by_path[source_path]
        key = (id(old_shard), id(new_shard))
        if key in checked_pairs:
            continue
        checked_pairs.add(key)
        diff_old_new_shards(old_shard, new_shard)
