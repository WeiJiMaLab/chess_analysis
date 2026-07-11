"""Adversarial canary tests for the ``packhistory_trees`` REPLAY LOOP ITSELF.

Scope, deliberately narrower than the two existing leakage test files in this
directory: ``test_no_future_leakage.py`` and ``test_canary_future_leakage.py``
both poison the *final_value/final_wdl baseline* or update-log entries and
check DOWNSTREAM CONSUMERS (``ControllerEpisodeDataset``, ``pack_history.py``'s
``_build_tree_n``/``_forward_filled_wdl_at_step``) -- i.e. they assume the
update log itself is trustworthy and test the readers. This file instead
attacks the SOURCE that produces the update log:
``preprocess_mc/pack.py``'s ``_replay_backprop_history`` (and, for one test,
``_build_compact_trajectory`` around it). If ``step_index`` assignment or the
"final" value/WDL baseline were ever computed from something other than the
tree's own real, already-elapsed expansion history -- e.g. from
``oracle_root_q_trace`` (a legitimate FULL-TREE-DEPENDENT quantity, but only
as a training TARGET, never as a per-step INPUT column), or from a
sort-by-magnitude shortcut, or from a truncation-unstable computation -- the
existing two test files would not necessarily catch it, because they never
poison anything upstream of the update log's own construction.

Five canaries, each independent:

1. ``test_canary_truncated_replay_matches_full_replay_up_to_truncation`` --
   replays the SAME real tree twice, once with the full expansion order and
   once with only the first ``S`` expansions, and proves the two runs are
   BIT-IDENTICAL for every ``step_index < S``. If ``_replay_backprop_history``
   ever used any information that implicitly depends on expansions AFTER the
   one currently being processed (array length, a global pass over the whole
   list before looping, etc.), truncating the input would change early rows
   and this would fail.

2. ``test_canary_late_touched_node_poison_does_not_leak_backward`` -- poisons
   the STATIC leaf value (an impossible sentinel) of one real node that is
   first touched by backprop only in the second half of the 96-step search,
   and proves no update-log row anywhere (any node) with ``step_index``
   strictly before that node's own first-touch step shows any trace of it --
   with a positive control proving the poison IS reachable at/after that step.

3. ``test_canary_update_log_step_partition_matches_independent_reconstruction``
   -- for every step of several real trees, independently (NOT by calling
   ``_ancestor_edge_path`` or any of pack.py's own helpers -- a plain
   parent-pointer walk written fresh in this test file) recomputes which
   nodes SHOULD receive an update-log row at that step, and checks it against
   the ACTUAL set of node ids the real update log assigns to that
   ``step_index``. This is the closest thing to "re-derive ground truth from
   scratch and diff against production" available without re-running search.

4. ``test_canary_final_value_is_last_entry_not_an_extremum`` -- for real nodes
   whose replayed q_value trajectory is genuinely non-monotonic (goes up AND
   down across their own update-log entries), confirms
   ``_replay_backprop_history``'s ``final_value``/``final_wdl`` (the baseline
   ``_build_compact_trajectory`` bakes into ``node_features``) equals the
   CHRONOLOGICALLY LAST entry, not e.g. the max/min-magnitude entry -- ruling
   out a "sorted by final magnitude" style implementation that would
   coincidentally match on monotonic trajectories but diverge here.

5. ``test_canary_oracle_trace_poison_does_not_reach_update_log`` -- poisons
   ``record.oracle_root_q_trace`` / ``oracle_final_root_q_values`` /
   ``oracle_best_move_index`` (the DP-oracle-adjacent, legitimately
   full-tree-dependent fields) and confirms the poison does NOT reach the
   update log's ``q_value``/``wdl`` columns or the ``node_features`` baseline
   built from them -- while confirming (positive control) it DOES reach
   ``halt_rewards``, which is BY DESIGN allowed to depend on the full tree
   (it's a training target, not a per-step state input). This directly tests
   "what to hunt for" item 3: oracle machinery may legitimately feed targets,
   but must never bleed into the per-step VALUE columns.

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_canary_packhistory_trees_replay.py -v
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from cts.data.preprocess_mc.pack import (
    _VALUE_FEATURE_NAME,
    _build_compact_trajectory,
    _ordered_expansion_parent_ids,
    _replay_backprop_history,
    build_compact_trajectory,
)

_HUMAN_TREES_DIR = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
_SCHEMA = tree_encoder_feature_schema()

# Impossible sentinel: 'value' is win-loss in [-1, 1] (see teacher_targets.py's
# VALUE_SCALAR_TO_CENTIPAWNS comment); -999 can never occur naturally.
_SENTINEL = -999.0


def _sample_human_tree_paths(n: int = 12) -> List[str]:
    """First ``n`` files by the ``{idx:06d}_root_{idx}.pt`` naming convention.

    Same approach as ``test_packhistory_trees.py``'s own helper: constructs
    candidate paths directly rather than listing the ~1.08M-entry directory.
    """
    paths = []
    idx = 0
    while len(paths) < n and idx < n * 50:
        candidate = os.path.join(_HUMAN_TREES_DIR, f"{idx:06d}_root_{idx}.pt")
        if os.path.exists(candidate):
            paths.append(candidate)
        idx += 1
    return paths


def _require_real_source_trees() -> str:
    if not os.path.isdir(_HUMAN_TREES_DIR):
        return f"{_HUMAN_TREES_DIR!r} is not reachable from this host"
    return ""


def _load_sample_records(n: int = 12) -> List[Tuple[str, RawPretrainExampleRecord]]:
    out: List[Tuple[str, RawPretrainExampleRecord]] = []
    for path in _sample_human_tree_paths(n):
        try:
            record = RawPretrainExampleRecord.load(path)
        except Exception:  # noqa: BLE001 -- a malformed/legacy file shouldn't kill the sample
            continue
        out.append((path, record))
    return out


def _rows_by_key(replay: Dict[str, "object"]) -> Dict[Tuple[int, int], Tuple[int, float, Tuple[float, float, float]]]:
    """Map (node_id, step_index) -> (visit_count, q_value, wdl) for one replay() output."""
    out: Dict[Tuple[int, int], Tuple[int, float, Tuple[float, float, float]]] = {}
    steps = replay["update_log_step_index"].tolist()
    nodes = replay["update_log_node_id"].tolist()
    visits = replay["update_log_visit_count"].tolist()
    qvals = replay["update_log_q_value"].tolist()
    wdls = replay["update_log_wdl"].tolist()
    for step, node, visit, q, wdl in zip(steps, nodes, visits, qvals, wdls):
        out[(node, step)] = (visit, q, tuple(wdl))
    return out


_pytestmark_real = pytest.mark.skipif(_require_real_source_trees() != "", reason=_require_real_source_trees())


# --------------------------------------------------------------------------- #
# Canary 1: truncated replay is causally deterministic
# --------------------------------------------------------------------------- #
@_pytestmark_real
def test_canary_truncated_replay_matches_full_replay_up_to_truncation():
    """Replay a real tree twice: once with the full 96-expansion order, once
    truncated to the first S. Every row with step_index < S must be
    bit-identical between the two runs -- proving _replay_backprop_history
    never consults anything beyond "the expansions processed so far" (e.g.
    the FULL array's length, or a pass over later elements) to compute an
    early row. A positive control confirms the truncated run really is
    missing later information (not vacuously identical because nothing was
    cut).
    """
    samples = _load_sample_records(12)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees")

    checked_trees = 0
    for path, record in samples:
        eids = _ordered_expansion_parent_ids(record)
        if len(eids) < 20:
            continue
        truncate_at = len(eids) // 2

        full_replay = _replay_backprop_history(record, eids, value_feature=_VALUE_FEATURE_NAME)
        truncated_replay = _replay_backprop_history(
            record, eids[:truncate_at], value_feature=_VALUE_FEATURE_NAME
        )

        full_rows = _rows_by_key(full_replay)
        truncated_rows = _rows_by_key(truncated_replay)

        # Every row the truncated run produced with step_index < truncate_at
        # must appear, byte-for-byte, in the full run.
        early_truncated_rows = {k: v for k, v in truncated_rows.items() if k[1] < truncate_at}
        assert early_truncated_rows, f"{path}: truncated run produced no early rows -- fixture too small"
        for key, value in early_truncated_rows.items():
            assert key in full_rows, f"{path}: {key} present in truncated run but missing from full run"
            assert full_rows[key] == value, (
                f"{path}: row {key} differs between full and truncated replay "
                f"(full={full_rows[key]}, truncated={value}) -- the replay is not causally "
                f"local, an early step_index's row depends on information beyond that step."
            )
        # Truncated run must not itself have manufactured any row at/after
        # the cutoff (sanity on the truncation, not the causality claim).
        assert all(key[1] < truncate_at for key in truncated_rows), (
            f"{path}: truncated replay (expansions[:{truncate_at}]) produced a row at/after "
            f"the cutoff -- truncation isn't doing what this test assumes"
        )

        # Positive control: the full run must have rows at/after the cutoff
        # that are simply ABSENT from the truncated run (proves truncation
        # really did remove real information, so the "identical before the
        # cutoff" result above isn't vacuous).
        late_full_rows = {k for k in full_rows if k[1] >= truncate_at}
        assert late_full_rows, f"{path}: full run has no rows at/after the cutoff -- pick a different tree"
        checked_trees += 1

    assert checked_trees > 0, "no real tree was long enough to exercise this canary"


# --------------------------------------------------------------------------- #
# Canary 2: poisoning a late-first-touched node's static leaf value must not
# leak into any update-log row before that node's own first appearance.
# --------------------------------------------------------------------------- #
def _find_late_touched_node(record: RawPretrainExampleRecord, eids: List[int], min_step: int) -> Optional[Tuple[int, int]]:
    """Find (node_id, first_touch_step) for a real node whose first update-log
    entry lands at step_index > min_step (i.e. only touched by backprop in
    roughly the second half of the search)."""
    replay = _replay_backprop_history(record, eids, value_feature=_VALUE_FEATURE_NAME)
    ptr = replay["node_update_ptr"].tolist()
    steps = replay["update_log_step_index"].tolist()
    num_nodes = int(record.parent_index.shape[0])
    for node_id in range(1, num_nodes):
        lo, hi = ptr[node_id], ptr[node_id + 1]
        if hi > lo and steps[lo] > min_step:
            return node_id, steps[lo]
    return None


@_pytestmark_real
def test_canary_late_touched_node_poison_does_not_leak_backward():
    """Poison the STATIC leaf value of a real node that is only ever first
    touched by backprop in roughly the second half of the search (step_index
    > 60 out of 96), with an impossible sentinel, and confirm no update-log
    row for ANY node, at ANY step_index strictly before that node's own
    first-touch step, shows a trace of it.

    This is the direct analogue, at the SOURCE (the raw per-node static
    feature that feeds _replay_backprop_history), of the "poison the on-disk
    quantity that leaked" methodology in test_canary_future_leakage.py -- but
    targeting the replay loop itself rather than a downstream reader, and
    poisoning a node-local input rather than the aggregate baseline array.
    """
    samples = _load_sample_records(12)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees")

    checked_any = False
    for path, record in samples:
        eids = _ordered_expansion_parent_ids(record)
        found = _find_late_touched_node(record, eids, min_step=60)
        if found is None:
            continue
        node_id, first_touch_step = found

        value_col = record.feature_names.index(_VALUE_FEATURE_NAME)

        # Two independently loaded copies: one left clean, one poisoned --
        # avoids any risk of shared-tensor mutation bleeding between them.
        clean_record = RawPretrainExampleRecord.load(path)
        poisoned_record = RawPretrainExampleRecord.load(path)
        original_value = float(poisoned_record.node_features[node_id, value_col].item())
        assert abs(original_value - _SENTINEL) > 1.0, "picked a node whose real value coincidentally collides with the sentinel"
        poisoned_record.node_features[node_id, value_col] = _SENTINEL

        clean_replay = _replay_backprop_history(clean_record, eids, value_feature=_VALUE_FEATURE_NAME)
        poisoned_replay = _replay_backprop_history(poisoned_record, eids, value_feature=_VALUE_FEATURE_NAME)

        clean_rows = _rows_by_key(clean_replay)
        poisoned_rows = _rows_by_key(poisoned_replay)
        assert set(clean_rows) == set(poisoned_rows), (
            f"{path}: poisoning a static leaf value changed WHICH (node, step) rows exist -- "
            f"should only ever change VALUES of existing rows, never the log's own shape"
        )

        diverging_keys = [key for key in clean_rows if clean_rows[key] != poisoned_rows[key]]
        # --- Main claim: nothing before the node's own first-touch step diverges. ---
        early_divergences = [key for key in diverging_keys if key[1] < first_touch_step]
        assert not early_divergences, (
            f"{path}: future leakage in the replay SOURCE: poisoning node {node_id}'s static "
            f"value (first touched at step {first_touch_step}) changed rows at earlier steps: "
            f"{early_divergences[:10]} -- some node's update-log entry, tagged with a step_index "
            f"strictly before the poisoned node ever entered the search, is computed using "
            f"information that (per the real backprop history) didn't exist yet."
        )
        # --- Positive control: the poison IS reachable, at/after that step. ---
        assert diverging_keys, (
            f"{path}: poisoning node {node_id}'s static value changed nothing at all -- "
            f"this canary is vacuous (either the node is unreachable or the poison never "
            f"propagates anywhere, which would itself be a different bug)"
        )
        assert all(key[1] >= first_touch_step for key in diverging_keys), (
            f"{path}: unexpected divergence keys outside the expected causal window: {diverging_keys[:10]}"
        )
        checked_any = True

    assert checked_any, "no real tree in the sample had a node first-touched after step 60 -- sample too shallow"


# --------------------------------------------------------------------------- #
# Canary 3: per-step touched-node set matches an INDEPENDENT reconstruction
# --------------------------------------------------------------------------- #
def _expected_touched_nodes_at_step(parent_index: List[int], expanded_node: int) -> List[int]:
    """Independent (does NOT call pack.py's _ancestor_edge_path) reconstruction
    of which nodes should receive an update-log row when ``expanded_node`` is
    expanded: every node on its own root-to-self ancestor chain, EXCLUDING
    the root itself (the root has no incoming edge to back up into).

    Plain parent-pointer walk, written fresh here rather than trusting the
    exact same helper the production code uses -- if pack.py's own
    ``_ancestor_edge_path`` had a systematic bug (e.g. off-by-one dropping
    the leaf itself, or including one extra ancestor), reusing it in this
    test would hide the bug rather than catch it.
    """
    chain: List[int] = []
    current = expanded_node
    while parent_index[current] != -1:
        chain.append(current)
        current = parent_index[current]
    return chain  # order doesn't matter for the set-equality check below


@_pytestmark_real
def test_canary_update_log_step_partition_matches_independent_reconstruction():
    """For every step of several real trees, independently recompute (via a
    plain parent-pointer walk, not by calling any of pack.py's own path
    helpers) the SET of nodes that should get an update-log row at that
    step_index, and check it against the actual set the real production
    replay assigns to that step_index.

    This is an exhaustive, whole-trajectory cross-check (every one of the 96
    steps, not a sample) of the single most safety-critical invariant in the
    whole design: "step k's rows come from exactly expansion k's real
    ancestor path, nothing more, nothing less, nothing from any other step."
    """
    samples = _load_sample_records(8)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees")

    checked_steps = 0
    for path, record in samples:
        eids = _ordered_expansion_parent_ids(record)
        replay = _replay_backprop_history(record, eids, value_feature=_VALUE_FEATURE_NAME)
        parent_index = record.parent_index.tolist()

        actual_by_step: Dict[int, set] = {}
        for node, step in zip(replay["update_log_node_id"].tolist(), replay["update_log_step_index"].tolist()):
            actual_by_step.setdefault(step, set()).add(node)

        for step_index, expanded_node in enumerate(eids):
            expected = set(_expected_touched_nodes_at_step(parent_index, expanded_node))
            actual = actual_by_step.get(step_index, set())
            assert actual == expected, (
                f"{path}: step {step_index} (expanding node {expanded_node}): actual update-log "
                f"nodes {sorted(actual)} != independently-reconstructed expected ancestor set "
                f"{sorted(expected)} -- either a wrong-step tag (future/past leakage) or a wrong "
                f"path (structural bug), not distinguishable from this check alone but both are "
                f"real problems this check would catch."
            )
            checked_steps += 1

    assert checked_steps > 500, f"sanity check on the check itself -- only covered {checked_steps} steps"


# --------------------------------------------------------------------------- #
# Canary 4: final_value/final_wdl is the LAST entry, not an extremum
# --------------------------------------------------------------------------- #
def _find_nonmonotonic_node(replay: Dict[str, "object"], num_nodes: int) -> Optional[int]:
    """Find a node with >=3 update-log entries whose q_value sequence goes
    both up and down (not monotonic), and whose last value differs from both
    its own max and min -- so "final == last" and "final == extremum" make
    different, checkable predictions."""
    ptr = replay["node_update_ptr"].tolist()
    qvals_all = replay["update_log_q_value"].tolist()
    for node_id in range(1, num_nodes):
        lo, hi = ptr[node_id], ptr[node_id + 1]
        if hi - lo < 3:
            continue
        qvals = qvals_all[lo:hi]
        diffs = [qvals[i + 1] - qvals[i] for i in range(len(qvals) - 1)]
        if any(d > 0 for d in diffs) and any(d < 0 for d in diffs):
            last = qvals[-1]
            if abs(last - max(qvals)) > 1e-9 and abs(last - min(qvals)) > 1e-9:
                return node_id
    return None


@_pytestmark_real
def test_canary_final_value_is_last_entry_not_an_extremum():
    """For a real node whose replayed q_value trajectory is genuinely
    non-monotonic (goes up AND down across its own updates) and whose FINAL
    value is neither the max nor the min of that trajectory, confirm
    replay()'s reported ``final_value``/``final_wdl`` (the quantity
    _build_compact_trajectory bakes into node_features as the baseline)
    equals the CHRONOLOGICALLY LAST entry.

    Rules out any implementation that picks "the biggest-magnitude update"
    or "the most extreme update" instead of "the update from the real, last
    expansion event" -- a mistake that would be invisible on a monotonic
    trajectory (where last == extremum coincide) but is exactly the kind of
    "implicitly encodes later/aggregate information" shortcut item 1 of this
    audit's brief asks about.
    """
    samples = _load_sample_records(12)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees")

    checked_any = False
    for path, record in samples:
        eids = _ordered_expansion_parent_ids(record)
        replay = _replay_backprop_history(record, eids, value_feature=_VALUE_FEATURE_NAME)
        num_nodes = int(record.parent_index.shape[0])
        node_id = _find_nonmonotonic_node(replay, num_nodes)
        if node_id is None:
            continue

        ptr = replay["node_update_ptr"].tolist()
        lo, hi = ptr[node_id], ptr[node_id + 1]
        last_q = float(replay["update_log_q_value"].tolist()[hi - 1])
        last_wdl = tuple(replay["update_log_wdl"].tolist()[hi - 1])

        final_value = float(replay["final_value"][node_id])
        final_wdl = tuple(float(x) for x in replay["final_wdl"][node_id])

        assert final_value == pytest.approx(last_q, abs=1e-6), (
            f"{path}: node {node_id}: final_value={final_value} != chronologically-last "
            f"q_value={last_q} -- final_value is not simply 'the last real backup', which "
            f"would be consistent with a magnitude/extremum-based shortcut instead."
        )
        for component_index in range(3):
            assert final_wdl[component_index] == pytest.approx(last_wdl[component_index], abs=1e-6), (
                f"{path}: node {node_id}: final_wdl component {component_index} doesn't match "
                f"the chronologically-last entry"
            )
        checked_any = True

    assert checked_any, "no real tree in the sample had a suitably non-monotonic node -- sample too small/uniform"


# --------------------------------------------------------------------------- #
# Canary 5: oracle-trace poisoning must not reach the update log's VALUE
# columns (it legitimately reaches halt_rewards -- a positive control).
# --------------------------------------------------------------------------- #
@_pytestmark_real
def test_canary_oracle_trace_poison_does_not_reach_update_log():
    """Poison the DP-oracle-adjacent fields (``oracle_root_q_trace``,
    ``oracle_final_root_q_values``, ``oracle_best_move_index``) -- quantities
    that are BY DESIGN computed from the complete, final tree and legitimately
    feed training TARGETS (``halt_rewards``) -- and confirm the poison does
    NOT reach the update log's ``q_value``/``wdl`` columns or the
    ``node_features`` value/WDL baseline built from them.

    Positive control: the SAME poison DOES change ``halt_rewards`` (proving
    the poison is reachable through the pipeline at all, and specifically
    through the one field that's supposed to depend on it) -- so a "nothing
    changed anywhere" degenerate implementation would not pass this test
    vacuously.
    """
    samples = _load_sample_records(12)
    if not samples:
        pytest.skip("no loadable sample records found under human_trees")

    checked_any = False
    for path, record in samples:
        clean_record = RawPretrainExampleRecord.load(path)
        poisoned_record = RawPretrainExampleRecord.load(path)

        # Poison the oracle-adjacent fields with an impossible sentinel,
        # preserving shapes (these are validated tensors -- _validate() would
        # reject a shape change, but not an out-of-range value).
        poisoned_record.oracle_root_q_trace.fill_(_SENTINEL)
        poisoned_record.oracle_final_root_q_values.fill_(_SENTINEL)
        # oracle_best_move_index must stay a valid index into oracle_root_moves
        # (an int32 tensor validated for shape only, not value range beyond
        # that) -- shift every entry to a different-but-valid index so the
        # move IDENTITY used for the halt-reward lookup changes, without
        # crashing PretrainExample-style validation that doesn't apply to
        # this record type. If there's only one root move, skip (nothing to
        # perturb).
        num_moves = len(poisoned_record.oracle_root_moves)
        if num_moves < 2:
            continue
        original_index = poisoned_record.oracle_best_move_index.clone()
        # RawPretrainExampleRecord is a frozen dataclass -- object.__setattr__
        # is the documented escape hatch its own __post_init__ uses internally.
        object.__setattr__(poisoned_record, "oracle_best_move_index", (original_index + 1) % num_moves)

        clean_trajectory = build_compact_trajectory(clean_record, _SCHEMA, source_path=path)
        poisoned_trajectory = build_compact_trajectory(poisoned_record, _SCHEMA, source_path=path)
        if clean_trajectory is None or poisoned_trajectory is None:
            continue

        # --- Main claim: update log VALUE columns and the node_features
        # baseline built from them are untouched. ---
        for field in ("update_log_q_value", "update_log_wdl", "update_log_step_index", "update_log_node_id"):
            clean_arr = clean_trajectory[field]
            poisoned_arr = poisoned_trajectory[field]
            assert clean_arr.shape == poisoned_arr.shape, f"{path}: {field} shape changed by oracle-trace poison"
            assert (clean_arr == poisoned_arr).all(), (
                f"{path}: future/oracle leakage: {field} differs after poisoning "
                f"oracle_root_q_trace/oracle_final_root_q_values/oracle_best_move_index -- "
                f"the update log's per-step VALUE columns must be computed purely from real "
                f"replayed backprop, never from the full-tree DP-oracle machinery."
            )
        assert (clean_trajectory["node_features"] == poisoned_trajectory["node_features"]).all(), (
            f"{path}: node_features baseline (final_value/final_wdl) differs after the oracle-trace "
            f"poison -- the same leak, one layer up."
        )

        # --- Positive control: halt_rewards DOES change (it's legitimately
        # sourced from oracle_final_root_q_values / oracle_best_move_index). ---
        halt_changed = not (clean_trajectory["halt_rewards"] == poisoned_trajectory["halt_rewards"]).all()
        assert halt_changed, (
            f"{path}: poisoning oracle_final_root_q_values/oracle_best_move_index left "
            f"halt_rewards completely unchanged -- this canary isn't exercising the intended "
            f"code path (build_compact_trajectory may not actually be reading these fields "
            f"the way this test assumes), so the 'no leak into update log' result above isn't "
            f"informative."
        )
        checked_any = True

    assert checked_any, "no real tree in the sample survived poisoning + rebuild -- sample too small/degenerate"
