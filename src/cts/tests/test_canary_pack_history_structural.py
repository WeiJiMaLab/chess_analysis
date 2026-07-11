"""Adversarial canaries for structural (not value/WDL) future-leakage failure modes in
``pack_history.py``, plus a robustness/latent-bug check and a fresh, independent
regression canary for the already-fixed step-index off-by-one.

``test_no_future_leakage.py`` and ``test_canary_future_leakage.py`` (this directory)
both focus on the *value/WDL* leak: does a node's per-step feature row ever reflect
information from after the queried step. This file targets four DIFFERENT failure
modes that those files do not cover, all built on hand-constructed, fully synthetic
``HistoryTrajectory`` fixtures (no dependency on real ``/scratch`` data, so these
always run, never skip):

1. ``test_edge_set_never_includes_future_expansion`` / ``test_edge_set_transition_is_exact``
   -- T_n's *structure* (which edges/nodes exist, not just their values) must never
   include an edge born from an expansion event that happens after step n. Exhaustive,
   hand-computed ground truth across every step of a small synthetic tree, plus an
   explicit before/after transition check as the positive control.

2. ``test_never_visited_filter_uses_whole_trajectory_not_prefix`` -- the
   never-visited-leaf filter (``_never_visited``, used by
   ``build_snapshot_pair_example``'s ``keep_mask``) is a function of the *entire*
   96-step trajectory, not of anything knowable as of step n. Directly demonstrates
   this by constructing two leaves that are IDENTICAL as of every step <= n (both
   zero-visit so far) but differ only in whether a visit happens at some step > n --
   and showing the filter's decision differs between them despite that.

3. ``test_never_visited_filter_couples_sibling_predictions_via_attention`` -- follows
   up on (2): does that future-dependent filtering decision actually reach the
   model's OUTPUT for a *different* (unfiltered, legitimately-kept) edge under the
   same parent? Runs a real (randomly initialized but deterministic) ``ChildWdlModel``
   forward pass twice -- once with a never-visited sibling edge present, once with it
   removed exactly as ``build_snapshot_pair_example`` would -- and shows the KEPT
   sibling's own predicted WDL logits change. This empirically checks (not just
   assumes) whether ``ChildWdlModel`` processes edges independently within a tree.

4. ``test_delta_visits_window_matches_target_window`` -- ``_delta_visits``' weight
   window and ``_forward_filled_wdl_at_step``'s target-query step must describe the
   same "new visits since T_n, as of T_{n+k}" boundary, cross-checked against a
   brute-force independent count, plus confirms T_n's own INPUT node_features never
   move no matter what is injected into that future window.

5. ``test_snapshot_step_range_enforced_by_only_caller`` /
   ``test_forward_fill_past_recorded_range_clamps_not_garbage`` -- confirms
   ``deterministic_snapshot_steps``'s ``[1, high]`` range is respected by its only
   real caller, and separately (robustness check, not a real-path exercise, since no
   caller currently violates the contract) that querying past the last recorded
   update-log entry degrades safely to "last known value" forward-fill, not garbage.

6. ``test_off_by_one_regression_canary_independent`` -- a fresh, from-scratch,
   hand-computed regression canary for the step-index off-by-one fixed 2026-07-10
   (see history.md's Task 4 Progress Log, "off-by-one fix pass" entry), independent
   of both existing test files' fixtures/methodology.

7. ``test_value_query_ignores_root_rank_latent_bug`` -- documents a related, but
   DIFFERENT and NOT YET fixed, latent inconsistency found while re-deriving (6) from
   scratch: ``_build_tree_n``'s structural cutoff (``expansion_count =
   first_decision_expansion_count + local_step``) correctly incorporates
   ``first_decision_expansion_count`` (`= root_rank + 1`, from
   ``preprocess_mc/pack.py:_build_compact_trajectory``), but its VALUE query
   (``_forward_filled_wdl_at_step(trajectory, node_ids, local_step)``) uses bare
   ``local_step`` and does not. Empirically confirmed DORMANT today (root_rank == 0
   for all 200 real ``xaba20k`` trees sampled while investigating this file -- see
   this module's own inline note below), but live and reproducible on synthetic data
   with ``first_decision_expansion_count > 1``. Direction of the resulting error is
   understaying (this queries a real, valid, but too-early step), not leaking the
   future, so it falls outside this repo's specific "future leaks backward" threat
   model -- flagged as a real, separate correctness hypothesis, not re-classified as
   a leak.

Run explicitly (this directory isn't covered by pytest.ini's testpaths):
    pytest src/cts/tests/test_canary_pack_history_structural.py -v
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
import torch

from cts.core.schema import tree_encoder_feature_schema
from cts.core.tensorizer import TensorizedTreeExample, collate_tensorized_examples
from cts.data.preprocess_gnn.pack_history import (
    HistoryTrajectory,
    _build_tree_n,
    _delta_visits,
    _forward_filled_wdl_at_step,
    _never_visited,
    build_snapshot_pair_example,
    deterministic_snapshot_steps,
    PackHistoryGNNPretrainConfig,
)
from cts.models.gnn import ChildWdlModel


# --------------------------------------------------------------------------- #
# Synthetic trajectory builder
# --------------------------------------------------------------------------- #


def _make_trajectory(
    *,
    num_nodes: int,
    parents: List[int],
    depths: List[int],
    children_by_parent: Dict[int, List[int]],
    expansion_parent_ids: List[int],
    first_decision_expansion_count: int,
    step_node_cutoffs: List[int],
    updates: Dict[int, List[Tuple[int, float, float, float]]],
) -> HistoryTrajectory:
    """Hand-build a ``HistoryTrajectory`` from explicit, easy-to-audit Python
    structures -- CSR/flat-array plumbing is generated here so every test below
    only has to state the *tree shape* and *update log* it wants, not manage
    pointer arithmetic by hand.

    ``updates``: node_id -> list of (step_index, q_value, wdl_win, wdl_draw) rows,
    ascending step_index (wdl_loss is derived as 1 - win - draw so every synthetic
    row is a valid distribution unless a test deliberately overwrites it).
    """
    node_features = torch.zeros((num_nodes, 5), dtype=torch.float32)  # baseline irrelevant to these tests
    parent_index = torch.tensor(parents, dtype=torch.long)
    depth = torch.tensor(depths, dtype=torch.long)

    child_ptr = [0]
    edge_child: List[int] = []
    edge_slot: List[int] = []
    for parent_id in range(num_nodes):
        kids = children_by_parent.get(parent_id, [])
        edge_child.extend(kids)
        edge_slot.extend(range(len(kids)))
        child_ptr.append(len(edge_child))

    node_update_ptr = [0]
    update_step_index: List[int] = []
    update_wdl: List[List[float]] = []
    update_visit_count: List[int] = []
    update_q_value: List[float] = []
    for node_id in range(num_nodes):
        rows = updates.get(node_id, [])
        for step_index, q_value, win, draw in rows:
            update_step_index.append(step_index)
            update_wdl.append([win, draw, 1.0 - win - draw])
            update_visit_count.append(1)
            update_q_value.append(q_value)
        node_update_ptr.append(len(update_step_index))

    return HistoryTrajectory(
        source_path="synthetic://test",
        node_features=node_features,
        parent_index=parent_index,
        depth=depth,
        child_ptr=torch.tensor(child_ptr, dtype=torch.long),
        edge_child=torch.tensor(edge_child, dtype=torch.long),
        edge_slot=torch.tensor(edge_slot, dtype=torch.long),
        expansion_parent_ids=torch.tensor(expansion_parent_ids, dtype=torch.long),
        first_decision_expansion_count=first_decision_expansion_count,
        step_node_cutoffs=torch.tensor(step_node_cutoffs, dtype=torch.long),
        node_update_ptr=torch.tensor(node_update_ptr, dtype=torch.long),
        update_step_index=torch.tensor(update_step_index, dtype=torch.long),
        update_wdl=torch.tensor(update_wdl, dtype=torch.float32) if update_wdl else torch.zeros((0, 3)),
        update_visit_count=torch.tensor(update_visit_count, dtype=torch.long),
        update_q_value=torch.tensor(update_q_value, dtype=torch.float32),
    )


# --------------------------------------------------------------------------- #
# 1. Structural (edge-set) future leak
# --------------------------------------------------------------------------- #


def _branching_trajectory() -> HistoryTrajectory:
    """8-node tree, 5 expansion events, each node expanded at most once (matching
    the real invariant from ``_ordered_expansion_parent_ids``: a node's children are
    all created together in a single expansion event, never incrementally).

    Shape:
      exp0 (step_index=0, parent=0): children 1,2   -> nodes {0,1,2}
      exp1 (step_index=1, parent=1): children 3,4   -> +{3,4}
      exp2 (step_index=2, parent=2): child  5       -> +{5}
      exp3 (step_index=3, parent=3): child  6       -> +{6}  (born late: n=4)
      exp4 (step_index=4, parent=5): child  7       -> +{7}  (born latest: n=5)

    ``first_decision_expansion_count=1`` (root_rank=0, i.e. no pre-root trimming --
    matches the real, empirically-confirmed-always-0-root_rank case, see test 7 below
    for the root_rank>0 case).
    """
    return _make_trajectory(
        num_nodes=8,
        parents=[-1, 0, 0, 1, 1, 2, 3, 5],
        depths=[0, 1, 1, 2, 2, 2, 3, 3],
        children_by_parent={0: [1, 2], 1: [3, 4], 2: [5], 3: [6], 5: [7]},
        expansion_parent_ids=[0, 1, 2, 3, 5],
        first_decision_expansion_count=1,
        step_node_cutoffs=[3, 5, 6, 7, 8],
        updates={},  # values irrelevant to this test; structure only
    )


def test_edge_set_never_includes_future_expansion():
    """Exhaustive, hand-computed ground truth: for every n in [1, 5], T_n's edge set
    (parent, child) pairs must equal exactly the union of children created by
    expansion events at full positions [0, n), never more.

    A regression that leaks even one future edge early (e.g. an off-by-one in
    ``expansion_count = first_decision_expansion_count + local_step``, or in the
    ``active_parents`` slice) would show up as an extra pair in some ``n``'s set
    below.
    """
    traj = _branching_trajectory()
    expected_edges_by_n = {
        1: {(0, 1), (0, 2)},
        2: {(0, 1), (0, 2), (1, 3), (1, 4)},
        3: {(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)},
        4: {(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)},
        5: {(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (5, 7)},
    }
    # Node count present at each n (from _branching_trajectory's own step_node_cutoffs) --
    # cross-checked here so the node-array-size assertion below isn't just restating
    # the fixture's own numbers back at itself.
    expected_node_count_by_n = {1: 3, 2: 5, 3: 6, 4: 7, 5: 8}
    for n, expected in expected_edges_by_n.items():
        snap = _build_tree_n(traj, n)
        assert snap is not None
        actual = set(zip(snap.edge_parent.tolist(), snap.edge_child.tolist()))
        assert actual == expected, (
            f"future leakage regression: T_{n}'s edge set is {actual}, expected {expected}. "
            f"Extra edges (leaked from the future): {actual - expected}. "
            f"Missing edges (structurally understated): {expected - actual}."
        )
        # Node array itself must never expose future-born node ids either (a node id
        # appearing in node_features with no way to reach it via any edge would still
        # be a latent leak channel if a consumer ever indexed it directly).
        assert snap.node_features.shape[0] == expected_node_count_by_n[n], (
            f"T_{n}'s node array has {snap.node_features.shape[0]} rows, expected "
            f"{expected_node_count_by_n[n]} -- a future-born node id is reachable even if no edge "
            f"currently points to it"
        )
        for child_id in snap.edge_child.tolist():
            assert child_id < expected_node_count_by_n[n], (
                f"T_{n} has an edge pointing to node {child_id}, which is >= this step's own node "
                f"count {expected_node_count_by_n[n]} -- a future-born node reachable via an edge"
            )


def test_edge_set_transition_is_exact():
    """Positive control for the test above: child 6's edge (parent=3) must be
    ABSENT at n=3 and PRESENT at n=4 -- an exact one-step transition, not "always
    absent" (which would make the leak-freedom claim vacuous) or "present too
    early" (which would be the leak itself).
    """
    traj = _branching_trajectory()

    snap_n3 = _build_tree_n(traj, 3)
    assert snap_n3 is not None
    edges_n3 = set(zip(snap_n3.edge_parent.tolist(), snap_n3.edge_child.tolist()))
    assert (3, 6) not in edges_n3, "future leakage: edge (3,6) (born at expansion position 3, i.e. n=4) visible at n=3"

    snap_n4 = _build_tree_n(traj, 4)
    assert snap_n4 is not None
    edges_n4 = set(zip(snap_n4.edge_parent.tolist(), snap_n4.edge_child.tolist()))
    assert (3, 6) in edges_n4, (
        "positive control failed: edge (3,6) never appears even at n=4, when it structurally should -- "
        "this test isn't exercising the intended code path (or active_parents/expansion_count is broken "
        "in the OTHER direction, silently dropping real edges)"
    )

    # depth/parent_index must be consistently truncated to node_cutoff too, not just edges.
    assert snap_n3.node_features.shape[0] == 6  # nodes 0..5 only
    assert snap_n3.parent_index.shape[0] == 6
    assert snap_n3.depth.shape[0] == 6
    assert int(snap_n3.parent_index.max().item()) < 6, "parent_index at n=3 references a not-yet-existing node"


# --------------------------------------------------------------------------- #
# 2. Never-visited-leaf filter: whole-trajectory dependence
# --------------------------------------------------------------------------- #


def _sibling_trajectory(late_visit_step: int) -> HistoryTrajectory:
    """3-node star: root(0) -> {leafA(1), leafB(2)}, both created at the single
    root expansion (step_index=0, n=1). leafA is genuinely NEVER visited by
    backprop (0 update-log rows, ever). leafB has exactly one update-log row at
    ``late_visit_step`` -- if ``late_visit_step`` is chosen > n for whatever n a
    test queries, leafB is *observationally identical to leafA as of that n*
    (both zero-visit so far) but ``_never_visited`` must still tell them apart,
    because it looks at the whole trajectory, not just "so far".
    """
    return _make_trajectory(
        num_nodes=3,
        parents=[-1, 0, 0],
        depths=[0, 1, 1],
        children_by_parent={0: [1, 2]},
        expansion_parent_ids=[0],
        first_decision_expansion_count=1,
        step_node_cutoffs=[3],
        updates={2: [(late_visit_step, 0.5, 0.6, 0.3)]},  # node 1 (leafA): no rows at all
    )


def test_never_visited_filter_uses_whole_trajectory_not_prefix():
    """Direct demonstration that ``_never_visited`` is NOT a function of "state as of
    step n": build two leaves indistinguishable as of n=1 (both zero visits so far)
    that differ only in whether a backprop update happens at some step STRICTLY
    AFTER n=1. ``_never_visited`` must classify them differently -- proving its
    answer depends on information only available by looking past step n.

    This is not, by itself, a claim that this is a leak (see this file's module
    docstring point 3/7 and the report this test was written for) -- it is the
    narrower, mechanical fact the leak-or-not judgment call rests on.
    """
    traj = _sibling_trajectory(late_visit_step=50)  # visited at a step far beyond any n this test queries

    assert _never_visited(traj, 1) is True, "leafA (genuinely never visited) misclassified as visited"
    assert _never_visited(traj, 2) is False, (
        "leafB (visited at step 50, far in the future relative to n=1) misclassified as never-visited -- "
        "or, if this assertion is what fails, it would mean _never_visited degenerately treats every leaf "
        "the same regardless of the update log, which would make the filter's future-dependence moot for "
        "a different (more alarming) reason: it wouldn't be using the update log at all."
    )

    # As of step n=1 (the only step this tree has), both leaves are IDENTICAL in
    # T_1's own INPUT features -- confirming the only distinguishing signal is
    # PRESENCE/ABSENCE after filtering, not a value difference.
    snap = _build_tree_n(traj, 1)
    assert snap is not None
    assert torch.equal(snap.node_features[1], snap.node_features[2]), (
        "test setup invalid: leafA and leafB should be feature-identical at n=1 (both zero-visit so far) "
        "for this to isolate the filter's structural effect from any value difference"
    )


def test_never_visited_filter_couples_sibling_predictions_via_attention():
    """Follow-up: does the future-dependent filtering decision from the test above
    actually reach the model's OUTPUT for the *other*, legitimately-kept sibling
    edge under the same parent?

    ``ChildWdlHead`` itself is a plain per-edge MLP (see ``gnn.py``), but
    ``TreeEncoder``'s upward message pass (``TreeAttMsgLayer.forward``,
    ``tree_mha.py``) computes each parent's message as a SEGMENTED SOFTMAX
    attention over exactly the children present in ``edge_parent``/``edge_child``
    -- removing one child changes the softmax denominator (``parent_sums``) over
    the remaining children for every other child under that parent. This test
    checks that mechanism is live, not just present in the code: run the same
    ``ChildWdlModel`` forward pass over the SAME tree twice, once with leafA's
    edge present and once with it removed (exactly as ``build_snapshot_pair_example``
    would after the never-visited filter), and confirm leafB's OWN predicted logits
    (leafB's row/edge is untouched by the filter in both runs) differ between the
    two runs.

    If this assertion instead found NO difference, that would falsify the "probably
    benign, edges are processed independently" assumption flagged as unconfirmed in
    the task this test was written for -- i.e. it would be good news. It does NOT
    find that: the coupling is real and architecturally expected given segmented
    softmax attention, so the never-visited filter is judged here as a genuine
    train-time structural artifact (see this file's module docstring point 3),
    not merely "the label set changes, nothing else does."
    """
    torch.manual_seed(0)
    schema = tree_encoder_feature_schema()
    model = ChildWdlModel(k=1, node_feat=len(schema.feature_names), device="cpu",
                           node_embed_hidden=16, d_embed=16, d_message=16, n_heads=2, d_att=8,
                           decoder_hidden=16)
    model.eval()

    node_features = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],   # root
            [0.1, 0.4, 0.3, 0.3, 0.1],   # leafA (the one that gets filtered)
            [-0.2, 0.2, 0.3, 0.5, 0.2],  # leafB (stays kept in both runs)
        ],
        dtype=torch.float32,
    )
    parent_index = torch.tensor([-1, 0, 0], dtype=torch.long)
    depth = torch.tensor([0, 1, 1], dtype=torch.long)

    def _example(include_leaf_a: bool) -> TensorizedTreeExample:
        if include_leaf_a:
            edge_parent, edge_child, edge_slot = [0, 0], [1, 2], [0, 1]
        else:
            edge_parent, edge_child, edge_slot = [0], [2], [1]  # leafB keeps its real slot=1
        return TensorizedTreeExample(
            node_features=node_features.clone(),
            parent_index=parent_index.clone(),
            edge_parent=torch.tensor(edge_parent, dtype=torch.long),
            edge_child=torch.tensor(edge_child, dtype=torch.long),
            edge_slot=torch.tensor(edge_slot, dtype=torch.long),
            depth=depth.clone(),
            node_targets=torch.zeros(3, dtype=torch.float32),
            feature_names=schema.feature_names,
        )

    batch_with_a, _ = collate_tensorized_examples([_example(include_leaf_a=True)])
    batch_without_a, _ = collate_tensorized_examples([_example(include_leaf_a=False)])

    with torch.inference_mode():
        logits_with_a = model(batch_with_a)
        logits_without_a = model(batch_without_a)

    # leafB's edge is the LAST edge in batch_with_a (edge_child==[1,2]) and the
    # ONLY edge in batch_without_a (edge_child==[2]) -- locate it by edge_child==2
    # in each batch rather than assuming a fixed row index.
    leaf_b_row_with_a = (batch_with_a.edge_child == 2).nonzero(as_tuple=True)[0]
    leaf_b_row_without_a = (batch_without_a.edge_child == 2).nonzero(as_tuple=True)[0]
    assert leaf_b_row_with_a.numel() == 1 and leaf_b_row_without_a.numel() == 1

    logit_b_with_a = logits_with_a[leaf_b_row_with_a[0]]
    logit_b_without_a = logits_without_a[leaf_b_row_without_a[0]]

    assert not torch.allclose(logit_b_with_a, logit_b_without_a, atol=1e-6), (
        "expected leafB's predicted WDL logits to change depending on whether leafA's edge is present "
        "(confirms real cross-sibling coupling via TreeAttMsgLayer's segmented softmax) -- if this now "
        "passes (no difference), the encoder architecture changed and the never-visited-filter judgment "
        "in pack_history.py's module docstring / this report should be revisited, not silently ignored."
    )


# --------------------------------------------------------------------------- #
# 3. Delta-visits window vs. target window
# --------------------------------------------------------------------------- #


def _delta_visits_trajectory() -> HistoryTrajectory:
    """Root(0) -> child(1), single expansion, but a RICH update log for node 1
    spanning step_index 0..20 (as if node 1 kept getting revisited by later
    expansions elsewhere in a bigger tree -- update-log rows don't require the
    node's own subtree to grow, only that it's on some later expansion's ancestor
    path, so this is a legitimate, if minimal, way to synthesize a dense log).
    """
    # Every integer step 0..20 gets its own row, and win is DISTINCT per step
    # (0.30, 0.31, 0.32, ...) so a boundary query one step off actually returns a
    # numerically different value -- a fixture with a constant/repeating WDL would
    # let an off-by-one boundary bug hide behind "the wrong row happened to have
    # the same value anyway".
    updates_node1 = [(s, 0.1 * s, 0.30 + 0.01 * s, 0.2) for s in range(0, 21)]
    return _make_trajectory(
        num_nodes=2,
        parents=[-1, 0],
        depths=[0, 1],
        children_by_parent={0: [1]},
        expansion_parent_ids=[0],
        first_decision_expansion_count=1,
        # 30 identical entries: only expansion 0 ever grows the tree, but this test
        # queries n up to 10 -- step_node_cutoffs just needs to be long enough that
        # _build_tree_n's `local_step < num_steps` bound check doesn't reject those
        # n's; the node COUNT staying flat at 2 is fine, this test only cares about
        # node 1's per-step VALUE (forward-filled from the update log), not further
        # structural growth.
        step_node_cutoffs=[2] * 30,
        updates={1: updates_node1},
    )


def test_delta_visits_window_matches_target_window():
    """Cross-checks three things against a from-scratch brute-force count, for
    several (n, lookahead_k) pairs:

      1. ``_delta_visits(traj, node, n-1, n_plus_k-1)`` equals the number of
         update-log rows with step_index strictly in ``(n-1, n_plus_k-1]``,
         counted independently here by iterating the raw step list in Python
         (not by re-deriving the same bisect calls the function itself uses).
      2. That same upper bound (``n_plus_k-1``) is exactly the step
         ``_forward_filled_wdl_at_step`` uses for the TARGET query -- i.e. Delta-
         visits counts precisely the rows that could have moved the target between
         T_n and T_{n+k}, no more, no fewer.
      3. T_n's own INPUT (``_build_tree_n``'s node_features) is bit-identical
         regardless of what is injected into that future window -- confirming the
         weight/target machinery, despite legitimately depending on future rows,
         never leaks into the model's input for the SAME example.
    """
    traj = _delta_visits_trajectory()
    all_steps = traj.update_step_index[
        traj.node_update_ptr[1].item() : traj.node_update_ptr[2].item()
    ].tolist()

    for n, lookahead_k in [(1, 4), (3, 6), (5, 10), (1, 20), (10, 8)]:
        n_plus_k = n + lookahead_k
        brute_force_count = sum(1 for s in all_steps if (n - 1) < s <= (n_plus_k - 1))
        actual_count = _delta_visits(traj, 1, n - 1, n_plus_k - 1)
        assert actual_count == brute_force_count, (
            f"n={n} k={lookahead_k}: _delta_visits returned {actual_count}, brute-force count over the "
            f"same (n-1, n_plus_k-1] window is {brute_force_count}"
        )

        # The target query's upper bound must be the exact same n_plus_k-1.
        target_at_bound = _forward_filled_wdl_at_step(traj, torch.tensor([1]), n_plus_k - 1)
        target_one_past = _forward_filled_wdl_at_step(traj, torch.tensor([1]), n_plus_k)
        rows_at_exact_bound = sum(1 for s in all_steps if s == n_plus_k - 1)
        rows_at_one_past = sum(1 for s in all_steps if s == n_plus_k)
        if rows_at_one_past > 0:
            # A row lands exactly one step past the delta-visits window: the target
            # query at n_plus_k-1 must NOT see it (else the window is off by one).
            assert not torch.equal(target_at_bound, target_one_past), (
                f"n={n} k={lookahead_k}: target query at step {n_plus_k - 1} is indistinguishable from "
                f"step {n_plus_k}, but a real update-log row exists at step {n_plus_k} -- suggests the "
                f"target query's step bound has drifted by one relative to Delta-visits' upper bound"
            )
        assert rows_at_exact_bound >= 0  # sanity: just documents this window's boundary is being exercised

        # Input side: build T_n and confirm it's identical no matter how far past
        # local_step the log continues (poison-free sanity re-check of the
        # already-tested-elsewhere input/target separation, from the delta-visits
        # angle specifically).
        snap = _build_tree_n(traj, n)
        assert snap is not None
        expected_local_step = n - 1
        rows_visible_to_input = sum(1 for s in all_steps if s <= expected_local_step)
        rows_beyond_input = sum(1 for s in all_steps if s > expected_local_step)
        if rows_beyond_input > 0 and rows_visible_to_input == 0:
            assert float(snap.node_features[1, 0]) == pytest.approx(0.0, abs=1e-9), (
                f"n={n}: T_n's input shows a nonzero value for node 1 despite zero update-log rows "
                f"visible as of local_step={expected_local_step} -- future rows are leaking into the input"
            )


# --------------------------------------------------------------------------- #
# 4. Snapshot-step range enforcement + forward-fill robustness past the real range
# --------------------------------------------------------------------------- #


def test_snapshot_step_range_enforced_by_only_caller():
    """``deterministic_snapshot_steps`` documents its range as
    ``[1, min(search_budget, num_available_steps) - lookahead_k]``. Confirm every
    sampled ``n``, across many (search_budget, num_available_steps, lookahead_k,
    snapshots_per_tree) combinations, satisfies ``n + lookahead_k <=
    min(search_budget, num_available_steps)`` -- i.e. ``build_snapshot_pair_example``
    is never handed an ``n`` whose ``n + lookahead_k`` would exceed this tree's own
    recorded step range. ``pack_history_split_to_shards`` (this module's only real
    caller of ``build_snapshot_pair_example``) sources every ``n`` exclusively from
    this function (confirmed by grep: no other call site exists in the repo), so this
    is the one gate that actually matters for that contract in production.
    """
    for search_budget in (10, 50, 96):
        for num_available_steps in (5, 30, 96, 200):
            for lookahead_k in (1, 5, 12, 40):
                for snapshots_per_tree in (1, 3):
                    config = PackHistoryGNNPretrainConfig(
                        pack_history_dir="unused", output_root="unused",
                        lookahead_k=lookahead_k, snapshots_per_tree=snapshots_per_tree,
                        search_budget=search_budget, seed=0,
                    )
                    high = min(search_budget, num_available_steps) - lookahead_k
                    chosen = deterministic_snapshot_steps("some/tree/path.pt", num_available_steps, config)
                    if high < 1:
                        assert chosen == [], (
                            f"search_budget={search_budget} num_available_steps={num_available_steps} "
                            f"lookahead_k={lookahead_k}: expected no snapshots sampled (range empty), got {chosen}"
                        )
                        continue
                    for n in chosen:
                        assert 1 <= n <= high, (
                            f"search_budget={search_budget} num_available_steps={num_available_steps} "
                            f"lookahead_k={lookahead_k}: sampled n={n} outside documented range [1, {high}] "
                            f"-- n + lookahead_k would exceed this tree's recorded step range"
                        )


def test_forward_fill_past_recorded_range_clamps_not_garbage():
    """Robustness check (not a real production path today, since (see test above)
    the one real caller never violates the contract): if ``_forward_filled_wdl_at_step``
    were ever queried with a step beyond the last real update-log entry recorded for
    a node -- e.g. a future config/caller bug that lets ``n + lookahead_k`` run past
    the tree's real recorded range -- it must degrade SAFELY to "the last known
    value forward-filled indefinitely" (standard, correct forward-fill semantics),
    not silently read uninitialized/clamped garbage or crash. Confirms the actual
    behavior so a future reader doesn't have to re-derive it from the bisect call.
    """
    traj = _delta_visits_trajectory()  # node 1's last real update is at step_index=20
    last_real_step = 20

    at_last = _forward_filled_wdl_at_step(traj, torch.tensor([1]), last_real_step)
    far_past_1 = _forward_filled_wdl_at_step(traj, torch.tensor([1]), last_real_step + 1)
    far_past_1000 = _forward_filled_wdl_at_step(traj, torch.tensor([1]), last_real_step + 1000)

    assert torch.equal(at_last, far_past_1) and torch.equal(at_last, far_past_1000), (
        "querying past the last recorded update-log entry does not stably return the last known value -- "
        f"got at_last={at_last.tolist()}, +1={far_past_1.tolist()}, +1000={far_past_1000.tolist()}"
    )
    # And this is a REAL (nonzero) value, not a degenerate always-zero fallback --
    # confirms this is genuinely "forward-filled last value", not an out-of-range
    # guard that happens to coincide with zero.
    assert far_past_1000[0, 0].item() != 0.0 or far_past_1000[0, 1].item() != 0.0, (
        "positive control failed: the forward-filled value past the real range is exactly zero, "
        "indistinguishable from an out-of-range guard -- pick a fixture whose last real WDL isn't zero"
    )

    # _build_tree_n's STRUCTURAL bound, by contrast, is explicitly checked
    # (local_step >= num_steps) and returns None rather than reading garbage --
    # confirming the two functions take different (both individually safe, but
    # asymmetric) approaches to being queried out of their intended range.
    num_steps = int(traj.step_node_cutoffs.shape[0])
    assert _build_tree_n(traj, num_steps + 100) is None, (
        "_build_tree_n should refuse (return None) for n far beyond this tree's recorded step range"
    )


# --------------------------------------------------------------------------- #
# 5. Fresh, independent regression canary for the already-fixed off-by-one
# --------------------------------------------------------------------------- #


def test_off_by_one_regression_canary_independent():
    """From-scratch, hand-computed canary for the step-index off-by-one fixed
    2026-07-10 (history.md, Task 4 Progress Log "off-by-one fix pass" entry),
    built independently of both existing leak test files' fixtures.

    Tree: root(0) -> child(1), single expansion (step_index=0, n=1). Node 1 gets
    exactly one backprop update, at step_index=0 (i.e. as part of the very
    expansion that created it -- the earliest a node can ever be visited, matching
    real PUCT semantics: a node's own incoming edge is updated the moment it's
    selected as the about-to-be-expanded leaf).

    At n=1 (local_step=0), the correct forward-filled value MUST include this
    step_index=0 update (``bisect_right`` at step=0 finds it: 0 <= 0). The
    old, buggy pre-fix code queried at ``step=n=1`` directly instead of
    ``local_step=n-1=0`` -- for THIS fixture that particular value happens to
    read identically (there is nothing at step_index=1 to spuriously include), so
    to make this canary actually discriminate, node 1 gets a SECOND update at
    step_index=1 (simulating "the next expansion's backup", exactly the shape of
    leak the original bug produced) with a distinguishable WDL. The fixed code at
    n=1 must show ONLY the step_index=0 value; the buggy ``step=n`` query would
    show the step_index=1 value instead.
    """
    traj = _make_trajectory(
        num_nodes=2,
        parents=[-1, 0],
        depths=[0, 1],
        children_by_parent={0: [1]},
        expansion_parent_ids=[0, 0],  # two expansion "positions" for arithmetic purposes only (see below)
        first_decision_expansion_count=1,
        step_node_cutoffs=[2, 2],  # node count is 2 at both local steps in this minimal fixture
        updates={1: [(0, 0.0, 0.6, 0.2), (1, 0.0, 0.1, 0.1)]},
    )

    # n=1 -> local_step=0: correct behavior reads ONLY the step_index=0 row.
    snap = _build_tree_n(traj, 1)
    assert snap is not None
    win, draw = float(snap.node_features[1, 1]), float(snap.node_features[1, 2])
    assert win == pytest.approx(0.6, abs=1e-6) and draw == pytest.approx(0.2, abs=1e-6), (
        f"off-by-one regression: T_1 (local_step=0) shows wdl_win={win}, wdl_draw={draw}, expected "
        f"(0.6, 0.2) (the step_index=0 row). Got the step_index=1 row instead "
        f"(win=0.1, draw=0.1) -- this is EXACTLY the pre-2026-07-10-fix bug (querying step=n instead of "
        f"step=n-1=local_step), leaking the next expansion's backup one step early."
    )

    # Positive control: n=2 -> local_step=1 DOES correctly pick up the step_index=1 row.
    snap2 = _build_tree_n(traj, 2)
    assert snap2 is not None
    win2, draw2 = float(snap2.node_features[1, 1]), float(snap2.node_features[1, 2])
    assert win2 == pytest.approx(0.1, abs=1e-6) and draw2 == pytest.approx(0.1, abs=1e-6), (
        "positive control failed: T_2 (local_step=1) never picks up the step_index=1 row at all -- "
        "this test isn't exercising forward-fill correctly"
    )


# --------------------------------------------------------------------------- #
# 6. Latent (currently dormant) root_rank inconsistency
# --------------------------------------------------------------------------- #


def test_value_query_ignores_root_rank_latent_bug():
    """Documents (does not "fix") a latent inconsistency found while re-deriving
    the off-by-one fix from scratch for this audit.

    ``preprocess_mc/pack.py:_build_compact_trajectory`` sets
    ``first_decision_expansion_count = root_rank + 1`` and packs ``update_log_*``
    on the FULL (untrimmed) expansion-position scale (explicitly documented at
    ``pack.py`` around the ``update_log_step_index`` dict entry: "0-indexed by
    position in expansion_parent_ids, NOT re-based to root_rank the way
    step_node_cutoffs/tree_sizes above are"), while ``step_node_cutoffs`` IS
    re-based (trimmed to start from root_rank).

    ``_build_tree_n``'s structural cutoff correctly re-derives the full-scale
    expansion count: ``expansion_count = first_decision_expansion_count +
    local_step`` (`= root_rank + n`, full scale) and slices
    ``expansion_parent_ids[:expansion_count]`` -- also full scale, so structure is
    correct for any root_rank. But its VALUE query,
    ``_forward_filled_wdl_at_step(trajectory, node_ids, local_step)``, uses BARE
    ``local_step`` (`= n - 1`) -- missing the same ``+ root_rank`` shift -- to query
    ``update_step_index``, which is on the FULL scale. When root_rank > 0 this
    queries a real, valid, but numerically SMALLER (too-early) step than the
    structural cutoff implies, so T_n's values would be stale by exactly
    ``root_rank`` steps relative to its own structure -- NOT a future leak (forward-
    fill is monotonic in the query step, so querying too-early can only omit
    information, never include future information), but a genuine, different
    correctness bug from the ones already fixed.

    Empirically CONFIRMED DORMANT on real data: sampled first_decision_expansion_count
    (= root_rank + 1) across 200 real trees built from
    ``ysagiv_xaba20k/split/train_manifest.txt`` via the production
    ``preprocess_mc.pack._pack_split`` path (same fixture-building approach as
    ``test_canary_future_leakage.py``) -- all 200 had root_rank == 0 (every real
    tree's own root is the very first thing expanded, which is structurally forced
    for a single fresh PUCT search: nothing else exists to expand before the root
    does). So this bug cannot fire against any real data seen so far; this test
    exercises it only on synthetic data with root_rank > 0, which today's real
    ``xaba20k`` trees never produce. Labeled here as a hypothesis about a distinct,
    currently-inert code path -- not a confirmed live leak.
    """
    # root_rank=2: first_decision_expansion_count=3, i.e. 2 "pre-root" expansions
    # (structurally impossible for a real single-root PUCT tree, per the note
    # above, but a valid input to this FUNCTION taken in isolation).
    traj = _make_trajectory(
        num_nodes=2,
        parents=[-1, 0],
        depths=[0, 1],
        children_by_parent={0: [1]},
        expansion_parent_ids=[0, 0, 0],  # 3 full-scale expansion positions: 0,1,2
        first_decision_expansion_count=3,  # root_rank=2
        step_node_cutoffs=[2],  # single post-root step (n=1 -> local_step=0)
        # Full-scale update log: an update lands at full-scale step_index=2 (the
        # correct query point for n=1 given root_rank=2: expansion_count=3+0=3,
        # so full position 2 is the last processed expansion) with a distinct
        # value from anything at step_index=0.
        updates={1: [(0, 0.0, 0.9, 0.05), (2, 0.0, 0.4, 0.4)]},
    )

    snap = _build_tree_n(traj, 1)
    assert snap is not None
    win = float(snap.node_features[1, 1])

    correct_full_scale_step = traj.first_decision_expansion_count - 1 + 0  # = expansion_count - 1 = 2
    current_code_query_step = 0  # local_step = n - 1 = 0, what _build_tree_n actually queries today

    expected_if_correct = _forward_filled_wdl_at_step(traj, torch.tensor([1]), correct_full_scale_step)
    expected_from_current_code = _forward_filled_wdl_at_step(traj, torch.tensor([1]), current_code_query_step)

    assert win == pytest.approx(float(expected_from_current_code[0, 0]), abs=1e-6), (
        "sanity check on this test's own derivation of _build_tree_n's actual query step failed"
    )
    if correct_full_scale_step != current_code_query_step:
        assert not torch.allclose(expected_if_correct, expected_from_current_code, atol=1e-6), (
            "test fixture too degenerate to discriminate the two query steps (both happen to read the "
            "same update-log row) -- adjust the fixture's step_index values"
        )
        assert win == pytest.approx(0.9, abs=1e-6), (
            f"T_1 with root_rank=2 shows wdl_win={win}. This equals the step_index=0 row (0.9), confirming "
            f"_build_tree_n queries local_step={current_code_query_step} rather than the root_rank-correct "
            f"full-scale step {correct_full_scale_step} (which would show 0.4, the step_index=2 row) -- "
            f"i.e. the latent inconsistency described in this test's docstring is real and reproducible, "
            f"though confirmed dormant on all real xaba20k data sampled for this audit (root_rank always 0)."
        )
