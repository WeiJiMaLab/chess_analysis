"""Standalone, retraining-free replay of the PUCT teacher-search backup rule
(see ``cts.data.preprocess_gnn.teacher_targets.PUCTSearch``/``_backpropagate_path``),
so that a saved tree's node VALUES can be relabeled (e.g. temperature-scaled
``tanh(cp_order / T)`` instead of the WDL-derived ``value`` column) and re-backed-up
WITHOUT re-running Stockfish or PUCT search itself.

Context (saturation hypothesis, see ``saturation_hypothesis.md`` at repo root): the
shipped ``value`` feature is Stockfish WDL win_prob - loss_prob, which saturates to
|value| >= 0.99 for ~56% of nodes because the WDL curve is essentially a step
function around ~300cp. This module lets us ask "what would the halt-reward
trajectory have looked like if node values were desaturated cp-scores instead" --
holding tree SHAPE (topology, expansion order) exactly fixed, since PUCT's actual
expansion decisions during generation were made under the OLD, saturating value.
This is a first-order approximation, not a full re-generation: a fully correct
experiment would re-run PUCT search under the new value function, which changes
SELECTION (which nodes get expanded at all), not just labels. That is out of scope
here (see task description / ``saturation_hypothesis.md``).

## The backup rule being replayed

``PUCTSearch.on_expand``/``on_terminal_leaf`` call ``_backpropagate_path``, which
walks from a just-visited node up to the root, incrementing each edge's
``visit_count`` by 1 and ``total_value`` by the node's OWN static value
(sign-flipped once per ply, since side-to-move alternates). This happens exactly
once per node that was ever selected as a leaf during generation -- i.e. every node
with ``is_expanded=True`` (its own value backed up at the moment its children were
added) plus every childless node with ``is_terminal=True`` (backed up when
discovered terminal, no children added). Both facts are directly readable off a
saved ``.pt`` record's ``is_expanded``/``is_terminal``/``child_ptr`` arrays --
no need to re-run the search.

Because backup is a pure running SUM, the FINAL per-edge ``total_value``/
``visit_count`` (hence ``oracle_final_root_q_values``) is exactly order-independent.
The per-STEP trace (``oracle_root_q_trace``/``oracle_best_move_index``), however, IS
order-dependent -- it's read after each real expansion, so we need *a* fully-
determined event order consistent with the tree's causal structure. Real expansions
are ordered by their first child's node id (children get contiguous ids at
creation, so sorting parents by their first child's id recovers the true expansion
order -- this is the exact method ``cts.data.preprocess_mc.pack._ordered_expansion_parent_ids``
uses, validated by production code). Terminal-leaf-only backups have no children to
sort by; they're interleaved into the combined order by their OWN node id (a
node cannot be discovered terminal before it exists). This is an approximation
for the relative order of terminal leaves specifically -- see module docstring
caveat above -- but terminal leaves are rare (~0.1-0.5% of nodes in this corpus)
and, being a pure order dependence, only affects which trace WINDOW their small
contribution first appears in, not the final totals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class ReplayResult:
    """Output of :func:`replay_puct_backup`.

    All arrays are plain numpy. ``root_children`` gives the node ids (in the fixed
    canonical order used for every other array's move axis) of the root's children
    -- i.e. the "moves" the trace/final-Q vectors are indexed by.
    """

    root_children: np.ndarray  # [M] int -- node ids of root's children, canonical move order
    expansion_parent_ids: np.ndarray  # [T] int -- real-expansion node ids, in reconstructed expansion order
    root_q_trace: np.ndarray  # [T, M] float64 -- root's per-move Q after each real expansion
    best_move_index: np.ndarray  # [T] int -- argmax(root_q_trace[t]) per step (first-max tie-break)
    final_root_q: np.ndarray  # [M] float64 -- root's per-move Q after ALL backups (real + terminal)


def ordered_expansion_parent_ids(
    parent_index: np.ndarray,
    child_ptr: np.ndarray,
    children_index: np.ndarray,
    is_expanded: np.ndarray,
) -> List[int]:
    """Reconstruct expansion order from CSR child arrays.

    Byte-for-byte the same logic as
    ``cts.data.preprocess_mc.pack._ordered_expansion_parent_ids``: children get
    contiguous ids at creation, so sorting parents-that-got-children by their
    FIRST child's id recovers the original expansion sequence.
    """
    num_nodes = int(parent_index.shape[0])
    pairs = []
    for node_id in range(num_nodes):
        start, end = int(child_ptr[node_id]), int(child_ptr[node_id + 1])
        if bool(is_expanded[node_id]) and end > start:
            pairs.append((int(children_index[start]), node_id))
    pairs.sort()
    return [node_id for _, node_id in pairs]


def replay_puct_backup(
    parent_index: np.ndarray,
    child_ptr: np.ndarray,
    children_index: np.ndarray,
    is_expanded: np.ndarray,
    is_terminal: np.ndarray,
    node_values: np.ndarray,
) -> ReplayResult:
    """Replay PUCT's accumulate-along-root-path backup with an arbitrary per-node
    value array, holding tree topology (and hence expansion order) fixed.

    Args:
        parent_index: [N] int, -1 for the root.
        child_ptr: [N+1] int CSR pointer.
        children_index: [num_edges] int, flat child-id list in slot order.
        is_expanded: [N] bool -- node got real children during generation.
        is_terminal: [N] bool -- node was flagged terminal (depth cap / real
            terminal / no children from provider).
        node_values: [N] float -- the value to back up for THIS node (its own
            static value, whatever value function is being tested).
    """
    num_nodes = int(parent_index.shape[0])
    parent_index = np.asarray(parent_index, dtype=np.int64)
    child_ptr = np.asarray(child_ptr, dtype=np.int64)
    children_index = np.asarray(children_index, dtype=np.int64)
    is_expanded = np.asarray(is_expanded, dtype=bool)
    is_terminal = np.asarray(is_terminal, dtype=bool)
    node_values = np.asarray(node_values, dtype=np.float64)

    root_candidates = np.where(parent_index < 0)[0]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one root (parent_index == -1), found {len(root_candidates)}.")
    root_id = int(root_candidates[0])

    child_counts = child_ptr[1:] - child_ptr[:-1]
    root_children = children_index[child_ptr[root_id]:child_ptr[root_id + 1]].copy()
    if len(root_children) == 0:
        raise ValueError("Root has no children -- cannot compute root Q values.")

    expansion_parent_ids = ordered_expansion_parent_ids(parent_index, child_ptr, children_index, is_expanded)
    expansion_set = set(expansion_parent_ids)

    terminal_leaf_ids = [
        int(node_id) for node_id in range(num_nodes)
        if bool(is_terminal[node_id]) and int(child_counts[node_id]) == 0 and node_id not in expansion_set
    ]

    # Combined causal order: real expansions keyed by their first child's id
    # (validated method, see docstring); terminal leaves keyed by their own id
    # (an approximation -- see module docstring).
    first_child_of = {}
    for node_id in expansion_parent_ids:
        first_child_of[node_id] = int(children_index[child_ptr[node_id]])
    events = [(first_child_of[node_id], node_id, True) for node_id in expansion_parent_ids]
    events += [(node_id, node_id, False) for node_id in terminal_leaf_ids]
    events.sort(key=lambda e: e[0])

    edge_total_value = np.zeros(num_nodes, dtype=np.float64)
    edge_visit_count = np.zeros(num_nodes, dtype=np.int64)

    def _root_q_now() -> np.ndarray:
        out = np.empty(len(root_children), dtype=np.float64)
        for i, child_id in enumerate(root_children):
            vc = edge_visit_count[child_id]
            if vc > 0:
                out[i] = edge_total_value[child_id] / vc
            else:
                out[i] = -node_values[child_id]  # FPU seed: child's own static value, sign-flipped
        return out

    trace_rows: List[np.ndarray] = []
    best_moves: List[int] = []

    for _, node_id, is_real_expansion in events:
        value = node_values[node_id]
        cur = node_id
        while cur != root_id:
            value = -value
            edge_total_value[cur] += value
            edge_visit_count[cur] += 1
            cur = int(parent_index[cur])
        if is_real_expansion:
            row = _root_q_now()
            trace_rows.append(row)
            best_moves.append(int(np.argmax(row)))  # first-max tie-break (ties are measure-zero on continuous values)

    final_root_q = _root_q_now()

    return ReplayResult(
        root_children=np.asarray(root_children, dtype=np.int64),
        expansion_parent_ids=np.asarray(expansion_parent_ids, dtype=np.int64),
        root_q_trace=np.asarray(trace_rows, dtype=np.float64) if trace_rows else np.zeros((0, len(root_children))),
        best_move_index=np.asarray(best_moves, dtype=np.int64),
        final_root_q=final_root_q,
    )


def build_trajectory(
    parent_index: np.ndarray,
    child_ptr: np.ndarray,
    children_index: np.ndarray,
    is_expanded: np.ndarray,
    is_terminal: np.ndarray,
    depth: np.ndarray,
    node_values: np.ndarray,
) -> dict | None:
    """Convert a replayed tree into the per-trajectory dict consumed by
    ``analysis.evaluate``'s ``_regret_at``/``_return_curves``/``fit_singlehalt_stop``
    (``halt_rewards``/``tree_sizes``/``heights``/``widths``/``time_budgets``).

    Mirrors ``cts.data.preprocess_mc.pack._build_compact_trajectory``: trims
    everything before the ROOT's own expansion (the controller's step 0), and
    ``halt_rewards[t] = final_root_q[best_move_index[t]]`` -- the FINAL,
    fully-searched quality of whichever move looked best at intermediate step t.

    Returns None if the tree has no expansions or no root expansion (mirrors the
    production function's contract).
    """
    result = replay_puct_backup(parent_index, child_ptr, children_index, is_expanded, is_terminal, node_values)
    if len(result.expansion_parent_ids) == 0:
        return None
    root_id = int(np.where(np.asarray(parent_index, dtype=np.int64) < 0)[0][0])
    try:
        root_rank = int(np.where(result.expansion_parent_ids == root_id)[0][0])
    except IndexError:
        return None

    child_ptr = np.asarray(child_ptr, dtype=np.int64)
    child_counts_by_parent = {
        node_id: int(child_ptr[node_id + 1] - child_ptr[node_id]) for node_id in result.expansion_parent_ids.tolist()
    }
    node_cutoffs = []
    current_nodes = 1
    for node_id in result.expansion_parent_ids.tolist():
        current_nodes += child_counts_by_parent[node_id]
        node_cutoffs.append(current_nodes)
    trimmed_node_cutoffs = node_cutoffs[root_rank:]
    trimmed_best_move_index = result.best_move_index[root_rank:]
    halt_rewards = result.final_root_q[trimmed_best_move_index]

    depth = np.asarray(depth, dtype=np.int64)
    heights, widths = [], []
    for node_cutoff in trimmed_node_cutoffs:
        prefix_depth = depth[:node_cutoff]
        heights.append(int(prefix_depth.max()))
        widths.append(int(np.bincount(prefix_depth).max()))

    num_steps = len(trimmed_node_cutoffs)
    starting_budget = num_steps
    return {
        "num_steps": num_steps,
        "halt_rewards": halt_rewards.tolist(),
        "tree_sizes": [int(v) for v in trimmed_node_cutoffs],
        "heights": heights,
        "widths": widths,
        "time_budgets": list(range(starting_budget, starting_budget - num_steps, -1)),
        "starting_budget": starting_budget,
    }


def tanh_cp_value(cp_order: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled desaturated value: ``tanh(cp_order / temperature)``.

    Equivalent to ``2 * sigmoid(2*cp_order/temperature) - 1``; bounded to [-1, 1]
    like the original WDL-derived ``value`` column, but with a much gentler
    saturation curve controlled by ``temperature``.
    """
    return np.tanh(np.asarray(cp_order, dtype=np.float64) / float(temperature))


def saturation_fraction(values: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of ``values`` with ``|value| >= threshold``."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.abs(values) >= threshold))
