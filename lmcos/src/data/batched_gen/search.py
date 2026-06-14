"""Frontier-batched, eval-agnostic tree generation (report §1.3, Phase 1).

``generate_trees_batched`` holds up to ``max_concurrent`` independent trees in
flight and advances them in lockstep: every outer step runs each tree's PUCT
descent to the one leaf it wants next, gathers *all* trees' required position
evaluations into a single ``evaluator.evaluate([...])`` call, scatters the
results back, and advances each tree exactly one expansion (expand · backprop ·
record-trace). Each tree advances in strict sequential order — we never batch
*within* a tree — so every trajectory is identical to a purely sequential run
(report §1.4: batching changes only the grouping of independent evaluations,
never their values or ordering).

Parity by construction (the L1 contract, report §2): the only new code here is
the lockstep driver. Selection, child preparation, prior normalization,
backprop, oracle-trace recording, target consolidation, value-gap and
policy-drift derivation are all the *same* functions the legacy sequential path
calls, imported from ``teacher_targets``. A per-tree ``_EvaluatorBackedProvider``
shapes the evaluator's numbers into feature dicts byte-identically to the
legacy provider, and the per-tree budget RNG is reseeded from the input ``rng``
in input order, reproducing the legacy seeding. Hence, for a single FEN, the
output is byte-identical to::

    build_pretrain_example(fen, evaluator_to_provider(evaluator),
                           search_config, node_budget_distribution,
                           rng, root_position_id)

which is the acceptance criterion a sibling test asserts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cts.core.tree import SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    EdgeStats,
    GeneratedTree,
    NodeBudgetDistribution,
    PretrainExample,
    TeacherSearchConfig,
    _backpropagate_path,
    _has_expandable_frontier,
    _maybe_static_node_wdl,
    _prepare_children,
    _root_q_values_from_edge_stats,
    _select_leaf_by_puct,
    _static_node_value,
    consolidate_generated_tree,
    node_value_gap_features,
)

from .evaluator import Evaluator, evaluator_to_provider


@dataclass
class _TreeInFlight:
    """Mutable per-tree state for one in-flight search.

    Mirrors, field for field, the local state of one call to
    ``generate_partial_tree_from_provider``; the batched driver simply steps
    many of these in lockstep instead of running them one at a time. Each tree
    owns its own evaluator-backed provider so feature dicts are produced by the
    exact legacy code path.
    """

    fen: str
    root_position_id: Optional[str]
    provider: object  # _EvaluatorBackedProvider; legacy TreeExpansionProvider surface
    sampled_node_budget: int
    tree: SearchTree
    edge_stats: Dict[Tuple[int, int], EdgeStats] = field(default_factory=dict)
    num_expansions: int = 0
    simulations: int = 0
    finished: bool = False
    oracle_trace_expansion_counts: List[int] = field(default_factory=list)
    oracle_root_moves: List[str] = field(default_factory=list)
    oracle_root_q_trace: List[List[float]] = field(default_factory=list)
    oracle_best_move_trace: List[str] = field(default_factory=list)
    oracle_root_visits_trace: List[List[int]] = field(default_factory=list)

    @property
    def max_simulations(self) -> int:
        # Same simulation cap as the legacy loop, guarding against a frontier
        # that never exhausts (e.g. depth-capped trees re-selecting leaves).
        return self.sampled_node_budget * 50

    def record_oracle_root_trace(self) -> None:
        """Snapshot the root's per-move Q/visits after an expansion.

        Byte-for-byte the same logic as the nested ``_record_oracle_root_trace``
        in ``generate_partial_tree_from_provider``: freeze the canonical move
        order on the first call, then append a Q row, the argmax move, and the
        visit row each subsequent step.
        """
        if self.tree.root_id is None:
            return
        root_children = self.tree.root_children()
        if not root_children:
            return
        if not self.oracle_root_moves:
            self.oracle_root_moves = []
            for child_id in root_children:
                move_uci = self.tree.get_node(child_id).incoming_move_uci
                if move_uci is None:
                    raise ValueError(f"Root child {child_id} is missing an incoming move.")
                self.oracle_root_moves.append(str(move_uci))
        root_q_values = _root_q_values_from_edge_stats(self.tree, self.edge_stats)
        row = [float(root_q_values[move]) for move in self.oracle_root_moves]
        best_move = self.oracle_root_moves[max(range(len(row)), key=row.__getitem__)]
        self.oracle_trace_expansion_counts.append(self.num_expansions)
        self.oracle_root_q_trace.append(row)
        self.oracle_best_move_trace.append(best_move)
        visits_row = [
            int(self.edge_stats.get((self.tree.root_id, child_id), EdgeStats()).visit_count)
            for child_id in root_children
        ]
        self.oracle_root_visits_trace.append(visits_row)

    def to_generated_tree(self) -> GeneratedTree:
        """Package the finished state into the legacy ``GeneratedTree`` carrier."""
        return GeneratedTree(
            tree=self.tree,
            edge_stats=self.edge_stats,
            sampled_node_budget=self.sampled_node_budget,
            num_expansions=self.num_expansions,
            oracle_trace_expansion_counts=self.oracle_trace_expansion_counts,
            oracle_root_moves=self.oracle_root_moves,
            oracle_root_q_trace=self.oracle_root_q_trace,
            oracle_best_move_trace=self.oracle_best_move_trace,
            oracle_root_visits_trace=self.oracle_root_visits_trace,
        )


def generate_trees_batched(
    fens: Sequence[str],
    evaluator: Evaluator,
    search_config: TeacherSearchConfig,
    node_budget_distribution: NodeBudgetDistribution,
    *,
    rng: random.Random,
    max_concurrent: int = 1024,
    root_position_ids: Optional[Sequence[str]] = None,
) -> List[PretrainExample]:
    """Generate one ``PretrainExample`` per FEN via the frontier-batched loop.

    Args:
        fens: root FENs; one example is produced per FEN, in input order.
        evaluator: batched neural evaluator (mock / cached / lc0-backed).
        search_config: PUCT/depth/budget settings (shared with the legacy path).
        node_budget_distribution: per-tree expansion-budget sampler.
        rng: RNG for budget sampling. Reseeded per tree in input order so a
            given seed yields the same budgets — hence the same trees — as the
            legacy sequential path.
        max_concurrent: cap on trees held in flight at once (continuous
            batching tops the pool back up as trees finish). Does not affect
            any tree's result, only memory and batch size.
        root_position_ids: optional per-FEN human-readable ids (defaults to the
            FEN), threaded into example metadata exactly as the legacy path.

    Returns:
        ``PretrainExample`` list aligned with ``fens``.
    """
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive.")
    if root_position_ids is not None and len(root_position_ids) != len(fens):
        raise ValueError("root_position_ids must align with fens.")

    # --- Reproduce the legacy per-tree budget seeding ---
    # The sequential path samples one budget per tree off the shared ``rng`` in
    # input order. We draw the budgets here in the same order so the sampled
    # budget for tree i is identical regardless of batching.
    per_tree_budgets = [node_budget_distribution.sample(rng) for _ in fens]

    pending = list(range(len(fens)))  # indices not yet admitted into the pool
    examples: List[Optional[PretrainExample]] = [None] * len(fens)
    in_flight: List[_TreeInFlight] = []

    def _admit(slot_index: int) -> _TreeInFlight:
        fen = fens[slot_index]
        provider = evaluator_to_provider(evaluator)
        return _TreeInFlight(
            fen=fen,
            root_position_id=(root_position_ids[slot_index] if root_position_ids is not None else None),
            provider=provider,
            sampled_node_budget=per_tree_budgets[slot_index],
            tree=SearchTree(
                root_fen=fen,
                root_scalar_features=provider.root_features(fen),
                root_metadata=provider.root_metadata(fen),
            ),
        )

    # Track which output slot each in-flight tree fills, so a finished tree
    # writes back to the right position and a fresh tree can take its place.
    slot_of: Dict[int, int] = {}

    def _fill_pool() -> None:
        while len(in_flight) < max_concurrent and pending:
            slot_index = pending.pop(0)
            tree_state = _admit(slot_index)
            slot_of[id(tree_state)] = slot_index
            in_flight.append(tree_state)

    _fill_pool()

    while in_flight:
        _run_one_lockstep_step(in_flight, evaluator, search_config)

        # Harvest finished trees, write their examples back to the right slot,
        # then refill the pool (continuous batching).
        still_running: List[_TreeInFlight] = []
        for tree_state in in_flight:
            if tree_state.finished:
                slot_index = slot_of.pop(id(tree_state))
                examples[slot_index] = _assemble_example(tree_state, search_config)
            else:
                still_running.append(tree_state)
        in_flight[:] = still_running
        _fill_pool()

    assert all(example is not None for example in examples)
    return [example for example in examples if example is not None]


def _run_one_lockstep_step(
    in_flight: Sequence[_TreeInFlight],
    evaluator: Evaluator,
    search_config: TeacherSearchConfig,
) -> None:
    """Advance every in-flight tree by exactly one expansion, batching the eval.

    Two passes per outer step:

    1. **Select** (per tree, no eval): run each tree's PUCT descent to its
       chosen leaf. Gather every FEN whose evaluation that leaf's expansion
       will require — the leaf's own children — into one set.
    2. **One forward pass**: ``evaluator.evaluate([...])`` over the gathered
       FENs, warming the shared evaluator's cache. Each tree's provider then
       reads its numbers from that cache, so the actual expand/backprop/record
       for each tree consumes exactly the batched results.

    Because each tree's provider memoizes per-FEN, the expansion in pass 2 does
    not re-hit the accelerator; the single ``evaluate`` call here is the whole
    batch for this step.
    """
    # --- Pass 1: selection + gather the FENs this step will evaluate ---
    selections: List[Optional[Tuple[int, List[Tuple[int, int]]]]] = []
    per_tree_child_fens: List[List[str]] = []
    fens_to_eval: List[str] = []
    seen_fens = set()
    for tree_state in in_flight:
        selection = _select_for_step(tree_state, search_config)
        selections.append(selection)
        if selection is None:
            per_tree_child_fens.append([])
            continue
        leaf_id, _path = selection
        leaf = tree_state.tree.get_node(leaf_id)
        # A leaf that will actually expand needs its children evaluated; a
        # depth-capped or terminal leaf does not. The provider enumerates the
        # exact non-terminal child FENs expansion will request, *without*
        # itself calling the evaluator (it reuses the cached parent eval).
        if leaf.is_terminal or leaf.depth >= search_config.max_depth:
            per_tree_child_fens.append([])
            continue
        child_fens = tree_state.provider.nonterminal_child_fens(leaf.fen)
        per_tree_child_fens.append(child_fens)
        for child_fen in child_fens:
            if child_fen not in seen_fens:
                seen_fens.add(child_fen)
                fens_to_eval.append(child_fen)

    # --- One forward pass over the whole step's batch ---
    # Every position any tree needs this step is scored in a single call; the
    # results are then scattered into each tree's provider cache so pass 2 does
    # no further evaluation (report §1.3, step 3).
    if fens_to_eval:
        batched_evals = evaluator.evaluate(fens_to_eval)
        eval_by_fen = dict(zip(fens_to_eval, batched_evals))
        for tree_state, child_fens in zip(in_flight, per_tree_child_fens):
            for child_fen in child_fens:
                tree_state.provider.seed_eval_cache(child_fen, eval_by_fen[child_fen])

    # --- Pass 2: expand · backprop · record, per tree, reading the batch ---
    for tree_state, selection in zip(in_flight, selections):
        if selection is None:
            tree_state.finished = True
            continue
        _advance_one_expansion(tree_state, selection, search_config)


def _select_for_step(
    tree_state: _TreeInFlight,
    search_config: TeacherSearchConfig,
) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    """Run one tree's stop-check + PUCT selection; None means the tree is done.

    Mirrors the loop guard and ``_select_leaf_by_puct`` call at the top of the
    legacy ``while`` body, including the simulation-cap break.
    """
    if tree_state.num_expansions >= tree_state.sampled_node_budget:
        return None
    if not _has_expandable_frontier(tree_state.tree, search_config):
        return None
    tree_state.simulations += 1
    if tree_state.simulations > tree_state.max_simulations:
        return None
    return _select_leaf_by_puct(tree_state.tree, tree_state.edge_stats, search_config)


def _advance_one_expansion(
    tree_state: _TreeInFlight,
    selection: Tuple[int, List[Tuple[int, int]]],
    search_config: TeacherSearchConfig,
) -> None:
    """Expand one leaf (or terminate it) and backprop — the legacy step body.

    This is a transcription of one iteration of the ``while`` loop inside
    ``generate_partial_tree_from_provider``: same terminal handling, same
    ``_prepare_children`` normalization, same edge-stat initialization, same
    backprop, same oracle-trace recording on a real expansion.
    """
    node_id, path = selection
    tree = tree_state.tree
    node = tree.get_node(node_id)
    leaf_value = _static_node_value(tree, node_id, search_config.value_feature)
    leaf_wdl = _maybe_static_node_wdl(tree, node_id)

    if node.is_terminal or node.depth >= search_config.max_depth:
        node.is_terminal = True
        _backpropagate_path(tree_state.edge_stats, path, leaf_value, leaf_wdl)
        return

    raw_children = tree_state.provider.expand_node(node.fen, node.depth)
    children = _prepare_children(raw_children, prior_feature=search_config.prior_feature)
    if not children:
        node.is_terminal = True
        _backpropagate_path(tree_state.edge_stats, path, leaf_value, leaf_wdl)
        return

    child_ids = tree.add_children(node_id, children)
    for child_id in child_ids:
        tree_state.edge_stats[(node_id, child_id)] = EdgeStats()
    tree_state.num_expansions += 1
    _backpropagate_path(tree_state.edge_stats, path, leaf_value, leaf_wdl)
    tree_state.record_oracle_root_trace()


def _assemble_example(
    tree_state: _TreeInFlight,
    search_config: TeacherSearchConfig,
) -> PretrainExample:
    """Turn a finished in-flight tree into a ``PretrainExample``.

    Transcribes the budget-driven branch and post-processing of
    ``build_pretrain_example``: ``consolidate_generated_tree`` for node/edge
    targets, the same metadata stamps, the same oracle-trace export, the same
    value-gap and root policy-drift derivation. Reuses the legacy helpers so
    the assembled example matches the sequential path field for field.
    """
    generated_tree = tree_state.to_generated_tree()
    tree = generated_tree.tree
    teacher_result = consolidate_generated_tree(generated_tree, search_config)

    metadata = {
        "root_position_id": tree_state.root_position_id or tree_state.fen,
        "search_config_id": search_config.search_config_id,
        "provider_metadata": dict(tree_state.provider.provider_metadata()),
        "target_generation_version": search_config.target_normalization_version,
    }

    oracle_trace_expansion_counts = list(generated_tree.oracle_trace_expansion_counts)
    oracle_root_moves = list(generated_tree.oracle_root_moves)
    oracle_root_q_trace = [list(row) for row in generated_tree.oracle_root_q_trace]
    oracle_best_move_trace = list(generated_tree.oracle_best_move_trace)
    oracle_final_root_q_values = _root_q_values_from_edge_stats(tree, generated_tree.edge_stats)
    metadata["oracle_trace_generation_version"] = "generated_search_root_q_v1"

    value_gap = node_value_gap_features(
        tree, teacher_result.edge_stats, value_feature=search_config.value_feature
    )

    policy_drift = _root_policy_drift(generated_tree, tree)

    return PretrainExample(
        tree=tree,
        node_target_values=teacher_result.node_target_values,
        edge_wdl_targets={},
        metadata=metadata,
        oracle_trace_expansion_counts=oracle_trace_expansion_counts,
        oracle_root_moves=oracle_root_moves,
        oracle_root_q_trace=oracle_root_q_trace,
        oracle_best_move_trace=oracle_best_move_trace,
        oracle_final_root_q_values=oracle_final_root_q_values,
        value_gap=value_gap,
        policy_drift=policy_drift,
    )


def _root_policy_drift(generated_tree: GeneratedTree, tree: SearchTree) -> List[float]:
    """Root policy-drift vector, transcribed from ``build_pretrain_example``.

    KL between the earliest root-visit distribution with >= ``n_min`` total
    visits and the final root-visit distribution. NaN for every node except
    the root (and the root too, when the early threshold is never reached).
    """
    policy_drift = [float("nan")] * tree.num_nodes()
    if tree.root_id is None or not generated_tree.oracle_root_visits_trace:
        return policy_drift

    n_min = 10
    early_visits = None
    for visits_row in generated_tree.oracle_root_visits_trace:
        if sum(visits_row) >= n_min:
            early_visits = visits_row
            break
    if early_visits is None:
        return policy_drift

    late_visits = [
        int(generated_tree.edge_stats.get((tree.root_id, child_id), EdgeStats()).visit_count)
        for child_id in tree.root_children()
    ]
    from cts.data.preprocess_gnn.policy_drift import compute_policy_drift

    drift_m = compute_policy_drift(
        node_id=tree.root_id,
        early_visits=early_visits,
        late_visits=late_visits,
        n_min=n_min,
    )
    if drift_m is not None:
        policy_drift[tree.root_id] = drift_m.kl_divergence
    return policy_drift


__all__ = ["generate_trees_batched"]
