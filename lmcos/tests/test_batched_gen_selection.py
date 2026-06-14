"""T-sel / T-prior / T-trace: batched search internals match the legacy helpers.

Report §4 IDs covered:
  - T-sel  : batched PUCT leaf selection picks the SAME leaf as the legacy
             ``_select_leaf_by_puct`` given a fixed synthetic tree + fixed edge
             stats, including tie-break (earliest child on equal score) and
             canonical child ordering.
  - T-prior: prior normalization + budget truncation + child ordering identical
             to the legacy ``normalize_prior_scores`` / ``_prepare_children``.
  - T-trace: per-step oracle trace (expansion counts / root-q rows / best-move /
             root-move columns) recorded at identical step indices and aligned,
             versus the legacy generation path.

These exercise the search-logic layer (L1). They use a deterministic
``MockEvaluator`` and small synthetic trees / FENs; no engine/GPU/network.
"""

from __future__ import annotations

import random

import pytest

from cts.core.tree import ExpansionChild, SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    EdgeStats,
    _prepare_children,
    _select_leaf_by_puct,
    normalize_prior_scores,
)

from cts.data.batched_gen.evaluator import MockEvaluator
from cts.data.batched_gen.search import generate_trees_batched

from test_batched_gen_common import (
    SMALL_FEN_LIST,
    legacy_example,
    make_budget,
    make_config,
)


def _node_features(value: float, prior: float, w: float, d: float, ll: float) -> dict:
    total = w + d + ll
    return {
        "value": value,
        "prior": prior,
        "wdl_win": w / total,
        "wdl_draw": d / total,
        "wdl_loss": ll / total,
    }


def _build_two_level_tree() -> SearchTree:
    """A small fixed tree: root -> {a, b, c}, with a -> {a1, a2}.

    Child UCI strings are chosen so insertion order already equals UCI-sorted
    order, which is what the canonical pipeline assumes.
    """
    tree = SearchTree(root_fen="root", root_scalar_features=_node_features(0.0, 1.0, 1, 1, 1))
    tree.add_children(
        0,
        [
            ExpansionChild("a2a3", "fa", _node_features(0.1, 0.5, 6, 2, 2)),
            ExpansionChild("b2b3", "fb", _node_features(-0.2, 0.3, 3, 4, 3)),
            ExpansionChild("c2c3", "fc", _node_features(0.05, 0.2, 5, 2, 3)),
        ],
    )
    tree.add_children(
        1,
        [
            ExpansionChild("a3a4", "faa", _node_features(0.2, 0.6, 7, 2, 1)),
            ExpansionChild("a7a6", "fab", _node_features(-0.1, 0.4, 3, 3, 4)),
        ],
    )
    return tree


# --------------------------------------------------------------------------- #
# T-sel
# --------------------------------------------------------------------------- #
def _batched_select(tree, edge_stats, config):
    """Resolve the batched selection routine, whatever the impl named it.

    The contract specifies batched selection must match ``_select_leaf_by_puct``;
    the public entry is ``generate_trees_batched`` but a low-level selection
    helper is expected for the unit-level T-sel test. We try the conventional
    names and skip cleanly if none is exported (the parity is then still
    covered end-to-end by T-replay).
    """
    import cts.data.batched_gen.search as search_mod

    for name in ("select_leaf_by_puct", "_select_leaf_by_puct", "batched_select_leaf"):
        fn = getattr(search_mod, name, None)
        if fn is not None:
            return fn
    pytest.skip(
        "No low-level batched selection helper exported by cts.data.batched_gen.search; "
        "selection parity is covered end-to-end by T-replay."
    )


def test_selection_matches_legacy_empty_stats() -> None:
    """T-sel: with zero stats everywhere, batched selection == legacy selection.

    With all-zero visits the PUCT score reduces to ``c_puct * prior`` so the
    highest-prior child wins; ties must break to the earliest child.
    """
    tree = _build_two_level_tree()
    config = make_config()
    edge_stats = {
        (0, 1): EdgeStats(), (0, 2): EdgeStats(), (0, 3): EdgeStats(),
        (1, 4): EdgeStats(), (1, 5): EdgeStats(),
    }
    select = _batched_select(tree, edge_stats, config)
    leaf, path = select(tree, edge_stats, config)
    exp_leaf, exp_path = _select_leaf_by_puct(tree, edge_stats, config)
    assert leaf == exp_leaf
    assert list(path) == list(exp_path)


def test_selection_matches_legacy_with_visits() -> None:
    """T-sel: with mixed visit counts, batched selection == legacy selection."""
    tree = _build_two_level_tree()
    config = make_config()
    edge_stats = {
        (0, 1): EdgeStats(visit_count=3, total_value=0.9),
        (0, 2): EdgeStats(visit_count=1, total_value=-0.2),
        (0, 3): EdgeStats(visit_count=2, total_value=0.3),
        (1, 4): EdgeStats(visit_count=1, total_value=0.2),
        (1, 5): EdgeStats(),
    }
    select = _batched_select(tree, edge_stats, config)
    leaf, path = select(tree, edge_stats, config)
    exp_leaf, exp_path = _select_leaf_by_puct(tree, edge_stats, config)
    assert leaf == exp_leaf
    assert list(path) == list(exp_path)


def test_selection_tie_break_earliest_child() -> None:
    """T-sel: equal PUCT scores must select the earliest (lowest-id) child.

    Two root children with identical priors and identical zero stats produce
    identical scores; ``_select_leaf_by_puct`` keeps strict ``>`` so the first
    child wins. The batched routine must reproduce that exactly.
    """
    tree = SearchTree(root_fen="root", root_scalar_features=_node_features(0.0, 1.0, 1, 1, 1))
    tree.add_children(
        0,
        [
            ExpansionChild("a2a3", "fa", _node_features(0.0, 0.5, 1, 1, 1)),
            ExpansionChild("b2b3", "fb", _node_features(0.0, 0.5, 1, 1, 1)),
        ],
    )
    config = make_config()
    edge_stats = {(0, 1): EdgeStats(), (0, 2): EdgeStats()}
    select = _batched_select(tree, edge_stats, config)
    leaf, path = select(tree, edge_stats, config)
    exp_leaf, exp_path = _select_leaf_by_puct(tree, edge_stats, config)
    assert leaf == exp_leaf == 1, "tie must break to the earliest child (id 1)"
    assert list(path) == list(exp_path)


# --------------------------------------------------------------------------- #
# T-prior
# --------------------------------------------------------------------------- #
def _batched_prepare_children(config):
    """Resolve the batched prior-preparation helper, or skip if not exported."""
    import cts.data.batched_gen.search as search_mod
    import cts.data.batched_gen.evaluator as eval_mod

    for mod in (search_mod, eval_mod):
        for name in ("prepare_children", "_prepare_children", "normalize_children_priors"):
            fn = getattr(mod, name, None)
            if fn is not None:
                return fn
    pytest.skip(
        "No batched prior-preparation helper exported; prior parity is covered "
        "end-to-end by T-replay (node features include normalized priors)."
    )


def test_normalize_prior_scores_used_consistently() -> None:
    """T-prior: the legacy ``normalize_prior_scores`` is the canonical contract.

    Documents the exact normalization the batched loop must reproduce: a
    non-negative vector is rescaled by its sum (not softmaxed).
    """
    raw = [3.0, 1.0, 0.0, 6.0]
    out = normalize_prior_scores(raw)
    assert out == pytest.approx([0.3, 0.1, 0.0, 0.6])
    assert sum(out) == pytest.approx(1.0)


def test_prepare_children_priors_and_ordering() -> None:
    """T-prior: batched child prep matches legacy ``_prepare_children`` exactly.

    Same input children -> identical move order and identical normalized priors.
    """
    config = make_config()
    children = [
        ExpansionChild("a2a3", "fa", {"value": 0.1, "prior": 4.0}),
        ExpansionChild("b2b3", "fb", {"value": 0.2, "prior": 1.0}),
        ExpansionChild("c2c3", "fc", {"value": 0.3, "prior": 3.0}),
    ]
    expected = _prepare_children(children, prior_feature=config.prior_feature)

    prepare = _batched_prepare_children(config)
    # The batched helper signature is expected to mirror the legacy one.
    got = prepare(children, prior_feature=config.prior_feature)

    assert [c.move_uci for c in got] == [c.move_uci for c in expected]
    for g, e in zip(got, expected):
        assert g.scalar_features[config.prior_feature] == pytest.approx(
            e.scalar_features[config.prior_feature]
        )
        assert g.fen == e.fen


# --------------------------------------------------------------------------- #
# T-trace
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fen", SMALL_FEN_LIST[:2])
def test_oracle_trace_aligned_with_legacy(fen: str) -> None:
    """T-trace: oracle trace arrays line up step-for-step with the legacy path.

    The per-step oracle trace cannot be reconstructed from the final tree
    (§1.1) so the batched loop must record it online. We assert the batched
    trace equals the legacy trace at identical step indices and that all
    columns stay aligned with ``oracle_root_moves``.
    """
    mock = MockEvaluator()
    config = make_config()
    budget = make_budget()
    seed = 555
    root_id = f"trace::{fen}"

    expected = legacy_example(fen, mock, config, budget, seed=seed, root_position_id=root_id)
    got = generate_trees_batched(
        [fen], mock, config, budget, rng=random.Random(seed), root_position_ids=[root_id]
    )[0]

    assert got.oracle_trace_expansion_counts == expected.oracle_trace_expansion_counts
    assert got.oracle_root_moves == expected.oracle_root_moves
    assert got.oracle_best_move_trace == expected.oracle_best_move_trace
    assert len(got.oracle_root_q_trace) == len(expected.oracle_root_q_trace)
    for rg, re_ in zip(got.oracle_root_q_trace, expected.oracle_root_q_trace):
        assert len(rg) == len(got.oracle_root_moves)
        assert rg == pytest.approx(re_, abs=0.0, rel=0.0)

    # Internal alignment invariants the PretrainExample post-init also enforces.
    counts = got.oracle_trace_expansion_counts
    assert all(b > a for a, b in zip(counts, counts[1:])), "counts strictly increasing"
    assert all(m in got.oracle_root_moves for m in got.oracle_best_move_trace)
