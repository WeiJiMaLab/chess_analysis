"""Shared fixtures/helpers for the BeFS/centipawn provenance test battery.

Everything here goes through the REAL pipeline types (``SearchTree``,
``PretrainExample``, ``RawPretrainExampleRecord``) so a hand-built fixture
payload is bit-compatible with what the committed writer produces — the
tests never invent their own on-disk schema.

Sign/POV convention exercised throughout (established from the code, see
``StockfishDirectEvalProvider`` + ``_prune_children`` + ``_select_leaf_by_befs``):

  * Every per-node scalar (``value``, ``cp``, ``mate``, ``cp_order``, WDL) is
    stored from the SIDE-TO-MOVE POV of that node's own position — i.e. the
    mover AT that node, not the root mover.  Nothing is root-normalized.
  * Consequently a root child that is GOOD for the root mover carries a
    NEGATIVE value/cp_order (the child's mover — the opponent — is losing),
    and the negamax "best child for the parent" is the MIN-value child.
  * ``cp_order`` = ``cp`` for non-mate scores; mate-in-k (moves, mover POV)
    folds to ``sign * (20000 - k)``; a checkmated terminal is ``-20000``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cts.core.providers.base import TreeExpansionProvider
from cts.core.providers.common import append_move_to_position_spec
from cts.core.tree import ExpansionChild, SearchTree
from cts.data.preprocess_gnn.teacher_targets import (
    PretrainExample,
    RawPretrainExampleRecord,
)

# A couple of syntactically complete FENs (>= 4 fields) for fixture roots.
FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_MID = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def feats(
    value: float,
    wdl: Tuple[float, float, float],
    *,
    cp: Optional[float] = None,
    mate: Optional[float] = None,
    cp_order: Optional[float] = None,
    prior: float = 1.0,
) -> Dict[str, float]:
    """Build a node feature dict in the exact key vocabulary of the providers.

    ``cp`` XOR ``mate`` must be provided (that is the parser's invariant);
    ``cp_order`` defaults to ``cp`` when only ``cp`` is given.
    """
    win, draw, loss = wdl
    total = win + draw + loss
    assert total > 0
    p_win, p_draw, p_loss = win / total, draw / total, loss / total
    out: Dict[str, float] = {
        "value": float(value),
        "wdl_win": p_win,
        "wdl_draw": p_draw,
        "wdl_loss": p_loss,
        "wdl_var": (p_win + p_loss) - (p_win - p_loss) ** 2,
        "prior": float(prior),
    }
    assert abs((p_win - p_loss) - value) < 1e-9, "fixture wdl must reproduce value"
    if cp is not None:
        out["cp"] = float(cp)
    if mate is not None:
        out["mate"] = float(mate)
    resolved_order = cp_order if cp_order is not None else cp
    assert resolved_order is not None, "fixture node needs cp or an explicit cp_order"
    out["cp_order"] = float(resolved_order)
    return out


def build_root_children_payload(
    root_fen: str,
    root_feats: Mapping[str, float],
    children: Sequence[Tuple[str, Mapping[str, float], bool]],
    final_q: Mapping[str, float],
) -> Dict[str, Any]:
    """Build an on-disk payload dict via the REAL writer for a 1-ply tree.

    Args:
        children: (move_uci, scalar_features, is_terminal) triples.
        final_q: root-POV oracle_final_root_q_values, keyed by move.
    """
    tree = SearchTree(root_fen=root_fen, root_scalar_features=dict(root_feats))
    tree.add_children(
        0,
        [
            ExpansionChild(
                move_uci=move,
                fen=append_move_to_position_spec(root_fen, move),
                scalar_features=dict(child_feats),
                is_terminal=terminal,
            )
            for move, child_feats, terminal in children
        ],
    )
    moves = [move for move, _, _ in children]
    row = [float(final_q[move]) for move in moves]
    best = moves[int(np.argmax(row))]
    example = PretrainExample(
        tree=tree,
        node_target_values=[0.0] * tree.num_nodes(),
        metadata={"root_position_id": "fixture", "search_config_id": "fixture_befs"},
        oracle_trace_expansion_counts=[1],
        oracle_root_moves=moves,
        oracle_root_q_trace=[row],
        oracle_best_move_trace=[best],
        oracle_final_root_q_values={move: float(final_q[move]) for move in moves},
    )
    return RawPretrainExampleRecord.from_example(example).to_payload()


# ---------------------------------------------------------------------------
# Scripted (engine-free) provider
# ---------------------------------------------------------------------------

class ScriptedProvider(TreeExpansionProvider):
    """Deterministic provider: expansions are a literal fen -> children table.

    Children are given as (move_uci, scalar_features, is_terminal); the child
    fen is derived with the same ``append_move_to_position_spec`` the real
    providers use, so saved trees remain FEN-compact-reconstructable.
    """

    def __init__(
        self,
        root_fen: str,
        root_feats: Mapping[str, float],
        script: Mapping[str, Sequence[Tuple[str, Mapping[str, float], bool]]],
    ) -> None:
        self.root_fen = root_fen
        self._root_feats = dict(root_feats)
        self._script = {fen: list(kids) for fen, kids in script.items()}

    def root_features(self, fen: str) -> Mapping[str, float]:
        assert fen == self.root_fen
        return dict(self._root_feats)

    def expand_node(self, fen, depth, max_children=None):
        if fen not in self._script:
            raise AssertionError(
                f"ScriptedProvider was asked to expand an unscripted node: {fen!r} "
                f"(depth={depth}). The search left the scripted region — this is "
                "itself a selection-order failure."
            )
        kids = self._script[fen]
        return [
            ExpansionChild(
                move_uci=move,
                fen=append_move_to_position_spec(fen, move),
                scalar_features=dict(child_feats),
                is_terminal=terminal,
            )
            for move, child_feats, terminal in kids
        ]

    def provider_metadata(self):
        return {"engine_kind": "scripted"}


def make_line(*specs: Tuple[str, Mapping[str, float]]) -> List[Tuple[str, Mapping[str, float], bool]]:
    """Convenience: non-terminal scripted children from (move, feats) pairs."""
    return [(move, child_feats, False) for move, child_feats in specs]


# ---------------------------------------------------------------------------
# Independent BeFS replay (payload-only; deliberately NOT importing the
# implementation's selector so it is a genuinely independent oracle).
# ---------------------------------------------------------------------------

def true_expansion_order(payload: Mapping[str, Any]) -> List[int]:
    """Ground-truth expansion order from the stored tree.

    Node ids are assigned in insertion order and a parent's children are
    appended as one contiguous block at the moment of its expansion, so
    ordering expanded parents by their smallest child id recovers the
    expansion sequence exactly.
    """
    parent = payload["parent_index"].numpy()
    first_child: Dict[int, int] = {}
    for child_id, parent_id in enumerate(parent.tolist()):
        if parent_id < 0:
            continue
        first_child.setdefault(int(parent_id), int(child_id))
    return [parent_id for _, parent_id in sorted((cid, pid) for pid, cid in first_child.items())]


def replay_befs_expansion_order(payload: Mapping[str, Any], max_depth: int) -> List[int]:
    """Re-run greedy BeFS purely from the stored static ``cp_order`` values.

    Negamax convention: descend into the OPEN child with the smallest
    (cp_order, incoming_move_uci, node_id).  A leaf is open iff it is
    non-terminal and under the depth cap; an internal (sim-expanded) node is
    open iff any child is open.  Children only become visible once their
    parent has been expanded in the simulation, mirroring generation.
    """
    parent = payload["parent_index"].numpy()
    names = list(payload["feature_names"])
    cp_order = payload["node_features"].numpy()[:, names.index("cp_order")].astype(float)
    terminal = payload["is_terminal"].numpy()
    depth = payload["depth"].numpy()
    moves = list(payload["incoming_moves"])

    children: Dict[int, List[int]] = {}
    for child_id, parent_id in enumerate(parent.tolist()):
        if parent_id >= 0:
            children.setdefault(int(parent_id), []).append(int(child_id))

    n_expansions = len(true_expansion_order(payload))
    sim_expanded: set[int] = set()

    def is_open(node: int) -> bool:
        if node in sim_expanded:
            return any(is_open(child) for child in children.get(node, []))
        return (not bool(terminal[node])) and int(depth[node]) < max_depth

    order: List[int] = []
    for _ in range(n_expansions):
        if not is_open(0):
            break
        node = 0
        while node in sim_expanded:
            open_children = [child for child in children.get(node, []) if is_open(child)]
            assert open_children, f"replay invariant violated at node {node}"
            node = min(
                open_children,
                key=lambda child: (cp_order[child], moves[child] or "", child),
            )
        order.append(node)
        sim_expanded.add(node)
    return order
