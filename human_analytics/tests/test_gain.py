"""Regression tests for the FIXED Gain (voc) definition in
tree_values_analysis._tree_voc_and_gap (diagnose_gain.md Part 2 §3).

Gain is read from ONE growing oracle tree:
    a_shallow = first root move the search expanded (argmin first_visit over the q-trace)
    a_deep    = best @ 96 expansions = argmax(oracle_final_root_q_values)
    Gain      = final_Q[a_deep] − final_Q[a_shallow]   (≥ 0 by construction)

The bug these pin: the OLD definition picked a_shallow = the 1-ply value-head best, which is
frequently a root move the deep search NEVER visited, so final_Q[a_shallow] was the
uninitialized 0.0 → Gain = (deep≈+1) − 0.0 ≈ 1.0, a spurious point-mass. The fix reads
a_shallow from what the search actually expanded (nonzero q-trace), so it is always a visited
move with a real converged Q.

Tiny hand-built trees only — no 100k real-tree load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import _tree_voc_and_gap  # noqa: E402


def _make_tree(*, child_values, final_q, q_trace, root_moves=None):
    """Minimal payload dict matching what _tree_voc_and_gap reads.

    node_features: row 0 = root, rows 1.. = children, one column 'value'.
      child 'value' is from the CHILD's side, so parent-perspective 1-ply backup = −value.
    final_q: oracle_final_root_q_values, one entry per ROOT MOVE (parent perspective).
    q_trace: (n_steps, n_root_moves) — a move is 'visited' on the first step its entry != 0.
    """
    n_kids = len(child_values)
    node_features = torch.tensor([[0.0]] + [[v] for v in child_values], dtype=torch.float64)
    parent_index = torch.tensor([-1] + [0] * n_kids, dtype=torch.long)
    if root_moves is None:
        root_moves = [f"m{i}" for i in range(len(final_q))]
    return {
        "feature_names": ["value"],
        "node_features": node_features,
        "parent_index": parent_index,
        "oracle_final_root_q_values": np.asarray(final_q, dtype=float),
        "oracle_root_q_trace": np.asarray(q_trace, dtype=float),
        "oracle_root_moves": list(root_moves),
        "incoming_moves": {i + 1: f"m{i}" for i in range(n_kids)},
    }


def test_gain_shallow_is_visited():
    """THE regression: the 1-ply value-head best is an UNVISITED root move (final_Q 0.0).
    The fixed Gain must pick a_shallow = the first move the search expanded (NOT the
    unvisited one) and must NOT return ≈1.0 from a 0.0 lookup.

    Setup: 3 root moves. The value head most-prefers move 2 (a child value of −0.9 ⇒
    parent backup +0.9), but the deep search abandoned move 2 — its final_Q is the
    uninitialized 0.0 and it never appears in the q-trace. The search actually expanded
    move 0 first (a proven win, final_Q +1.0)."""
    # child values (child side): move0 strong→ child value −1 (parent +1); move1 mid;
    # move2 looks best to the 1-ply head (child value −0.9 → parent +0.9) but is UNVISITED.
    child_values = [-1.0, -0.2, -0.9]
    final_q = [1.0, 0.1, 0.0]            # move2's deep Q is the unvisited 0.0
    # q-trace: only moves 0 and 1 ever get a nonzero entry; move0 first.
    q_trace = [
        [1.0, 0.0, 0.0],   # step 0: move0 expanded
        [1.0, 0.1, 0.0],   # step 1: move1 expanded; move2 NEVER
    ]
    payload = _make_tree(child_values=child_values, final_q=final_q, q_trace=q_trace)
    voc, _ = _tree_voc_and_gap(payload)
    # a_deep = argmax(final_q) = move0 (1.0); a_shallow = first visited = move0 ⇒ Gain 0.
    assert voc == pytest.approx(0.0, abs=1e-9)
    # explicitly NOT the old artifact:
    assert voc < 0.95, "Gain pinned ≈1.0 — the unvisited-0.0 bug is back"


def test_gain_nonnegative():
    """Gain ≥ 0 for any well-formed growing tree (a_deep = argmax(final_Q))."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = rng.integers(2, 8)
        final_q = rng.uniform(-1, 1, size=k)
        # random visit order; make at least 2 moves visited
        q_trace = np.zeros((k, k))
        order = rng.permutation(k)
        for step, mv in enumerate(order):
            q_trace[step:, mv] = final_q[mv] if final_q[mv] != 0 else 0.01
        child_values = (-final_q).tolist()  # arbitrary but consistent shape
        payload = _make_tree(child_values=child_values, final_q=final_q.tolist(),
                             q_trace=q_trace.tolist())
        voc, _ = _tree_voc_and_gap(payload)
        assert not np.isnan(voc)
        assert voc >= -1e-9, f"Gain negative: {voc}"


def test_gain_matches_growing_tree_identity():
    """Gain == final_Q[argmax] − final_Q[first-expanded], exactly, for a known tree."""
    final_q = [0.3, 0.8, -0.2, 0.5]
    # first-expanded = move2 (its q-trace entry is nonzero earliest)
    q_trace = [
        [0.0, 0.0, -0.2, 0.0],   # step 0: move2 first
        [0.0, 0.8, -0.2, 0.0],   # step 1: move1
        [0.0, 0.8, -0.2, 0.5],   # step 2: move3
    ]
    child_values = [-v for v in final_q]
    payload = _make_tree(child_values=child_values, final_q=final_q, q_trace=q_trace)
    voc, _ = _tree_voc_and_gap(payload)
    a_deep = int(np.argmax(final_q))          # move1 (0.8)
    a_shallow = 2                              # first nonzero q-trace column
    assert voc == pytest.approx(final_q[a_deep] - final_q[a_shallow], abs=1e-12)
    assert voc == pytest.approx(0.8 - (-0.2), abs=1e-12)  # == 1.0, a REAL gain (move2 visited)


def test_gain_nan_on_degenerate_tree():
    """A root with <2 children (no real search) yields NaN, not a fabricated number."""
    payload = _make_tree(child_values=[-0.5], final_q=[0.5], q_trace=[[0.5]])
    voc, gap = _tree_voc_and_gap(payload)
    assert np.isnan(voc)
    assert np.isnan(gap)
