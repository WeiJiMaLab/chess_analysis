"""
Engine evaluation primitives for chess positions.

Central abstraction: PositionEval — all quantities computed in a single
evaluate_position() call (2–4 engine calls total).

Quantities:
    e_win_best         V_deep(a_deep)        win prob of best move at depth_deep
    e_win_second_best  V_deep(rank-2 move)   win prob of second-best move
    e_win_taken        V_deep(move_taken)    win prob of the move actually played
    voc                V_deep(a_deep) - V_deep(a_shallow)  ≥ 0  (Russek et al.)

Derived (properties, not stored separately):
    mq      = e_win_taken - e_win_best   ≤ 0  (0 = optimal move played)
    toptwo  = e_win_best - e_win_second_best  ≥ 0  (how decisively best beats 2nd)

Win probability convention: (wins + 0.5 * draws) / total from engine WDL,
always from the *side to move's* perspective before any move is made.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.engine


def _win_prob(info: chess.engine.InfoDict, board: chess.Board) -> float | None:
    """Win probability for the side to move from a single engine InfoDict."""
    if board.is_checkmate():
        return 0.0
    if board.is_game_over():
        return 0.5
    score = info.get("score")
    if score is None:
        return None
    wdl = score.pov(board.turn).wdl()
    return (wdl.wins + 0.5 * wdl.draws) / wdl.total()


@dataclass
class PositionEval:
    """All engine-derived quantities for one (position, move) pair."""
    e_win_best: float | None
    e_win_second_best: float | None
    e_win_taken: float | None
    voc: float | None

    @property
    def mq(self) -> float | None:
        """MQ = e_win_taken - e_win_best  (≤ 0; 0 = optimal move).

        Clamped to ≤ 0: Stockfish's multipv and root_moves searches use
        different aspiration windows and hash states, so e_win_taken from a
        root_moves fallback can numerically exceed e_win_best from multipv by
        a tiny amount.  The clamp enforces the theoretical guarantee.
        """
        if self.e_win_taken is None or self.e_win_best is None:
            return None
        return min(0.0, self.e_win_taken - self.e_win_best)

    @property
    def toptwo(self) -> float | None:
        """toptwo = e_win_best - e_win_second_best  (≥ 0)."""
        if self.e_win_best is None or self.e_win_second_best is None:
            return None
        return max(0.0, self.e_win_best - self.e_win_second_best)


def evaluate_position(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    depth_deep: int = 5,
    depth_shallow: int = 1,
    nodes_deep: int | None = None,
    nodes_shallow: int | None = None,
) -> PositionEval:
    """
    Evaluate one (position, move) pair and return all engine quantities.

    Limit type is controlled by the nodes_* parameters:
      - If nodes_deep / nodes_shallow are given, use node-budget limits
        (Limit(nodes=N)) instead of depth limits.  This means the effective
        search depth scales inversely with branching factor, so the VOC
        comparison becomes sensitive to position complexity.
      - Default (nodes_* = None): depth-based limits.

    Engine call budget (2–4 total):
      1. shallow limit multipv=1  →  a_shallow  (VOC denominator move)
      2. deep limit   multipv=2  →  e_win_best, e_win_second_best, a_deep;
                                    also covers e_win_taken and V_deep(a_shallow)
                                    if either falls in the top-2 lines
      3. root_moves=[move_taken] fallback  (only if move_taken not in top-2)
      4. root_moves=[a_shallow]  fallback  (only if a_shallow not in top-2)
    """
    if board.is_game_over():
        p = 0.5 if not board.is_checkmate() else 0.0
        return PositionEval(e_win_best=p, e_win_second_best=p, e_win_taken=p, voc=0.0)

    use_nodes = nodes_deep is not None
    limit_shallow = chess.engine.Limit(nodes=nodes_shallow or 1) if use_nodes else chess.engine.Limit(depth=depth_shallow)
    limit_deep = chess.engine.Limit(nodes=nodes_deep) if use_nodes else chess.engine.Limit(depth=depth_deep)

    # --- Step 1: shallow search → a_shallow ---
    shallow_info = engine.analyse(board, limit_shallow, multipv=1)
    if not shallow_info:
        return PositionEval(None, None, None, None)
    pv_shallow = shallow_info[0].get("pv")
    a_shallow = pv_shallow[0] if pv_shallow else None

    # --- Step 2: deep search multipv=2 ---
    deep_info = engine.analyse(board, limit_deep, multipv=2)
    if not deep_info:
        return PositionEval(None, None, None, None)

    e_win_best = _win_prob(deep_info[0], board)
    e_win_second_best = _win_prob(deep_info[1], board) if len(deep_info) > 1 else e_win_best

    pv_deep = deep_info[0].get("pv")
    a_deep = pv_deep[0] if pv_deep else None

    # Scan top-2 lines for e_win_taken and V_deep(a_shallow)
    e_win_taken = None
    v_deep_shallow = None
    for info in deep_info:
        pv = info.get("pv")
        if not pv:
            continue
        first_move = pv[0]
        val = _win_prob(info, board)
        if first_move == move:
            e_win_taken = val
        if a_shallow is not None and first_move == a_shallow:
            v_deep_shallow = val

    # --- Step 3: fallback for e_win_taken ---
    if e_win_taken is None:
        fb = engine.analyse(board, limit_deep, root_moves=[move])
        e_win_taken = _win_prob(fb, board) if fb else None

    # --- Step 4: fallback for V_deep(a_shallow) ---
    if a_shallow is not None and v_deep_shallow is None:
        fb = engine.analyse(board, limit_deep, root_moves=[a_shallow])
        v_deep_shallow = _win_prob(fb, board) if fb else None

    # VOC
    if e_win_best is None or a_shallow is None or v_deep_shallow is None:
        voc_val = None
    elif a_shallow == a_deep:
        voc_val = 0.0
    else:
        voc_val = max(0.0, e_win_best - v_deep_shallow)

    return PositionEval(
        e_win_best=e_win_best,
        e_win_second_best=e_win_second_best,
        e_win_taken=e_win_taken,
        voc=voc_val,
    )


# ---------------------------------------------------------------------------
# Backward-compatible standalone helpers
# ---------------------------------------------------------------------------

def move_quality(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    depth: int = 5,
) -> float | None:
    """MQ = e_win_taken - e_win_best  (≤ 0). Thin wrapper around evaluate_position."""
    return evaluate_position(board, move, engine, depth_deep=depth).mq


def voc(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth_deep: int = 5,
    depth_shallow: int = 1,
) -> float | None:
    """VOC = V_deep(a_deep) - V_deep(a_shallow)  (≥ 0). Wrapper around evaluate_position."""
    dummy_move = next(iter(board.legal_moves), None)
    if dummy_move is None:
        return None
    return evaluate_position(board, dummy_move, engine, depth_deep=depth_deep,
                             depth_shallow=depth_shallow).voc
