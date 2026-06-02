"""
Move Quality (MQ) and Value of Computation (VOC) using chess engines.

MQ(S, m):
    How good was move m from state S, relative to the engine best?
    MQ = e_win_taken - e_win_best  (always ≤ 0; 0 = optimal move played).
    Both values are win-probabilities from the *current player's perspective*,
    evaluated at depth `depth` before any move is made.

VOC(S):
    How much is gained by thinking deeply (Russek et al.)?
    VOC = V_deep(a_deep) - V_deep(a_shallow)  (always ≥ 0).
    a_shallow = argmax_a V_shallow(a)  (best move at depth_shallow = 1)
    a_deep    = argmax_a V_deep(a)     (best move at depth_deep   = 15)
    Both V_deep values are evaluated at depth_deep so they are comparable.

Win probability convention (same as script_engine_eval.py):
    _win_prob uses WDL from the *side to move's* perspective:
        (wins + 0.5 * draws) / total
"""

from __future__ import annotations

import chess
import chess.engine


def _win_prob(info: chess.engine.InfoDict, board: chess.Board) -> float | None:
    """Win probability for the side to move from a single InfoDict."""
    if board.is_checkmate():
        return 0.0
    if board.is_game_over():
        return 0.5
    score = info.get("score")
    if score is None:
        return None
    wdl = score.pov(board.turn).wdl()
    return (wdl.wins + 0.5 * wdl.draws) / wdl.total()


def move_quality(
    board: chess.Board,
    move: chess.Move,
    engine: chess.engine.SimpleEngine,
    depth: int = 15,
) -> float | None:
    """
    MQ = e_win_taken - e_win_best  (≤ 0; 0 = optimal move played).

    Evaluates the position with multipv=2 at `depth`. If `move` is not among
    the top-2 lines, a dedicated root_moves search recovers e_win_taken.

    Returns None for terminal positions or evaluation failures.
    """
    if board.is_game_over():
        return None

    info_list = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    if not info_list:
        return None

    e_win_best = _win_prob(info_list[0], board)
    if e_win_best is None:
        return None

    e_win_taken = None
    for info in info_list:
        pv = info.get("pv")
        if pv and pv[0] == move:
            e_win_taken = _win_prob(info, board)
            break

    if e_win_taken is None:
        fallback = engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
        e_win_taken = _win_prob(fallback, board) if fallback else None

    if e_win_taken is None:
        return None

    return e_win_taken - e_win_best


def voc(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth_deep: int = 15,
    depth_shallow: int = 1,
) -> float | None:
    """
    VOC = V_deep(a_deep) - V_deep(a_shallow)  (≥ 0).

    Step 1: depth_shallow search → a_shallow (best move at low depth).
    Step 2: depth_deep multipv=2 search → a_deep, V_deep(a_deep).
    Step 3: if a_shallow ≠ a_deep, recover V_deep(a_shallow) via root_moves search.

    Returns None for terminal positions or evaluation failures.
    """
    if board.is_game_over():
        return None

    # Step 1: find a_shallow
    shallow_info = engine.analyse(board, chess.engine.Limit(depth=depth_shallow), multipv=1)
    if not shallow_info:
        return None
    pv_shallow = shallow_info[0].get("pv") if shallow_info else None
    if not pv_shallow:
        return None
    a_shallow = pv_shallow[0]

    # Step 2: find a_deep and V_deep(a_deep) with multipv=2
    deep_info = engine.analyse(board, chess.engine.Limit(depth=depth_deep), multipv=2)
    if not deep_info:
        return None

    v_deep_best = _win_prob(deep_info[0], board)
    if v_deep_best is None:
        return None

    pv_deep = deep_info[0].get("pv")
    if not pv_deep:
        return None
    a_deep = pv_deep[0]

    if a_shallow == a_deep:
        return 0.0

    # Step 3: V_deep(a_shallow) — check multipv results first, then fall back
    v_deep_shallow = None
    for info in deep_info:
        pv = info.get("pv")
        if pv and pv[0] == a_shallow:
            v_deep_shallow = _win_prob(info, board)
            break

    if v_deep_shallow is None:
        fallback = engine.analyse(board, chess.engine.Limit(depth=depth_deep), root_moves=[a_shallow])
        v_deep_shallow = _win_prob(fallback, board) if fallback else None

    if v_deep_shallow is None:
        return None

    return max(0.0, v_deep_best - v_deep_shallow)
