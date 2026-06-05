"""Board- and tree-level features shared by A0a / A1 / converged_expansions analyses."""
from __future__ import annotations

import re

import chess
import numpy as np
import torch

_RE_WHITE = re.compile(r"[RNBQK]")
_RE_BLACK = re.compile(r"[rnbqk]")

# Human RT correlations from Stockfish depth=5 (human_analytics, n=1M, lichess 10+0).
HUMAN_RT_CORRELATIONS = {
    "branching": +0.195,
    "material": +0.039,
    "gain_depth": +0.096,
    "toptwo": -0.064,
}


def extract_board_features(fen: str) -> dict:
    """Position features from a root FEN string."""
    parts = fen.split()
    side = parts[1] if len(parts) > 1 else "w"
    placement = parts[0]
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    move_ply = (fullmove - 1) * 2 + (0 if side == "w" else 1)

    board = chess.Board(fen)
    n_possible = board.legal_moves.count()
    flat = placement.replace("/", "")
    n_self = len((_RE_WHITE if side == "w" else _RE_BLACK).findall(flat))

    return {"n_possible_moves": n_possible, "n_self_pieces_exc_pawns": n_self, "move_ply": move_ply}


def extract_tree_features(t: dict) -> dict:
    """Lc0-native position features from oracle_root_q_trace and oracle_best_move_index.

    toptwo_equiv: gap between top-2 deep Q-values (oracle_final_root_q_values).
    gain_depth_equiv: Q_final[best_idx[-1]] - Q_final[best_idx[1]] (VOC analog).
    """
    q = t["oracle_root_q_trace"]
    best_idx = t["oracle_best_move_index"]
    final_q = q[-1]

    nonzero = final_q[final_q != 0]
    if len(nonzero) >= 2:
        v = nonzero.topk(2).values
        toptwo = float((v[0] - v[1]).abs().item())
    else:
        toptwo = float("nan")

    if len(best_idx) >= 2:
        deep_q = float(final_q[int(best_idx[-1].item())].item())
        shallow_q = float(final_q[int(best_idx[1].item())].item())
        gain = deep_q - shallow_q
    else:
        gain = float("nan")

    return {"toptwo_equiv": toptwo, "gain_depth_equiv": gain}
