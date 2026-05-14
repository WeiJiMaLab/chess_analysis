"""Shared UCI-protocol parsing and position-spec helpers.

Utility layer sitting between the chess engine (which speaks UCI text lines)
and the rest of the CTS pipeline (which works with FEN strings, UCI moves,
and value scalars). Centralizes the regexes for engine output, the
``"<fen> ||moves|| <m1> <m2> ..."`` position-spec encoding used to thread
move histories through the codebase, and the conversion of engine cp/mate
scores into the unified [-1, 1] value range. Importing ``chess`` is
optional so callers without the dependency can still touch position-spec
strings (but board-level helpers will return ``None``).
"""

from __future__ import annotations

import re
from typing import List, Optional

try:
    import chess
except ImportError:  # pragma: no cover - exercised on cluster when dependency is installed.
    # python-chess is an optional dep; functions that need a real board
    # degrade to returning None when it's missing.
    chess = None


# Regex fragment matching a single UCI move (with optional promotion suffix).
UCI_MOVE_PATTERN = r"[a-h][1-8][a-h][1-8][qrbn]?"
# `info` line carrying a centipawn/mate score and (optionally) the principal-variation first move.
SCORE_LINE_RE = re.compile(
    rf"info .*score (?P<kind>cp|mate) (?P<score>-?\d+).*(?: pv (?P<move>{UCI_MOVE_PATTERN}))?"
)
# Per-move stats line emitted by some engines: move, prior probability P, and Q value.
MOVE_STATS_RE = re.compile(
    rf"(?P<move>{UCI_MOVE_PATTERN}).*?\bP[:=]\s*(?P<prior>-?\d+(?:\.\d+)?%?).*?\bQ[:=]\s*(?P<q>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# `wdl` triple (win/draw/loss counts) reported by engines that surface it.
WDL_RE = re.compile(
    r"\bwdl\s+(?P<win>\d+)\s+(?P<draw>\d+)\s+(?P<loss>\d+)\b",
    re.IGNORECASE,
)
# Final `bestmove <uci>` line that closes a UCI search.
BESTMOVE_RE = re.compile(r"^bestmove\s+(?P<move>\S+)")
# Delimiter splitting the root FEN from the trailing move list inside a position spec string.
POSITION_SPEC_SEPARATOR = " ||moves|| "


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def append_move_to_position_spec(position_spec: str, move: str) -> str:
    """Return a new position spec with ``move`` appended to its move list.

    If the spec already has a move list, the move is added to the end;
    otherwise the separator is introduced and ``move`` becomes the first move.

    Args:
        position_spec: existing position spec (FEN, or FEN + separator + moves).
        move: UCI move to append.
    """
    if POSITION_SPEC_SEPARATOR in position_spec:
        root_fen, move_text = position_spec.split(POSITION_SPEC_SEPARATOR, 1)
        moves = move_text.split()
        moves.append(move)
        return root_fen + POSITION_SPEC_SEPARATOR + " ".join(moves)
    return position_spec + POSITION_SPEC_SEPARATOR + move


def position_spec_to_uci_command(position_spec: str) -> str:
    """Format a position spec into the corresponding UCI ``position`` command string."""
    if POSITION_SPEC_SEPARATOR in position_spec:
        root_fen, move_text = position_spec.split(POSITION_SPEC_SEPARATOR, 1)
        return f"position fen {root_fen} moves {move_text}"
    return f"position fen {position_spec}"


def split_position_spec(position_spec: str) -> tuple[str, List[str]]:
    """Split a position spec into (root FEN, list of UCI moves). Empty move list if no separator."""
    if POSITION_SPEC_SEPARATOR not in position_spec:
        return position_spec, []
    root_fen, move_text = position_spec.split(POSITION_SPEC_SEPARATOR, 1)
    return root_fen, move_text.split()


def board_from_position_spec(position_spec: str):
    """Materialize a ``chess.Board`` from a position spec, or ``None`` on failure.

    Returns ``None`` if python-chess isn't installed or the FEN/moves don't
    parse — callers must handle the ``None`` case rather than assume a board.
    """
    if chess is None:
        return None

    root_fen, moves = split_position_spec(position_spec)
    try:
        board = chess.Board(root_fen)
        for move in moves:
            board.push_uci(move)
    except ValueError:
        return None
    return board


def terminal_value_from_board(board) -> Optional[float]:
    """Return the side-to-move value of a terminal position, or ``None`` if non-terminal.

    Uses ``+1`` for a win, ``-1`` for a loss, and ``0`` for a draw, all
    from the perspective of the side to move on ``board``. Returns ``None``
    when the game isn't over (or when python-chess is unavailable).
    """
    if chess is None:
        return None

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def terminal_value_from_position_spec(position_spec: str) -> Optional[float]:
    """Convenience wrapper: parse the spec and return its terminal value (or ``None``)."""
    board = board_from_position_spec(position_spec)
    if board is None:
        return None
    return terminal_value_from_board(board)


def uci_score_to_value(kind: str, raw_score: int) -> float:
    """Map a UCI score (cp or mate) into the unified [-1, 1] value range.

    Mate scores collapse to the sign of the mate (``+1`` for mate-in-N for
    side to move, ``-1`` for being mated). Centipawn scores are divided by
    1000 and clamped, matching the engine→value convention used elsewhere.

    Args:
        kind: ``"cp"`` or ``"mate"`` as emitted by the UCI info line.
        raw_score: signed integer score from the same info line.
    """
    if kind == "mate":
        return 1.0 if raw_score > 0 else -1.0
    return clamp(raw_score / 1000.0, -1.0, 1.0)


def analysis_has_no_legal_move(lines) -> bool:
    """Return True if the UCI output indicates the position has no legal move.

    Engines signal this with one of the sentinel ``bestmove`` payloads
    (``0000``, ``(none)``, ``none``) when faced with checkmate or stalemate.
    """
    for line in lines:
        match = BESTMOVE_RE.match(line)
        if match is None:
            continue
        move = match.group("move")
        return move in {"0000", "(none)", "none"}
    return False
