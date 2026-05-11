from __future__ import annotations

import re
from typing import List, Optional

try:
    import chess
except ImportError:  # pragma: no cover - exercised on cluster when dependency is installed.
    chess = None


UCI_MOVE_PATTERN = r"[a-h][1-8][a-h][1-8][qrbn]?"
SCORE_LINE_RE = re.compile(
    rf"info .*score (?P<kind>cp|mate) (?P<score>-?\d+).*(?: pv (?P<move>{UCI_MOVE_PATTERN}))?"
)
MOVE_STATS_RE = re.compile(
    rf"(?P<move>{UCI_MOVE_PATTERN}).*?\bP[:=]\s*(?P<prior>-?\d+(?:\.\d+)?%?).*?\bQ[:=]\s*(?P<q>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
WDL_RE = re.compile(
    r"\bwdl\s+(?P<win>\d+)\s+(?P<draw>\d+)\s+(?P<loss>\d+)\b",
    re.IGNORECASE,
)
BESTMOVE_RE = re.compile(r"^bestmove\s+(?P<move>\S+)")
POSITION_SPEC_SEPARATOR = " ||moves|| "


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def append_move_to_position_spec(position_spec: str, move: str) -> str:
    if POSITION_SPEC_SEPARATOR in position_spec:
        root_fen, move_text = position_spec.split(POSITION_SPEC_SEPARATOR, 1)
        moves = move_text.split()
        moves.append(move)
        return root_fen + POSITION_SPEC_SEPARATOR + " ".join(moves)
    return position_spec + POSITION_SPEC_SEPARATOR + move


def position_spec_to_uci_command(position_spec: str) -> str:
    if POSITION_SPEC_SEPARATOR in position_spec:
        root_fen, move_text = position_spec.split(POSITION_SPEC_SEPARATOR, 1)
        return f"position fen {root_fen} moves {move_text}"
    return f"position fen {position_spec}"


def split_position_spec(position_spec: str) -> tuple[str, List[str]]:
    if POSITION_SPEC_SEPARATOR not in position_spec:
        return position_spec, []
    root_fen, move_text = position_spec.split(POSITION_SPEC_SEPARATOR, 1)
    return root_fen, move_text.split()


def board_from_position_spec(position_spec: str):
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
    if chess is None:
        return None

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def terminal_value_from_position_spec(position_spec: str) -> Optional[float]:
    board = board_from_position_spec(position_spec)
    if board is None:
        return None
    return terminal_value_from_board(board)


def uci_score_to_value(kind: str, raw_score: int) -> float:
    if kind == "mate":
        return 1.0 if raw_score > 0 else -1.0
    return clamp(raw_score / 1000.0, -1.0, 1.0)


def analysis_has_no_legal_move(lines) -> bool:
    for line in lines:
        match = BESTMOVE_RE.match(line)
        if match is None:
            continue
        move = match.group("move")
        return move in {"0000", "(none)", "none"}
    return False
