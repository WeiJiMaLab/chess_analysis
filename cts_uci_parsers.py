from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from cts_uci_common import (
    MOVE_STATS_RE,
    MULTIPV_RE,
    SCORE_LINE_RE,
    WDL_RE,
    append_move_to_position_spec,
    clamp,
    uci_score_to_value,
)
from tree import ExpansionChild


@dataclass
class UciAnalysis:
    root_value: float
    children: List[ExpansionChild]
    raw_lines: List[str]


class UciAnalysisParser:
    def parse(self, lines: Sequence[str], fen: str) -> UciAnalysis:
        raise NotImplementedError


class StockfishAnalysisParser(UciAnalysisParser):
    def parse(self, lines: Sequence[str], fen: str) -> UciAnalysis:
        root_value = None
        children_by_move: Dict[str, float] = {}

        for line in lines:
            score_match = SCORE_LINE_RE.search(line)
            if score_match is not None and root_value is None:
                root_value = uci_score_to_value(
                    score_match.group("kind"),
                    int(score_match.group("score")),
                )

            multipv_match = MULTIPV_RE.search(line)
            if multipv_match is None:
                continue

            move = multipv_match.group("move")
            value = uci_score_to_value(
                multipv_match.group("kind"),
                int(multipv_match.group("score")),
            )
            children_by_move.setdefault(move, value)

        if root_value is None:
            raise ValueError("Could not parse a root value from Stockfish output.")
        if not children_by_move:
            raise ValueError("Could not parse child candidates from Stockfish output.")

        children = [
            ExpansionChild(
                move_uci=move,
                fen=append_move_to_position_spec(fen, move),
                scalar_features={"value": value, "prior": value},
            )
            for move, value in children_by_move.items()
        ]
        return UciAnalysis(root_value=root_value, children=children, raw_lines=list(lines))


class Lc0AnalysisParser(UciAnalysisParser):
    def parse(self, lines: Sequence[str], fen: str) -> UciAnalysis:
        root_value = None
        children_by_move: Dict[str, ExpansionChild] = {}

        for line in lines:
            score_match = SCORE_LINE_RE.search(line)
            if score_match is not None and root_value is None:
                root_value = uci_score_to_value(
                    score_match.group("kind"),
                    int(score_match.group("score")),
                )

            move_stats_match = MOVE_STATS_RE.search(line)
            if move_stats_match is None:
                continue

            move = move_stats_match.group("move")
            prior_text = move_stats_match.group("prior")
            if prior_text.endswith("%"):
                prior_score = float(prior_text[:-1]) / 100.0
            else:
                prior_score = float(prior_text)
            q_value = clamp(float(move_stats_match.group("q")), -1.0, 1.0)
            children_by_move.setdefault(
                move,
                ExpansionChild(
                    move_uci=move,
                    fen=append_move_to_position_spec(fen, move),
                    scalar_features={"value": q_value, "prior": prior_score},
                ),
            )

        if root_value is None:
            raise ValueError("Could not parse a root value from lc0 output.")
        if not children_by_move:
            raise ValueError("Could not parse child priors/Q values from lc0 output.")

        return UciAnalysis(
            root_value=root_value,
            children=list(children_by_move.values()),
            raw_lines=list(lines),
        )


class Lc0NoSearchAnalysisParser(UciAnalysisParser):
    def parse(self, lines: Sequence[str], fen: str) -> UciAnalysis:
        root_value = None
        children_by_move: Dict[str, ExpansionChild] = {}

        for line in lines:
            score_match = SCORE_LINE_RE.search(line)
            if score_match is not None and root_value is None:
                root_value = uci_score_to_value(
                    score_match.group("kind"),
                    int(score_match.group("score")),
                )

            move_stats_match = MOVE_STATS_RE.search(line)
            if move_stats_match is None:
                continue

            move = move_stats_match.group("move")
            prior_text = move_stats_match.group("prior")
            if prior_text.endswith("%"):
                prior_score = float(prior_text[:-1]) / 100.0
            else:
                prior_score = float(prior_text)
            children_by_move.setdefault(
                move,
                ExpansionChild(
                    move_uci=move,
                    fen=append_move_to_position_spec(fen, move),
                    scalar_features={"prior": prior_score},
                ),
            )

        if root_value is None:
            raise ValueError("Could not parse a root value from lc0 output.")
        if not children_by_move:
            raise ValueError("Could not parse child priors from lc0 output.")

        return UciAnalysis(
            root_value=root_value,
            children=list(children_by_move.values()),
            raw_lines=list(lines),
        )


def parse_root_value_from_lines(lines: Sequence[str]) -> float:
    return parse_root_value_features_from_lines(lines, require_wdl=False)["value"]


def value_features_from_wdl(win: float, draw: float, loss: float) -> Dict[str, float]:
    total = win + draw + loss
    if total <= 0.0:
        raise ValueError("WDL counts must have positive mass.")
    p_win = win / total
    p_draw = draw / total
    p_loss = loss / total
    value = p_win - p_loss
    variance = (p_win + p_loss) - value * value
    return {
        "value": value,
        "wdl_win": p_win,
        "wdl_draw": p_draw,
        "wdl_loss": p_loss,
        "wdl_var": variance,
    }


def terminal_value_features(value: float) -> Dict[str, float]:
    if value > 0.0:
        return value_features_from_wdl(1.0, 0.0, 0.0)
    if value < 0.0:
        return value_features_from_wdl(0.0, 0.0, 1.0)
    return value_features_from_wdl(0.0, 1.0, 0.0)


def parse_root_value_features_from_lines(
    lines: Sequence[str],
    *,
    require_wdl: bool = True,
) -> Dict[str, float]:
    for line in lines:
        wdl_match = WDL_RE.search(line)
        if wdl_match is None:
            continue
        return value_features_from_wdl(
            float(wdl_match.group("win")),
            float(wdl_match.group("draw")),
            float(wdl_match.group("loss")),
        )

    if require_wdl:
        for line in lines:
            score_match = SCORE_LINE_RE.search(line)
            if score_match is None:
                continue
            if score_match.group("kind") == "mate":
                return terminal_value_features(
                    uci_score_to_value(
                        score_match.group("kind"),
                        int(score_match.group("score")),
                    )
                )
        raise ValueError("Could not parse WDL statistics from engine output.")

    for line in lines:
        score_match = SCORE_LINE_RE.search(line)
        if score_match is not None:
            return {
                "value": uci_score_to_value(
                    score_match.group("kind"),
                    int(score_match.group("score")),
                )
            }
    raise ValueError("Could not parse a root value from engine output.")
