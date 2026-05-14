"""Parsers that turn raw UCI engine output into tree-building primitives.

This module replaces the older unified ``Lc0AnalysisParser`` which conflated
prior parsing and value parsing and silently inverted perspective in some
cases. The split here is deliberate: ``parse_no_search_analysis`` consumes
the ``go nodes 1`` output of a policy engine (root value + child moves with
their priors) and ``parse_root_value_features_from_lines`` consumes the root
output of a separate value engine (WDL triple), so the two perspectives
never get mixed up at this boundary. Sits between the engine-driver layer
and the tree-construction layer in the data-generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .common import (
    MOVE_STATS_RE,
    SCORE_LINE_RE,
    WDL_RE,
    append_move_to_position_spec,
    uci_score_to_value,
)
from ..tree import ExpansionChild


@dataclass
class UciAnalysis:
    """Result of parsing a single engine analysis turn at one node.

    Bundles the root's scalar value with the expansion children produced by
    the engine's move list, plus the original lines for debugging/logging.
    """

    root_value: float  # scalar value at the analyzed position, from the engine's perspective
    children: List[ExpansionChild]  # one entry per child move reported by the engine, with prior already attached
    raw_lines: List[str]  # original engine output lines, kept for diagnostics


def parse_no_search_analysis(lines: Sequence[str], fen: str) -> UciAnalysis:
    """Parse lc0's ``go nodes 1`` (zero-search) output into a ``UciAnalysis``.

    Extracts the root score and every child move's prior in a single pass.
    Used during dataset generation to seed a node's children with the
    policy engine's priors without paying for any tree search.

    Args:
        lines: raw UCI output lines from a single ``go nodes 1`` call.
        fen: FEN of the position being analyzed; used to derive each
            child's resulting FEN by appending its move.
    """
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
        # lc0 prints priors either as "12.34%" or as a raw fraction;
        # normalize both forms to [0, 1].
        if prior_text.endswith("%"):
            prior_score = float(prior_text[:-1]) / 100.0
        else:
            prior_score = float(prior_text)
        # setdefault: if the engine repeats a move across lines, keep
        # the first occurrence's prior rather than overwriting it.
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
    """Return just the scalar root value, falling back to ``score`` when no WDL is present."""
    return parse_root_value_features_from_lines(lines, require_wdl=False)["value"]


def value_features_from_wdl(win: float, draw: float, loss: float) -> Dict[str, float]:
    """Normalize raw WDL counts into the canonical feature dict.

    Converts raw win/draw/loss masses (counts or probabilities) into the
    five features the encoder consumes: ``value`` (win-loss), the
    normalized WDL triple, and the variance of the outcome distribution.

    Args:
        win: raw win mass; need not be normalized.
        draw: raw draw mass.
        loss: raw loss mass.
    """
    total = win + draw + loss
    if total <= 0.0:
        raise ValueError("WDL counts must have positive mass.")
    p_win = win / total
    p_draw = draw / total
    p_loss = loss / total
    value = p_win - p_loss
    # Variance of the outcome (treating win=+1, draw=0, loss=-1):
    # E[x^2] = p_win + p_loss, then subtract (E[x])^2.
    variance = (p_win + p_loss) - value * value
    return {
        "value": value,
        "wdl_win": p_win,
        "wdl_draw": p_draw,
        "wdl_loss": p_loss,
        "wdl_var": variance,
    }


def terminal_value_features(value: float) -> Dict[str, float]:
    """Construct the canonical feature dict for a terminal position.

    Maps a deterministic terminal outcome to a degenerate WDL: a win is
    all win-mass, a loss is all loss-mass, anything else is treated as a
    forced draw. Used when the engine reports a mate score instead of WDL.
    """
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
    """Extract the root's value features from a value engine's output lines.

    Preference order: real WDL line first, mate score as a terminal
    fallback, then a plain centipawn/score line if ``require_wdl=False``.
    Used for the root position only — child priors come from the policy
    engine via ``parse_no_search_analysis``.

    Args:
        lines: raw UCI output lines from the value engine at the root.
        require_wdl: if True, fail when no WDL or mate line is present.
            If False, accept a bare score line and return ``{"value": ...}``.
    """
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
        # No WDL line found, but a mate score collapses to a known terminal
        # outcome, so accept it as a degenerate WDL distribution.
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

    # WDL not required: fall back to the first plain score line and return
    # only the scalar value (no WDL triple, no variance).
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
