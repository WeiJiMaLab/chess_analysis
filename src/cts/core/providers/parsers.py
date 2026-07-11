"""``parse_no_search_analysis`` handles the policy engine's output (priors);
``parse_root_value_features_from_lines`` handles the separate value engine's
(WDL) -- kept split so the two perspectives never mix at this boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .common import (
    MOVE_STATS_RE,
    SCORE_LINE_RE,
    WDL_RE,
    append_move_to_position_spec,
    uci_score_to_value,
)
from ..tree import ExpansionChild


# Centipawn band reserved for mate scores in ``cp_order``. Stockfish's maximum
# *real* centipawn magnitude is VALUE_MAX_EVAL * 100 / PawnValueEg ≈
# 32000 * 100 / 208 ≈ 15385 (see stockfish/src/uci.cpp:309), so mapping a
# "mate in N (moves)" score to ``±(20000 − N)`` keeps every mate strictly above
# (below, when getting mated) every possible non-mate score, while closer mates
# rank higher — preserving Stockfish's own mate-distance ordering.
MATE_CP_ORDER_BAND = 20000.0


def score_order_features(kind: str, raw_score: int) -> Dict[str, float]:
    """Map a UCI ``score cp N`` / ``score mate N`` into native-unit features.

    Emits (additively; never part of ``TREE_ENCODER_FEATURE_NAMES``):
      - ``cp``: the raw centipawn score, side-to-move POV. Only present for
        non-mate scores.
      - ``mate``: the signed mate distance in moves (positive = side to move
        mates, negative = side to move gets mated). Only present for mate
        scores.
      - ``cp_order``: a single scalar usable as a search priority in
        centipawn units: equal to ``cp`` for non-mate scores, and
        ``sign(mate) * (MATE_CP_ORDER_BAND - |mate|)`` for mate scores.
        Always present.

    Missing keys are simply absent from the dict (node feature tensors store
    NaN for absent features), never ``None`` — ``SearchTree`` scalar features
    must be numeric.
    """
    if kind == "mate":
        distance = abs(int(raw_score))
        # ``mate 0`` means the side to move is checkmated; treat it as a loss.
        sign = 1.0 if int(raw_score) > 0 else -1.0
        return {
            "mate": float(raw_score),
            "cp_order": sign * (MATE_CP_ORDER_BAND - float(distance)),
        }
    return {"cp": float(raw_score), "cp_order": float(raw_score)}


def apply_tanh_cp_feature(features: Dict[str, float], temperature: Optional[float]) -> Dict[str, float]:
    """Add a ``tanh_cp_value`` feature, a temperature-desaturated alternative to ``value``.

    ``tanh_cp_value = tanh(cp_order / temperature)`` -- same formula, and same
    intent (T3's "Related discovery": the shipped ``value`` feature saturates
    to ``|value| >= 0.99`` for ~56% of nodes because Stockfish's WDL curve is
    essentially a step function around ~300cp, while ``cp_order`` is
    continuous), as ``analysis.relabel_replay.tanh_cp_value`` -- deliberately
    NOT imported from there: ``cts`` is the lower-level package and must not
    depend on ``analysis`` (the diagnostic layer built on top of it), so the
    one-line formula is duplicated here rather than re-derived differently.
    See ``cp_regen.md`` (Agent 3, plan.md) for why this exists: unlike
    ``analysis.relabel_replay``'s shape-frozen relabel-and-replay probe, this
    feature is computed at GENERATION time so that PUCT's own selection (not
    just the final label) can respond to the desaturated value when
    ``value_feature: tanh_cp_value`` is selected in ``BuildTreeConfig``.

    Returns ``features`` unchanged (same dict, no copy) when ``temperature``
    is ``None`` -- the default, zero-cost no-op path used by every existing
    config that doesn't opt into this feature. Requires ``cp_order`` to
    already be present in ``features`` (true for every code path that calls
    this: ``score_order_features``/``terminal_value_features`` both always
    emit it).
    """
    if temperature is None:
        return features
    if "cp_order" not in features:
        raise KeyError("apply_tanh_cp_feature requires 'cp_order' in features; got keys: " + ", ".join(features))
    features = dict(features)
    features["tanh_cp_value"] = math.tanh(features["cp_order"] / float(temperature))
    return features


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

    Also emits the native-unit features (see ``score_order_features``): a
    decided terminal is a mate at distance 0 (``mate=0.0`` — the sign of the
    outcome lives in ``cp_order = ±MATE_CP_ORDER_BAND``, since a float can't
    carry a signed zero reliably); a draw is ``cp=0``/``cp_order=0``.
    """
    if value > 0.0:
        # Side to move has won with no move to make — unreachable in chess,
        # kept for interface symmetry.
        return {**value_features_from_wdl(1.0, 0.0, 0.0), "mate": 0.0, "cp_order": MATE_CP_ORDER_BAND}
    if value < 0.0:
        # Checkmate: side to move is mated right now.
        return {**value_features_from_wdl(0.0, 0.0, 1.0), "mate": 0.0, "cp_order": -MATE_CP_ORDER_BAND}
    # Stalemate / draw.
    return {**value_features_from_wdl(0.0, 1.0, 0.0), "cp": 0.0, "cp_order": 0.0}


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

    In addition to the WDL-derived features, the engine's native ``score
    cp|mate`` is captured ADDITIVELY (``cp``/``mate``/``cp_order``, see
    ``score_order_features``) when present — UCI ``info`` lines carry the
    score alongside the wdl triple. Engines that emit WDL without a score
    line (e.g. lc0 valuehead mode) simply omit these keys.
    """
    for line in lines:
        wdl_match = WDL_RE.search(line)
        if wdl_match is None:
            continue
        features = value_features_from_wdl(
            float(wdl_match.group("win")),
            float(wdl_match.group("draw")),
            float(wdl_match.group("loss")),
        )
        # Prefer the score on the same info line as the WDL (they describe
        # the same evaluation); fall back to the first score line anywhere.
        score_match = SCORE_LINE_RE.search(line)
        if score_match is None:
            for other_line in lines:
                score_match = SCORE_LINE_RE.search(other_line)
                if score_match is not None:
                    break
        if score_match is not None:
            features.update(
                score_order_features(score_match.group("kind"), int(score_match.group("score")))
            )
        return features

    if require_wdl:
        # No WDL line found, but a mate score collapses to a known terminal
        # outcome, so accept it as a degenerate WDL distribution. The native
        # mate distance overrides the distance-0 stub from
        # ``terminal_value_features``.
        for line in lines:
            score_match = SCORE_LINE_RE.search(line)
            if score_match is None:
                continue
            if score_match.group("kind") == "mate":
                raw_score = int(score_match.group("score"))
                return {
                    **terminal_value_features(
                        uci_score_to_value(score_match.group("kind"), raw_score)
                    ),
                    **score_order_features("mate", raw_score),
                }
        raise ValueError("Could not parse WDL statistics from engine output.")

    # WDL not required: fall back to the first plain score line and return
    # the scalar value (no WDL triple, no variance) plus the native score.
    for line in lines:
        score_match = SCORE_LINE_RE.search(line)
        if score_match is not None:
            return {
                "value": uci_score_to_value(
                    score_match.group("kind"),
                    int(score_match.group("score")),
                ),
                **score_order_features(score_match.group("kind"), int(score_match.group("score"))),
            }
    raise ValueError("Could not parse a root value from engine output.")
