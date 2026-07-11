from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from .common import (
    POSITION_SPEC_SEPARATOR,
    UCI_MOVE_PATTERN,
    analysis_has_no_legal_move,
    append_move_to_position_spec,
    board_from_position_spec,
    chess,
    position_spec_to_uci_command,
    split_position_spec,
    terminal_value_from_board,
    terminal_value_from_position_spec,
)
from .parsers import (
    UciAnalysis,
    parse_no_search_analysis,
    parse_root_value_features_from_lines,
    parse_root_value_from_lines,
    terminal_value_features,
)
from .process import UciEngineConfig, UciEngineProcess
from .base import TreeExpansionProvider
from ..tree import ExpansionChild


class Lc0DirectEvalProvider(TreeExpansionProvider):
    """``TreeExpansionProvider`` that reads priors and WDL straight from ``lc0``.

    Holds two long-lived engine processes (prior + value) and parses their
    UCI output into the feature dicts the tree generator expects. Caches are
    per-instance and unbounded — fine because tree generation visits a
    bounded set of FENs per tree, and the provider is discarded between
    trees in production.
    """

    def __init__(
        self,
        prior_engine: UciEngineProcess,
        value_engine: UciEngineProcess,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Args:
            prior_engine: ``lc0`` process configured for ``go nodes 1`` so it
                emits per-move priors without doing any search.
            value_engine: separate ``lc0`` process used solely to read the
                root WDL valuehead for a position.
            metadata: free-form provider tags surfaced via ``provider_metadata``
                (e.g. engine version, weights file) — copied into trees on disk.
        """
        self.prior_engine = prior_engine
        self.value_engine = value_engine
        self._metadata = dict(metadata or {})
        # Per-FEN caches. Tree generation revisits the same positions
        # (transpositions, repeated rollouts) so caching cuts engine calls
        # by a large factor in practice.
        self._prior_cache: Dict[str, UciAnalysis] = {}
        self._value_cache: Dict[str, Mapping[str, float]] = {}
        self._terminal_cache: Dict[str, Optional[float]] = {}
        # Optional debug log of every FEN the value engine is asked about.
        # Off by default; flip the env var to inspect what the value engine
        # actually sees when chasing a discrepancy.
        self._value_query_log_path = Path("logs/valuehead_queries.log")
        self._enable_value_query_log = os.environ.get("CTS_LOG_VALUEHEAD_QUERIES", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if self._enable_value_query_log:
            self._value_query_log_path.parent.mkdir(parents=True, exist_ok=True)

    def root_features(self, fen: str) -> Mapping[str, float]:
        """Return the per-node features stored on the root of a fresh tree.

        For terminal positions (checkmate/stalemate) we synthesize the
        feature dict directly from the known game result; otherwise we ask
        the value engine. ``prior=1.0`` is the convention for the root since
        a root has no incoming move and therefore no real prior.
        """
        terminal_value = self._terminal_value_for_fen(fen)
        if terminal_value is not None:
            features = terminal_value_features(terminal_value)
        else:
            features = self._value_features_for_fen(fen)
        return {**features, "prior": 1.0}

    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        """Return one ``ExpansionChild`` per legal move from ``fen``.

        Calls the prior engine to enumerate candidate moves with their
        priors, then queries the value engine for each child's WDL. Children
        that turn out to be terminal short-circuit the value call.

        Args:
            fen: parent FEN to expand.
            depth: parent's depth in the tree (unused here; part of the
                ``TreeExpansionProvider`` contract for providers that care).
            max_children: optional cap on how many children to return; the
                first ``max_children`` from the prior engine's ordering.
        """
        prior_analysis = self._prior_analysis_for_fen(fen)
        selected_children = list(prior_analysis.children)
        if max_children is not None:
            selected_children = selected_children[:max_children]
        children = []
        parent_board = board_from_position_spec(fen)
        for child in selected_children:
            scalar_features = dict(child.scalar_features)
            is_terminal = child.is_terminal
            # Cheap terminal check first — if the move ends the game we
            # already know the value exactly and skip the value engine call.
            terminal_value = self._terminal_value_for_child(child, parent_board)
            if terminal_value is not None:
                scalar_features.update(terminal_value_features(terminal_value))
                is_terminal = True
                children.append(
                    ExpansionChild(
                        move_uci=child.move_uci,
                        fen=child.fen,
                        scalar_features=scalar_features,
                        metadata=dict(child.metadata or {}),
                        is_terminal=is_terminal,
                    )
                )
                continue
            try:
                scalar_features.update(self._value_features_for_fen(child.fen))
            except RuntimeError:
                # Value engine failed. Sometimes that's because the position
                # is actually terminal and the engine refused to evaluate it;
                # try the prior engine for confirmation and pick up the
                # terminal value from there. If that doesn't explain it,
                # re-raise — something else is wrong.
                terminal_value = self._terminal_value_from_prior(child.fen)
                if terminal_value is None:
                    raise
                scalar_features.update(terminal_value_features(terminal_value))
                is_terminal = True
            children.append(
                ExpansionChild(
                    move_uci=child.move_uci,
                    fen=child.fen,
                    scalar_features=scalar_features,
                    metadata=dict(child.metadata or {}),
                    is_terminal=is_terminal,
                )
            )
        return children

    def provider_metadata(self) -> Mapping[str, str]:
        """Return a fresh copy of the provider tags supplied at construction time."""
        return dict(self._metadata)

    def clear_caches(self) -> None:
        """Drop all cached prior/value/terminal lookups. Used between trees."""
        self._prior_cache.clear()
        self._value_cache.clear()
        self._terminal_cache.clear()

    def _prior_analysis_for_fen(self, fen: str) -> UciAnalysis:
        """Memoized: parsed prior-engine output (children + priors) for ``fen``."""
        if fen not in self._prior_cache:
            lines = self.prior_engine.analyse(fen)
            self._prior_cache[fen] = parse_no_search_analysis(lines, fen)
        return self._prior_cache[fen]

    def _value_features_for_fen(self, fen: str) -> Mapping[str, float]:
        """Memoized: WDL valuehead reading for ``fen`` from the value engine.

        On engine failure we close-and-restart the process once and retry;
        long-lived ``lc0`` processes occasionally wedge and a fresh start
        clears it. A second failure propagates as ``RuntimeError`` for the
        caller to interpret (often as "this position is actually terminal").
        """
        if fen not in self._value_cache:
            if self._enable_value_query_log:
                with self._value_query_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{fen}\n")
            try:
                lines = self.value_engine.analyse(fen)
            except RuntimeError:
                self.value_engine.close()
                self.value_engine.start()
                lines = self.value_engine.analyse(fen)
            self._value_cache[fen] = parse_root_value_features_from_lines(lines, require_wdl=True)
        return self._value_cache[fen]

    def _terminal_value_from_prior(self, fen: str) -> Optional[float]:
        """Ask the prior engine whether ``fen`` is terminal; return its value if so.

        Used as a fallback when the value engine refuses to evaluate a
        position. Returns ``None`` if the prior engine reports legal moves,
        i.e. the position is not in fact terminal.
        """
        lines = self.prior_engine.analyse(fen)
        if not analysis_has_no_legal_move(lines):
            return None
        return parse_root_value_from_lines(lines)

    def _terminal_value_for_fen(self, fen: str) -> Optional[float]:
        """Memoized: terminal value of ``fen`` (or ``None`` if non-terminal)."""
        if fen not in self._terminal_cache:
            self._terminal_cache[fen] = terminal_value_from_position_spec(fen)
        return self._terminal_cache[fen]

    def _terminal_value_for_child(self, child: ExpansionChild, parent_board) -> Optional[float]:
        """Cheap terminal check for a child position.

        Prefers reusing the parent's ``python-chess`` board and pushing the
        move (much faster than reparsing the child FEN from scratch); falls
        back to FEN parsing if the move push fails or the board library is
        unavailable. Result is cached on the child's FEN.
        """
        if child.fen in self._terminal_cache:
            return self._terminal_cache[child.fen]

        terminal_value = None
        if parent_board is not None and chess is not None:
            try:
                child_board = parent_board.copy(stack=False)
                child_board.push_uci(child.move_uci)
                terminal_value = terminal_value_from_board(child_board)
            except ValueError:
                terminal_value = None

        if terminal_value is None:
            terminal_value = terminal_value_from_position_spec(child.fen)

        self._terminal_cache[child.fen] = terminal_value
        return terminal_value


__all__ = ["Lc0DirectEvalProvider"]
