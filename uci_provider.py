from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from cts_uci_common import (
    BESTMOVE_RE,
    MOVE_STATS_RE,
    MULTIPV_RE,
    POSITION_SPEC_SEPARATOR,
    SCORE_LINE_RE,
    UCI_MOVE_PATTERN,
    analysis_has_no_legal_move,
    append_move_to_position_spec,
    board_from_position_spec,
    chess,
    clamp as _clamp,
    position_spec_to_uci_command,
    split_position_spec,
    terminal_value_from_board,
    terminal_value_from_position_spec,
    uci_score_to_value as _uci_score_to_value,
)
from cts_uci_parsers import (
    Lc0AnalysisParser,
    Lc0NoSearchAnalysisParser,
    StockfishAnalysisParser,
    UciAnalysis,
    UciAnalysisParser,
    parse_root_value_from_lines,
)
from cts_uci_process import UciEngineConfig, UciEngineProcess
from supervised_branch import TreeExpansionProvider
from tree import ExpansionChild


class UciTreeExpansionProvider(TreeExpansionProvider):
    def __init__(
        self,
        engine: UciEngineProcess,
        parser: UciAnalysisParser,
        metadata: Optional[Mapping[str, str]] = None,
        populate_child_values_from_root_eval: bool = False,
    ) -> None:
        self.engine = engine
        self.parser = parser
        self._analysis_cache: Dict[str, UciAnalysis] = {}
        self._terminal_cache: Dict[str, Optional[float]] = {}
        self._metadata = dict(metadata or {})
        self.populate_child_values_from_root_eval = populate_child_values_from_root_eval

    def root_features(self, fen: str) -> Mapping[str, float]:
        terminal_value = self._terminal_value_for_fen(fen)
        if terminal_value is not None:
            return {"value": terminal_value, "prior": 1.0}
        analysis = self._analysis_for_fen(fen)
        return {"value": analysis.root_value, "prior": 1.0}

    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        analysis = self._analysis_for_fen(fen)
        if not self.populate_child_values_from_root_eval:
            children = list(analysis.children)
            if max_children is None:
                return children
            return children[:max_children]

        expanded_children = []
        selected_children = list(analysis.children)
        if max_children is not None:
            selected_children = selected_children[:max_children]
        for child in selected_children:
            child_value = self.root_features(child.fen)["value"]
            scalar_features = dict(child.scalar_features)
            scalar_features["value"] = child_value
            expanded_children.append(
                ExpansionChild(
                    move_uci=child.move_uci,
                    fen=child.fen,
                    scalar_features=scalar_features,
                    metadata=dict(child.metadata),
                    is_terminal=child.is_terminal,
                )
            )
        return expanded_children

    def provider_metadata(self) -> Mapping[str, str]:
        return dict(self._metadata)

    def clear_caches(self) -> None:
        self._analysis_cache.clear()
        self._terminal_cache.clear()

    def _analysis_for_fen(self, fen: str) -> UciAnalysis:
        if fen not in self._analysis_cache:
            lines = self.engine.analyse(fen)
            self._analysis_cache[fen] = self.parser.parse(lines, fen)
        return self._analysis_cache[fen]

    def _terminal_value_for_fen(self, fen: str) -> Optional[float]:
        if fen not in self._terminal_cache:
            self._terminal_cache[fen] = terminal_value_from_position_spec(fen)
        return self._terminal_cache[fen]


class Lc0DirectEvalProvider(TreeExpansionProvider):
    def __init__(
        self,
        prior_engine: UciEngineProcess,
        value_engine: UciEngineProcess,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.prior_engine = prior_engine
        self.value_engine = value_engine
        self.prior_parser = Lc0NoSearchAnalysisParser()
        self._metadata = dict(metadata or {})
        self._prior_cache: Dict[str, UciAnalysis] = {}
        self._value_cache: Dict[str, float] = {}
        self._terminal_cache: Dict[str, Optional[float]] = {}
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
        terminal_value = self._terminal_value_for_fen(fen)
        if terminal_value is not None:
            return {"value": terminal_value, "prior": 1.0}
        return {"value": self._value_for_fen(fen), "prior": 1.0}

    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        prior_analysis = self._prior_analysis_for_fen(fen)
        selected_children = list(prior_analysis.children)
        if max_children is not None:
            selected_children = selected_children[:max_children]
        children = []
        parent_board = board_from_position_spec(fen)
        for child in selected_children:
            scalar_features = dict(child.scalar_features)
            fallback_value = scalar_features.get("value")
            is_terminal = child.is_terminal
            terminal_value = self._terminal_value_for_child(child, parent_board)
            if terminal_value is not None:
                scalar_features["value"] = terminal_value
                is_terminal = True
                children.append(
                    ExpansionChild(
                        move_uci=child.move_uci,
                        fen=child.fen,
                        scalar_features=scalar_features,
                        metadata=dict(child.metadata),
                        is_terminal=is_terminal,
                    )
                )
                continue
            try:
                scalar_features["value"] = self._value_for_fen(child.fen, fallback_value=fallback_value)
            except RuntimeError:
                terminal_value = self._terminal_value_from_prior(child.fen)
                if terminal_value is None:
                    raise
                scalar_features["value"] = terminal_value
                is_terminal = True
            children.append(
                ExpansionChild(
                    move_uci=child.move_uci,
                    fen=child.fen,
                    scalar_features=scalar_features,
                    metadata=dict(child.metadata),
                    is_terminal=is_terminal,
                )
            )
        return children

    def provider_metadata(self) -> Mapping[str, str]:
        return dict(self._metadata)

    def clear_caches(self) -> None:
        self._prior_cache.clear()
        self._value_cache.clear()
        self._terminal_cache.clear()

    def _prior_analysis_for_fen(self, fen: str) -> UciAnalysis:
        if fen not in self._prior_cache:
            lines = self.prior_engine.analyse(fen)
            self._prior_cache[fen] = self.prior_parser.parse(lines, fen)
        return self._prior_cache[fen]

    def _value_for_fen(self, fen: str, fallback_value: Optional[float] = None) -> float:
        if fen not in self._value_cache:
            if self._enable_value_query_log:
                with self._value_query_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{fen}\n")
            try:
                lines = self.value_engine.analyse(fen)
            except RuntimeError as exc:
                self.value_engine.close()
                self.value_engine.start()
                try:
                    lines = self.value_engine.analyse(fen)
                except RuntimeError:
                    if fallback_value is None:
                        raise
                    print(
                        "warning: valuehead failed; falling back to classic child value "
                        f"for fen={fen!r}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._value_cache[fen] = fallback_value
                    return self._value_cache[fen]
            self._value_cache[fen] = parse_root_value_from_lines(lines)
        return self._value_cache[fen]

    def _terminal_value_from_prior(self, fen: str) -> Optional[float]:
        lines = self.prior_engine.analyse(fen)
        if not _analysis_has_no_legal_move(lines):
            return None
        return parse_root_value_from_lines(lines)

    def _terminal_value_for_fen(self, fen: str) -> Optional[float]:
        if fen not in self._terminal_cache:
            self._terminal_cache[fen] = terminal_value_from_position_spec(fen)
        return self._terminal_cache[fen]

    def _terminal_value_for_child(self, child: ExpansionChild, parent_board) -> Optional[float]:
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


def _analysis_has_no_legal_move(lines: Sequence[str]) -> bool:
    return analysis_has_no_legal_move(lines)


__all__ = [
    "BESTMOVE_RE",
    "Lc0AnalysisParser",
    "Lc0DirectEvalProvider",
    "Lc0NoSearchAnalysisParser",
    "MOVE_STATS_RE",
    "MULTIPV_RE",
    "POSITION_SPEC_SEPARATOR",
    "SCORE_LINE_RE",
    "StockfishAnalysisParser",
    "UCI_MOVE_PATTERN",
    "UciAnalysis",
    "UciAnalysisParser",
    "UciEngineConfig",
    "UciEngineProcess",
    "UciTreeExpansionProvider",
    "_analysis_has_no_legal_move",
    "_clamp",
    "_uci_score_to_value",
    "append_move_to_position_spec",
    "board_from_position_spec",
    "chess",
    "parse_root_value_from_lines",
    "position_spec_to_uci_command",
    "split_position_spec",
    "terminal_value_from_board",
    "terminal_value_from_position_spec",
]
