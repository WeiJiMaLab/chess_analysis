from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from cts_uci_common import (
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
from cts_uci_parsers import (
    Lc0NoSearchAnalysisParser,
    UciAnalysis,
    parse_root_value_features_from_lines,
    parse_root_value_from_lines,
    terminal_value_features,
)
from cts_uci_process import UciEngineConfig, UciEngineProcess
from cts_pretrain import TreeExpansionProvider
from tree import ExpansionChild


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
        self._value_cache: Dict[str, Mapping[str, float]] = {}
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
        prior_analysis = self._prior_analysis_for_fen(fen)
        selected_children = list(prior_analysis.children)
        if max_children is not None:
            selected_children = selected_children[:max_children]
        children = []
        parent_board = board_from_position_spec(fen)
        for child in selected_children:
            scalar_features = dict(child.scalar_features)
            is_terminal = child.is_terminal
            terminal_value = self._terminal_value_for_child(child, parent_board)
            if terminal_value is not None:
                scalar_features.update(terminal_value_features(terminal_value))
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
                scalar_features.update(self._value_features_for_fen(child.fen))
            except RuntimeError:
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

    def _value_features_for_fen(self, fen: str) -> Mapping[str, float]:
        if fen not in self._value_cache:
            if self._enable_value_query_log:
                with self._value_query_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{fen}\n")
            try:
                lines = self.value_engine.analyse(fen)
            except RuntimeError:
                self.value_engine.close()
                self.value_engine.start()
                try:
                    lines = self.value_engine.analyse(fen)
                except RuntimeError:
                    raise
            self._value_cache[fen] = parse_root_value_features_from_lines(lines, require_wdl=True)
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
    "Lc0DirectEvalProvider",
    "Lc0NoSearchAnalysisParser",
    "POSITION_SPEC_SEPARATOR",
    "UCI_MOVE_PATTERN",
    "UciAnalysis",
    "UciEngineConfig",
    "UciEngineProcess",
    "append_move_to_position_spec",
    "board_from_position_spec",
    "chess",
    "parse_root_value_from_lines",
    "parse_root_value_features_from_lines",
    "position_spec_to_uci_command",
    "split_position_spec",
    "terminal_value_from_board",
    "terminal_value_from_position_spec",
]
