"""Production tree-expansion provider backed by a Stockfish subprocess.

Plugs into the ``TreeExpansionProvider`` interface that tree generation calls
to expand a node: given a FEN, return root WDL features and a list of
``ExpansionChild`` records for each legal move.

Can be configured to limit its playing strength to a particular Elo rating
to simulate human-like evaluations rather than superhuman play.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Optional, Sequence

from .base import TreeExpansionProvider
from .common import (
    append_move_to_position_spec,
    board_from_position_spec,
    chess,
    terminal_value_from_board,
    terminal_value_from_position_spec,
)
from .parsers import (
    apply_tanh_cp_feature,
    parse_root_value_features_from_lines,
    terminal_value_features,
)
from .process import UciEngineProcess
from ..tree import ExpansionChild


class StockfishDirectEvalProvider(TreeExpansionProvider):
    """``TreeExpansionProvider`` that reads evaluations and WDL features from ``Stockfish``.

    Configurable to use a specific search node/depth budget and a target Elo strength.
    Caches evaluations to avoid redundant subprocess calls during rollouts.
    """

    def __init__(
        self,
        engine: UciEngineProcess,
        search_limit_nodes: int = 100,
        search_limit_depth: Optional[int] = None,
        elo: Optional[int] = None,
        metadata: Optional[Mapping[str, str]] = None,
        tanh_cp_temperature: Optional[float] = None,
    ) -> None:
        """
        Args:
            engine: ``UciEngineProcess`` instance running Stockfish.
            search_limit_nodes: Number of nodes to search per evaluation.
            search_limit_depth: Optional fixed depth to search per evaluation.
            elo: Optional playing strength limit in Elo (e.g., 1800).
                Configures UCI_LimitStrength and UCI_Elo on startup.
            metadata: Free-form provider tags surfaced via ``provider_metadata``.
            tanh_cp_temperature: when set, every node's feature dict also gets
                a ``tanh_cp_value = tanh(cp_order / tanh_cp_temperature)``
                feature (see ``apply_tanh_cp_feature``) -- a desaturated
                alternative to the WDL-derived ``value`` column. ``None``
                (default) is a zero-cost no-op; existing configs are
                unaffected. Select it for search/backup via
                ``BuildTreeConfig.value_feature = "tanh_cp_value"``.
        """
        self.engine = engine
        self.search_limit_nodes = search_limit_nodes
        self.search_limit_depth = search_limit_depth
        self.elo = elo
        self.tanh_cp_temperature = tanh_cp_temperature
        self._metadata = dict(metadata or {})

        # Always enable WDL output for Stockfish since the parser requires it
        self.engine.start()
        self.engine._send("setoption name UCI_ShowWDL value true")
        if elo is not None:
            self.engine._send("setoption name UCI_LimitStrength value true")
            self.engine._send(f"setoption name UCI_Elo value {elo}")
        self.engine._send("isready")
        self.engine._read_until("readyok")

        self._value_cache: Dict[str, Mapping[str, float]] = {}
        self._terminal_cache: Dict[str, Optional[float]] = {}

    def root_features(self, fen: str) -> Mapping[str, float]:
        """Return the per-node features stored on the root of a fresh tree."""
        terminal_value = self._terminal_value_for_fen(fen)
        if terminal_value is not None:
            features = terminal_value_features(terminal_value)
        else:
            features = self._value_features_for_fen(fen)
        features = apply_tanh_cp_feature(features, self.tanh_cp_temperature)
        return {**features, "prior": 1.0}

    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        """Return one ``ExpansionChild`` per legal move from ``fen``."""
        board = board_from_position_spec(fen)
        if board is None:
            raise ValueError(f"Could not parse board from FEN: {fen}")

        legal_moves = list(board.legal_moves)
        if max_children is not None:
            legal_moves = legal_moves[:max_children]

        children = []
        n_moves = len(legal_moves)
        # Uniform priors for Stockfish since it lacks a policy head
        prior = 1.0 / n_moves if n_moves > 0 else 1.0

        for move in legal_moves:
            move_uci = move.uci()
            child_fen = append_move_to_position_spec(fen, move_uci)
            scalar_features = {"prior": prior}

            # Cheap terminal check first to avoid engine calls on mate/stalemate
            terminal_value = self._terminal_value_for_child(child_fen, move_uci, board)
            if terminal_value is not None:
                scalar_features.update(terminal_value_features(terminal_value))
                is_terminal = True
            else:
                scalar_features.update(self._value_features_for_fen(child_fen))
                is_terminal = False
            scalar_features = apply_tanh_cp_feature(scalar_features, self.tanh_cp_temperature)

            children.append(
                ExpansionChild(
                    move_uci=move_uci,
                    fen=child_fen,
                    scalar_features=scalar_features,
                    is_terminal=is_terminal,
                )
            )
        return children

    def provider_metadata(self) -> Mapping[str, str]:
        """Return provider tags including engine kind and Elo settings."""
        meta = dict(self._metadata)
        meta["engine_kind"] = "stockfish"
        if self.elo is not None:
            meta["elo"] = str(self.elo)
        return meta

    def clear_caches(self) -> None:
        """Drop all cached lookups."""
        self._value_cache.clear()
        self._terminal_cache.clear()

    def _value_features_for_fen(self, fen: str) -> Mapping[str, float]:
        """Memoized: WDL features for ``fen`` from the Stockfish engine."""
        if fen not in self._value_cache:
            # Temporarily override search limits to the configured nodes/depth
            orig_config = self.engine.config
            self.engine.config = dataclasses.replace(
                orig_config,
                depth=self.search_limit_depth,
                nodes=self.search_limit_nodes,
                movetime_ms=0,
                multipv=1,
            )
            try:
                lines = self.engine.analyse(fen)
            finally:
                # Always restore original configuration
                self.engine.config = orig_config

            self._value_cache[fen] = parse_root_value_features_from_lines(lines, require_wdl=True)
        return self._value_cache[fen]

    def _terminal_value_for_fen(self, fen: str) -> Optional[float]:
        """Memoized: terminal value of ``fen``."""
        if fen not in self._terminal_cache:
            self._terminal_cache[fen] = terminal_value_from_position_spec(fen)
        return self._terminal_cache[fen]

    def _terminal_value_for_child(self, child_fen: str, move_uci: str, parent_board) -> Optional[float]:
        """Cheap terminal check for a child position reusing the parent board."""
        if child_fen in self._terminal_cache:
            return self._terminal_cache[child_fen]

        terminal_value = None
        if parent_board is not None and chess is not None:
            try:
                child_board = parent_board.copy(stack=False)
                child_board.push_uci(move_uci)
                terminal_value = terminal_value_from_board(child_board)
            except ValueError:
                terminal_value = None

        if terminal_value is None:
            terminal_value = terminal_value_from_position_spec(child_fen)

        self._terminal_cache[child_fen] = terminal_value
        return terminal_value
