from __future__ import annotations

import re
import subprocess
import chess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from metacontrol.core.schemas import ChildInfo

class TreeExpansionProvider(ABC):
    """Abstraction over the chess engine."""

    @abstractmethod
    def evaluate_root(self, fen: str) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        """Return (value, optional_wdl) for the root position."""

    @abstractmethod
    def expand(self, fen: str, depth: int) -> List[ChildInfo]:
        """Return child expansions for a position, sorted by prior descending."""


class UciExpansionProvider(TreeExpansionProvider):
    """Base class for Universal Chess Interface (UCI) engines."""
    
    def __init__(
        self, 
        engine_path: str, 
        nodes: Optional[int] = None, 
        movetime_ms: Optional[int] = None,
        options: Optional[Dict[str, str]] = None
    ):
        self.engine_path = engine_path
        self.nodes, self.movetime_ms = nodes, movetime_ms
        self.options = options or {}
        self._process = self._start_engine()

    def _start_engine(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [self.engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        self._send(proc, "uci")
        self._wait_for(proc, "uciok")
        for k, v in self.options.items():
            self._send(proc, f"setoption name {k} value {v}")
        self._send(proc, "isready")
        self._wait_for(proc, "readyok")
        return proc

    def _send(self, proc: subprocess.Popen, cmd: str):
        if proc.stdin:
            proc.stdin.write(f"{cmd}\n")

    def _wait_for(self, proc: subprocess.Popen, target: str) -> List[str]:
        lines = []
        while True:
            line = proc.stdout.readline().strip()
            lines.append(line)
            if target in line: return lines
            if not line and proc.poll() is not None:
                raise RuntimeError(f"Engine died. Last output: {lines[-5:]}")

    def _get_analysis(self, fen: str) -> List[str]:
        self._send(self._process, f"position fen {fen}")
        cmd = "go"
        if self.nodes: cmd += f" nodes {self.nodes}"
        if self.movetime_ms: cmd += f" movetime {self.movetime_ms}"
        self._send(self._process, cmd)
        return self._wait_for(self._process, "bestmove")

    def _parse_wdl(self, match: re.Match) -> Tuple[float, float, float]:
        w, d, l = int(match.group("win")), int(match.group("draw")), int(match.group("loss"))
        total = w + d + l
        return (w/total, d/total, l/total)

    def evaluate_root(self, fen: str) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        board = chess.Board(fen)
        if outcome := board.outcome(claim_draw=True):
            if outcome.winner is None: return 0.0, (0.0, 1.0, 0.0)
            return (1.0, (1.0, 0.0, 0.0)) if outcome.winner == board.turn else (-1.0, (0.0, 0.0, 1.0))

        children = self.expand(fen, 0)
        if not children: return 0.0, None
        # expand() returns child perspective, so flip back for root
        wdl = children[0].wdl
        root_wdl = (wdl[2], wdl[1], wdl[0]) if wdl else None
        return -children[0].value, root_wdl

    @abstractmethod
    def expand(self, fen: str, depth: int) -> List[ChildInfo]:
        """Subclasses must implement expansion logic."""

    def __del__(self):
        if hasattr(self, '_process') and self._process:
            try:
                self._send(self._process, "quit")
                self._process.terminate()
                self._process.wait(timeout=1)
            except: pass


class LC0ExpansionProvider(UciExpansionProvider):
    """Lc0-specific provider with VerboseMoveStats parsing."""
    
    MOVE_RE = re.compile(r"info string\s+(?P<move>[a-h][1-8][a-h][1-8][qrbn]?)\s+.*?\(P:\s*(?P<p>[\d.%]+)\).*?\(Q:\s*(?P<q>[\d.-]+)\)", re.I)
    WDL_RE = re.compile(r"wdl\s+(?P<win>\d+)\s+(?P<draw>\d+)\s+(?P<loss>\d+)", re.I)
    WL_D_RE = re.compile(r"\(WL:\s*(?P<wl>[\d.-]+)\).*?\(D:\s*(?P<d>[\d.-]+)\)", re.I)

    def __init__(self, engine_path: str, weights_path: str, nodes: int = 64):
        super().__init__(engine_path, nodes=nodes, options={
            "WeightsFile": weights_path, 
            "MultiPV": "8", 
            "VerboseMoveStats": "true",
            "UCI_ShowWDL": "true"
        })

    def evaluate_root(self, fen: str) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        board = chess.Board(fen)
        if outcome := board.outcome(claim_draw=True):
            if outcome.winner is None: return 0.0, (0.0, 1.0, 0.0)
            return (1.0, (1.0, 0.0, 0.0)) if outcome.winner == board.turn else (-1.0, (0.0, 0.0, 1.0))

        lines = self._get_analysis(fen)
        for line in reversed(lines):
            if wm := self.WDL_RE.search(line):
                wdl = self._parse_wdl(wm)
                return wdl[0] - wdl[2], wdl
        return 0.0, None

    def _parse_wl_d(self, match: re.Match) -> Tuple[float, float, float]:
        try:
            wl, d = float(match.group("wl")), float(match.group("d"))
            w = (1.0 - d + wl) / 2.0
            l = 1.0 - d - w
            return w, d, l
        except ValueError:
            return None

    def expand(self, fen: str, depth: int) -> List[ChildInfo]:
        board, children = chess.Board(fen), {}
        if board.is_game_over(): return []

        for line in self._get_analysis(fen):
            if m := self.MOVE_RE.search(line):
                move = m.group("move")
                prior = float(m.group("p")[:-1])/100 if m.group("p").endswith("%") else float(m.group("p"))
                q_val = float(m.group("q"))
                val = -q_val
                
                # Try standard WDL first, then WL/D format, then fallback to Q
                wdl = None
                if wm := self.WDL_RE.search(line):
                    wdl = self._parse_wdl(wm)
                elif wm := self.WL_D_RE.search(line):
                    wdl = self._parse_wl_d(wm)
                
                if wdl is None:
                    # Fallback to Q-based WDL if search is shallow (assume 0 draw)
                    wdl = ((1.0 + q_val)/2.0, 0.0, (1.0 - q_val)/2.0)
                
                wdl = (wdl[2], wdl[1], wdl[0]) # Flip for child
                
                if move not in children:
                    try:
                        board.push_uci(move)
                        children[move] = ChildInfo(move, board.fen(), val, prior, wdl, board.is_game_over())
                        board.pop()
                    except ValueError: continue
        return list(children.values())


class StockfishExpansionProvider(UciExpansionProvider):
    """Stockfish-specific provider with MultiPV and WDL parsing."""
    
    INFO_RE = re.compile(r"multipv (?P<rank>\d+).*?wdl (?P<win>\d+) (?P<draw>\d+) (?P<loss>\d+).*?pv (?P<move>\w+)", re.I)

    def __init__(self, engine_path: str, nodes: int = 1000):
        super().__init__(engine_path, nodes=nodes, options={"MultiPV": "8", "UCI_ShowWDL": "true"})

    def expand(self, fen: str, depth: int) -> List[ChildInfo]:
        board, children = chess.Board(fen), {}
        if board.is_game_over(): return []

        for line in self._get_analysis(fen):
            if m := self.INFO_RE.search(line):
                move, rank = m.group("move"), int(m.group("rank"))
                # wdl from parent perspective
                pwdl = self._parse_wdl(m)
                # child perspective
                wdl = (pwdl[2], pwdl[1], pwdl[0])
                val = wdl[0] - wdl[2]
                
                if move not in children:
                    try:
                        board.push_uci(move)
                        children[move] = ChildInfo(move, board.fen(), val, 1.0/rank, wdl, board.is_game_over())
                        board.pop()
                    except ValueError: continue
        return sorted(children.values(), key=lambda x: x.prior, reverse=True)
