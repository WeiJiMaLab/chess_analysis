from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import List, Mapping, Optional

from cts_uci_common import position_spec_to_uci_command


@dataclass(frozen=True)
class UciEngineConfig:
    engine_path: str
    engine_kind: str = "lc0"
    engine_mode: Optional[str] = None
    movetime_ms: int = 200
    multipv: int = 8
    depth: Optional[int] = None
    nodes: Optional[int] = None
    weights_path: Optional[str] = None
    uci_options: Mapping[str, str] = field(default_factory=dict)
    set_multipv: bool = True
    enable_verbose_move_stats: bool = True

    def __post_init__(self) -> None:
        if self.engine_kind not in ("lc0", "stockfish"):
            raise ValueError("engine_kind must be 'lc0' or 'stockfish'.")
        if self.movetime_ms <= 0 and self.depth is None and self.nodes is None:
            raise ValueError("At least one search limit must be provided.")
        if self.multipv <= 0:
            raise ValueError("multipv must be positive.")


class UciEngineProcess:
    def __init__(self, config: UciEngineConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None
        self._recent_lines: List[str] = []
        self._current_fen: Optional[str] = None

    def __enter__(self) -> "UciEngineProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            return

        command = [self.config.engine_path]
        if self.config.engine_mode is not None:
            command.append(self.config.engine_mode)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")

        if self.config.weights_path is not None:
            self._set_option("WeightsFile", self.config.weights_path)
        if self.config.set_multipv:
            self._set_option("MultiPV", str(self.config.multipv))
        if self.config.engine_kind == "lc0" and self.config.enable_verbose_move_stats:
            self._set_option("VerboseMoveStats", "true")
        for key, value in self.config.uci_options.items():
            self._set_option(key, value)
        self._send("isready")
        self._read_until("readyok")

    def close(self) -> None:
        if self.process is None:
            return
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            pass
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(self.process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
        self.process = None

    def analyse(self, fen: str) -> List[str]:
        if self.process is None:
            self.start()
        assert self.process is not None

        self._current_fen = fen
        self._send(position_spec_to_uci_command(fen))
        command_parts = ["go"]
        if self.config.movetime_ms > 0:
            command_parts.extend(["movetime", str(self.config.movetime_ms)])
        if self.config.depth is not None:
            command_parts.extend(["depth", str(self.config.depth)])
        if self.config.nodes is not None:
            command_parts.extend(["nodes", str(self.config.nodes)])
        self._send(" ".join(command_parts))

        lines = []
        while True:
            line = self._readline()
            lines.append(line)
            if line.startswith("bestmove "):
                self._current_fen = None
                break
        return lines

    def _set_option(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Engine process is not running.")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if line == "":
            return_code = self.process.poll()
            recent_output = "\n".join(self._recent_lines[-20:])
            current_fen = self._current_fen or "<none>"
            raise RuntimeError(
                "Unexpected EOF from engine process. "
                f"engine_path={self.config.engine_path!r} "
                f"engine_mode={self.config.engine_mode!r} "
                f"return_code={return_code!r} "
                f"current_fen={current_fen!r} "
                f"recent_output={recent_output!r}"
            )
        stripped = line.strip()
        self._recent_lines.append(stripped)
        if len(self._recent_lines) > 200:
            self._recent_lines = self._recent_lines[-200:]
        return stripped

    def _read_until(self, token: str) -> List[str]:
        lines = []
        while True:
            line = self._readline()
            lines.append(line)
            if line == token:
                return lines
