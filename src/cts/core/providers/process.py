"""Process death is recovered by restarting on the next ``analyse`` call."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import List, Mapping, Optional

from .common import position_spec_to_uci_command


@dataclass(frozen=True)
class UciEngineConfig:
    """Immutable configuration bundle for ``UciEngineProcess``.

    Captures everything needed to spawn the engine and run a single
    ``analyse`` call: the binary path, optional subcommand mode, search
    limits (movetime / depth / nodes), MultiPV width, and any extra
    ``setoption`` overrides. Frozen so a single config can be shared
    across multiple processes without spooky mutation.
    """

    engine_path: str  # absolute path to the engine binary (e.g. lc0, stockfish)
    engine_kind: str = "lc0"  # which engine family this is; gates engine-specific options
    engine_mode: Optional[str] = None  # optional subcommand passed as argv[1] (e.g. "backendbench")
    movetime_ms: int = 200  # per-position search budget in milliseconds; 0 disables
    multipv: int = 8  # number of principal variations to report
    depth: Optional[int] = None  # optional fixed-depth search limit
    nodes: Optional[int] = None  # optional fixed-node search limit
    weights_path: Optional[str] = None  # lc0 weights file path; passed via setoption WeightsFile
    uci_options: Mapping[str, str] = field(default_factory=dict)  # extra setoption name→value pairs applied after the defaults
    set_multipv: bool = True  # whether to emit setoption MultiPV at startup
    enable_verbose_move_stats: bool = True  # lc0-only: turn on VerboseMoveStats so child stats appear in info lines

    def __post_init__(self) -> None:
        # Validate at construction so misconfigurations fail before a
        # subprocess is ever spawned. At least one search limit must be
        # finite or the engine would search forever on ``go``.
        if self.engine_kind not in ("lc0", "stockfish"):
            raise ValueError("engine_kind must be 'lc0' or 'stockfish'.")
        if self.movetime_ms <= 0 and self.depth is None and self.nodes is None:
            raise ValueError("At least one search limit must be provided.")
        if self.multipv <= 0:
            raise ValueError("multipv must be positive.")


class UciEngineProcess:
    """Owns a single long-lived UCI engine subprocess.

    Usable as a context manager (``with UciEngineProcess(cfg) as eng:``)
    or by calling ``start`` / ``close`` explicitly. ``analyse(fen)`` is
    the main entry point; it returns the raw stdout lines up to and
    including the terminating ``bestmove`` line so callers can parse
    whichever info format they need. A rolling buffer of recent output
    is kept to help diagnose unexpected EOFs from the engine.
    """

    def __init__(self, config: UciEngineConfig) -> None:
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None  # live Popen handle, or None when not running
        self._recent_lines: List[str] = []  # bounded ring of recent stdout lines for crash diagnostics
        self._current_fen: Optional[str] = None  # FEN being analysed right now; only populated mid-analyse

    def __enter__(self) -> "UciEngineProcess":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Spawn the engine and complete the UCI handshake. Idempotent: a no-op if already running."""
        if self.process is not None:
            return

        # Build argv: just the binary, optionally followed by a subcommand
        # (e.g. lc0's ``backendbench``). stderr is merged into stdout so
        # all diagnostic output flows through a single readline loop.
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

        # Apply options in a deterministic order: weights first (lc0 needs
        # them to initialize), then MultiPV, then engine-specific toggles,
        # then user-supplied overrides which can shadow any of the above.
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
        """Tear down the subprocess, swallowing any errors. Idempotent."""
        if self.process is None:
            return
        # Best-effort graceful shutdown: send ``quit``, then terminate, then
        # close pipes. Every step is wrapped so a dead engine doesn't keep
        # us from cleaning up the rest.
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
        """Analyse a single position and return all engine output up to ``bestmove``.

        Auto-starts the engine if it's not already running. The returned
        list contains every stdout line emitted in response to the
        ``position`` + ``go`` pair, in order, with the terminating
        ``bestmove ...`` line as the last element.

        Args:
            fen: FEN string (or position spec accepted by
                ``position_spec_to_uci_command``) identifying the position.
        """
        if self.process is None:
            self.start()
        assert self.process is not None

        # Track the in-flight FEN so EOF errors can report what was being
        # analysed when the engine died.
        self._current_fen = fen
        self._send(position_spec_to_uci_command(fen))
        # Assemble the ``go`` command from whichever search limits are set.
        # Multiple limits can coexist; the engine stops at the first one hit.
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
        """Send a single ``setoption name <name> value <value>`` line."""
        self._send(f"setoption name {name} value {value}")

    def _send(self, command: str) -> None:
        """Write a UCI command line to the engine's stdin and flush."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Engine process is not running.")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _readline(self) -> str:
        """Read one stripped line from the engine; raise with diagnostics on EOF.

        An empty read means the engine closed its stdout (typically because
        it crashed or was killed). The error message includes the return
        code, the FEN being analysed, and the tail of recent output so the
        caller has enough context to root-cause the failure.
        """
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
        # Maintain a bounded ring of recent lines (last 200) for crash
        # diagnostics without unbounded memory growth on long-running engines.
        self._recent_lines.append(stripped)
        if len(self._recent_lines) > 200:
            self._recent_lines = self._recent_lines[-200:]
        return stripped

    def _read_until(self, token: str) -> List[str]:
        """Read lines until one exactly equals ``token``; return all lines including the terminator."""
        lines = []
        while True:
            line = self._readline()
            lines.append(line)
            if line == token:
                return lines
