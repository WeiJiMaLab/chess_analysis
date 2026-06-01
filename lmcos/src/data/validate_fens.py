"""Filter candidate root FENs by lc0 best-move-instability checks.

Consumes the candidate list produced by ``sample_root_fens_from_della.py``
and runs each FEN through lc0's value head to keep only positions where
the engine's preferred move is sensitive to small perturbations — those
are the "interesting" roots where the controller has a meaningful
decision to make. FENs that lc0 rejects outright (bad string, no legal
move) are flagged as failures and printed for inspection.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ValidateFensConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fens: str
    engine_path: str
    weights_path: str
    sample_size: int = 200
    seed: int = 0


def sample_fens(path: Path, sample_size: int, seed: int) -> list[tuple[int, str]]:
    """Read ``path`` and return up to ``sample_size`` ``(line_no, fen)`` pairs.

    Line numbers are 1-indexed and preserved so failure output points back
    to the source file. Blank lines are skipped. If the file has fewer
    lines than ``sample_size``, every non-blank line is returned.
    """
    lines = [(idx + 1, line.strip()) for idx, line in enumerate(path.open("r", encoding="utf-8")) if line.strip()]
    if sample_size >= len(lines):
        return lines
    rng = random.Random(seed)
    return rng.sample(lines, sample_size)


def validate_one(engine_path: str, weights_path: str, fen: str) -> tuple[bool, str]:
    """Run a single FEN through lc0 and return ``(ok, captured_output)``.

    Drives lc0 via UCI on stdin: load weights, set the position, ask for a
    one-node search, and quit. We treat the position as bad if lc0 exits
    non-zero or its output contains any of the known failure markers
    (parse error, illegal side-to-move, or the null bestmove that lc0
    emits when no legal reply exists).
    """
    command = [engine_path, "valuehead"]
    # UCI script: handshake, point at the weights, load the position,
    # request a one-node search to force a bestmove, then exit cleanly.
    payload = (
        "uci\n"
        f"setoption name WeightsFile value {weights_path}\n"
        "isready\n"
        f"position fen {fen}\n"
        "go nodes 1\n"
        "quit\n"
    )
    proc = subprocess.run(
        command,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = proc.stdout
    # Substrings lc0 emits when the FEN is unusable. "bestmove 0000" is
    # the null move lc0 returns when no legal continuation exists.
    bad_markers = [
        "Bad fen string",
        "invalid side to move",
        "bestmove 0000",
    ]
    ok = proc.returncode == 0 and not any(marker in output for marker in bad_markers)
    return ok, output


def main(config: ValidateFensConfig) -> None:
    """Sample FENs, validate each through lc0, and print a summary plus first 10 failures."""
    sampled = sample_fens(Path(config.fens), config.sample_size, config.seed)
    failures: list[tuple[int, str, str]] = []

    for line_no, fen in sampled:
        ok, output = validate_one(config.engine_path, config.weights_path, fen)
        if not ok:
            failures.append((line_no, fen, output))

    # Summary first so it's visible even when failure detail is long.
    # Cap printed failures at 10 so a broken weights file doesn't flood the log.
    print(f"checked={len(sampled)} failures={len(failures)}")
    for line_no, fen, output in failures[:10]:
        print(f"line={line_no}")
        print(f"fen={fen}")
        print(output.strip())
        print("---")


if __name__ == "__main__":
    from cts._config import run_with_config_cli
    run_with_config_cli(ValidateFensConfig, main)
