from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a sample of root FENs by asking lc0 to parse them.")
    parser.add_argument("--fens", required=True)
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def sample_fens(path: Path, sample_size: int, seed: int) -> list[tuple[int, str]]:
    lines = [(idx + 1, line.strip()) for idx, line in enumerate(path.open("r", encoding="utf-8")) if line.strip()]
    if sample_size >= len(lines):
        return lines
    rng = random.Random(seed)
    return rng.sample(lines, sample_size)


def validate_one(engine_path: str, weights_path: str, fen: str) -> tuple[bool, str]:
    command = [engine_path, "valuehead"]
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
    bad_markers = [
        "Bad fen string",
        "invalid side to move",
        "bestmove 0000",
    ]
    ok = proc.returncode == 0 and not any(marker in output for marker in bad_markers)
    return ok, output


def main() -> None:
    args = build_arg_parser().parse_args()
    sampled = sample_fens(Path(args.fens), args.sample_size, args.seed)
    failures: list[tuple[int, str, str]] = []

    for line_no, fen in sampled:
        ok, output = validate_one(args.engine_path, args.weights_path, fen)
        if not ok:
            failures.append((line_no, fen, output))

    print(f"checked={len(sampled)} failures={len(failures)}")
    for line_no, fen, output in failures[:10]:
        print(f"line={line_no}")
        print(f"fen={fen}")
        print(output.strip())
        print("---")


if __name__ == "__main__":
    main()
