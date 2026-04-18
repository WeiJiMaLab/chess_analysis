from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def _gather_examples(source_root: Path) -> list[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    examples = sorted(source_root.rglob("*.pt"))
    if not examples:
        raise ValueError(f"No .pt examples found under {source_root}")
    return examples


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _rewrite_file(source: Path, destination: Path) -> tuple[int, int]:
    example = torch.load(source, weights_only=False)
    before = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(example, destination)
    after = destination.stat().st_size
    return before, after


def _rewrite_in_place(source: Path) -> tuple[int, int]:
    temp_path = source.with_suffix(source.suffix + ".compact_tmp")
    if temp_path.exists():
        temp_path.unlink()
    before, after = _rewrite_file(source, temp_path)
    os.replace(temp_path, source)
    return before, after


def _rewrite_to_output(source: Path, source_root: Path, output_root: Path) -> tuple[int, int, Path]:
    destination = output_root / source.relative_to(source_root)
    before, after = _rewrite_file(source, destination)
    return before, after, destination


def _iter_selected(paths: Iterable[Path], limit: int | None) -> Iterable[Path]:
    if limit is None:
        yield from paths
        return
    for index, path in enumerate(paths):
        if index >= limit:
            break
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite raw pretrain examples using the compact serializer.")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory containing raw pretrain .pt examples.",
    )
    parser.add_argument(
        "--output-root",
        help="Write rewritten examples to a parallel directory tree under this root.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite examples atomically in place.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only rewrite the first N examples.")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="When using --output-root, skip files that already exist at the destination.",
    )
    args = parser.parse_args()

    if args.in_place == bool(args.output_root):
        raise ValueError("Specify exactly one of --in-place or --output-root.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when provided.")
    if args.log_interval <= 0:
        raise ValueError("log-interval must be positive.")

    source_root = Path(args.source_root)
    output_root = Path(args.output_root) if args.output_root else None
    examples = _gather_examples(source_root)

    total_before = 0
    total_after = 0
    rewritten = 0
    skipped = 0

    for index, source in enumerate(_iter_selected(examples, args.limit), start=1):
        if output_root is not None:
            destination = output_root / source.relative_to(source_root)
            if args.skip_existing and destination.exists():
                skipped += 1
                continue
            before, after, _ = _rewrite_to_output(source, source_root, output_root)
        else:
            before, after = _rewrite_in_place(source)

        rewritten += 1
        total_before += before
        total_after += after

        if rewritten % args.log_interval == 0:
            saved = total_before - total_after
            print(
                f"rewritten={rewritten} skipped={skipped} "
                f"before={_human_bytes(total_before)} after={_human_bytes(total_after)} "
                f"saved={_human_bytes(saved)}"
            )

    saved = total_before - total_after
    print(f"source_root={source_root}")
    if output_root is not None:
        print(f"output_root={output_root}")
    print(f"mode={'in_place' if args.in_place else 'copy'}")
    print(f"rewritten={rewritten}")
    print(f"skipped={skipped}")
    print(f"before_bytes={total_before}")
    print(f"after_bytes={total_after}")
    print(f"saved_bytes={saved}")
    if total_before > 0:
        print(f"saved_fraction={saved / total_before:.6f}")


if __name__ == "__main__":
    main()
