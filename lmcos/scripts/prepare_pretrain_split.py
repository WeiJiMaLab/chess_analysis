from __future__ import annotations

import argparse
import random
from pathlib import Path


def _gather_examples(source_root: Path) -> list[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    examples = sorted(source_root.rglob("*.pt"))
    if not examples:
        raise ValueError(f"No .pt examples found under {source_root}")
    return examples


def _write_manifest(examples: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for source in examples:
            handle.write(f"{source}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare flat train/validation splits from shard directories.")
    parser.add_argument(
        "--source-root",
        default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees",
    )
    parser.add_argument(
        "--split-root",
        default="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    source_root = Path(args.source_root)
    split_root = Path(args.split_root)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"

    examples = _gather_examples(source_root)
    rng = random.Random(args.seed)
    rng.shuffle(examples)

    validation_count = max(1, int(round(len(examples) * args.validation_fraction)))
    validation_examples = examples[:validation_count]
    train_examples = examples[validation_count:]
    if not train_examples:
        raise ValueError("Validation split consumed all examples; reduce validation_fraction.")

    if args.clear:
        for manifest in (train_manifest, validation_manifest):
            if manifest.exists():
                manifest.unlink()

    _write_manifest(train_examples, train_manifest)
    _write_manifest(validation_examples, validation_manifest)

    print(f"source_root={source_root}")
    print(f"split_root={split_root}")
    print(f"train_manifest={train_manifest}")
    print(f"validation_manifest={validation_manifest}")
    print(f"train_examples={len(train_examples)}")
    print(f"validation_examples={len(validation_examples)}")


if __name__ == "__main__":
    main()
