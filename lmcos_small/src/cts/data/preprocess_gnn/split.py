"""Split generated-tree shards into train/validation manifests.

Walks the generated-trees directory produced by
``submit_generate_dataset_shards.py``, shuffles the per-tree ``.pt`` files
with a fixed seed, and writes two newline-delimited manifests (train and
validation) into ``--split-root``. The split is at the example level rather
than the shard level so every shard contributes to both manifests. Output
manifests feed ``derive_pretrain_prefixes.py`` downstream.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees"
    split_root: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split"
    validation_fraction: float = 0.05
    seed: int = 0
    clear: bool = False
    # Optional path to a clean_trees.txt (one basename per line, e.g. from
    # filter_trees_by_trace). When set, the split is restricted to those trees
    # under source_root instead of every *.pt — this is how tree filtering feeds
    # the encoder/MC packs.
    include_list: Optional[str] = None


def _gather_examples(source_root: Path, include_list: Optional[str] = None) -> list[Path]:
    """Collect example ``.pt`` files under ``source_root``, sorted.

    With ``include_list``, take only the basenames it lists (the filtered subset);
    otherwise recursively glob every ``.pt``.
    """
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    if include_list:
        names = [ln.strip() for ln in Path(include_list).read_text().splitlines() if ln.strip()]
        examples = [source_root / name for name in names]
        missing = [str(p) for p in examples if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} trees from include_list {include_list!r} not under "
                f"{source_root} (e.g. {missing[:3]})"
            )
        if not examples:
            raise ValueError(f"include_list {include_list!r} is empty")
        # Sort so the seeded shuffle depends only on --seed, not list order.
        return sorted(examples)
    # Sort so the deterministic shuffle below depends only on --seed,
    # not on filesystem walk order (which varies across machines).
    examples = sorted(source_root.rglob("*.pt"))
    if not examples:
        raise ValueError(f"No .pt examples found under {source_root}")
    return examples


def _write_manifest(examples: list[Path], destination: Path) -> None:
    """Write one example path per line to ``destination`` (parents created as needed)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for source in examples:
            handle.write(f"{source}\n")


def main(config: SplitConfig) -> None:
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    source_root = Path(config.source_root)
    split_root = Path(config.split_root)
    train_manifest = split_root / "train_manifest.txt"
    validation_manifest = split_root / "validation_manifest.txt"

    examples = _gather_examples(source_root, config.include_list)
    # Seeded shuffle: identical (source_root, seed) reproduces the same split.
    rng = random.Random(config.seed)
    rng.shuffle(examples)

    # Round to nearest, but always keep at least one validation example so
    # tiny dev datasets still produce a non-empty validation manifest.
    validation_count = max(1, int(round(len(examples) * config.validation_fraction)))
    validation_examples = examples[:validation_count]
    train_examples = examples[validation_count:]
    if not train_examples:
        raise ValueError("Validation split consumed all examples; reduce validation_fraction.")

    # --clear removes stale manifests before writing so a smaller new split
    # can't leave half of the previous run's paths on disk.
    if config.clear:
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
    from cts._config import run_with_config_cli
    run_with_config_cli(SplitConfig, main)
