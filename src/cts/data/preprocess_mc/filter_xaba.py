"""Classifies each tree's root-best-move trace into one of three motifs -- "A"
(stable), "X*A" (switches, settles), "X*AB*A" (churns and revisits) -- and keeps
every tree except the "X*AB*A" ones."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from cts.data.preprocess_mc.pack import _source_top_level_root_churn_category


def churn_category(path: Path) -> str | None:
    record = RawPretrainExampleRecord.load(str(path))
    return _source_top_level_root_churn_category(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)  # matches filter_argmax.py -- unconstrained BLAS threading inflates wall-time per-call

    paths = sorted(args.trees_dir.rglob("*.pt"))
    kept = []
    n_skipped = 0
    category_counts: dict[str, int] = {}
    for i, path in enumerate(paths):
        try:
            category = churn_category(path)
        except Exception:
            n_skipped += 1
            continue
        if category is None:
            n_skipped += 1
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
        if category != "X*AB*A":
            kept.append(path.name)
        if (i + 1) % 20000 == 0:
            print(f"[{i + 1}/{len(paths)}] kept={len(kept)} skipped={n_skipped} categories={category_counts}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + ("\n" if kept else ""))
    print(
        f"DONE: {len(kept)}/{len(paths)} kept ({len(kept) / max(1, len(paths)):.3f}), "
        f"{n_skipped} skipped, categories={category_counts} -> {args.output}"
    )


if __name__ == "__main__":
    main()
