"""Select "informative-stopping" trees by a post-hoc read of the search's best-move trace.

A tree's ``oracle_best_move_index[i]`` is the argmax root move after expansion ``i+1``; the
trace has one entry per expansion (96 under the canonical budget). We gate the tree set used
to train BOTH the GNN encoder and the MC controller (matching ysagiv's "train GNN+MC on the
filtered subset" pipeline) by the **INTERSECTION** of two tree-level criteria:

1. **PUCT-stability** — *search materially changes the chosen action*::

       bmi[0] != bmi[-1]   AND   bmi[mid] != bmi[-1]          (mid = len // 2)

   Drops trees whose best move is already settled at the first expansion or by the midpoint
   (no deliberation needed). This is a post-hoc 96-step read of ysagiv's PUCT-stability filter
   (`oracle96_trace_filtered`); the original legacy variant ran a separate 16-node search.

2. **Monotone-convergence** — *once the eventual-best is first found, it is never abandoned*::

       gss = argmax(bmi == bmi[-1]);   require all(bmi[gss:] == bmi[-1])

   Drops "found-then-lost" oscillators (e.g. ``1 2 3 [4] 3 2 1 4 4 4`` — the 4 is hit
   transiently at step 3, lost, then re-found) where the greedy first-found stop is unreliable.

The episode-level **halt-reward-range** filter (``min_halt_reward_range``) is applied
SEPARATELY at MC-pack time, on top of this tree set.

Empirically on the 405K ``human_trees``: PUCT-stability ~32%, monotone ~65%, **intersection ~16%**.

    python -m cts.data.filter_trees_by_trace --config filter_trees.yaml
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


class FilterTreesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trees_dir: str = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees"
    output_dir: str  # writes clean_trees.txt (basenames) + filter_stats.json
    n_workers: int = 72
    min_steps: int = 3
    # Array-shard mode (CPU array job): when shard_index >= 0, this process
    # classifies only the trees[shard_index::num_shards] slice and writes a
    # shard-level shards/clean_trees_shard_<idx>.txt; merge_and_pack.slurm later
    # concatenates the shards into the final clean_trees.txt. When shard_index < 0
    # (default) the original single-process whole-set behaviour is used.
    num_shards: int = 1
    shard_index: int = -1  # skip degenerate traces shorter than this


def _classify(path: str) -> tuple[str, bool, bool]:
    """Return (basename, puct_stable, monotone) for one tree; (name, False, False) on error."""
    name = os.path.basename(path)
    try:
        bmi = np.asarray(torch.load(path, weights_only=False)["oracle_best_move_index"]).ravel()
    except Exception:
        return name, False, False
    if bmi.size < 3:
        return name, False, False
    final = bmi[-1]
    puct = bool(bmi[0] != final and bmi[bmi.size // 2] != final)
    gss = int(np.argmax(bmi == final))
    monotone = bool(np.all(bmi[gss:] == final))
    return name, puct, monotone


def main(config: FilterTreesConfig) -> None:
    trees = sorted(str(p) for p in Path(config.trees_dir).glob("*.pt"))
    out = Path(config.output_dir)

    if config.shard_index >= 0:
        _run_shard(config, trees, out)
        return

    out.mkdir(parents=True, exist_ok=True)
    print(f"[filter-trees] classifying {len(trees):,} trees with {config.n_workers} workers", flush=True)

    clean, n_puct, n_mono, n_both = [], 0, 0, 0
    with ProcessPoolExecutor(max_workers=config.n_workers) as ex:
        for i, (name, puct, mono) in enumerate(ex.map(_classify, trees, chunksize=64), 1):
            n_puct += puct
            n_mono += mono
            if puct and mono:
                n_both += 1
                clean.append(name)
            if i % 50000 == 0:
                print(f"[filter-trees] {i:,}/{len(trees):,} clean_so_far={len(clean):,}", flush=True)

    (out / "clean_trees.txt").write_text("\n".join(sorted(clean)) + "\n")
    n = len(trees)
    stats = {
        "trees_dir": config.trees_dir,
        "n_total": n,
        "puct_stability": n_puct,
        "monotone_convergence": n_mono,
        "intersection_clean": n_both,
        "frac_puct": n_puct / n,
        "frac_monotone": n_mono / n,
        "frac_clean": n_both / n,
        "criterion": "INTERSECTION(bmi[0]!=final and bmi[mid]!=final ; all(bmi[gss:]==final)); mid=len//2",
    }
    (out / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[filter-trees] clean={n_both:,}/{n:,} ({100*n_both/n:.1f}%) "
          f"[puct {100*n_puct/n:.1f}% · monotone {100*n_mono/n:.1f}%] -> {out}/clean_trees.txt", flush=True)


def _run_shard(config: FilterTreesConfig, trees: list[str], out: Path) -> None:
    """Classify only the trees[shard_index::num_shards] slice (CPU array task).

    Writes shards/clean_trees_shard_<idx>.txt (basenames) so a downstream merge
    can concatenate the shards into the canonical clean_trees.txt. Per-shard
    counts go to shards/shard_<idx>_stats.json for spot-checking.
    """
    if not 0 <= config.shard_index < config.num_shards:
        raise ValueError(
            f"shard_index={config.shard_index} out of range for num_shards={config.num_shards}"
        )
    shard = trees[config.shard_index :: config.num_shards]
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[filter-trees] shard {config.shard_index}/{config.num_shards}: "
        f"{len(shard):,}/{len(trees):,} trees, {config.n_workers} worker(s)",
        flush=True,
    )

    clean, n_puct, n_mono, n_both = [], 0, 0, 0
    if config.n_workers <= 1:
        results = (_classify(p) for p in shard)
    else:
        ex = ProcessPoolExecutor(max_workers=config.n_workers)
        results = ex.map(_classify, shard, chunksize=64)
    for name, puct, mono in results:
        n_puct += puct
        n_mono += mono
        if puct and mono:
            n_both += 1
            clean.append(name)
    if config.n_workers > 1:
        ex.shutdown()

    (shard_dir / f"clean_trees_shard_{config.shard_index}.txt").write_text(
        "\n".join(sorted(clean)) + "\n"
    )
    n = len(shard)
    stats = {
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
        "n_shard": n,
        "puct_stability": n_puct,
        "monotone_convergence": n_mono,
        "intersection_clean": n_both,
    }
    (shard_dir / f"shard_{config.shard_index}_stats.json").write_text(json.dumps(stats, indent=2))
    print(
        f"[filter-trees] shard {config.shard_index}: clean={n_both:,}/{n:,} "
        f"-> {shard_dir}/clean_trees_shard_{config.shard_index}.txt",
        flush=True,
    )


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(FilterTreesConfig, main)
