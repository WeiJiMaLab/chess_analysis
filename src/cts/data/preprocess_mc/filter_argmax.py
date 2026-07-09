""""Thinking helps" filter, run after gen_trees and before pack_trees (see
slurm/pipeline/argmax_filter.slurm + pipeline.yaml). Non-destructive: scans
every generated tree and writes an include_list of survivors' basenames for
split.py to read (config split.include_list) -- gen_trees's own output
directory (human_analysis.trees_default) is left untouched, so
engine_analysis_pwin/cp still see the full, unfiltered population.

For each tree, reconstructs the cost-adjusted return curve at a fixed
BudgetedOracleConfig using the exact same machinery mc_pack uses
(``build_compact_trajectory`` / ``budgeted_oracle_from_trajectory``), and
keeps the tree only if the TRUE oracle argmax step exceeds
``--argmax-threshold`` -- i.e. only positions where continued search
genuinely has a better stop point than the immediate static read. See
labnotebook 2026-07-07: the 900-FEN pilot found ``argmax > 2`` the
highest-yield cutoff (24.7% retained, SingleHalt* k*=7).

Usage: python -m cts.data.preprocess_mc.filter_argmax \
    --trees-dir <dir> --output <include_list.txt>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.data.preprocess_mc.pack import build_compact_trajectory, budgeted_oracle_from_trajectory

# Validated regime (labnotebook 2026-07-07): the fixed-readout, zero-maintenance
# curve is where the filter's yield/k* numbers were measured. Keep in lockstep
# with mc_pack.time_lambda / mc_pack.maintenance_scale if those are ever retuned.
_ORACLE_CONFIG = BudgetedOracleConfig(
    time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0
)
_STARTING_BUDGET = 96


def argmax_step(path: Path) -> int | None:
    record = RawPretrainExampleRecord.load(str(path))
    trajectory = build_compact_trajectory(record, source_path=str(path))
    if trajectory is None or trajectory["num_steps"] <= 0:
        return None
    policy = budgeted_oracle_from_trajectory(trajectory, _STARTING_BUDGET, _ORACLE_CONFIG)
    return int(policy.optimal_stop_step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--argmax-threshold", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(1)  # unconstrained BLAS threading measured ~5x wall-time inflation per-call

    paths = sorted(args.trees_dir.rglob("*.pt"))
    kept = []
    n_skipped = 0
    for i, path in enumerate(paths):
        try:
            step = argmax_step(path)
        except Exception:
            n_skipped += 1
            continue
        if step is None:
            n_skipped += 1
            continue
        if step > args.argmax_threshold:
            kept.append(path.name)
        if (i + 1) % 20000 == 0:
            print(f"[{i + 1}/{len(paths)}] kept={len(kept)} skipped={n_skipped}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + ("\n" if kept else ""))
    print(
        f"DONE: {len(kept)}/{len(paths)} kept ({len(kept) / max(1, len(paths)):.3f}), "
        f"{n_skipped} skipped -> {args.output}"
    )


if __name__ == "__main__":
    main()
