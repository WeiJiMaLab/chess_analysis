"""S-thru: trees/s across batch sizes — the 10-100x speedup claim.

Report §5 ID: **S-thru** — "100 trees at batch sizes {1, 32, 256, 1024};
trees/s + nvidia-smi util". What it proves: the central bet of R-BATCHGEN, that
batching the evaluation across independent trees turns a ~85%-idle accelerator
(batch 1) into a saturated one (large batch), buying the 10-100x trees/s lever
(report §1.2, go/no-go §8). It also picks the operating batch size N.

The batch-size knob is ``max_concurrent`` on ``generate_trees_batched``: it
caps how many trees are in flight, which sets the eval batch width per outer
step (~31 positions per tree). At ``max_concurrent=1`` the path degenerates to
sequential single-position evals — the apples-to-apples stand-in for the
lc0-UCI batch-1 baseline this whole effort replaces.

Uses the in-process ``NetEvaluator`` (batched lc0 net) because the speedup
lives in feeding it a full batch; if the net evaluator is still a scaffold the
script reports that clearly rather than silently measuring nothing.
"""

from __future__ import annotations

import argparse
import random
import time

from _gpu_util import sample_gpu_utilization_percent
from _lc0_baseline import build_search_config
from _smoke_common import (
    DEFAULT_MAX_NODES,
    DEFAULT_MIN_NODES,
    SmokeResult,
    add_common_arguments,
    import_symbol,
    run_smoke,
    sample_fens,
)

DEFAULT_BATCH_SIZES = (1, 32, 256, 1024)


def _build_net_evaluator(args):
    """Construct the in-process batched net evaluator on the requested device."""
    NetEvaluator = import_symbol("net_evaluator", "NetEvaluator")
    return NetEvaluator(args.lc0_weights, device=args.device)


def _time_one_batch_size(
    fens, evaluator, search_config, budget_distribution, max_concurrent, seed
):
    """Generate all trees at one ``max_concurrent`` and return (seconds, util)."""
    generate_trees_batched = import_symbol("search", "generate_trees_batched")
    root_ids = [f"root_{index}" for index in range(len(fens))]

    start = time.perf_counter()
    examples = generate_trees_batched(
        fens,
        evaluator,
        search_config,
        budget_distribution,
        rng=random.Random(seed),
        max_concurrent=max_concurrent,
        root_position_ids=root_ids,
    )
    # Sample GPU util right after the run completes; for a tighter steady-state
    # number an operator can watch ``nvidia-smi dmon`` alongside this script.
    util = sample_gpu_utilization_percent()
    elapsed = time.perf_counter() - start
    evaluator.clear_caches()
    return elapsed, util, len(examples)


def _main(args: argparse.Namespace) -> SmokeResult:
    from cts.data.preprocess_gnn.teacher_targets import NodeBudgetDistribution

    fens = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=args.n,
        seed=args.seed,
    )
    search_config = build_search_config("smoke_sthru")
    budget_distribution = NodeBudgetDistribution(DEFAULT_MIN_NODES, DEFAULT_MAX_NODES)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]

    evaluator = _build_net_evaluator(args)

    rows = []
    print(f"\n{'batch':>8} {'trees':>8} {'seconds':>10} {'trees/s':>10} {'gpu_util%':>10}")
    for max_concurrent in batch_sizes:
        elapsed, util, n_done = _time_one_batch_size(
            fens, evaluator, search_config, budget_distribution, max_concurrent, args.seed
        )
        trees_per_s = n_done / elapsed if elapsed > 0 else float("nan")
        util_str = "n/a" if util is None else f"{util:.0f}"
        print(f"{max_concurrent:>8} {n_done:>8} {elapsed:>10.2f} {trees_per_s:>10.2f} {util_str:>10}")
        rows.append(
            {
                "batch": max_concurrent,
                "trees": n_done,
                "seconds": round(elapsed, 3),
                "trees_per_s": round(trees_per_s, 3),
                "gpu_util_percent": util,
            }
        )

    baseline = next((row for row in rows if row["batch"] == 1), rows[0])
    best = max(rows, key=lambda row: row["trees_per_s"])
    speedup = best["trees_per_s"] / baseline["trees_per_s"] if baseline["trees_per_s"] else float("nan")
    return SmokeResult(
        smoke_id="S-thru",
        passed=None,  # measure-only; go/no-go is a human judgement vs the >=10x target
        summary={
            "n_trees": len(fens),
            "device": args.device,
            "rows": rows,
            "best_batch": best["batch"],
            "best_trees_per_s": best["trees_per_s"],
            "speedup_vs_batch1": round(speedup, 2),
            "go_threshold_note": "report §8: >=10x at a saturating batch is GO; <~5x is ABORT.",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=100)
    parser.add_argument(
        "--batch-sizes", type=str, default=",".join(str(value) for value in DEFAULT_BATCH_SIZES),
        help="Comma-separated max_concurrent values to sweep.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Net evaluator device (cuda/cpu).")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_smoke(_main, _parse_args()))
