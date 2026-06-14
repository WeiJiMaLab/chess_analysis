"""S-cpu: batched-CPU trees/s across thread counts — is the CPU fleet cheaper?

Report §5 ID: **S-cpu** — "Batched-CPU throughput on a short node (fp32 ONNX-RT,
K cores); trees/s". What it proves: whether the 1,400-core CPU fleet is the
cheaper lane than scarce GPUs once the eval is batched (report §1.2 close,
§5 S-cpu, library §6 "onnxruntime runs the same graph on CPU and GPU"). The
GPU lane wins on raw throughput, but if CPU trees/s-per-dollar beats the GPU
queue wait, the fleet is the pragmatic answer.

Sweep: for each thread count K, set the net evaluator to use K CPU threads and
time generating the tree set. We pin the native thread-pool env vars
(``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``) — the
knobs ONNX-RT and torch BLAS actually read at first use — so the CPU lane uses
exactly K cores. The ONNX backend is chosen because it is the CPU-fleet lane
(report §6: "onnxruntime runs the same graph on CPU and GPU").

NOTE: thread-count env vars are read by native libs at first use, so this script
times each K in a *fresh subprocess of itself* (``--worker``) to make the pin
actually take effect. Without the subprocess, the first K would lock the pools.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

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

DEFAULT_THREAD_COUNTS = (1, 2, 4, 8, 16)


def _run_worker(args) -> None:
    """Child mode: generate the trees with the (already-pinned) thread count, emit JSON.

    Runs in a subprocess so OMP/MKL thread pools are sized at import time to the
    requested K. Prints a single JSON line to stdout that the parent parses.
    """
    from cts.data.preprocess_gnn.teacher_targets import NodeBudgetDistribution

    NetEvaluator = import_symbol("net_evaluator", "NetEvaluator")
    NetBackend = import_symbol("net_evaluator", "NetBackend")
    generate_trees_batched = import_symbol("search", "generate_trees_batched")

    fens = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=args.n,
        seed=args.seed,
    )
    search_config = build_search_config("smoke_scpu")
    budget_distribution = NodeBudgetDistribution(DEFAULT_MIN_NODES, DEFAULT_MAX_NODES)
    # Thread count is set via the OMP/MKL env vars the parent pinned before
    # spawning this subprocess (native libs read them at first use).
    evaluator = NetEvaluator(args.lc0_weights, backend=NetBackend.ONNX, device="cpu")

    start = time.perf_counter()
    examples = generate_trees_batched(
        fens,
        evaluator,
        search_config,
        budget_distribution,
        rng=random.Random(args.seed),
        max_concurrent=args.max_concurrent,
        root_position_ids=[f"root_{index}" for index in range(len(fens))],
    )
    elapsed = time.perf_counter() - start
    print(json.dumps({"threads": args.worker_threads, "trees": len(examples), "seconds": elapsed}))


def _time_thread_count(args, threads: int) -> dict:
    """Spawn a worker subprocess with the thread pools pinned to ``threads``."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    command = [
        sys.executable, os.path.abspath(__file__),
        "--worker", "--worker-threads", str(threads),
        "--n", str(args.n),
        "--fen-pool", args.fen_pool,
        "--lc0-weights", args.lc0_weights,
        "--lc0-binary", args.lc0_binary,
        "--seed", str(args.seed),
        "--max-concurrent", str(args.max_concurrent),
    ]
    if args.fens:
        command += ["--fens", args.fens]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
    # The worker may print scaffold messages; the result is the last JSON line.
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"Worker produced no result line. stderr:\n{completed.stderr}")


def _main(args: argparse.Namespace) -> SmokeResult:
    thread_counts = [int(value) for value in args.threads.split(",")]
    rows = []
    print(f"\n{'threads':>8} {'trees':>8} {'seconds':>10} {'trees/s':>10}")
    for threads in thread_counts:
        result = _time_thread_count(args, threads)
        trees_per_s = result["trees"] / result["seconds"] if result["seconds"] > 0 else float("nan")
        print(f"{threads:>8} {result['trees']:>8} {result['seconds']:>10.2f} {trees_per_s:>10.2f}")
        rows.append(
            {
                "threads": threads,
                "trees": result["trees"],
                "seconds": round(result["seconds"], 3),
                "trees_per_s": round(trees_per_s, 3),
            }
        )

    best = max(rows, key=lambda row: row["trees_per_s"])
    return SmokeResult(
        smoke_id="S-cpu",
        passed=None,  # measure-only; lane choice is a cost comparison vs S-thru GPU
        summary={
            "n_trees": args.n,
            "rows": rows,
            "best_threads": best["threads"],
            "best_trees_per_s": best["trees_per_s"],
            "lane_note": "Compare best_trees_per_s * fleet_cores vs the GPU lane (S-thru).",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=100)
    parser.add_argument(
        "--threads", type=str, default=",".join(str(value) for value in DEFAULT_THREAD_COUNTS),
        help="Comma-separated CPU thread counts to sweep.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=256, help="Trees in flight per outer step."
    )
    # Internal: the parent re-invokes this script in a thread-pinned subprocess.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    if _args.worker:
        # Child path: do the timed run and emit JSON; no SmokeResult envelope.
        _run_worker(_args)
    else:
        raise SystemExit(run_smoke(_main, _args))
