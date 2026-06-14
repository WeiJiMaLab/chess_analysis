"""S-mem: peak RSS with N=1000 trees live — the continuous-batching budget.

Report §5 ID: **S-mem** — "N=1,000 live trees; peak RSS". What it proves: that
holding a wide batch of trees in flight (the prerequisite for continuous
batching, report §1.3) fits in memory. Report §9 claims trees are tiny
(~hundreds of bytes/node) so N can be thousands; this measures it instead of
assuming it, so the operator can size ``max_concurrent`` for the real run.

RSS sampling degrades gracefully: prefer ``psutil`` (gives a clean current RSS),
fall back to ``resource.getrusage`` (peak RSS, ``ru_maxrss`` — KB on Linux),
and finally parse ``/proc/self/status`` (``VmHWM``). We report the peak observed
across a few samples taken while the trees are alive.
"""

from __future__ import annotations

import argparse
import random

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


def _current_rss_mb():
    """Best-effort current RSS in MB, trying psutil -> /proc -> None."""
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024), "psutil"
    except Exception:
        pass
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0, "proc"
    except OSError:
        pass
    return None, "unavailable"


def _peak_rss_mb():
    """Peak RSS in MB via resource.getrusage (ru_maxrss is KB on Linux)."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _main(args: argparse.Namespace) -> SmokeResult:
    from cts.data.preprocess_gnn.teacher_targets import NodeBudgetDistribution

    NetEvaluator = import_symbol("net_evaluator", "NetEvaluator")
    generate_trees_batched = import_symbol("search", "generate_trees_batched")

    fens = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=args.n,
        seed=args.seed,
    )
    search_config = build_search_config("smoke_smem")
    budget_distribution = NodeBudgetDistribution(DEFAULT_MIN_NODES, DEFAULT_MAX_NODES)

    rss_before, source = _current_rss_mb()
    evaluator = NetEvaluator(args.lc0_weights, device=args.device)

    # Force the whole batch to be in flight simultaneously: max_concurrent >= N
    # means no tree finishes-and-frees before the others start, so the live-set
    # high-water mark is what we measure.
    examples = generate_trees_batched(
        fens,
        evaluator,
        search_config,
        budget_distribution,
        rng=random.Random(args.seed),
        max_concurrent=len(fens),
        root_position_ids=[f"root_{index}" for index in range(len(fens))],
    )
    rss_after, _ = _current_rss_mb()

    # Keep the examples referenced until after sampling so they count toward RSS.
    total_nodes = sum(example.tree.num_nodes() for example in examples)
    peak_mb = _peak_rss_mb()

    rss_delta = None
    if rss_before is not None and rss_after is not None:
        rss_delta = round(rss_after - rss_before, 1)
    return SmokeResult(
        smoke_id="S-mem",
        passed=None,  # measure-only
        summary={
            "n_trees": len(fens),
            "total_nodes": total_nodes,
            "rss_source": source,
            "rss_before_mb": None if rss_before is None else round(rss_before, 1),
            "rss_after_mb": None if rss_after is None else round(rss_after, 1),
            "rss_delta_mb": rss_delta,
            "peak_rss_mb_ru_maxrss": round(peak_mb, 1),
            "bytes_per_node_estimate": (
                None if rss_delta is None or total_nodes == 0
                else round(rss_delta * 1024 * 1024 / total_nodes, 1)
            ),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=1000)
    parser.add_argument("--device", type=str, default="cuda", help="Net evaluator device (cuda/cpu).")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_smoke(_main, _parse_args()))
