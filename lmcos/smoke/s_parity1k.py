"""S-parity1k: end-to-end trajectory identity, new (net) vs lc0-UCI baseline.

Report §5 ID: **S-parity1k** — "1,000 held-out FENs end-to-end; % trajectory-
identical; divergence characterization". What it proves: the L3 go/no-go
evidence of report §8 — either trajectory identity ~1.0, or every divergence is
traced to a near-tie PUCT decision and is therefore benign (report §2 L3,
§9 risk row). This is the headline parity number for the green-light decision.

Unlike S-1tree/S-Nlock (which pin the *same* eval oracle through both paths to
isolate search logic), this runs the NEW pipeline with its in-process
``NetEvaluator`` against the lc0-UCI baseline — so a divergence can come from
either search logic OR evaluator numerics, which is exactly what L3 is meant to
surface. For each forked tree we report the first expansion index where the
expansion order (and best-move trace) diverges, and the PUCT margin between the
top-two root children at that step as a proxy for "was this a near-tie?".
"""

from __future__ import annotations

import argparse
import random

from _lc0_baseline import build_old_examples, build_search_config
from _smoke_common import (
    DEFAULT_MAX_NODES,
    DEFAULT_MIN_NODES,
    SmokeResult,
    add_common_arguments,
    import_symbol,
    run_smoke,
    sample_fens,
)


def _expansion_move_sequence(example):
    """The ordered list of incoming moves = the tree's expansion trajectory.

    ``iter_nodes`` yields nodes in insertion (= expansion) order, so the move
    sequence is the time axis the controller is trained on. Two trees are
    trajectory-identical iff these sequences match.
    """
    return [node.incoming_move_uci for node in example.tree.iter_nodes()]


def _first_divergence_index(new_example, old_example):
    """First expansion index where the two trajectories differ, or None."""
    new_moves = _expansion_move_sequence(new_example)
    old_moves = _expansion_move_sequence(old_example)
    for index, (new_move, old_move) in enumerate(zip(new_moves, old_moves)):
        if new_move != old_move:
            return index
    if len(new_moves) != len(old_moves):
        return min(len(new_moves), len(old_moves))
    return None


def _root_q_margin_at_step(example, step_index):
    """Top-two root-Q margin at ``step_index`` of the oracle trace (near-tie probe).

    The oracle_root_q_trace is per-step root child Q-values aligned with
    oracle_root_moves. A small margin at the divergence step means PUCT was
    nearly indifferent there, so a forked choice is expected and benign.
    Returns None when the trace doesn't reach ``step_index``.
    """
    trace = example.oracle_root_q_trace
    counts = example.oracle_trace_expansion_counts
    if not trace or not counts:
        return None
    # Find the trace row recorded at/after the divergence expansion count.
    row_index = None
    for index, count in enumerate(counts):
        if count >= step_index:
            row_index = index
            break
    if row_index is None:
        row_index = len(trace) - 1
    row = sorted((float(value) for value in trace[row_index]), reverse=True)
    if len(row) < 2:
        return None
    return row[0] - row[1]


def _build_new_examples(fens, args, search_config, budget_distribution):
    """Full new pipeline: NetEvaluator + batched search."""
    NetEvaluator = import_symbol("net_evaluator", "NetEvaluator")
    generate_trees_batched = import_symbol("search", "generate_trees_batched")

    evaluator = NetEvaluator(args.lc0_weights, device=args.device)
    return generate_trees_batched(
        fens,
        evaluator,
        search_config,
        budget_distribution,
        rng=random.Random(args.seed),
        max_concurrent=args.max_concurrent,
        root_position_ids=[f"root_{index}" for index in range(len(fens))],
    )


def _main(args: argparse.Namespace) -> SmokeResult:
    from cts.data.preprocess_gnn.teacher_targets import NodeBudgetDistribution

    fens = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=args.n,
        seed=args.seed,
    )
    search_config = build_search_config("smoke_parity1k")
    budget_distribution = NodeBudgetDistribution(DEFAULT_MIN_NODES, DEFAULT_MAX_NODES)

    new_examples = _build_new_examples(fens, args, search_config, budget_distribution)
    old_examples = build_old_examples(
        fens,
        lc0_binary=args.lc0_binary,
        lc0_weights=args.lc0_weights,
        search_config=search_config,
        budget_distribution=budget_distribution,
        seed=args.seed,
    )

    identical = 0
    divergences = []
    near_tie_margins = []
    for index, (new_example, old_example) in enumerate(zip(new_examples, old_examples)):
        diverge_at = _first_divergence_index(new_example, old_example)
        if diverge_at is None:
            identical += 1
            continue
        margin = _root_q_margin_at_step(old_example, diverge_at)
        if margin is not None:
            near_tie_margins.append(margin)
        divergences.append(
            {"tree": index, "first_divergence_expansion": diverge_at, "root_q_margin": margin}
        )

    n = len(fens)
    fraction_identical = identical / n if n else float("nan")
    margins_sorted = sorted(near_tie_margins)
    median_margin = margins_sorted[len(margins_sorted) // 2] if margins_sorted else None
    return SmokeResult(
        smoke_id="S-parity1k",
        passed=None,  # go/no-go is a human read of identity + divergence benignity
        summary={
            "n_trees": n,
            "n_trajectory_identical": identical,
            "fraction_identical": round(fraction_identical, 4),
            "n_diverged": len(divergences),
            "median_root_q_margin_at_divergence": median_margin,
            "max_root_q_margin_at_divergence": max(near_tie_margins) if near_tie_margins else None,
            "divergences_sample": divergences[:20],
            "go_note": "report §8 L3: identity ~1.0 OR all divergences near-tie -> GO.",
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=1000)
    parser.add_argument("--device", type=str, default="cuda", help="Net evaluator device (cuda/cpu).")
    parser.add_argument(
        "--max-concurrent", type=int, default=1024, help="Trees in flight in the batched run."
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_smoke(_main, _parse_args()))
