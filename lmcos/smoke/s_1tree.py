"""S-1tree: generate ONE tree the new way vs the old way and diff everything.

Report §5 ID: **S-1tree** — "Generate 1 tree new vs old; diff tree, trace,
targets". What it proves: end-to-end wiring of the batched path plus the L1
(search-logic) and L3 (trajectory) contracts on a single, easy-to-inspect case.

The trap this guards against: a refactor that *runs* but silently produces a
different expansion order or trace. Because the controller's training unit is
the ordered trajectory (report §1.1), a single forked expansion ruins the
example. So we compare node-by-node and step-by-step and print the FIRST
mismatch — the operator wants "diverged at expansion k", not a bare ``!=``.

Both paths drive the *same* lc0-UCI evaluation oracle:
  - OLD: ``build_pretrain_example`` over the legacy ``Lc0DirectEvalProvider``.
  - NEW: ``generate_trees_batched`` over an ``Lc0UciEvaluator`` that wraps the
    *same* ``Lc0DirectEvalProvider`` (so the eval numbers are identical and any
    diff is pure search-logic, isolating L1 from L2 numerics).
"""

from __future__ import annotations

import argparse
import random

from _lc0_baseline import build_search_config, lc0_direct_provider
from _smoke_common import (
    DEFAULT_MAX_NODES,
    DEFAULT_MIN_NODES,
    SmokeResult,
    add_common_arguments,
    diff_pretrain_examples,
    import_symbol,
    run_smoke,
    sample_fens,
)


def _build_old_example(fen, args, search_config, budget_distribution):
    """Generate the baseline example via the legacy sequential lc0-UCI path."""
    from cts.data.preprocess_gnn.teacher_targets import build_pretrain_example

    with lc0_direct_provider(args.lc0_binary, args.lc0_weights) as provider:
        return build_pretrain_example(
            fen,
            provider,
            search_config,
            node_budget_distribution=budget_distribution,
            rng=random.Random(args.seed),
            root_position_id="root_0",
        )


def _build_new_example(fen, args, search_config, budget_distribution):
    """Generate the candidate example via the batched in-process path (N=1).

    ``Lc0UciEvaluator`` adapts the same two-engine lc0 provider to the batched
    ``Evaluator`` interface, so the new path consumes byte-identical lc0 evals.
    """
    Lc0UciEvaluator = import_symbol("evaluator", "Lc0UciEvaluator")
    generate_trees_batched = import_symbol("search", "generate_trees_batched")

    with lc0_direct_provider(args.lc0_binary, args.lc0_weights) as provider:
        evaluator = Lc0UciEvaluator(provider)
        examples = generate_trees_batched(
            [fen],
            evaluator,
            search_config,
            budget_distribution,
            rng=random.Random(args.seed),
            max_concurrent=1,
            root_position_ids=["root_0"],
        )
    return examples[0]


def _main(args: argparse.Namespace) -> SmokeResult:
    from cts.data.preprocess_gnn.teacher_targets import NodeBudgetDistribution

    fen = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=1,
        seed=args.seed,
    )[0]
    search_config = build_search_config("smoke_s1tree")
    budget_distribution = NodeBudgetDistribution(DEFAULT_MIN_NODES, DEFAULT_MAX_NODES)

    old_example = _build_old_example(fen, args, search_config, budget_distribution)
    new_example = _build_new_example(fen, args, search_config, budget_distribution)

    mismatches = diff_pretrain_examples(new_example, old_example)
    return SmokeResult(
        smoke_id="S-1tree",
        passed=not mismatches,
        summary={
            "fen": fen,
            "old_num_nodes": old_example.tree.num_nodes(),
            "new_num_nodes": new_example.tree.num_nodes(),
            "old_trace_len": len(old_example.oracle_best_move_trace),
            "new_trace_len": len(new_example.oracle_best_move_trace),
            "num_mismatches": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "all_mismatches": mismatches,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_smoke(_main, _parse_args()))
