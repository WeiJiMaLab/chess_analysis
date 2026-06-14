"""S-Nlock: N trees batched must each equal their sequential counterpart.

Report §5 ID: **S-Nlock** — "N=8 trees in lockstep; assert each == its
sequential counterpart". What it proves: batching changes only the *grouping*
of independent evaluations, never any tree's values or expansion order
(report §1.4). This is the operational sibling of the ``T-batch-inv`` and
``T-replay`` unit tests: if a tree's result depends on which other trees share
its batch, virtual-loss-style cross-talk has crept in and the dataset is
corrupt.

Method: build N baseline examples sequentially (one engine, one tree at a
time), then build the same N via ``generate_trees_batched`` with the batch
wide enough to hold them all in flight at once, and diff per tree. Both paths
thread a single ``random.Random(seed)`` through the FENs in input order (tree
*i*'s budget is the *i*-th draw), so budgets match across paths regardless of
completion order — the same convention as ``build_tree.py``.
"""

from __future__ import annotations

import argparse
import random

from _lc0_baseline import build_old_examples, build_search_config, lc0_direct_provider
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


def _build_new_examples(fens, args, search_config, budget_distribution):
    """Batched path: all N trees in flight at once.

    ``generate_trees_batched`` takes a single ``rng`` and draws one budget per
    tree in input order (tree *i* = *i*-th draw), the same convention the
    sequential baseline uses — so the two paths' budgets line up tree-for-tree.
    """
    Lc0UciEvaluator = import_symbol("evaluator", "Lc0UciEvaluator")
    generate_trees_batched = import_symbol("search", "generate_trees_batched")

    root_ids = [f"root_{index}" for index in range(len(fens))]
    with lc0_direct_provider(args.lc0_binary, args.lc0_weights) as provider:
        evaluator = Lc0UciEvaluator(provider)
        return generate_trees_batched(
            fens,
            evaluator,
            search_config,
            budget_distribution,
            rng=random.Random(args.seed),
            max_concurrent=len(fens),
            root_position_ids=root_ids,
        )


def _main(args: argparse.Namespace) -> SmokeResult:
    from cts.data.preprocess_gnn.teacher_targets import NodeBudgetDistribution

    fens = sample_fens(
        explicit_fens_path=args.fens,
        pool_path=args.fen_pool,
        count=args.n,
        seed=args.seed,
    )
    search_config = build_search_config("smoke_snlock")
    budget_distribution = NodeBudgetDistribution(DEFAULT_MIN_NODES, DEFAULT_MAX_NODES)

    old_examples = build_old_examples(
        fens,
        lc0_binary=args.lc0_binary,
        lc0_weights=args.lc0_weights,
        search_config=search_config,
        budget_distribution=budget_distribution,
        seed=args.seed,
    )
    new_examples = _build_new_examples(fens, args, search_config, budget_distribution)

    per_tree = []
    num_failed = 0
    for index, (new_example, old_example) in enumerate(zip(new_examples, old_examples)):
        mismatches = diff_pretrain_examples(new_example, old_example)
        if mismatches:
            num_failed += 1
        per_tree.append(
            {
                "tree": index,
                "ok": not mismatches,
                "first_mismatch": mismatches[0] if mismatches else None,
            }
        )

    failures = [entry for entry in per_tree if not entry["ok"]]
    return SmokeResult(
        smoke_id="S-Nlock",
        passed=num_failed == 0,
        summary={
            "n_trees": len(fens),
            "n_matched": len(fens) - num_failed,
            "n_mismatched": num_failed,
            "mismatched_trees": failures,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser, default_n=8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_smoke(_main, _parse_args()))
