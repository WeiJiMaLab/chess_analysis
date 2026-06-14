"""Shared lc0-UCI baseline construction for the parity smoke tests.

WHY a shared module: S-1tree, S-Nlock, and S-parity1k all need to build the
*legacy* ``PretrainExample`` via the two-engine ``Lc0DirectEvalProvider`` exactly
as ``build_tree.py``'s ``generate-dataset`` does, and they all need the matching
``TeacherSearchConfig`` / ``NodeBudgetDistribution``. Duplicating the engine
wiring three times would invite drift between the baseline a smoke test compares
against and the baseline scale-up actually produces. Defined once here.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator, List, Optional

from _smoke_common import (
    DEFAULT_C_PUCT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MULTIPV,
    DEFAULT_SEARCH_BUDGET,
)


def build_search_config(search_config_id: str):
    """TeacherSearchConfig mirroring build_tree.py generate-dataset defaults."""
    from cts.data.preprocess_gnn.teacher_targets import TeacherSearchConfig

    return TeacherSearchConfig(
        max_depth=DEFAULT_MAX_DEPTH,
        search_budget=DEFAULT_SEARCH_BUDGET,
        c_puct=DEFAULT_C_PUCT,
        prior_feature="prior",
        value_feature="value",
        target_normalization_version="v1",
        search_config_id=search_config_id,
    )


@contextmanager
def lc0_direct_provider(lc0_binary: str, lc0_weights: str) -> Iterator[object]:
    """Yield a two-engine ``Lc0DirectEvalProvider`` (classic priors + valuehead).

    Matches build_tree.py exactly: a classic-mode engine with multipv for
    sibling priors via verbose-move-stats, and a valuehead-mode engine with
    ``UCI_ShowWDL`` for the position value. Both run ``nodes=1`` (oracle, not
    search). Engines are torn down on context exit.
    """
    from cts.core.providers import (
        Lc0DirectEvalProvider,
        UciEngineConfig,
        UciEngineProcess,
    )

    prior_config = UciEngineConfig(
        engine_path=lc0_binary,
        engine_kind="lc0",
        engine_mode="classic",
        movetime_ms=0,
        multipv=DEFAULT_MULTIPV,
        depth=None,
        nodes=1,
        weights_path=lc0_weights,
        set_multipv=False,
        enable_verbose_move_stats=True,
    )
    value_config = UciEngineConfig(
        engine_path=lc0_binary,
        engine_kind="lc0",
        engine_mode="valuehead",
        movetime_ms=0,
        multipv=1,
        depth=None,
        nodes=1,
        weights_path=lc0_weights,
        uci_options={"UCI_ShowWDL": "true"},
        set_multipv=False,
        enable_verbose_move_stats=False,
    )
    with UciEngineProcess(prior_config) as prior_engine, UciEngineProcess(value_config) as value_engine:
        yield Lc0DirectEvalProvider(prior_engine, value_engine)


def build_old_examples(
    fens: List[str],
    *,
    lc0_binary: str,
    lc0_weights: str,
    search_config,
    budget_distribution,
    seed: int,
) -> list:
    """Sequential baseline: one ``build_pretrain_example`` per FEN.

    A single ``random.Random(seed)`` is threaded through the FENs in input
    order — exactly how ``build_tree.py`` generate-dataset and
    ``generate_trees_batched`` sample per-tree budgets (tree *i*'s budget is the
    *i*-th draw). This is what keeps the sequential baseline byte-comparable to
    the batched path; a per-tree ``seed + index`` scheme would NOT match the
    production generator. Search itself is rng-free, so one draw per tree.
    """
    from cts.data.preprocess_gnn.teacher_targets import build_pretrain_example

    examples = []
    with lc0_direct_provider(lc0_binary, lc0_weights) as provider:
        rng = random.Random(seed)
        for index, fen in enumerate(fens):
            example = build_pretrain_example(
                fen,
                provider,
                search_config,
                node_budget_distribution=budget_distribution,
                rng=rng,
                root_position_id=f"root_{index}",
            )
            examples.append(example)
            provider.clear_caches()
    return examples
