"""Batched, in-process tree generation (R-BATCHGEN).

The production tree generator (``cts.data.build_tree``) drives our own
sequential PUCT but queries lc0 one position at a time over a UCI text pipe;
~80-85% of the per-tree wall-clock is round-trip and batch-1 launch overhead,
not neural compute. This subpackage replaces that with an **in-process,
batched neural evaluator** and a **frontier-batched search loop** that holds
many independent trees in flight and gathers every tree's required position
evaluations into a single forward pass.

The load-bearing invariant is **search-logic parity** (report §2, layer L1):
for a *fixed* evaluation oracle, the new batched loop must produce a
``PretrainExample`` byte-identical to the legacy
``build_pretrain_example`` sequential loop. That is what lets us swap the
evaluation backend without re-baselining the search. To make the two paths
provably consume identical numbers, the batched loop reuses the very same
per-tree helpers from ``cts.data.preprocess_gnn.teacher_targets``, and
``evaluator_to_provider`` lets the legacy path be driven by the same fixed
``Evaluator``.

Public surface:

- ``evaluator``     — the batched ``Evaluator`` abstraction (mock / cached /
  lc0-UCI-backed) plus ``evaluator_to_provider`` (the parity bridge).
- ``search``        — ``generate_trees_batched``, the frontier driver.
- ``net_evaluator`` — in-process batched lc0 net (Phase 2 scaffold).
- ``encoding``      — lc0 112-plane input encoding (Phase 2 scaffold).

Importing this package must never require the heavy inference libraries
(lczerolens / onnxruntime / torch-for-inference); those are lazy-imported
inside ``net_evaluator`` / ``encoding`` only when a forward pass is run.
"""

from .evaluator import (
    CachedEvaluator,
    Evaluator,
    Lc0UciEvaluator,
    MockEvaluator,
    PositionEval,
    evaluator_to_provider,
)
from .search import generate_trees_batched

__all__ = [
    "CachedEvaluator",
    "Evaluator",
    "Lc0UciEvaluator",
    "MockEvaluator",
    "PositionEval",
    "evaluator_to_provider",
    "generate_trees_batched",
]
