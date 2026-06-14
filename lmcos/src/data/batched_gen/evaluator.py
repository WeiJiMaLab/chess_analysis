"""The batched evaluation abstraction that the new tree generator drives.

An :class:`Evaluator` turns a batch of FENs into a list of :class:`PositionEval`
(side-to-move priors + value + WDL) in one call. This is the dimension we
batch over the accelerator (report §1.2-1.3): selection/expansion/backprop stay
per-tree on the CPU, but every tree's required evaluations are gathered into a
single ``evaluate([...])`` call.

The pivotal piece for parity is :func:`evaluator_to_provider`. It wraps an
``Evaluator`` in an object satisfying the *exact* provider interface that the
legacy ``build_pretrain_example`` consumes (``root_features`` / ``expand_node``
/ ``clear_caches``), and it builds the per-node feature dicts with the *same*
helper functions the production ``Lc0DirectEvalProvider`` uses
(``value_features_from_wdl`` / ``terminal_value_features``). So the legacy
sequential loop and the new batched loop both ultimately read their numbers
from the same ``Evaluator`` and shape them identically — which is what makes
the L1 byte-exact replay test (report §4, T-replay) achievable.

Concrete evaluators here:

- :class:`MockEvaluator` — deterministic synthetic outputs (stable hash of the
  FEN), enumerating legal moves with python-chess. Reproducible across runs
  and processes, so it can stand in for lc0 in the parity tests.
- :class:`CachedEvaluator` — records an inner evaluator's outputs keyed by FEN
  and replays them deterministically (the "fixed evaluation oracle" of §2).
- :class:`Lc0UciEvaluator` — adapts the existing ``Lc0DirectEvalProvider`` so
  the batched loop can use the real engines (baseline / Phase 0.2).
"""

from __future__ import annotations

import abc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from cts.core.providers.base import TreeExpansionProvider
from cts.core.providers.common import (
    board_from_position_spec,
    terminal_value_from_position_spec,
)
from cts.core.providers.parsers import terminal_value_features, value_features_from_wdl
from cts.core.tree import ExpansionChild


@dataclass(frozen=True)
class PositionEval:
    """A single position's neural-evaluation result, side-to-move perspective.

    ``priors`` are the **raw** policy-head scores per legal ``move_uci`` (not
    normalized); the search loop normalizes them per-parent via the existing
    ``normalize_prior_scores`` helper, exactly as the legacy provider's priors
    are normalized. ``value`` is win-loss in ``[-1, 1]`` and ``wdl`` is the
    win/draw/loss triple summing to 1 — both from the side-to-move's view.
    """

    priors: Dict[str, float]
    value: float
    wdl: tuple  # (win, draw, loss), sums to 1, side-to-move perspective

    def __post_init__(self) -> None:
        # Validate at the boundary so a malformed eval fails here rather than
        # deep inside the search loop or the on-disk record validator.
        if len(self.wdl) != 3:
            raise ValueError("PositionEval.wdl must have length 3.")
        if not all(math.isfinite(component) for component in self.wdl):
            raise ValueError("PositionEval.wdl components must be finite.")
        if not math.isfinite(self.value):
            raise ValueError("PositionEval.value must be finite.")


class Evaluator(abc.ABC):
    """Batched neural evaluator: a list of FENs -> a list of ``PositionEval``."""

    @abc.abstractmethod
    def evaluate(self, fens: Sequence[str]) -> List[PositionEval]:
        """Return one ``PositionEval`` per input FEN, in input order."""
        raise NotImplementedError

    def clear_caches(self) -> None:
        """Release any per-position caches. Default: no-op."""
        return None


# --- The parity bridge: an Evaluator behind the legacy provider interface ---


def _value_features_from_eval(position_eval: PositionEval) -> Dict[str, float]:
    """Shape a ``PositionEval`` into the canonical value-feature dict.

    Uses the *same* helpers as ``Lc0DirectEvalProvider`` so the feature dict
    is byte-identical to what the legacy path would store on the node:
    ``value`` + ``wdl_win/draw/loss`` + ``wdl_var``.
    """
    win, draw, loss = position_eval.wdl
    return value_features_from_wdl(float(win), float(draw), float(loss))


class _EvaluatorBackedProvider(TreeExpansionProvider):
    """Legacy ``TreeExpansionProvider`` whose numbers come from an ``Evaluator``.

    This is the bridge that lets the *unmodified* sequential
    ``build_pretrain_example`` be driven by the very same fixed ``Evaluator``
    as the new batched loop. ``root_features`` and ``expand_node`` reproduce,
    feature-for-feature, what ``Lc0DirectEvalProvider`` produces, but read the
    priors/value/WDL from ``evaluator.evaluate`` instead of from lc0 over UCI.

    A terminal child short-circuits the value lookup exactly as the lc0
    provider does (``terminal_value_features`` from the known game result), so
    the two providers agree even on mate/stalemate leaves.
    """

    def __init__(self, evaluator: Evaluator, metadata: Optional[Mapping[str, str]] = None) -> None:
        self._evaluator = evaluator
        self._metadata = dict(metadata or {})
        # Cache the per-FEN eval so a position queried as a child and later as
        # a parent costs a single ``evaluate`` call — matching the lc0
        # provider's per-FEN memoization and keeping eval counts identical.
        self._eval_cache: Dict[str, PositionEval] = {}

    def _eval_for_fen(self, fen: str) -> PositionEval:
        if fen not in self._eval_cache:
            self._eval_cache[fen] = self._evaluator.evaluate([fen])[0]
        return self._eval_cache[fen]

    def seed_eval_cache(self, fen: str, position_eval: PositionEval) -> None:
        """Inject a pre-computed eval so a later ``expand_node`` is a cache hit.

        The frontier driver gathers a whole step's child FENs, evaluates them in
        one batched call, then seeds each tree's provider with the results so
        the subsequent per-tree expansion never re-hits the evaluator. This is
        what makes the single batched ``evaluate`` the *only* forward pass for
        the step, for any evaluator (not just self-caching ones).
        """
        self._eval_cache.setdefault(fen, position_eval)

    def root_features(self, fen: str) -> Mapping[str, float]:
        """Root feature dict: value/WDL features plus the sentinel ``prior=1.0``.

        Mirrors ``Lc0DirectEvalProvider.root_features``: a terminal root uses
        the degenerate WDL from its known result; otherwise the evaluator's
        value/WDL. The root has no incoming move, so its prior is the
        convention ``1.0``.
        """
        terminal_value = terminal_value_from_position_spec(fen)
        if terminal_value is not None:
            features = terminal_value_features(terminal_value)
        else:
            features = _value_features_from_eval(self._eval_for_fen(fen))
        return {**features, "prior": 1.0}

    def expand_node(
        self,
        fen: str,
        depth: int,
        max_children: Optional[int] = None,
    ) -> Sequence[ExpansionChild]:
        """One ``ExpansionChild`` per legal move, mirroring the lc0 provider.

        The child's raw prior is read from the *parent's* policy head
        (``parent_eval.priors[move]``); its value/WDL is read from the
        *child's* evaluation, except for terminal children whose value is
        known exactly and which skip the evaluator entirely.
        """
        parent_eval = self._eval_for_fen(fen)
        parent_board = board_from_position_spec(fen)
        moves = self._legal_moves_in_prior_order(fen, parent_eval)
        if max_children is not None:
            moves = moves[:max_children]

        children: List[ExpansionChild] = []
        for move_uci in moves:
            child_fen = self._child_position_spec(fen, move_uci)
            scalar_features: Dict[str, float] = {"prior": float(parent_eval.priors[move_uci])}
            terminal_value = self._child_terminal_value(parent_board, move_uci, child_fen)
            if terminal_value is not None:
                scalar_features.update(terminal_value_features(terminal_value))
                children.append(
                    ExpansionChild(
                        move_uci=move_uci,
                        fen=child_fen,
                        scalar_features=scalar_features,
                        metadata={},
                        is_terminal=True,
                    )
                )
                continue
            scalar_features.update(_value_features_from_eval(self._eval_for_fen(child_fen)))
            children.append(
                ExpansionChild(
                    move_uci=move_uci,
                    fen=child_fen,
                    scalar_features=scalar_features,
                    metadata={},
                    is_terminal=False,
                )
            )
        return children

    def nonterminal_child_fens(self, fen: str) -> List[str]:
        """Child FENs that ``expand_node`` will need the evaluator to score.

        Lets the batched driver discover the per-step evaluation set *without*
        triggering any per-child ``evaluate`` call: it reuses the cached parent
        eval (already present once the parent was itself evaluated) for the move
        list, then filters out terminal children whose value is known exactly.
        The batched loop warms these FENs in one ``evaluate`` call, after which
        the real ``expand_node`` reads them all from cache.
        """
        parent_eval = self._eval_for_fen(fen)
        parent_board = board_from_position_spec(fen)
        child_fens: List[str] = []
        for move_uci in self._legal_moves_in_prior_order(fen, parent_eval):
            child_fen = self._child_position_spec(fen, move_uci)
            if self._child_terminal_value(parent_board, move_uci, child_fen) is None:
                child_fens.append(child_fen)
        return child_fens

    def provider_metadata(self) -> Mapping[str, str]:
        return dict(self._metadata)

    def clear_caches(self) -> None:
        self._eval_cache.clear()
        self._evaluator.clear_caches()

    @staticmethod
    def _legal_moves_in_prior_order(fen: str, position_eval: PositionEval) -> List[str]:
        """Return the legal-move UCIs this provider will expand, in a stable order.

        ``priors`` keys are exactly the legal moves the evaluator scored. We
        order them by the canonical legal-move enumeration of python-chess so
        the order is deterministic and engine-independent (the teacher then
        sorts children by UCI for the on-disk slot order anyway).
        """
        board = board_from_position_spec(fen)
        if board is None:
            return sorted(position_eval.priors)
        ordered = [move.uci() for move in board.legal_moves if move.uci() in position_eval.priors]
        # Defensive: include any scored move python-chess didn't enumerate
        # (should not happen for a consistent evaluator) so we never silently
        # drop a prior.
        ordered.extend(sorted(set(position_eval.priors) - set(ordered)))
        return ordered

    @staticmethod
    def _child_position_spec(fen: str, move_uci: str) -> str:
        from cts.core.providers.common import append_move_to_position_spec

        return append_move_to_position_spec(fen, move_uci)

    @staticmethod
    def _child_terminal_value(parent_board, move_uci: str, child_fen: str) -> Optional[float]:
        """Side-to-move terminal value of a child, or None if it is non-terminal."""
        from cts.core.providers.common import chess, terminal_value_from_board

        if parent_board is not None and chess is not None:
            try:
                child_board = parent_board.copy(stack=False)
                child_board.push_uci(move_uci)
                return terminal_value_from_board(child_board)
            except ValueError:
                pass
        return terminal_value_from_position_spec(child_fen)


def evaluator_to_provider(
    evaluator: Evaluator,
    metadata: Optional[Mapping[str, str]] = None,
) -> TreeExpansionProvider:
    """Wrap an ``Evaluator`` in the legacy provider interface.

    The returned object can be passed straight to ``build_pretrain_example``,
    driving the *unchanged* sequential search from the same fixed evaluation
    oracle that the batched loop consumes. This is the foundation of the
    byte-exact replay test (report §4, T-replay): identical numbers in, so any
    output difference is a search-logic bug, not an evaluator difference.
    """
    return _EvaluatorBackedProvider(evaluator, metadata=metadata)


# --- Concrete evaluators ---


def _stable_unit_float(*parts: str) -> float:
    """Deterministic float in ``[0, 1)`` from a stable (cross-process) hash.

    Uses ``hashlib`` rather than Python's salted ``hash()`` so the same inputs
    yield the same number across runs and processes — a hard requirement for
    the mock to back reproducible parity tests.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    # Take 8 bytes as a 64-bit integer and scale into [0, 1).
    raw = int.from_bytes(digest[:8], "big")
    return raw / float(1 << 64)


class MockEvaluator(Evaluator):
    """Deterministic synthetic evaluator for tests and parity replay.

    For FENs present in an explicit ``table`` the stored ``PositionEval`` is
    returned verbatim. For everything else a reproducible ``PositionEval`` is
    derived from a stable hash of the FEN: one prior per *legal* move
    (enumerated with python-chess) and a value/WDL drawn deterministically
    from the same hash. The outputs depend only on the FEN, so two processes
    (the legacy and the batched paths) see identical numbers.
    """

    def __init__(self, table: Optional[Dict[str, PositionEval]] = None) -> None:
        self._table = dict(table or {})

    def evaluate(self, fens: Sequence[str]) -> List[PositionEval]:
        return [self._evaluate_one(fen) for fen in fens]

    def _evaluate_one(self, fen: str) -> PositionEval:
        if fen in self._table:
            return self._table[fen]
        return self._synthesize(fen)

    @staticmethod
    def _synthesize(fen: str) -> PositionEval:
        board = board_from_position_spec(fen)
        priors: Dict[str, float] = {}
        if board is not None:
            for move in board.legal_moves:
                move_uci = move.uci()
                # Raw, unnormalized prior in (0, 1]; the search loop normalizes.
                priors[move_uci] = _stable_unit_float(fen, "prior", move_uci) + 1e-6
        # Value/WDL from the same hash family. Build a positive WDL triple and
        # normalize it; derive value as win - loss so the two are consistent.
        win = _stable_unit_float(fen, "wdl", "win") + 1e-6
        draw = _stable_unit_float(fen, "wdl", "draw") + 1e-6
        loss = _stable_unit_float(fen, "wdl", "loss") + 1e-6
        total = win + draw + loss
        win, draw, loss = win / total, draw / total, loss / total
        value = win - loss
        return PositionEval(priors=priors, value=value, wdl=(win, draw, loss))


class CachedEvaluator(Evaluator):
    """Persistently caches an inner evaluator's outputs, keyed by FEN.

    On construction it loads any previously recorded outputs from ``path``; on
    a cache miss it calls the inner evaluator, stores the result, and flushes
    the cache to ``path``. Replaying a fully-populated cache is deterministic
    and inner-evaluator-free, which is exactly the "fixed evaluation oracle"
    the parity tests (report §2) replay through both search loops.

    The on-disk format is JSON (one object per FEN) so a recorded cache is
    human-inspectable and portable across machines/backends.
    """

    def __init__(self, inner: Evaluator, cache_path: str) -> None:
        self._inner = inner
        self._path = Path(cache_path)
        self._cache: Dict[str, PositionEval] = {}
        if self._path.exists():
            self._load()

    def evaluate(self, fens: Sequence[str]) -> List[PositionEval]:
        missing = [fen for fen in fens if fen not in self._cache]
        if missing:
            # Batch the misses through the inner evaluator in one call so the
            # cache-population path itself benefits from batching.
            for fen, position_eval in zip(missing, self._inner.evaluate(missing)):
                self._cache[fen] = position_eval
            self._flush()
        return [self._cache[fen] for fen in fens]

    def clear_caches(self) -> None:
        # Only the inner evaluator's volatile state is cleared; the persisted
        # replay cache is the whole point of this class and is kept.
        self._inner.clear_caches()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for fen, entry in payload.get("evals", {}).items():
            self._cache[fen] = PositionEval(
                priors={str(move): float(prior) for move, prior in entry["priors"].items()},
                value=float(entry["value"]),
                wdl=tuple(float(component) for component in entry["wdl"]),
            )

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "cts_batched_gen_eval_cache_v1",
            "evals": {
                fen: {
                    "priors": dict(position_eval.priors),
                    "value": position_eval.value,
                    "wdl": list(position_eval.wdl),
                }
                for fen, position_eval in self._cache.items()
            },
        }
        # Write atomically (temp + replace) so a crash mid-flush can't corrupt
        # an existing recorded cache.
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, self._path)


class Lc0UciEvaluator(Evaluator):
    """Adapt the existing ``Lc0DirectEvalProvider`` to the batched interface.

    This is the Phase 0.2 baseline: it produces ``PositionEval`` from the real
    lc0 engines, so the batched loop and the cached/mock replays all speak the
    same currency. Evaluation is still batch-1 under the hood (the provider
    queries lc0 over UCI one FEN at a time) — that is exactly the overhead the
    in-process ``NetEvaluator`` is meant to remove — but wrapping it here lets
    us *record* a ``CachedEvaluator`` from genuine lc0 outputs for parity work.

    ``priors`` are recovered as the per-child raw priors the provider parsed
    (already in ``[0, 1]``), and ``value``/``wdl`` from the provider's value
    features for the position.
    """

    def __init__(self, provider) -> None:
        # Typed loosely to avoid importing the lc0 module at package import
        # time; any object with the ``Lc0DirectEvalProvider`` surface works.
        self._provider = provider

    def evaluate(self, fens: Sequence[str]) -> List[PositionEval]:
        return [self._evaluate_one(fen) for fen in fens]

    def _evaluate_one(self, fen: str) -> PositionEval:
        # Children carry the per-move priors; the parent's value features give
        # value + WDL. Terminal positions have no children to expand, so fall
        # back to the root features (which encode the degenerate terminal WDL).
        priors: Dict[str, float] = {}
        for child in self._provider.expand_node(fen, depth=0):
            priors[child.move_uci] = float(child.scalar_features["prior"])
        root_features = self._provider.root_features(fen)
        value = float(root_features["value"])
        wdl = (
            float(root_features["wdl_win"]),
            float(root_features["wdl_draw"]),
            float(root_features["wdl_loss"]),
        )
        return PositionEval(priors=priors, value=value, wdl=wdl)

    def clear_caches(self) -> None:
        self._provider.clear_caches()


__all__ = [
    "PositionEval",
    "Evaluator",
    "MockEvaluator",
    "CachedEvaluator",
    "Lc0UciEvaluator",
    "evaluator_to_provider",
]
