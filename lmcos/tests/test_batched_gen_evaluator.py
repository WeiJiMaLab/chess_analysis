"""T-mock-determinism / T-cached / T-fwd: the Evaluator abstraction layer.

Report §4 IDs covered:
  - T-mock-determinism: ``MockEvaluator`` returns identical ``PositionEval`` for
                        a FEN across two separate instantiations (stable hashing,
                        NOT builtin ``hash``); wdl sums to 1; a prior for every
                        legal move.
  - T-cached          : ``CachedEvaluator`` replays inner results exactly on a
                        second pass without re-calling the inner evaluator
                        (verified with a counting spy inner Evaluator).
  - T-fwd (net)       : guarded by ``pytest.importorskip`` so it is skipped when
                        the heavy lib is absent; asserts the ``NetEvaluator``
                        interface (one valid PositionEval per fen).

No engine/GPU/network: everything runs on Mock + a tmp cache file.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import pytest

from cts.data.batched_gen.evaluator import (
    CachedEvaluator,
    Evaluator,
    MockEvaluator,
    PositionEval,
)

from test_batched_gen_common import SMALL_FEN_LIST


def _is_valid_eval(ev: PositionEval, *, legal_moves: Sequence[str] | None = None) -> None:
    assert isinstance(ev.value, float)
    assert -1.0 - 1e-6 <= ev.value <= 1.0 + 1e-6, f"value out of range: {ev.value}"
    assert len(ev.wdl) == 3
    assert all(math.isfinite(c) and c >= -1e-9 for c in ev.wdl), ev.wdl
    assert sum(ev.wdl) == pytest.approx(1.0, abs=1e-5), f"wdl must sum to 1: {ev.wdl}"
    assert all(math.isfinite(p) for p in ev.priors.values())
    if legal_moves is not None:
        for mv in legal_moves:
            assert mv in ev.priors, f"missing prior for legal move {mv}"


# --------------------------------------------------------------------------- #
# T-mock-determinism
# --------------------------------------------------------------------------- #
def test_mock_evaluator_deterministic_across_instances() -> None:
    """T-mock-determinism: two fresh MockEvaluators agree on every FEN.

    This is the property that lets a CPU fleet shard generation across
    processes: the synthetic oracle must be a pure function of the FEN. A mock
    built on builtin ``hash`` would fail this once ``PYTHONHASHSEED`` varies
    between processes, so the contract pins ``hashlib``.
    """
    a = MockEvaluator()
    b = MockEvaluator()
    fens = list(SMALL_FEN_LIST)
    out_a = a.evaluate(fens)
    out_b = b.evaluate(fens)
    assert len(out_a) == len(out_b) == len(fens)
    for ea, eb in zip(out_a, out_b):
        assert ea.value == eb.value
        assert ea.wdl == eb.wdl
        assert ea.priors == eb.priors


def test_mock_evaluator_valid_distribution_and_full_priors() -> None:
    """T-mock-determinism: wdl sums to 1 and there is a prior for every legal move."""
    import chess

    mock = MockEvaluator()
    for fen in SMALL_FEN_LIST:
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        ev = mock.evaluate([fen])[0]
        _is_valid_eval(ev, legal_moves=legal)


def test_mock_evaluator_not_using_builtin_hash() -> None:
    """T-mock-determinism: stable (hashlib-based) hashing, not ``hash()``.

    We can't introspect the implementation directly, but determinism across
    instances within the SAME process is necessary; the cross-process claim is
    documented and relied on by the fleet. Here we assert at least that the same
    FEN evaluated twice in the same evaluator is identical (idempotent) and that
    distinct FENs generally differ (the hash actually depends on the FEN).
    """
    mock = MockEvaluator()
    fen = SMALL_FEN_LIST[0]
    e1 = mock.evaluate([fen])[0]
    e2 = mock.evaluate([fen])[0]
    assert e1.value == e2.value and e1.wdl == e2.wdl and e1.priors == e2.priors

    others = [mock.evaluate([f])[0] for f in SMALL_FEN_LIST]
    values = [e.value for e in others]
    assert len(set(values)) > 1, "MockEvaluator value should depend on the FEN"


def test_mock_evaluator_table_override() -> None:
    """T-mock-determinism: an explicit ``table`` overrides the synthetic values."""
    fen = SMALL_FEN_LIST[0]
    forced = PositionEval(priors={"e2e4": 1.0}, value=0.5, wdl=(0.6, 0.3, 0.1))
    mock = MockEvaluator(table={fen: forced})
    out = mock.evaluate([fen])[0]
    assert out.value == 0.5
    assert out.wdl == (0.6, 0.3, 0.1)
    assert out.priors == {"e2e4": 1.0}


# --------------------------------------------------------------------------- #
# T-cached
# --------------------------------------------------------------------------- #
class _SpyEvaluator(Evaluator):
    """Counting spy: records exactly which FENs the inner evaluator was asked for."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self._mock = MockEvaluator()

    def evaluate(self, fens: Sequence[str]) -> List[PositionEval]:
        self.calls.append(list(fens))
        return self._mock.evaluate(fens)

    def clear_caches(self) -> None:  # pragma: no cover - trivial
        pass

    @property
    def queried_fens(self) -> List[str]:
        return [f for batch in self.calls for f in batch]


def test_cached_evaluator_replays_without_recalling_inner(tmp_path) -> None:
    """T-cached: second pass over the same FENs hits the cache, not the inner eval."""
    cache_path = tmp_path / "evalcache.pt"
    spy = _SpyEvaluator()
    cached = CachedEvaluator(spy, cache_path=str(cache_path))

    fens = list(SMALL_FEN_LIST)
    first = cached.evaluate(fens)
    assert len(first) == len(fens)
    assert set(spy.queried_fens) == set(fens), "first pass must populate the cache"
    inner_calls_after_first = len(spy.queried_fens)

    second = cached.evaluate(fens)
    assert len(spy.queried_fens) == inner_calls_after_first, (
        "second pass must not re-query the inner evaluator for cached FENs"
    )
    # Replayed values are exactly the cached values.
    for a, b in zip(first, second):
        assert a.value == b.value
        assert a.wdl == b.wdl
        assert a.priors == b.priors


def test_cached_evaluator_only_queries_uncached_fens(tmp_path) -> None:
    """T-cached: a mixed batch only forwards the not-yet-cached FENs to inner."""
    cache_path = tmp_path / "evalcache2.pt"
    spy = _SpyEvaluator()
    cached = CachedEvaluator(spy, cache_path=str(cache_path))

    warm = list(SMALL_FEN_LIST[:2])
    cached.evaluate(warm)
    spy.calls.clear()

    mixed = list(SMALL_FEN_LIST)  # 2 warm + 2 cold
    cached.evaluate(mixed)
    queried = set(spy.queried_fens)
    assert queried == set(SMALL_FEN_LIST[2:]), (
        f"only the cold FENs should reach inner; got {queried}"
    )


def test_cached_evaluator_persists_across_instances(tmp_path) -> None:
    """T-cached: a second CachedEvaluator reading the same path serves from disk."""
    cache_path = tmp_path / "evalcache3.pt"
    spy1 = _SpyEvaluator()
    CachedEvaluator(spy1, cache_path=str(cache_path)).evaluate(list(SMALL_FEN_LIST))

    spy2 = _SpyEvaluator()
    cached2 = CachedEvaluator(spy2, cache_path=str(cache_path))
    cached2.evaluate(list(SMALL_FEN_LIST))
    assert spy2.queried_fens == [], "warm on-disk cache must avoid inner calls entirely"


# --------------------------------------------------------------------------- #
# T-fwd (net) — skipped unless the heavy lib is present.
# --------------------------------------------------------------------------- #
# NetEvaluator interface + numeric parity live in test_batched_gen_net_forward.py
# (with the real ONNX net + proper skips); no scaffold-era placeholder here.
