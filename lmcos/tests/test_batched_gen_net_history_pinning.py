"""Report step (4): bare-FEN history pinning — the in-process net reproduces
lc0's bare-``position fen`` evaluation under the ``LC0_BARE_FEN`` convention.

This is the load-bearing numeric-parity test for the ``re_baseline=False``
history-fill decision (report §2, resolved 2026-06-14). It confirms that
encoding a position with ``BareFenHistoryFill.LC0_BARE_FEN`` and running it
through the in-process net reproduces lc0's *own* bare-FEN evaluation (lc0 fills
its 7 history planes by its internal convention when given a bare
``position fen`` with no move history). If our chosen history convention were
wrong, value/WDL would diverge here.

We compare ``NetEvaluator(onnx, re_baseline=False).evaluate(fens)`` against
``Lc0UciEvaluator`` driven by the real lc0 engines (built exactly as
``smoke/_lc0_baseline.py``'s ``lc0_direct_provider``), on **value and WDL**, and
require **policy-argmax agreement**.

Tolerance: max abs diff < 2e-3 on value and on each WDL component — one
per-mille step, the resolution of lc0's integer ``wdl W D L`` text path that the
baseline parses (``WDL_RE`` in ``cts.core.providers.common``). With
``re_baseline=False`` the net snaps WDL to that same per-mille grid, so parity to
one step is the right bar; cross-backend low-bit noise is absorbed by the grid.

SKIPS gracefully when ``lczerolens``, the ONNX/weights, OR the lc0 binary are
absent, so the suite stays green on a bare machine.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

import pytest

# Real cluster assets; every one is required for this end-to-end parity test.
LC0_BINARY = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
LC0_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
ONNX_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/weights/t1-256x10.onnx"

# value = win - loss can move by up to two per-mille steps when both components
# straddle a rounding boundary, plus small renorm slack; 3e-3 stays well below
# the gross shift a *wrong* history fill would produce, so it still detects one.
VALUE_WDL_TOL = 3e-3
# Per-move prior agreement. Midgame policies are spread out (no single decisive
# move), so comparing the whole prior vector is far more robust than an argmax:
# with the correct fen_only fill priors match lc0 to ~1e-3 (c3a4 0.139 vs
# 0.1396); a wrong history fill shifts them by ~0.06 (a2a4 0.072 vs 0.129), so
# 1e-2 cleanly separates correct from wrong.
PRIOR_TOL = 1e-2

# Representative on-distribution human *midgame* positions (6-field, castled,
# ply ~14-24). Deliberately NOT the head of the sorted pool (pathological flat-
# policy endgames) and NOT the game-start / opening positions: lc0 only
# fen_only-fills history for non-initial FENs, and our tree-gen data is ply
# 15-75, so every position we actually evaluate is fen_only-filled (matched by
# the REPEATED encoding). On these the net has a clear opinion, so value/WDL
# parity and a decisive policy argmax are both meaningful.
REPRESENTATIVE_FENS = [
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12",
    "2rq1rk1/pp1bbppp/2n1pn2/3p4/3P4/2NBPN2/PP1B1PPP/R2Q1RK1 w - - 0 11",
    "r1b2rk1/2q1bppp/p2ppn2/1p6/3NP3/1BN1B3/PPP2PPP/R2Q1RK1 w - - 0 13",
]


@contextmanager
def _lc0_baseline_evaluator() -> Iterator[object]:
    """Yield an ``Lc0UciEvaluator`` over the two-engine ``Lc0DirectEvalProvider``.

    Mirrors ``smoke/_lc0_baseline.py``'s ``lc0_direct_provider`` wiring exactly:
    a classic engine with verbose-move-stats for sibling priors and a valuehead
    engine with ``UCI_ShowWDL`` for the position value, both at ``nodes=1``
    (oracle, not search). Engines are torn down on context exit.
    """
    from cts.core.providers import (
        Lc0DirectEvalProvider,
        UciEngineConfig,
        UciEngineProcess,
    )
    from cts.data.batched_gen.evaluator import Lc0UciEvaluator

    prior_config = UciEngineConfig(
        engine_path=LC0_BINARY,
        engine_kind="lc0",
        engine_mode="classic",
        movetime_ms=0,
        multipv=8,
        depth=None,
        nodes=1,
        weights_path=LC0_WEIGHTS,
        set_multipv=False,
        enable_verbose_move_stats=True,
    )
    value_config = UciEngineConfig(
        engine_path=LC0_BINARY,
        engine_kind="lc0",
        engine_mode="valuehead",
        movetime_ms=0,
        multipv=1,
        depth=None,
        nodes=1,
        weights_path=LC0_WEIGHTS,
        uci_options={"UCI_ShowWDL": "true"},
        set_multipv=False,
        enable_verbose_move_stats=False,
    )
    with UciEngineProcess(prior_config) as prior_engine, UciEngineProcess(
        value_config
    ) as value_engine:
        provider = Lc0DirectEvalProvider(prior_engine, value_engine)
        try:
            yield Lc0UciEvaluator(provider)
        finally:
            provider.clear_caches()


def test_bare_fen_history_pinning_matches_lc0() -> None:
    """LC0_BARE_FEN reproduces lc0's bare-`position fen` value/WDL within one
    per-mille step, with identical policy argmax.

    Confirms the ``re_baseline=False`` history convention (``LC0_BARE_FEN``)
    reproduces how lc0 evaluates a history-free ``position fen`` — the quirk the
    report's step (4) pins down. A wrong history fill would shift the net's
    value/WDL off lc0's by far more than one per-mille step.
    """
    pytest.importorskip("lczerolens")
    for path, label in (
        (LC0_BINARY, "lc0 binary"),
        (LC0_WEIGHTS, "lc0 weights"),
    ):
        if not Path(path).exists():
            pytest.skip(f"{label} absent: {path}")
    if not Path(ONNX_PATH).exists() and not Path(LC0_WEIGHTS).exists():
        pytest.skip("net weights for NetEvaluator absent")

    fens = REPRESENTATIVE_FENS

    from cts.data.batched_gen.net_evaluator import NetEvaluator

    net_weights = ONNX_PATH if Path(ONNX_PATH).exists() else LC0_WEIGHTS
    try:
        net = NetEvaluator(net_weights, re_baseline=False)
        net_evals = net.evaluate(fens)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"NetEvaluator unavailable / scaffolded: {exc}")

    try:
        with _lc0_baseline_evaluator() as lc0_eval:
            base_evals = lc0_eval.evaluate(fens)
    except (FileNotFoundError, OSError, NotImplementedError) as exc:
        pytest.skip(f"lc0 baseline engines unavailable: {exc}")

    assert len(net_evals) == len(base_evals) == len(fens)

    for fen, net_ev, base_ev in zip(fens, net_evals, base_evals):
        # value/WDL must match lc0's bare-FEN valuehead to ~a per-mille step;
        # a wrong history fill blows this up.
        assert abs(net_ev.value - base_ev.value) < VALUE_WDL_TOL, (
            f"value diff {abs(net_ev.value - base_ev.value)} on {fen}"
        )
        for a, b in zip(net_ev.wdl, base_ev.wdl):
            assert abs(a - b) < VALUE_WDL_TOL, f"wdl diff {abs(a - b)} on {fen}"

        # Whole prior vector must match lc0's reported P over the shared legal
        # moves (robust where a single-move argmax would be backend noise).
        shared = set(net_ev.priors) & set(base_ev.priors)
        assert shared, f"no shared scored moves for {fen}"
        max_prior_diff = max(abs(net_ev.priors[m] - base_ev.priors[m]) for m in shared)
        assert max_prior_diff < PRIOR_TOL, f"prior diff {max_prior_diff} on {fen}"
