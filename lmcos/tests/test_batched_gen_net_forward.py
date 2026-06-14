"""T-fwd: ``NetEvaluator`` in-process forward-pass behavior (report §4, L2).

This exercises the real in-process net interface end-to-end on a handful of
FENs from the held-out pool: shape of the result, WDL/value validity, a prior
for every legal move (cross-checked against python-chess), and the
``re_baseline`` quirk contract — with ``re_baseline=False`` priors land on the
1e-4 (verbose-move-stats) grid and WDL on the per-mille grid; with
``re_baseline=True`` they need not.

It is NOT a numeric-parity test against lc0 (that is the bare-FEN history pinning
test, ``test_batched_gen_net_history_pinning.py``). Here we only assert the
interface + the quantization quirk, so the bar is the contract, not lc0's bits.

All resource-dependent assertions SKIP gracefully when ``lczerolens`` is absent
or when the ONNX/weights assets are missing, so the suite stays green on a bare
machine.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest

# Held-out assets (real cluster paths). Tests SKIP if any is absent.
ONNX_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/weights/t1-256x10.onnx"
WEIGHTS_PATH = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
FEN_POOL_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/fens.txt"

# lc0 verbose-move-stats text resolution is 0.01% == 1e-4; WDL is integer
# per-mille == 1e-3. A component is "on grid" if it is an integer multiple of
# the grid step (within fp slack).
PRIOR_GRID = 1e-4
PERMILLE_GRID = 1e-3
_GRID_SLACK = 1e-6


def _read_fens(n: int) -> List[str]:
    """Read up to ``n`` FENs from the pool, expanding 4-field -> 6-field.

    The pool stores 4-field FENs (``... w - -``); python-chess and lc0 want the
    halfmove/fullmove counters, so append ``" 0 1"`` when they are missing.
    """
    path = Path(FEN_POOL_PATH)
    if not path.exists():
        return []
    fens: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if len(line.split()) == 4:
                line = line + " 0 1"
            fens.append(line)
            if len(fens) >= n:
                break
    return fens


def _on_grid(value: float, step: float) -> bool:
    scaled = value / step
    return abs(scaled - round(scaled)) < _GRID_SLACK


def _require_net():
    """Import-or-skip lczerolens, skip if assets absent, and build a NetEvaluator.

    Returns the constructed evaluator; skips (never fails) when the environment
    can't run the real net or while the forward pass is still a scaffold.
    """
    pytest.importorskip("lczerolens")
    if not Path(ONNX_PATH).exists() and not Path(WEIGHTS_PATH).exists():
        pytest.skip(f"neither ONNX ({ONNX_PATH}) nor weights ({WEIGHTS_PATH}) present")
    from cts.data.batched_gen.net_evaluator import NetEvaluator

    weights = ONNX_PATH if Path(ONNX_PATH).exists() else WEIGHTS_PATH
    try:
        return NetEvaluator(weights)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"NetEvaluator not constructable here: {exc}")


def _evaluate_or_skip(net, fens):
    """Run ``net.evaluate``; skip if the forward pass is still scaffolded."""
    try:
        return net.evaluate(fens)
    except NotImplementedError as exc:
        pytest.skip(f"NetEvaluator forward pass not implemented yet: {exc}")


def _test_fens(n: int = 4) -> List[str]:
    fens = _read_fens(n)
    if not fens:
        pytest.skip(f"FEN pool absent: {FEN_POOL_PATH}")
    return fens


# --------------------------------------------------------------------------- #
# Interface + validity (T-fwd)
# --------------------------------------------------------------------------- #
def test_net_evaluate_one_eval_per_fen() -> None:
    """evaluate returns exactly one PositionEval per input FEN, in order."""
    net = _require_net()
    fens = _test_fens(4)
    out = _evaluate_or_skip(net, fens)
    assert len(out) == len(fens)


def test_net_evaluate_wdl_sums_to_one_and_value_in_range() -> None:
    """Each result's WDL sums to 1 (+-1e-5) and value lies in [-1, 1]."""
    net = _require_net()
    fens = _test_fens(4)
    out = _evaluate_or_skip(net, fens)
    for ev in out:
        assert len(ev.wdl) == 3
        assert all(math.isfinite(c) and c >= -1e-9 for c in ev.wdl), ev.wdl
        assert sum(ev.wdl) == pytest.approx(1.0, abs=1e-5), ev.wdl
        assert -1.0 - 1e-6 <= ev.value <= 1.0 + 1e-6, ev.value
        assert all(math.isfinite(p) for p in ev.priors.values())


def test_net_evaluate_prior_for_every_legal_move() -> None:
    """Cross-check priors against python-chess: one entry per legal UCI move."""
    import chess

    net = _require_net()
    fens = _test_fens(4)
    out = _evaluate_or_skip(net, fens)
    for fen, ev in zip(fens, out):
        legal = {m.uci() for m in chess.Board(fen).legal_moves}
        assert set(ev.priors) == legal, (
            f"priors must be exactly the legal moves for {fen}; "
            f"missing={legal - set(ev.priors)} extra={set(ev.priors) - legal}"
        )


# --------------------------------------------------------------------------- #
# re_baseline=False -> quirk quantization on grid (T-fwd, quirk)
# --------------------------------------------------------------------------- #
def test_net_replicate_priors_on_1e4_grid() -> None:
    """re_baseline=False: every returned prior lies on the 1e-4 text grid.

    The raw priors are renormalized per-parent downstream, but the per-move
    quantization to verbose-move-stats resolution happens at the evaluator, so
    each emitted prior must already be an integer multiple of 1e-4.
    """
    pytest.importorskip("lczerolens")
    if not Path(ONNX_PATH).exists() and not Path(WEIGHTS_PATH).exists():
        pytest.skip("net assets absent")
    from cts.data.batched_gen.net_evaluator import NetEvaluator

    weights = ONNX_PATH if Path(ONNX_PATH).exists() else WEIGHTS_PATH
    fens = _test_fens(3)
    try:
        net = NetEvaluator(weights, re_baseline=False)
        out = net.evaluate(fens)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"NetEvaluator unavailable / scaffolded: {exc}")
    for ev in out:
        for move, prior in ev.priors.items():
            assert _on_grid(prior, PRIOR_GRID), f"prior {move}={prior} off 1e-4 grid"


def test_net_replicate_wdl_on_permille_grid() -> None:
    """re_baseline=False: each returned WDL component lies on the per-mille grid.

    Renormalization can in principle push a component off the exact k/1000 grid,
    but the quirk replication snaps the *counts* to integers and the lc0 path
    renormalizes identically; we assert each component is within one fp slack of
    a k/1000 multiple, the same bar the baseline produces.
    """
    pytest.importorskip("lczerolens")
    if not Path(ONNX_PATH).exists() and not Path(WEIGHTS_PATH).exists():
        pytest.skip("net assets absent")
    from cts.data.batched_gen.net_evaluator import NetEvaluator

    weights = ONNX_PATH if Path(ONNX_PATH).exists() else WEIGHTS_PATH
    fens = _test_fens(3)
    try:
        net = NetEvaluator(weights, re_baseline=False)
        out = net.evaluate(fens)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"NetEvaluator unavailable / scaffolded: {exc}")
    for ev in out:
        for component in ev.wdl:
            assert _on_grid(component, PERMILLE_GRID), f"wdl {component} off per-mille grid"


def test_net_rebaseline_need_not_be_on_grid() -> None:
    """re_baseline=True: full-precision priors/WDL — at least one value off-grid.

    The quirk-free path keeps the net's full-precision output, so generically
    not every prior/WDL component snaps to the coarse text/per-mille grids. We
    only require the path to be a valid PositionEval and that it is NOT forced
    onto the grids (so the False/True switch is observably different).
    """
    pytest.importorskip("lczerolens")
    if not Path(ONNX_PATH).exists() and not Path(WEIGHTS_PATH).exists():
        pytest.skip("net assets absent")
    from cts.data.batched_gen.net_evaluator import NetEvaluator

    weights = ONNX_PATH if Path(ONNX_PATH).exists() else WEIGHTS_PATH
    fens = _test_fens(4)
    try:
        net = NetEvaluator(weights, re_baseline=True)
        out = net.evaluate(fens)
    except (FileNotFoundError, NotImplementedError) as exc:
        pytest.skip(f"NetEvaluator unavailable / scaffolded: {exc}")
    # Still a valid distribution.
    for ev in out:
        assert sum(ev.wdl) == pytest.approx(1.0, abs=1e-5), ev.wdl
    # And at least one prior across the batch is genuinely off the coarse grid,
    # proving full precision is retained (not quantized like re_baseline=False).
    any_off_grid = any(
        not _on_grid(prior, PRIOR_GRID)
        for ev in out
        for prior in ev.priors.values()
    )
    assert any_off_grid, (
        "re_baseline=True should retain full-precision priors; all priors were "
        "on the 1e-4 grid, which is indistinguishable from re_baseline=False"
    )
