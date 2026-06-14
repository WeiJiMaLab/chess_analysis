"""T-fwd (quirk): ``quantize_wdl_to_uci_permille`` — the WDL counterpart of the
prior text-resolution quantizer (report §2, re_baseline decision 2026-06-14).

The lc0-UCI value path prints WDL as three INTEGER per-mille counts
(``wdl W D L`` with ``W,D,L`` in ``[0, 1000]``, see ``WDL_RE`` in
``cts.core.providers.common`` and ``parse_root_value_features_from_lines``),
which ``value_features_from_wdl`` then renormalizes. So with ``re_baseline=False``
the in-process net must round each WDL component to the k/1000 grid and
renormalize to sum 1 before it can match the baseline bit-for-bit.

These tests are pure (no engine / GPU / heavy lib), mirroring the existing
prior-quantizer tests in ``test_batched_gen_rebaseline.py`` in style. They do
NOT duplicate those; they cover the NEW ``quantize_wdl_to_uci_permille`` symbol.

Contract under test::

    quantize_wdl_to_uci_permille(win, draw, loss) -> (win', draw', loss')
      - each raw component is rounded to k/1000 (integer per-mille) BEFORE renorm
      - the returned triple is renormalized to sum to 1 (side-to-move)
      - idempotent on an already-on-grid, already-normalized triple
"""

from __future__ import annotations

import math

import pytest

from cts.data.batched_gen.net_evaluator import quantize_wdl_to_uci_permille


def _on_permille_grid(value: float, *, tol: float = 1e-9) -> bool:
    """True iff ``value`` is an integer multiple of 1e-3 (the per-mille grid)."""
    scaled = value * 1000.0
    return abs(scaled - round(scaled)) < 1e-6 or math.isclose(scaled, round(scaled), abs_tol=tol)


def test_wdl_quantizer_rounds_each_component_to_permille() -> None:
    """Each raw component snaps to the nearest k/1000 before renormalization.

    0.5004 -> 500/1000, 0.2999 -> 300/1000, 0.1997 -> 200/1000. Those integer
    counts already sum to 1000, so renormalization is a no-op and the returned
    triple is exactly (0.5, 0.3, 0.2).
    """
    win, draw, loss = quantize_wdl_to_uci_permille(0.5004, 0.2999, 0.1997)
    assert (win, draw, loss) == pytest.approx((0.500, 0.300, 0.200), abs=1e-12)


def test_wdl_quantizer_renormalizes_to_sum_one() -> None:
    """When the rounded per-mille counts don't sum to 1000, the result renorms.

    Raw masses needn't be normalized on input (lc0's are counts); the returned
    triple must always sum to 1 so it is a valid ``PositionEval.wdl``.
    """
    for raw in [
        (0.6, 0.3, 0.2),  # sums to 1.1
        (0.1, 0.1, 0.1),  # sums to 0.3
        (0.3334, 0.3333, 0.3333),
        (0.9, 0.0501, 0.0501),
    ]:
        out = quantize_wdl_to_uci_permille(*raw)
        assert len(out) == 3
        assert all(math.isfinite(c) and c >= 0.0 for c in out), out
        assert sum(out) == pytest.approx(1.0, abs=1e-9), f"{raw} -> {out}"


def test_wdl_quantizer_is_idempotent_on_grid_triple() -> None:
    """Quantizing an already-on-grid, already-normalized triple is a fixed point."""
    for triple in [
        (0.500, 0.300, 0.200),
        (0.001, 0.998, 0.001),
        (0.333, 0.334, 0.333),
        (1.000, 0.000, 0.000),
    ]:
        out = quantize_wdl_to_uci_permille(*triple)
        again = quantize_wdl_to_uci_permille(*out)
        assert out == pytest.approx(again, abs=1e-12), f"{triple}: {out} != {again}"


def test_wdl_quantizer_handles_near_zero_component() -> None:
    """A component below 0.5/1000 rounds to 0/1000 (prints as ``0``) and drops out.

    The remaining mass renormalizes across the surviving components and still
    sums to 1; the near-zero component lands at exactly 0.0.
    """
    win, draw, loss = quantize_wdl_to_uci_permille(0.5, 0.5, 0.0004)
    assert loss == pytest.approx(0.0, abs=1e-12), (win, draw, loss)
    assert sum((win, draw, loss)) == pytest.approx(1.0, abs=1e-9)


def test_wdl_quantizer_value_lands_on_grid() -> None:
    """``value = win - loss`` of the quantized triple lies on the per-mille grid.

    Both ``win`` and ``loss`` are k/1000 after rounding; for a triple whose
    rounded counts already sum to 1000 (no renorm) their difference is exactly
    an integer multiple of 1e-3 — the same grid the lc0-UCI ``value`` lands on.
    """
    win, draw, loss = quantize_wdl_to_uci_permille(0.625, 0.250, 0.125)
    # 625 + 250 + 125 = 1000, so renorm is a no-op and win/loss stay on grid.
    value = win - loss
    assert _on_permille_grid(win), win
    assert _on_permille_grid(loss), loss
    assert _on_permille_grid(value), value
    assert value == pytest.approx(0.500, abs=1e-12)


def test_wdl_quantizer_components_are_on_grid_when_counts_sum_to_1000() -> None:
    """Each returned component is on the k/1000 grid when no renorm is needed."""
    for triple in [
        (0.500, 0.300, 0.200),
        (0.700, 0.200, 0.100),
        (0.001, 0.001, 0.998),
    ]:
        for component in quantize_wdl_to_uci_permille(*triple):
            assert _on_permille_grid(component), (triple, component)
