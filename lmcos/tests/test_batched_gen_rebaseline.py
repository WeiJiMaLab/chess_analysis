"""re_baseline switch: bit-for-bit replication of the lc0-UCI quirks by default.

Report §2 (resolved 2026-06-14): the priority is bit-for-bit replication, so
``re_baseline=False`` is the default and reproduces two UCI-path quirks —
(a) priors rounded to verbose-move-stats text resolution, and (b) lc0's bare-FEN
history fill. These tests cover the backend-independent pieces of that contract
(the net forward itself is a Phase-2 scaffold, exercised by T-fwd).
"""

from __future__ import annotations

from cts.data.batched_gen.encoding import BareFenHistoryFill, history_fill_for
from cts.data.batched_gen.net_evaluator import (
    NetEvaluator,
    quantize_prior_to_uci_text_resolution,
)


# --------------------------------------------------------------------------- #
# (a) prior quantization to lc0 verbose-move-stats text resolution
# --------------------------------------------------------------------------- #
def test_quantizer_rounds_to_two_decimal_percent() -> None:
    """12.3349% -> "12.33%" -> 0.1233; 12.3351% -> "12.34%" -> 0.1234."""
    assert quantize_prior_to_uci_text_resolution(0.123349) == 0.1233
    assert quantize_prior_to_uci_text_resolution(0.123351) == 0.1234


def test_quantizer_prunes_tiny_priors_to_zero() -> None:
    """A prior below 0.005% prints as "0.00%" and reparses to exactly 0.0."""
    assert quantize_prior_to_uci_text_resolution(0.00004) == 0.0


def test_quantizer_is_idempotent() -> None:
    """Quantizing an already-quantized value is a fixed point (1e-4 grid)."""
    for p in (0.5, 0.1234, 0.0001, 0.9999):
        q = quantize_prior_to_uci_text_resolution(p)
        assert quantize_prior_to_uci_text_resolution(q) == q


def test_quantizer_output_lands_on_1e4_grid() -> None:
    """Every output is an exact multiple of 1e-4 (two decimals of a percent)."""
    for p in (0.123349, 0.4417, 0.000051, 0.876543):
        q = quantize_prior_to_uci_text_resolution(p)
        assert round(q * 10000) == q * 10000 or abs(q * 10000 - round(q * 10000)) < 1e-6


# --------------------------------------------------------------------------- #
# (b) history-fill switch: replicate lc0 by default, canonical when re-baselining
# --------------------------------------------------------------------------- #
def test_history_fill_default_replicates_lc0() -> None:
    """re_baseline=False -> match lc0's actual bare-FEN behavior."""
    assert history_fill_for(False) is BareFenHistoryFill.LC0_BARE_FEN


def test_history_fill_rebaseline_uses_canonical() -> None:
    """re_baseline=True -> the defined canonical fill."""
    assert history_fill_for(True) is BareFenHistoryFill.REPEAT


# --------------------------------------------------------------------------- #
# NetEvaluator carries the flag (default replicate) without loading heavy libs
# --------------------------------------------------------------------------- #
def test_net_evaluator_default_is_replicate() -> None:
    """Constructing a NetEvaluator defaults to re_baseline=False and imports
    no heavy libs (model is loaded lazily on first evaluate)."""
    ev = NetEvaluator("/nonexistent/weights.pb.gz")
    assert ev._re_baseline is False


def test_net_evaluator_rebaseline_flag_settable() -> None:
    ev = NetEvaluator("/nonexistent/weights.pb.gz", re_baseline=True)
    assert ev._re_baseline is True
