"""TDD-lite sanity tests for the real-generation-time ``tanh_cp_value`` value
feature (Agent 3 / ``cp_regen.md``, plan.md). Unlike ``CP``'s
``analysis.relabel_replay`` (shape-frozen relabel-and-replay), this feature is
computed at GENERATION time inside ``cts.core.providers`` so PUCT's own
selection can respond to the desaturated value, not just its final label --
see ``apply_tanh_cp_feature`` in ``cts/core/providers/parsers.py`` and the
``StockfishDirectEvalProvider(tanh_cp_temperature=...)``/
``BuildTreeConfig(tanh_cp_temperature=...)`` wiring in ``stockfish.py``/
``build_tree.py``.

Three layers, matching this repo's TDD-lite convention (a few targeted
sanity tests, not an exhaustive suite):
  1. The pure feature function (``apply_tanh_cp_feature``) against the exact
     formula already tested in ``analysis.relabel_replay.tanh_cp_value``
     (CP's replay probe) -- same math, independently implemented in ``cts``
     to avoid a cts -> analysis dependency (see ``apply_tanh_cp_feature``'s
     docstring).
  2. ``StockfishDirectEvalProvider`` end-to-end feature wiring against a fake
     in-process engine (no real Stockfish subprocess) -- confirms the
     temperature actually reaches ``root_features``/``expand_node``.
  3. ``BuildTreeConfig``/``generate_dataset_stockfish_command``'s fail-fast
     validation: selecting ``value_feature="tanh_cp_value"`` without a
     temperature must raise before any engine/file I/O happens.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.relabel_replay import tanh_cp_value
from cts.core.providers.parsers import apply_tanh_cp_feature
from cts.core.providers.process import UciEngineConfig
from cts.core.providers.stockfish import StockfishDirectEvalProvider
from cts.data.build_tree import BuildTreeConfig, generate_dataset_stockfish_command


# ===========================================================================
# 1. Pure feature function vs. CP's already-tested formula
# ===========================================================================
def test_apply_tanh_cp_feature_matches_relabel_replay_formula():
    for cp_order in [-5000.0, -268.0, 0.0, 137.0, 4000.0]:
        for temperature in [100.0, 300.0, 600.0]:
            features = apply_tanh_cp_feature({"cp_order": cp_order, "value": 0.5}, temperature)
            expected = float(tanh_cp_value(np.array([cp_order]), temperature)[0])
            assert features["tanh_cp_value"] == pytest.approx(expected, abs=1e-12)
            assert features["tanh_cp_value"] == pytest.approx(math.tanh(cp_order / temperature), abs=1e-12)
            # Original keys untouched.
            assert features["cp_order"] == cp_order
            assert features["value"] == 0.5


def test_apply_tanh_cp_feature_noop_when_temperature_none():
    features = {"cp_order": 268.0, "value": 0.9}
    out = apply_tanh_cp_feature(features, None)
    assert out is features  # same object, zero-cost no-op
    assert "tanh_cp_value" not in out


def test_apply_tanh_cp_feature_requires_cp_order():
    with pytest.raises(KeyError):
        apply_tanh_cp_feature({"value": 0.5}, 300.0)


def test_apply_tanh_cp_feature_bounded_like_value():
    # Desaturation must stay in [-1, 1] like the WDL "value" column, since
    # PUCT's exploration formula assumes a bounded, roughly stationary scale
    # (config_minply15_maxply75.yaml's treegen.value_feature comment).
    for cp_order in [-20000.0, -300.0, 0.0, 300.0, 20000.0]:
        val = apply_tanh_cp_feature({"cp_order": cp_order}, 300.0)["tanh_cp_value"]
        assert -1.0 <= val <= 1.0


# ===========================================================================
# 2. Provider wiring: temperature reaches root_features / expand_node
# ===========================================================================
class _FakeEngine:
    """Stand-in for ``UciEngineProcess``: enough surface for
    ``StockfishDirectEvalProvider.__init__``'s handshake plus one
    ``analyse()`` call, no real Stockfish subprocess involved."""

    def __init__(self):
        self.config = UciEngineConfig(engine_path="fake", engine_kind="stockfish")
        self._sent = []

    def start(self):
        pass

    def _send(self, line):
        self._sent.append(line)

    def _read_until(self, token):
        pass

    def analyse(self, fen):
        # A plausible Stockfish info line: cp=137 with a WDL triple that is
        # NOT saturated (so the test can tell "value" and "tanh_cp_value"
        # apart), same regex shape ``parse_root_value_features_from_lines``
        # expects (see cts/core/providers/common.py's SCORE_LINE_RE/WDL_RE).
        return [
            "info depth 1 seldepth 1 multipv 1 score cp 137 wdl 550 300 150 "
            "nodes 1 nps 1000 pv e2e4"
        ]


def test_stockfish_provider_root_features_include_tanh_cp_value_when_configured():
    engine = _FakeEngine()
    provider = StockfishDirectEvalProvider(engine, tanh_cp_temperature=300.0)
    features = provider.root_features("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert features["cp_order"] == pytest.approx(137.0)
    assert features["tanh_cp_value"] == pytest.approx(math.tanh(137.0 / 300.0), abs=1e-9)
    # The un-desaturated WDL "value" is still present, unaffected.
    assert "value" in features


def test_stockfish_provider_root_features_omit_tanh_cp_value_by_default():
    engine = _FakeEngine()
    provider = StockfishDirectEvalProvider(engine)  # tanh_cp_temperature=None default
    features = provider.root_features("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert "tanh_cp_value" not in features


# ===========================================================================
# 3. Config-level fail-fast validation
# ===========================================================================
def test_generate_dataset_stockfish_rejects_tanh_cp_value_without_temperature():
    config = BuildTreeConfig(
        command="generate-dataset",
        provider="stockfish",
        value_feature="tanh_cp_value",
        tanh_cp_temperature=None,
        fens="/nonexistent/fens.txt",  # never reached -- validation must fail first
        output_dir="/tmp/does_not_matter",
    )
    with pytest.raises(ValueError, match="tanh_cp_temperature"):
        generate_dataset_stockfish_command(config)


def test_build_tree_config_accepts_tanh_cp_value_with_temperature():
    # Just construction + the field round-trip; no engine I/O.
    config = BuildTreeConfig(
        command="generate-dataset",
        provider="stockfish",
        value_feature="tanh_cp_value",
        tanh_cp_temperature=300.0,
        fens="/nonexistent/fens.txt",
        output_dir="/tmp/does_not_matter",
    )
    assert config.value_feature == "tanh_cp_value"
    assert config.tanh_cp_temperature == 300.0
