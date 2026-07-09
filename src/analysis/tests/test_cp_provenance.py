"""Skeptical provenance tests for the centipawn tree pipeline (commit 23832b3).

Engine-free battery: exercises the real parser (`cts.core.providers.parsers`),
the real writer (`RawPretrainExampleRecord`), and the real pack tensorizer —
hunting POV/sign errors, mate-handling errors, and silent unit mixups.

POV/SIGN CONVENTION UNDER TEST (this file is the executable documentation):
  * ``cp`` / ``mate`` / ``cp_order`` are stored from the side-to-move POV of
    the node's OWN position (same convention as ``value``); nothing is
    normalized to the root mover.
  * Mate band: ``cp_order = sign(mate) * (20000 - |mate distance in moves|)``;
    checkmated-now terminal = -20000; stalemate/draw terminal = cp 0.

(The BeFS-specific selection-semantics battery that used to live here —
``TestBefsSelection`` plus its ``_befs_config``/``_generate`` helpers — was
removed 2026-07-08 along with the BeFS search rule itself: a confirmed,
unfixed tunnel-vision bug, and BeFS was already unreachable from the live
pipeline. See labnotebook.md.)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from analysis.tests._cp_fixtures import (
    FEN_MID,
    FEN_START,
    ScriptedProvider,
    feats,
    make_line,
    replay_befs_expansion_order,
    true_expansion_order,
)
from cts.core.providers.parsers import (
    MATE_CP_ORDER_BAND,
    parse_root_value_features_from_lines,
    score_order_features,
    terminal_value_features,
)
from cts.core.schema import TREE_ENCODER_FEATURE_NAMES, tree_encoder_feature_schema
from cts.data.preprocess_gnn.pack import raise_if_example_cannot_pack
from cts.data.preprocess_gnn.teacher_targets import (
    NodeBudgetDistribution,
    RawPretrainExampleRecord,
    TeacherSearchConfig,
    build_pretrain_example,
    generate_partial_tree_from_provider,
)

import random

WDL_FEATURES = ("value", "wdl_win", "wdl_draw", "wdl_loss", "wdl_var")


def _assert_cp_xor_mate(features: dict) -> None:
    assert ("cp" in features) != ("mate" in features), (
        f"cp/mate must be mutually exclusive, got keys {sorted(features)}"
    )
    assert "cp_order" in features and math.isfinite(features["cp_order"])


# ---------------------------------------------------------------------------
# 1. Parser: cp/mate capture on synthetic UCI lines (the real parse function)
# ---------------------------------------------------------------------------

class TestParserCpMateCapture:
    def test_cp_line_captures_cp_and_full_wdl(self):
        lines = [
            "info depth 1 seldepth 1 multipv 1 score cp 44 wdl 132 856 12 nodes 20 nps 20000 time 1 pv e2e4",
            "bestmove e2e4",
        ]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        for name in WDL_FEATURES:
            assert name in features, f"missing encoder feature {name}"
        assert features["cp"] == 44.0
        assert features["cp_order"] == 44.0
        assert "mate" not in features
        assert features["value"] == pytest.approx((132 - 12) / 1000.0)
        assert features["wdl_win"] == pytest.approx(0.132)
        _assert_cp_xor_mate(features)

    def test_negative_cp_keeps_sign(self):
        lines = ["info depth 1 score cp -873 wdl 0 40 960 nodes 1", "bestmove a1a2"]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        assert features["cp"] == -873.0
        assert features["cp_order"] == -873.0
        assert features["value"] == pytest.approx(-0.96)

    def test_mate_line_folds_into_band(self):
        lines = ["info depth 1 score mate 3 wdl 1000 0 0 nodes 1 pv a1a8", "bestmove a1a8"]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        assert features["mate"] == 3.0
        assert features["cp_order"] == 20000.0 - 3.0
        assert "cp" not in features, "mate scores must NOT populate cp"
        assert features["value"] == pytest.approx(1.0)
        assert features["wdl_win"] == pytest.approx(1.0)
        _assert_cp_xor_mate(features)

    def test_getting_mated_is_negative_band(self):
        lines = ["info depth 1 score mate -2 wdl 0 0 1000 nodes 1", "bestmove h8h7"]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        assert features["mate"] == -2.0
        assert features["cp_order"] == -(20000.0 - 2.0)
        assert features["value"] == pytest.approx(-1.0)
        _assert_cp_xor_mate(features)

    def test_mate_zero_means_mated_now(self):
        # UCI "mate 0": the side to move is checkmated. Must land at the very
        # bottom of the band, below every mate-in-k-against.
        features = score_order_features("mate", 0)
        assert features["cp_order"] == -20000.0
        assert features["mate"] == 0.0

    def test_mate_fallback_without_wdl_line(self):
        # Some engines report a bare mate score with no wdl triple; the parser
        # must synthesize the degenerate WDL AND keep the true mate distance.
        lines = ["info depth 245 score mate 5 nodes 1 pv a1a8", "bestmove a1a8"]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        assert features["value"] == pytest.approx(1.0)
        assert features["wdl_win"] == pytest.approx(1.0)
        assert features["mate"] == 5.0
        assert features["cp_order"] == 20000.0 - 5.0
        _assert_cp_xor_mate(features)

    def test_score_taken_from_same_line_as_wdl(self):
        # A stale score on an earlier info line must not be paired with the
        # WDL of a later line when that line carries its own score.
        lines = [
            "info depth 1 score cp 10 nodes 1",
            "info depth 2 score cp 99 wdl 500 0 500 nodes 2",
            "bestmove e2e4",
        ]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        assert features["cp"] == 99.0

    def test_wdl_only_line_falls_back_to_any_score_line(self):
        lines = [
            "info depth 1 wdl 300 400 300 nodes 1",
            "info string extra",
            "info depth 1 score cp -50 nodes 1",
            "bestmove e2e4",
        ]
        features = parse_root_value_features_from_lines(lines, require_wdl=True)
        assert features["cp"] == -50.0
        assert features["value"] == pytest.approx(0.0)


class TestTerminalFeatures:
    def test_checkmated_terminal(self):
        features = terminal_value_features(-1.0)
        assert features["value"] == pytest.approx(-1.0)
        assert features["wdl_loss"] == pytest.approx(1.0)
        assert features["cp_order"] == -MATE_CP_ORDER_BAND
        assert features["mate"] == 0.0
        assert "cp" not in features
        _assert_cp_xor_mate(features)

    def test_stalemate_terminal_is_cp_zero(self):
        features = terminal_value_features(0.0)
        assert features["cp"] == 0.0
        assert features["cp_order"] == 0.0
        assert "mate" not in features
        assert features["wdl_draw"] == pytest.approx(1.0)
        _assert_cp_xor_mate(features)

    def test_terminal_mate_below_every_mate_in_k(self):
        # Being mated NOW (-20000) must rank strictly below being mated in 1
        # (-19999): otherwise a search would prefer standing in checkmate over
        # a defensible mate-in-1.
        mated_now = terminal_value_features(-1.0)["cp_order"]
        mated_in_1 = score_order_features("mate", -1)["cp_order"]
        assert mated_now < mated_in_1


# ---------------------------------------------------------------------------
# 2. Mate band ordering (the ±(20000−k) ladder)
# ---------------------------------------------------------------------------

class TestMateBandOrdering:
    def test_winning_side_ladder(self):
        mate_in_1 = score_order_features("mate", 1)["cp_order"]
        mate_in_3 = score_order_features("mate", 3)["cp_order"]
        cp_1500 = score_order_features("cp", 1500)["cp_order"]
        assert mate_in_1 > mate_in_3 > cp_1500
        assert mate_in_1 == 19999.0
        assert mate_in_3 == 19997.0

    def test_losing_side_ladder(self):
        mated_in_1 = score_order_features("mate", -1)["cp_order"]
        mated_in_3 = score_order_features("mate", -3)["cp_order"]
        cp_minus_1500 = score_order_features("cp", -1500)["cp_order"]
        assert mated_in_1 < mated_in_3 < cp_minus_1500
        assert mated_in_1 == -19999.0

    def test_band_clears_stockfishs_maximum_real_cp(self):
        # Stockfish's max real |cp| is ~15385 (VALUE_MAX_EVAL*100/PawnValueEg).
        # Every mate at any plausible distance must outrank it.
        max_real_cp = 15385
        deep_mate = score_order_features("mate", 1000)["cp_order"]  # absurdly deep
        assert deep_mate == 19000.0
        assert deep_mate > max_real_cp
        assert score_order_features("mate", -1000)["cp_order"] < -max_real_cp

    def test_band_symmetry(self):
        for k in (1, 2, 7, 40):
            assert score_order_features("mate", k)["cp_order"] == -score_order_features("mate", -k)["cp_order"]


# ---------------------------------------------------------------------------
# 3. Encoder schema must be untouched (cp fields are additive-only)
# ---------------------------------------------------------------------------

class TestEncoderSchemaFrozen:
    def test_feature_names_exact(self):
        assert TREE_ENCODER_FEATURE_NAMES == ("value", "wdl_win", "wdl_draw", "wdl_loss", "wdl_var")

    def test_cp_fields_not_in_encoder_schema(self):
        for name in ("cp", "mate", "cp_order"):
            assert name not in TREE_ENCODER_FEATURE_NAMES


# ---------------------------------------------------------------------------
# 4. Shared fixtures for the writer/pack tests below
#
# (The BeFS selection-semantics battery that used to occupy this section —
# ``TestBefsSelection`` plus its ``_befs_config``/``_generate``/``_leafish``
# helpers — was removed 2026-07-08 along with the BeFS search rule itself.)
# ---------------------------------------------------------------------------

def _puct_config(max_depth: int, value_feature: str = "cp_order") -> TeacherSearchConfig:
    return TeacherSearchConfig(
        max_depth=max_depth,
        search_budget=64,
        c_puct=1.0,
        selection="puct",
        value_feature=value_feature,
        search_config_id="test_puct",
    )


ROOT_FEATS = feats(0.0, (0.2, 0.6, 0.2), cp=0.0)

# ---------------------------------------------------------------------------
# 5. Writer round-trip + pack compatibility + replay-oracle self-test
# ---------------------------------------------------------------------------

def _scripted_example():
    """A full build_pretrain_example run (the committed generation path) on a
    scripted tree containing cp, mate, and terminal-mate nodes."""
    root = FEN_MID
    fen_a = root + " ||moves|| a2a3"
    fen_b = root + " ||moves|| b2b3"
    provider = ScriptedProvider(
        root,
        ROOT_FEATS,
        {
            root: make_line(
                ("a2a3", feats(-0.5, (0.1, 0.3, 0.6), cp=-120.0)),
                ("b2b3", feats(0.2, (0.4, 0.4, 0.2), cp=60.0)),
            ),
            fen_a: [
                ("h7h6", feats(0.4, (0.6, 0.2, 0.2), cp=80.0), False),
                # terminal checkmate leaf (mover mated now)
                ("g7g5", feats(-1.0, (0.0, 0.0, 1.0), mate=0.0, cp_order=-20000.0), True),
            ],
            fen_a + " h7h6": make_line(
                ("c2c3", feats(-0.3, (0.2, 0.3, 0.5), cp=-40.0)),
                # deep mate score (non-terminal): mover mates in 2
                ("d2d4", feats(1.0, (1.0, 0.0, 0.0), mate=2.0, cp_order=19998.0)),
            ),
            fen_b: make_line(("h7h6", feats(0.0, (0.3, 0.4, 0.3), cp=0.0))),
        },
    )
    return build_pretrain_example(
        root,
        provider,
        _puct_config(max_depth=3),
        node_budget_distribution=NodeBudgetDistribution(3, 3),
        rng=random.Random(0),
        root_position_id="scripted_fixture",
        include_edge_wdl_targets=True,
    )


class TestWriterAndPackCompat:
    def test_payload_round_trip_preserves_cp_features(self):
        example = _scripted_example()
        record = RawPretrainExampleRecord.from_example(example)
        payload = record.to_payload()
        names = list(payload["feature_names"])
        for name in ("cp", "mate", "cp_order", *WDL_FEATURES, "prior"):
            assert name in names, f"payload lost feature {name}"
        node_features = payload["node_features"].numpy()
        cp_col = node_features[:, names.index("cp")]
        mate_col = node_features[:, names.index("mate")]
        order_col = node_features[:, names.index("cp_order")]
        # cp XOR mate per node; cp_order always finite.
        assert np.all(np.isfinite(order_col))
        assert np.all(np.isfinite(cp_col) != np.isfinite(mate_col)), (
            "every node must carry exactly one of cp/mate"
        )
        # The terminal mate leaf survived the round trip at exactly -20000.
        assert (order_col == -20000.0).any()
        # Round-trip back to an example keeps the same scalar dicts.
        rehydrated = RawPretrainExampleRecord.from_payload(payload).to_pretrain_example()
        for node_a, node_b in zip(example.tree.iter_nodes(), rehydrated.tree.iter_nodes()):
            assert node_a.scalar_features == pytest.approx(node_b.scalar_features)

    def test_pack_tensorizer_ignores_additive_cp_columns(self):
        example = _scripted_example()
        record = RawPretrainExampleRecord.from_example(example)
        schema = tree_encoder_feature_schema()
        # Must not raise: the 5 encoder columns are fully populated even
        # though cp/mate columns contain NaN (they are additive extras).
        raise_if_example_cannot_pack(record, schema, example_path_label="scripted_fixture")
        tensorized = record.to_tensorized_tree_example(schema)
        assert tensorized.node_features.shape[1] == 5
        assert tuple(tensorized.feature_names) == TREE_ENCODER_FEATURE_NAMES
        assert torch.isfinite(tensorized.node_features).all(), (
            "NaNs from the additive cp/mate columns leaked into encoder input"
        )
        # Column content equals the schema projection of the raw features.
        raw_names = list(record.feature_names)
        for column, name in enumerate(TREE_ENCODER_FEATURE_NAMES):
            expected = record.node_features[:, raw_names.index(name)]
            assert torch.equal(tensorized.node_features[:, column], expected)

    @pytest.mark.skip(
        reason=(
            "replay_befs_expansion_order (in _cp_fixtures.py) reimplements the OLD "
            "frozen-static-value BeFS selection rule, not the corrected genuine-minimax "
            "backup rule now used by generate_partial_tree_from_provider. It is not called "
            "by any production/validation pipeline (grepped: only this test uses it), so "
            "leaving it stale is safe for now, but it will no longer match real trees "
            "generated post-fix. Needs a rewrite (dynamic backed-up-value replay, not a "
            "frozen static comparison) before it's used to validate any new corpus again."
        )
    )
    def test_replay_oracle_matches_on_scripted_tree(self):
        # Self-test of the independent replay implementation used against the
        # real Stockfish pilot trees: on a scripted befs tree it must
        # reproduce the recorded expansion order exactly.
        example = _scripted_example()
        payload = RawPretrainExampleRecord.from_example(example).to_payload()
        truth = true_expansion_order(payload)
        replay = replay_befs_expansion_order(payload, max_depth=3)
        assert replay == truth
        assert len(truth) == 3
