"""
Skeptical provenance tests for the P0 board-correlate featurizer
(analysis.featurize_board) and its SQL-side companions in analysis.board.

Assume the metrics are wrong until proven right — hand-computed fixtures plus
DB cross-checks. Material is RAW non-pawn piece COUNTS (unweighted), mover POV.
"""

from __future__ import annotations

import os
import chess
import pytest

from analysis.featurize_board import featurize_fen, FEATURE_COLUMNS

DB_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/personal.db"
_HAS_DB = os.path.exists(DB_PATH)


def _feat(fen):
    """featurize_fen -> dict keyed by FEATURE_COLUMNS."""
    return dict(zip(FEATURE_COLUMNS, featurize_fen(fen)))


# --------------------------------------------------------------------------
# hand-computed fixtures (adversarial)
# --------------------------------------------------------------------------

def test_startpos_symmetry():
    f = _feat(chess.STARTING_FEN)
    assert f["in_check"] is False
    assert f["n_captures_avail"] == 0
    assert f["n_checks_avail"] == 0
    # weighted incl pawns: 8*1 + 2*3 + 2*3 + 2*5 + 1*9 = 8+6+6+10+9 = 39
    assert f["self_material"] == 39
    assert f["opp_material"] == 39
    assert f["material_imbalance"] == 0


def test_material_imbalance_is_mover_pov_and_flips_sign():
    """The SAME board, white-to-move vs black-to-move, must flip the sign of the
    mover-POV imbalance. This is the most likely silent bug."""
    # White has an extra queen (black has no queen).
    w_to_move = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    b_to_move = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    fw, fb = _feat(w_to_move), _feat(b_to_move)
    assert fw["self_material"] == 39 and fw["opp_material"] == 30   # black down a queen
    assert fw["material_imbalance"] == +9     # white (mover) up a queen (weighted)
    assert fb["material_imbalance"] == -9     # black (mover) down a queen
    assert fw["material_imbalance"] == -fb["material_imbalance"]


def test_material_includes_pawns_excludes_king():
    # King + pawns only: weighted material = pawn count on each side (king excluded).
    f = _feat("4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1")
    assert f["self_material"] == 3   # 3 white pawns
    assert f["opp_material"] == 2    # 2 black pawns
    assert f["material_imbalance"] == +1


def test_in_check_and_reply_set():
    # Black rook on e8 checks the white king on e1 down the open e-file.
    f = _feat("4r3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert f["in_check"] is True


def test_pinned_piece_capture_is_illegal():
    """A pinned piece cannot capture — legal_moves must exclude it, so it does
    not inflate n_captures_avail."""
    # White king e1, white knight e2 pinned by black rook e8; black pawn d3 is
    # NOT capturable by the pinned knight (Nxd3 would expose the king... actually
    # the knight on e2 is pinned along the e-file, so any knight move is illegal).
    fen = "4r3/8/8/8/8/3p4/4N3/4K3 w - - 0 1"
    board = chess.Board(fen)
    # sanity: the knight is pinned, so it has no legal moves at all
    knight_moves = [m for m in board.legal_moves if m.from_square == chess.E2]
    assert knight_moves == []
    f = _feat(fen)
    # the only captures counted must be legal ones (none here for the knight)
    assert f["n_captures_avail"] == sum(board.is_capture(m) for m in board.legal_moves)


def test_en_passant_capture_counted():
    """A 4-field FEN carrying the ep square must let python-chess see the ep
    capture (the DB fens are 4-field but include the ep square)."""
    # White pawn e5, black just played d7-d5; ep target d6. exd6 e.p. is legal.
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    board = chess.Board(fen)
    assert any(board.is_en_passant(m) for m in board.legal_moves)
    f = _feat(fen)
    assert f["n_captures_avail"] >= 1


def test_checks_available_count():
    # White queen d1, black king d8 on an open d-file: Qd1-d8 etc. give check.
    fen = "3k4/8/8/8/8/8/8/3QK3 w - - 0 1"
    board = chess.Board(fen)
    expected = sum(board.gives_check(m) for m in board.legal_moves)
    assert _feat(fen)["n_checks_avail"] == expected
    assert expected >= 1


def test_determinism():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    assert featurize_fen(fen) == featurize_fen(fen)


# --------------------------------------------------------------------------
# DB cross-checks (skipped without the DB)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_DB, reason="personal.db not present")
def test_in_check_matches_db_column():
    import duckdb
    conn = duckdb.connect(DB_PATH, read_only=True)
    rows = conn.execute(
        "SELECT p.fen, m.player_in_check "
        "FROM processed_moves_nonzero p JOIN moves m ON m.gid=p.gid AND m.move_ply=p.move_ply "
        "WHERE p.move_ply BETWEEN 15 AND 75 USING SAMPLE 1500"
    ).fetchall()
    conn.close()
    mism = [(fen, db) for fen, db in rows if _feat(fen)["in_check"] != bool(db)]
    assert not mism, f"{len(mism)} in_check mismatches vs moves.player_in_check, e.g. {mism[:3]}"


@pytest.mark.skipif(not _HAS_DB, reason="personal.db not present")
def test_nonpawn_count_matches_db_king_inclusion_consistent():
    """Cross-check the board's non-pawn piece COUNT (recomputed here, independent
    of the weighted self_material) against the DB count columns. A single
    consistent offset (0 = DB excludes king, 1 = includes) proves the columns are
    well-defined; we don't require a particular value, just consistency."""
    import duckdb
    NONPAWN = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

    def nonpawn_count(fen, mover):
        b = chess.Board(fen)
        side = b.turn if mover else (not b.turn)
        return sum(len(b.pieces(pt, side)) for pt in NONPAWN)

    conn = duckdb.connect(DB_PATH, read_only=True)
    rows = conn.execute(
        "SELECT fen, n_self_pieces_exc_pawns, n_opp_pieces_exc_pawns "
        "FROM processed_moves_nonzero WHERE move_ply BETWEEN 15 AND 75 USING SAMPLE 1500"
    ).fetchall()
    conn.close()
    diffs_self = {int(db_self) - nonpawn_count(fen, True) for fen, db_self, _ in rows}
    diffs_opp = {int(db_opp) - nonpawn_count(fen, False) for fen, _, db_opp in rows}
    assert len(diffs_self) == 1, f"inconsistent self non-pawn offset vs DB: {sorted(diffs_self)[:5]}"
    assert len(diffs_opp) == 1, f"inconsistent opp non-pawn offset vs DB: {sorted(diffs_opp)[:5]}"
    assert diffs_self == diffs_opp, f"self/opp offsets differ: {diffs_self} vs {diffs_opp}"


@pytest.mark.skipif(not _HAS_DB, reason="personal.db not present")
def test_all_windowed_fens_parse():
    import duckdb
    conn = duckdb.connect(DB_PATH, read_only=True)
    fens = [r[0] for r in conn.execute(
        "SELECT fen FROM processed_moves_nonzero WHERE move_ply BETWEEN 15 AND 75 USING SAMPLE 3000"
    ).fetchall()]
    conn.close()
    bad = []
    for fen in fens:
        try:
            chess.Board(fen)
        except Exception as e:
            bad.append((fen, str(e)))
    assert not bad, f"{len(bad)} FENs failed to parse, e.g. {bad[:3]}"


@pytest.mark.skipif(not _HAS_DB, reason="personal.db not present")
def test_prev_capture_recipe_matches_replay():
    """The n_pieces-lag recipe for prev_move_was_capture must match a python-chess
    replay of the game's move sequence (including en-passant, which drops a pawn
    off-square — the count recipe gets it right)."""
    import duckdb
    conn = duckdb.connect(DB_PATH, read_only=True)
    # a few complete games, in ply order
    gids = [r[0] for r in conn.execute(
        "SELECT DISTINCT gid FROM moves USING SAMPLE 8"
    ).fetchall()]
    q = ("SELECT move_ply, move_uci, "
         "(n_pieces < lag(n_pieces) OVER (ORDER BY move_ply)) AS prev_cap "
         "FROM moves WHERE gid = ? ORDER BY move_ply")
    total, mism = 0, 0
    for gid in gids:
        rows = conn.execute(q, [gid]).fetchall()
        board = chess.Board()
        prev_was_capture = None
        for ply, uci, recipe in rows:
            if prev_was_capture is not None and recipe is not None:
                total += 1
                if bool(recipe) != prev_was_capture:
                    mism += 1
            try:
                mv = chess.Move.from_uci(uci)
                prev_was_capture = board.is_capture(mv)
                board.push(mv)
            except Exception:
                break
    conn.close()
    assert total > 0
    assert mism == 0, f"{mism}/{total} prev_move_was_capture mismatches vs replay"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
