"""
Skeptical provenance tests for the single python board featurizer
(board.calc_captures_checks, captures/checks via legal-move enumeration)
and the SQL-side board columns it complements.

Assume the metrics are wrong until proven right — hand-computed fixtures plus
DB cross-checks.
"""

from __future__ import annotations

import os
import chess
import pytest

from analysis.board import calc_captures_checks

DB_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/personal.db"
_HAS_DB = os.path.exists(DB_PATH)


# --------------------------------------------------------------------------
# hand-computed fixtures (adversarial) — captures/checks only
# --------------------------------------------------------------------------

def test_startpos_has_no_captures_or_checks():
    caps, checks = calc_captures_checks(chess.STARTING_FEN)
    assert caps == 0
    assert checks == 0


def test_pinned_piece_capture_is_illegal():
    """A pinned piece cannot capture — legal_moves must exclude it, so it does
    not inflate the capture count."""
    # White king e1, white knight e2 pinned by black rook e8; the knight is pinned
    # along the e-file so every knight move (incl. any capture) is illegal.
    fen = "4r3/8/8/8/8/3p4/4N3/4K3 w - - 0 1"
    board = chess.Board(fen)
    knight_moves = [m for m in board.legal_moves if m.from_square == chess.E2]
    assert knight_moves == []
    caps, _ = calc_captures_checks(fen)
    assert caps == sum(board.is_capture(m) for m in board.legal_moves)


def test_en_passant_capture_counted():
    """A 4-field FEN carrying the ep square must let python-chess see the ep
    capture (the DB fens are 4-field but include the ep square)."""
    # White pawn e5, black just played d7-d5; ep target d6. exd6 e.p. is legal.
    fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
    board = chess.Board(fen)
    assert any(board.is_en_passant(m) for m in board.legal_moves)
    caps, _ = calc_captures_checks(fen)
    assert caps >= 1


def test_checks_available_count():
    # White queen d1, black king d8 on an open d-file: Qd1-d8 etc. give check.
    fen = "3k4/8/8/8/8/8/8/3QK3 w - - 0 1"
    board = chess.Board(fen)
    expected = sum(board.gives_check(m) for m in board.legal_moves)
    _, checks = calc_captures_checks(fen)
    assert checks == expected
    assert expected >= 1


def test_determinism():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    assert calc_captures_checks(fen) == calc_captures_checks(fen)


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
    mism = [(fen, db) for fen, db in rows if chess.Board(fen).is_check() != bool(db)]
    assert not mism, f"{len(mism)} in_check mismatches vs moves.player_in_check, e.g. {mism[:3]}"


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
