"""
Tests for n_self_pieces_exc_pawns / n_opp_pieces_exc_pawns SQL (preprocess_data).
"""

from __future__ import annotations

import unittest

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore


def _side_exc_counts(con: duckdb.DuckDBPyConnection, placement: str, player_white: bool) -> tuple[int, int]:
    row = con.execute(
        """
        SELECT
            CASE WHEN player_white THEN len(regexp_extract_all(replace(placement, '/', ''), '[RNBQK]'))
                 ELSE len(regexp_extract_all(replace(placement, '/', ''), '[rnbqk]'))
            END AS n_self,
            CASE WHEN player_white THEN len(regexp_extract_all(replace(placement, '/', ''), '[rnbqk]'))
                 ELSE len(regexp_extract_all(replace(placement, '/', ''), '[RNBQK]'))
            END AS n_opp
        FROM (SELECT ?::BOOLEAN AS player_white, ?::VARCHAR AS placement) t
        """,
        [player_white, placement],
    ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


@unittest.skipIf(duckdb is None, "duckdb not installed")
class TestSelfOppPiecesExcPawns(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()

    def tearDown(self) -> None:
        self.con.close()

    def test_starting_position_white_to_move(self) -> None:
        pl = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        s, o = _side_exc_counts(self.con, pl, True)
        self.assertEqual(s, 8)
        self.assertEqual(o, 8)

    def test_starting_position_black_to_move(self) -> None:
        pl = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        s, o = _side_exc_counts(self.con, pl, False)
        self.assertEqual(s, 8)
        self.assertEqual(o, 8)

    def test_white_king_only_white_moves(self) -> None:
        pl = "K7/8/8/8/8/8/8/8"
        s, o = _side_exc_counts(self.con, pl, True)
        self.assertEqual(s, 1)
        self.assertEqual(o, 0)

    def test_black_king_only_black_moves(self) -> None:
        pl = "8/8/8/8/8/8/8/k7"
        s, o = _side_exc_counts(self.con, pl, False)
        self.assertEqual(s, 1)
        self.assertEqual(o, 0)

    def test_black_king_only_but_white_flag_swaps_roles(self) -> None:
        pl = "8/8/8/8/8/8/8/k7"
        s, o = _side_exc_counts(self.con, pl, True)
        self.assertEqual(s, 0)
        self.assertEqual(o, 1)

    def test_mixed_board_counts_by_case(self) -> None:
        pl = "8/8/8/3N4/8/8/8/3Pk2K"
        s_w, o_w = _side_exc_counts(self.con, pl, True)
        self.assertEqual(s_w, 2)
        self.assertEqual(o_w, 1)
        s_b, o_b = _side_exc_counts(self.con, pl, False)
        self.assertEqual(s_b, 1)
        self.assertEqual(o_b, 2)


if __name__ == "__main__":
    unittest.main()
