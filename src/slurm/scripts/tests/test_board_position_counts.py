"""
Tests for DuckDB expressions that count pieces from FEN placement.

Must stay in sync with :func:`preprocess.process_moves` column definitions:
    n_pieces_on_board_inc_pawns, n_pieces_on_board_exc_pawns
"""

from __future__ import annotations

import unittest

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore


def _counts_from_placement(con: duckdb.DuckDBPyConnection, placement: str) -> tuple[int, int]:
    """Return (n_pieces_on_board_inc_pawns, n_pieces_on_board_exc_pawns) like ``processed_moves``."""
    row = con.execute(
        """
        SELECT
            len(regexp_extract_all(replace(placement, '/', ''), '[a-zA-Z]')) AS n_pieces_on_board_inc_pawns,
            len(regexp_extract_all(replace(placement, '/', ''), '[rnbqkRNBQK]')) AS n_pieces_on_board_exc_pawns
        FROM (SELECT ? AS placement) t
        """,
        [placement],
    ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1])


@unittest.skipIf(duckdb is None, "duckdb not installed")
class TestBoardPositionCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()

    def tearDown(self) -> None:
        self.con.close()

    def test_empty_board(self) -> None:
        n_inc, n_exc = _counts_from_placement(self.con, "8/8/8/8/8/8/8/8")
        self.assertEqual(n_inc, 0)
        self.assertEqual(n_exc, 0)

    def test_single_white_king(self) -> None:
        n_inc, n_exc = _counts_from_placement(self.con, "K7/8/8/8/8/8/8/8")
        self.assertEqual(n_inc, 1)
        self.assertEqual(n_exc, 1)

    def test_single_black_king(self) -> None:
        n_inc, n_exc = _counts_from_placement(self.con, "8/8/8/8/8/8/8/k7")
        self.assertEqual(n_inc, 1)
        self.assertEqual(n_exc, 1)

    def test_single_pawns_count_inc_only(self) -> None:
        n_inc_w, n_exc_w = _counts_from_placement(self.con, "8/P7/8/8/8/8/8/8")
        self.assertEqual(n_inc_w, 1)
        self.assertEqual(n_exc_w, 0)
        n_inc_b, n_exc_b = _counts_from_placement(self.con, "8/8/8/8/8/p7/8/8")
        self.assertEqual(n_inc_b, 1)
        self.assertEqual(n_exc_b, 0)

    def test_two_kings(self) -> None:
        n_inc, n_exc = _counts_from_placement(self.con, "K7/8/8/8/8/8/8/k7")
        self.assertEqual(n_inc, 2)
        self.assertEqual(n_exc, 2)

    def test_three_non_pawn_pieces(self) -> None:
        n_inc, n_exc = _counts_from_placement(self.con, "K6Q/8/8/8/8/8/8/k7")
        self.assertEqual(n_inc, 3)
        self.assertEqual(n_exc, 3)

    def test_mixed_pawn_and_pieces(self) -> None:
        """Knight + pawn + both kings: 4 inc, 3 exc."""
        n_inc, n_exc = _counts_from_placement(self.con, "8/8/8/3N4/8/8/8/3Pk2K")
        self.assertEqual(n_inc, 4)
        self.assertEqual(n_exc, 3)

    def test_starting_position(self) -> None:
        placement = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        n_inc, n_exc = _counts_from_placement(self.con, placement)
        self.assertEqual(n_inc, 32)
        self.assertEqual(n_exc, 16)

    def test_sparse_endgame(self) -> None:
        n_inc, n_exc = _counts_from_placement(self.con, "8/8/8/3k4/8/3K4/8/8")
        self.assertEqual(n_inc, 2)
        self.assertEqual(n_exc, 2)

    def test_promoted_queen_lowercase_preserved(self) -> None:
        """Black queen is 'q'; must still count in exc-pawn regex."""
        n_inc, n_exc = _counts_from_placement(self.con, "3q4/8/8/3k4/8/3K4/8/8")
        self.assertEqual(n_inc, 3)
        self.assertEqual(n_exc, 3)

    def test_ten_non_pawn_across_two_ranks(self) -> None:
        """Five black + five white non-pawn men; ranks sum to 8 squares each."""
        placement = "nqrbk3/8/8/8/8/8/8/NQRBK3"
        n_inc, n_exc = _counts_from_placement(self.con, placement)
        self.assertEqual(n_inc, 10)
        self.assertEqual(n_exc, 10)

    def test_exc_never_exceeds_inc(self) -> None:
        placements = [
            "8/8/8/8/8/8/8/8",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "1r2k2r/4npb1/pRPp3p/P2Pp1pn/1N4b1/2P2P1P/1K6/5BNR",
            "8/2p5/3p4/K2p4p/3Ppk2/8/8/8",
        ]
        for pl in placements:
            with self.subTest(placement=pl):
                n_inc, n_exc = _counts_from_placement(self.con, pl)
                self.assertLessEqual(n_exc, n_inc, msg=f"exc={n_exc} inc={n_inc}")


if __name__ == "__main__":
    unittest.main()
