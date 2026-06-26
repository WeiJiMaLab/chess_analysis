"""
Tests for board-level and database-level calculations in human_analytics.
Consolidates board piece counts, side-specific piece counts, and reservoir sampling logic.
"""

from __future__ import annotations

import unittest
import pytest

try:
    import duckdb
except ImportError:
    duckdb = None


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
class TestBoard(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect()

    def tearDown(self) -> None:
        self.con.close()

    # --- Board Piece Counts Tests ---
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
        n_inc, n_exc = _counts_from_placement(self.con, "3q4/8/8/3k4/8/3K4/8/8")
        self.assertEqual(n_inc, 3)
        self.assertEqual(n_exc, 3)

    def test_ten_non_pawn_across_two_ranks(self) -> None:
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

    # --- Side-Specific Piece Counts Tests ---
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


# --- Reservoir Sampling Tests (Self-contained in-memory mock) ---
@pytest.fixture
def mock_db_conn():
    conn = duckdb.connect()
    conn.execute("""
        CREATE TABLE processed_moves_nonzero (
            fen VARCHAR,
            move_ply INT,
            opponent_clock_time INT,
            move_time INT
        )
    """)
    matching_rows = [("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 30, 100, 10)] * 500
    non_matching_rows = [("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 30, 0, 0)] * 500
    
    for row in matching_rows + non_matching_rows:
        conn.execute("INSERT INTO processed_moves_nonzero VALUES (?, ?, ?, ?)", row)
    
    yield conn
    conn.close()


_WHERE = """
    move_ply BETWEEN 15 AND 75
    AND opponent_clock_time >= 60
    AND move_time > 0
"""


def test_filtered_subquery_then_sample_returns_exact_count(mock_db_conn):
    n = 200
    mock_db_conn.execute("SET enable_progress_bar = false")
    got = len(mock_db_conn.execute(f"""
        SELECT fen
        FROM (
            SELECT fen
            FROM processed_moves_nonzero
            WHERE {_WHERE}
        ) filtered
        USING SAMPLE {n} ROWS (RESERVOIR, 42)
    """).df())
    assert got == n


def test_sample_before_where_returns_fewer_rows(mock_db_conn):
    n = 200
    mock_db_conn.execute("SET enable_progress_bar = false")
    got = len(mock_db_conn.execute(f"""
        SELECT fen
        FROM processed_moves_nonzero
        WHERE {_WHERE}
        USING SAMPLE {n} ROWS (RESERVOIR, 42)
    """).df())
    assert got < n


if __name__ == "__main__":
    unittest.main()
