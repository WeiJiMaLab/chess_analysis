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


# --- Skeptical Robustness Tests for the Analyzer Pipeline ---
from analysis.utils.analysis import Analyzer, Variable

class TestAnalyzerCorrectness(unittest.TestCase):
    """Rigorous tests designed to verify math, row conservation, tie-safety, and SQL security in Analyzer."""

    def setUp(self) -> None:
        self.con = duckdb.connect()
        self.con.execute("""
            CREATE TABLE test_moves (
                gid INT,
                move_ply INT,
                move_time DOUBLE,
                x_val DOUBLE
            )
        """)

    def tearDown(self) -> None:
        self.con.close()

    def _populate_data(self, rows: list[tuple]) -> None:
        for row in rows:
            self.con.execute("INSERT INTO test_moves VALUES (?, ?, ?, ?)", row)

    def test_row_conservation_with_zero_inflation_and_edge_mass(self) -> None:
        """Verify that zero-inflation, interior binning, and edge-masses conserve the total row count."""
        # 100 rows total: 40 are exactly 0.0, 40 are normal (1.0 to 4.0), 20 are edge mass (>= 10.0)
        data = []
        for i in range(40):
            data.append((1, 20, 5.0, 0.0))  # zero-inflated lump
        for i in range(40):
            data.append((1, 20, 5.0, 1.0 + (i % 4)))  # interior
        for i in range(20):
            data.append((1, 20, 5.0, 10.0))  # edge mass
        self._populate_data(data)

        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=5,
            zero_inflated=True,
            zero_threshold=0.01,
            edge_mass=[(">=", 10.0)],
            ply_tertile_source="test_moves"
        )

        # The sum of all counts in the binned dataframe must exactly equal the total rows (100)
        total_binned_rows = analyzer.quantile_df["n"].sum()
        self.assertEqual(total_binned_rows, 100, f"Expected 100 rows, got {total_binned_rows} (leakage or duplication!)")

    def test_tie_safety_grouping(self) -> None:
        """Verify that tie_safe=True groups identical values into the same bin instead of splitting them."""
        # 100 rows: 80 rows have x=5.0, 20 rows have x=10.0
        data = [(1, 20, 5.0, 5.0)] * 80 + [(1, 20, 5.0, 10.0)] * 20
        self._populate_data(data)

        # In tie_safe mode, we should get exactly 2 bins because there are only 2 distinct values,
        # and no row with x=5.0 should be split into the x=10.0 bin.
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=5,
            tie_safe=True,
            ply_tertile_source="test_moves"
        )

        self.assertEqual(len(analyzer.quantile_df), 2)
        # Check that the bins contain exactly the 80 and 20 counts
        counts = sorted(analyzer.quantile_df["n"].tolist())
        self.assertEqual(counts, [20, 80])

    def test_empty_dataset_graceful_handling(self) -> None:
        """Verify that the analyzer does not crash when the filtered dataset is completely empty."""
        # No data populated, table is empty
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=5,
            ply_tertile_source="test_moves"
        )
        self.assertEqual(analyzer.n_moves, 0)
        self.assertTrue(analyzer.quantile_df.empty or len(analyzer.quantile_df) == 0)

    def test_null_value_omission(self) -> None:
        """Verify that NULL/None values are skipped and do not cause SQL crashes."""
        # 5 rows, one has NULL x_val
        data = [
            (1, 20, 5.0, 1.0),
            (1, 20, 5.0, 2.0),
            (1, 20, 5.0, None),
            (1, 20, 5.0, 4.0),
        ]
        self._populate_data(data)
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=2,
            ply_tertile_source="test_moves"
        )
        # The NULL row is ignored in counting because it is filtered out by the finite condition in the view definition
        self.assertEqual(analyzer.n_moves, 3)

    def test_nan_inf_handling(self) -> None:
        """Verify that NaN, Inf, and -Inf values in the database are handled gracefully."""
        data = [
            (1, 20, 5.0, 1.0),
            (1, 20, float('nan'), 2.0),
            (1, 20, 5.0, float('inf')),
            (1, 20, float('-inf'), 4.0),
        ]
        self._populate_data(data)
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=2,
            ply_tertile_source="test_moves"
        )
        # Verify the pipeline runs and does not raise an exception.
        # Valid numerical rows are processed.
        self.assertGreater(analyzer.n_moves, 0)

    def test_log_variable_non_positive_values(self) -> None:
        """Verify that non-positive values for log-transformed variables do not crash the pipeline."""
        data = [
            (1, 20, 5.0, 1.0),
            (1, 20, -5.0, 0.0),  # Non-positive values for x or y
            (1, 20, 0.0, -10.0),
            (1, 20, 10.0, 2.0),
        ]
        self._populate_data(data)
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val", is_log=True),
            y_var=Variable("move_time", is_log=True),
            n_bins=2,
            ply_tertile_source="test_moves"
        )
        # ln(0) or ln(negative) will yield NULL in DuckDB, which is skipped by ntile and aggregations.
        # The pipeline should complete without crashing.
        self.assertGreater(analyzer.n_moves, 0)

    def test_edge_mass_overlapping_and_disjoint(self) -> None:
        """Verify behavior of both disjoint and overlapping edge masses."""
        # 10 rows: x values from 1 to 10
        data = [(1, 20, 5.0, float(i)) for i in range(1, 11)]
        self._populate_data(data)

        # 1. Disjoint edge masses: should conserve exactly 10 rows.
        analyzer_disjoint = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=3,
            edge_mass=[("<=", 2.0), (">=", 9.0)],
            ply_tertile_source="test_moves"
        )
        self.assertEqual(analyzer_disjoint.quantile_df["n"].sum(), 10)

        # 2. Overlapping edge masses: a skeptical agent expects duplication to be handled or documented.
        # Let's verify the exact duplication behavior to be precise.
        analyzer_overlap = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=3,
            edge_mass=[(">=", 7.0), (">=", 9.0)],
            ply_tertile_source="test_moves"
        )
        # Rows with x >= 9.0 match both clauses, so they are duplicated in the union.
        # Rows with x=9.0, 10.0 (2 rows) are duplicated. Total rows = 10 + 2 = 12.
        self.assertEqual(analyzer_overlap.quantile_df["n"].sum(), 12)

    def test_single_distinct_value_tie_safety(self) -> None:
        """Verify that tie_safe grouping works correctly when there is only a single distinct x value."""
        data = [(1, 20, 5.0, 42.0)] * 50
        self._populate_data(data)
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=5,
            tie_safe=True,
            ply_tertile_source="test_moves"
        )
        # Should result in exactly 1 bin containing all 50 rows, without division by zero or out of bounds.
        self.assertEqual(len(analyzer.quantile_df), 1)
        self.assertEqual(analyzer.quantile_df.iloc[0]["n"], 50)

    def test_single_row_dataset_stddev_and_plotting(self) -> None:
        """Verify that a dataset with exactly 1 row (where stddev is NULL/NaN) does not crash plotting."""
        data = [(1, 20, 5.0, 10.0)]
        self._populate_data(data)
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=5,
            ply_tertile_source="test_moves"
        )
        self.assertEqual(analyzer.n_moves, 1)
        # stddev of a single row is NaN. Verify that we can still plot it without crashing.
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        analyzer.plot_quantile_bins(ax)
        plt.close(fig)

    def test_extreme_min_bin_count_filtering(self) -> None:
        """Verify that when min_bin_count filters out all bins, the plotting functions handle it gracefully."""
        data = [(1, 20, 5.0, float(i)) for i in range(10)]
        self._populate_data(data)
        analyzer = Analyzer(
            db_conn=self.con,
            table_name="test_moves",
            x_var=Variable("x_val"),
            y_var=Variable("move_time"),
            n_bins=5,
            min_bin_count=100,  # higher than any bin count (bins have size 2)
            ply_tertile_source="test_moves"
        )
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # Should not crash even though all bins are filtered out
        analyzer.plot_quantile_bins(ax)
        plt.close(fig)


