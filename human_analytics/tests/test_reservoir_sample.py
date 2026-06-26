"""Tests for DuckDB reservoir sampling on filtered subqueries."""

from __future__ import annotations

import duckdb
import pytest

# We run the tests on a mock in-memory database to keep them fast and deterministic.
@pytest.fixture
def mock_db_conn():
    conn = duckdb.connect()
    # Create a mock table matching TABLE_PROCESSED_MOVES_NONZERO
    conn.execute("""
        CREATE TABLE processed_moves_nonzero (
            fen VARCHAR,
            move_ply INT,
            opponent_clock_time INT,
            move_time INT
        )
    """)
    # Insert 1000 rows.
    # 500 rows match: move_ply=30 (BETWEEN 15 and 75), opponent_clock_time=100 (>= 60), move_time=10 (> 0)
    # 500 rows do not match: opponent_clock_time=0 and move_time=0
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
    """USING SAMPLE must be outside the filtered subquery (DuckDB applies it before WHERE)."""
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
    """Document the footgun: WHERE after USING SAMPLE filters the sample down."""
    n = 200
    mock_db_conn.execute("SET enable_progress_bar = false")
    got = len(mock_db_conn.execute(f"""
        SELECT fen
        FROM processed_moves_nonzero
        WHERE {_WHERE}
        USING SAMPLE {n} ROWS (RESERVOIR, 42)
    """).df())
    # Out of 1000 rows, 200 are sampled. Since only 50% match the WHERE clause,
    # the filtered result will contain approximately 100 rows, which is strictly less than 200.
    assert got < n
