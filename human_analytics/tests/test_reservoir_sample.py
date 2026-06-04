"""Tests for DuckDB reservoir sampling on filtered subqueries."""

from __future__ import annotations

import duckdb
import pytest

from utils.selected_db import SELECTED_DB_DEFAULT, TABLE_PROCESSED_MOVES_NONZERO

_WHERE = """
    move_ply BETWEEN 15 AND 75
    AND opponent_clock_time >= 60
    AND move_time > 0
"""


@pytest.mark.integration
def test_filtered_subquery_then_sample_returns_exact_count():
    """USING SAMPLE must be outside the filtered subquery (DuckDB applies it before WHERE)."""
    n = 200
    conn = duckdb.connect(SELECTED_DB_DEFAULT, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    got = len(conn.execute(f"""
        SELECT fen
        FROM (
            SELECT fen
            FROM {TABLE_PROCESSED_MOVES_NONZERO}
            WHERE {_WHERE}
        ) filtered
        USING SAMPLE {n} ROWS (RESERVOIR, 42)
    """).df())
    conn.close()
    assert got == n


@pytest.mark.integration
def test_sample_before_where_returns_fewer_rows():
    """Document the footgun: WHERE after USING SAMPLE filters the sample down."""
    n = 200
    conn = duckdb.connect(SELECTED_DB_DEFAULT, read_only=True)
    conn.execute("SET enable_progress_bar = false")
    got = len(conn.execute(f"""
        SELECT fen
        FROM {TABLE_PROCESSED_MOVES_NONZERO}
        WHERE {_WHERE}
        USING SAMPLE {n} ROWS (RESERVOIR, 42)
    """).df())
    conn.close()
    assert got < n
