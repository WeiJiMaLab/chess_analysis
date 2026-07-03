"""Lightweight DuckDB helpers shared across the pipeline (no plotting/chess deps)."""

from __future__ import annotations

import os

import duckdb


def sql_str(s: str) -> str:
    """Single-quoted SQL literal fragment."""
    return s.replace("'", "''")


def duckdb_connect_config(work_dir: str, threads: int, memory_limit: str) -> dict:
    """DuckDB ``connect`` config: spill/sort temp files live in ``work_dir``."""
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    return {
        "threads": int(threads),
        "memory_limit": str(memory_limit),
        "temp_directory": work_dir,
    }


def connect(database: str, work_dir: str, threads: int, memory_limit: str,
            read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open ``database`` with DuckDB spill/sort temp files routed to ``work_dir``."""
    return duckdb.connect(
        database=database,
        read_only=read_only,
        config=duckdb_connect_config(work_dir, threads, memory_limit),
    )
