"""Add `analysis/` to `sys.path` so pipeline scripts can import `utils`."""

from __future__ import annotations

import os
import sys


def src_root() -> str:
    """Absolute path to `chess_analysis/analysis`."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def project_root() -> str:
    """Absolute path to `chess_analysis` (repo root for `data/`, etc.)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def ensure_src() -> str:
    root = src_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
