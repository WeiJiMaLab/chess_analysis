"""Add `lmcos_small/human/` to `sys.path` so these pipeline scripts can import `utils`.

After the human_analytics -> lmcos_small merge this file lives at
`lmcos_small/slurm/human/scripts/_bootstrap.py`, while the `utils` package it
exposes lives at `lmcos_small/human/utils/`. The paths below are written
relative to this file so they survive the move.
"""

from __future__ import annotations

import os
import sys


def src_root() -> str:
    """Absolute path to `chess_analysis/lmcos_small/human` (holds the `utils` pkg)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "human")
    )


def project_root() -> str:
    """Absolute path to `chess_analysis` (repo root for `data/`, etc.)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )


def ensure_src() -> str:
    root = src_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
