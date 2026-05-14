"""Pytest bootstrap: put ``src/`` on ``sys.path`` so ``import cts.*`` works without ``pip install -e .``."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
