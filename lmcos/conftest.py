"""Pytest bootstrap: map ``import cts.*`` to ``src/`` and ``cts.analysis.*`` to ``analysis/``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ANALYSIS = ROOT / "analysis"


class _CtsPackageFinder:
    """Mirror ``package-dir`` in ``pyproject.toml`` without ``pip install -e .``."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "cts" and not fullname.startswith("cts."):
            return None
        if fullname == "cts.analysis" or fullname.startswith("cts.analysis."):
            suffix = fullname.removeprefix("cts.analysis.").split(".") if fullname != "cts.analysis" else []
            module_path = ANALYSIS.joinpath(*suffix) if suffix else ANALYSIS
        else:
            suffix = fullname.removeprefix("cts.").split(".") if fullname != "cts" else []
            module_path = SRC.joinpath(*suffix) if suffix else SRC
        init_py = module_path / "__init__.py"
        module_py = module_path.with_suffix(".py")
        if module_path.is_dir() and init_py.is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                init_py,
                submodule_search_locations=[str(module_path)],
            )
        if module_py.is_file():
            return importlib.util.spec_from_file_location(fullname, module_py)
        return None


if not any(isinstance(f, _CtsPackageFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _CtsPackageFinder())
