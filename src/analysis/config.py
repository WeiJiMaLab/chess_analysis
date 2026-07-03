"""Load and ${}-interpolate sections of the run config.

Shared by analysis (helpers.CONFIG) and preprocess so both read paths/params
from the same yaml with the same placeholder rules. Kept dependency-light (no
matplotlib/chess) so the preprocessing job can import it without the plotting stack.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config_allply.yaml"


def config_path() -> Path:
    """Active config file: $CONFIG if set, else the repo default."""
    env = os.environ.get("CONFIG")
    return Path(env) if env else _DEFAULT_CONFIG


def _interpolate(value, variables):
    """Recursively substitute ${key} / ${globals.key} placeholders."""
    if isinstance(value, str):
        def repl(match):
            name = match.group(1).removeprefix("globals.")
            return str(variables[name]) if name in variables else match.group(0)
        return re.sub(r"\$\{([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, variables) for v in value]
    return value


def load_config_section(section: str, path: str | os.PathLike | None = None) -> dict:
    """Return one top-level section of the run config with ${...} placeholders
    resolved against ``globals``."""
    path = Path(path) if path else config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    body = data.get(section)
    if not isinstance(body, dict):
        raise KeyError(f"'{section}' section missing from {path}")
    variables = dict(data.get("globals", {}))
    for _ in range(5):
        variables = {k: _interpolate(v, variables) for k, v in variables.items()}
    return _interpolate(body, variables)
