"""Shared config loading: YAML file + CLI ``--override`` mechanism + Pydantic validation.

Every CTS entry point uses this. The pattern at each call site is:

    from cts._config import load_config, run_with_config_cli

    class MyConfig(BaseModel):
        # … typed fields with defaults …
        encoder_checkpoint: str
        seed: int = 0

    def run(config: MyConfig) -> None:
        # ... actual work ...

    if __name__ == "__main__":
        run_with_config_cli(MyConfig, run)

The CLI accepts ``--config PATH`` (required, path to a YAML file) plus zero
or more ``--override key=value`` flags. Override keys support dot-notation
for nested fields (``--override head.hidden_dim=512``). Values are parsed as
YAML scalars so ``--override seed=7`` yields an int, ``--override use_flag=true``
yields a bool, etc. — no manual type coercion.

Pydantic validates the merged config and rejects unknown fields by default,
so a typo in a YAML file or a CLI override is caught at load time instead of
silently producing the wrong behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

C = TypeVar("C", bound=BaseModel)


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``target[a][b][c] = value`` for ``dotted_key = "a.b.c"``.

    Creates intermediate dicts as needed. Raises if any intermediate path
    points at a non-dict value, since silently overwriting it would erase
    fields the caller didn't mean to touch.
    """
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        elif not isinstance(existing, dict):
            raise ValueError(
                f"override path {dotted_key!r} traverses non-dict at {key!r}; "
                f"cannot descend into a scalar value."
            )
        cursor = existing
    cursor[keys[-1]] = value


def load_config(
    config_class: Type[C],
    path: str,
    overrides: Optional[List[str]] = None,
) -> C:
    """Load ``path`` as YAML, apply ``overrides``, validate against ``config_class``.

    Args:
        config_class: Pydantic model the merged data must satisfy.
        path: filesystem path to a YAML file. Empty/missing top-level
            content is treated as ``{}`` (the config is assumed to be
            fully-defaulted).
        overrides: list of ``"key=value"`` strings. ``key`` may be
            dot-separated for nested fields. ``value`` is parsed as a
            YAML scalar (``"7"`` -> int 7, ``"true"`` -> bool True,
            ``"foo bar"`` -> str ``"foo bar"``).

    Returns:
        A validated instance of ``config_class``.
    """
    raw_text = Path(path).read_text()
    data: Dict[str, Any] = yaml.safe_load(raw_text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path!r} must contain a YAML mapping at the top level.")
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"--override must be of the form 'key=value', got: {override!r}")
        key, value_text = override.split("=", 1)
        _set_nested(data, key.strip(), yaml.safe_load(value_text))
    return config_class.model_validate(data)


def run_with_config_cli(config_class: Type[C], runner: Callable[[C], None]) -> None:
    """Standard ``if __name__ == "__main__"`` wiring for any CTS entry point.

    Parses ``--config PATH`` and ``--override key=value`` (repeatable),
    loads the validated config, and hands it to ``runner``. Use this in
    every entry point so the CLI shape stays consistent across the project.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field (repeatable). Supports dot-notation for nested keys.",
    )
    args = parser.parse_args()
    config = load_config(config_class, args.config, args.override)
    runner(config)
