"""Shared config loading: YAML file + CLI mechanism + Pydantic validation.

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

Two CLI shapes are supported:

1. **Flat config** (the original contract): ``--config PATH`` points at a YAML
   file whose top level *is* the config. ``--override key=value`` (repeatable,
   dot-notation, YAML-parsed) patches individual fields.

2. **Merged config** (lmcos_small): ``--config PATH --stage NAME`` points at a
   multi-stage config — a ``globals`` block plus one section per stage — and
   ``NAME`` selects the section to validate. ``${...}`` placeholders are resolved
   against ``globals`` first. ``--set key=value`` patches a *global* before
   interpolation when dotted (e.g. ``--set globals.sf_elo=1800``) or the selected
   *section* when bare (e.g. ``--set num_workers=32``). This lets every entry
   point read one shared ``config.yaml`` directly — no intermediary rendered YAML.

Pydantic validates the result and rejects unknown fields, so a typo in a YAML
file or a CLI flag is caught at load time.
"""

from __future__ import annotations

import argparse
import re
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


def _parse_kv(item: str, flag: str) -> tuple[str, Any]:
    """Split a ``"key=value"`` CLI item; the value is parsed as a YAML scalar."""
    if "=" not in item:
        raise ValueError(f"{flag} must be of the form 'key=value', got: {item!r}")
    key, value_text = item.split("=", 1)
    return key.strip(), yaml.safe_load(value_text)


# --- merged-config helpers (the `--stage` path) -----------------------------

def _interpolate(value: Any, variables: Dict[str, Any]) -> Any:
    """Recursively substitute ``${key}`` / ``${globals.key}`` placeholders."""
    if isinstance(value, str):
        def repl(match: "re.Match[str]") -> str:
            name = match.group(1).removeprefix("globals.")
            return str(variables[name]) if name in variables else match.group(0)
        return re.sub(r"\$\{([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, variables) for v in value]
    return value


def _resolve_globals(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build the variable table from ``data['globals']``, resolving nested refs."""
    variables: Dict[str, Any] = {}
    for key, value in (data.get("globals") or {}).items():
        variables[key] = value
        variables[f"globals.{key}"] = value
    for _ in range(5):  # fixpoint for globals that reference other globals
        variables = {k: _interpolate(v, variables) for k, v in variables.items()}
        for key, value in list(variables.items()):
            clean = key.removeprefix("globals.")
            variables[clean] = value
            variables[f"globals.{clean}"] = value
    return variables


def _apply_variant(data: Dict[str, Any]) -> None:
    """$VARIANT overlay for a normative variant branch (see config ``variants:``).
    Kept in sync with render_stage.apply_variant: fork the run's OUTPUTS to a variant
    subdir (run_name → run_name/variants/<VARIANT>), PIN the variant's shared upstream
    dirs to the base run, apply its (dotted) overrides, then drop the block."""
    import os
    variant = os.environ.get("VARIANT")
    vspec = (data.get("variants") or {}).get(variant) if variant else None
    if vspec:
        base = _resolve_globals(data)
        g = data.setdefault("globals", {})
        g["run_name"] = f"{base['run_name']}/variants/{variant}"
        for key in vspec.get("share", []):
            if key in base:
                g[key] = base[key]                 # literal base path — reused, not forked
        for k, v in (vspec.get("overrides") or {}).items():
            _set_nested(data, k, v)                 # dotted, e.g. train.prune_eps
    data.pop("variants", None)


def load_config(
    config_class: Type[C],
    path: str,
    overrides: Optional[List[str]] = None,
    stage: Optional[str] = None,
    sets: Optional[List[str]] = None,
) -> C:
    """Load ``path`` as YAML and validate against ``config_class``.

    Args:
        config_class: Pydantic model the result must satisfy.
        path: filesystem path to a YAML file.
        overrides: ``"key=value"`` patches (dot-notation, YAML-parsed) applied to
            the validated mapping — the flat top level, or the selected section
            when ``stage`` is given.
        stage: when set, ``path`` is treated as a merged multi-stage config; the
            ``globals`` block is interpolated into every ``${...}`` and this named
            section is sliced out and validated.
        sets: ``"key=value"`` patches for the merged path — dotted keys patch the
            raw data (e.g. ``globals.sf_elo``) *before* interpolation; bare keys
            patch the selected section *after* it.

    Returns:
        A validated instance of ``config_class``.
    """
    data: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path!r} must contain a YAML mapping at the top level.")

    # Dotted --set (e.g. globals.sf_elo) patches the raw tree before interpolation.
    for item in sets or []:
        key, value = _parse_kv(item, "--set")
        if "." in key:
            _set_nested(data, key, value)

    _apply_variant(data)   # $VARIANT overlay (fork outputs, pin shared upstream, overrides)

    if stage is not None:
        data = _interpolate(data, _resolve_globals(data))
        if not isinstance(data.get(stage), dict):
            sections = sorted(k for k, v in data.items() if isinstance(v, dict) and k != "globals")
            raise ValueError(f"--stage {stage!r} is not a section in {path!r} (have: {sections})")
        result: Dict[str, Any] = dict(data[stage])
        for item in sets or []:  # bare --set patches the section
            key, value = _parse_kv(item, "--set")
            if "." not in key:
                result[key] = value
    else:
        result = data

    for override in overrides or []:  # --override patches the final mapping
        key, value = _parse_kv(override, "--override")
        _set_nested(result, key, value)
    return config_class.model_validate(result)


def run_with_config_cli(config_class: Type[C], runner: Callable[[C], None]) -> None:
    """Standard ``if __name__ == "__main__"`` wiring for any CTS entry point.

    Parses ``--config PATH`` plus the optional merged-config selectors
    (``--stage``, ``--set``) and the flat-config patcher (``--override``), loads
    the validated config, and hands it to ``runner``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--stage", default=None,
                        help="Select a section of a merged multi-stage config (with globals).")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="Merged-config patch: dotted keys hit globals (pre-interp), bare keys the section.")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Patch a field of the final config (repeatable, dot-notation).")
    args = parser.parse_args()
    config = load_config(config_class, args.config, args.override, stage=args.stage, sets=args.sets)
    runner(config)
