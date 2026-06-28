#!/usr/bin/env python3
"""Slice stage-sections or query values from a merged per-rung config.

    render_stage.py CONFIG STAGE --out OUT [--set KEY=VALUE ...]
    render_stage.py CONFIG --get KEY

CONFIG  a merged per-rung config (configs/elo{ELO}.yaml) containing a `globals`
        section and one section per stage (treegen, split, gnn_pack, mc_pack,
        encoder, materialize, train, eval).
STAGE   which top-level section to emit as flat YAML.
--out   where to write the flat YAML.
--get   query and print a single resolved value (e.g., globals.scratch_dir).
--set   inject/override a stage field (value is YAML-parsed).
"""
import argparse
import re
import sys
from pathlib import Path

import yaml


def interpolate(value, variables):
    """Recursively substitute ${key} or ${globals.key} placeholders in value."""
    if isinstance(value, str):
        def repl(match):
            name = match.group(1)
            clean_name = name.removeprefix("globals.")
            if clean_name in variables:
                return str(variables[clean_name])
            return match.group(0)
        return re.sub(r"\$\{([^}]+)\}", repl, value)
    elif isinstance(value, dict):
        return {k: interpolate(v, variables) for k, v in value.items()}
    elif isinstance(value, list):
        return [interpolate(v, variables) for v in value]
    return value


def set_nested(target, dotted_key, value):
    """Set value at nested dot-separated path in target dictionary."""
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def get_nested(target, dotted_key):
    """Retrieve a nested value from a dictionary using a dot-separated path."""
    keys = dotted_key.split(".")
    cursor = target
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError(f"Key {key!r} not found in path {dotted_key!r}")
        cursor = cursor[key]
    return cursor


def deep_merge(base, override):
    """Recursively merge ``override`` onto ``base`` (override wins; dicts merge)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_with_extends(path, _seen=None):
    """Load a config YAML, resolving a top-level ``extends: <file>`` chain.

    A per-rung spinoff (e.g. ``elo1800.yaml``) carries only ``extends: core.yaml``
    plus the handful of fields that differ (``globals.sf_elo``); the shared base
    lives once in ``core.yaml``. ``extends`` is resolved relative to the file that
    declares it; the spinoff is deep-merged *onto* the base so its values win.
    """
    path = Path(path)
    _seen = _seen or []
    if path in _seen:
        sys.exit(f"render_stage: circular extends via {path}")
    data = yaml.safe_load(path.read_text()) or {}
    base_ref = data.pop("extends", None)
    if base_ref is None:
        return data
    base = load_with_extends(path.parent / base_ref, _seen + [path])
    return deep_merge(base, data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render sections or query keys from merged configs.")
    ap.add_argument("config", help="Path to merged config YAML file.")
    ap.add_argument("stage", nargs="?", help="Stage section to extract (not required with --get).")
    ap.add_argument("--out", help="Path to output flat YAML (not required with --get).")
    ap.add_argument("--get", help="Dotted path key to query and print (e.g. globals.scratch_dir).")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="inject/override a stage field (repeatable); value is YAML-parsed")
    args = ap.parse_args()

    if not args.get and (not args.stage or not args.out):
        ap.error("Either --get or both STAGE and --out must be specified.")

    data = load_with_extends(args.config)

    # Apply dotted overrides to data (especially globals) before variables build & interpolation
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"render_stage: --set expects KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        key = key.strip()
        parsed_val = yaml.safe_load(value)
        if "." in key:
            set_nested(data, key, parsed_val)

    # Build and resolve the globals variables block (supporting nested references)
    variables = {}
    for k, v in data.get("globals", {}).items():
        variables[k] = v
        variables[f"globals.{k}"] = v

    for _ in range(5):
        variables = {k: interpolate(v, variables) for k, v in variables.items()}
        for k, v in variables.items():
            clean_k = k.removeprefix("globals.")
            variables[clean_k] = v
            variables[f"globals.{clean_k}"] = v

    # Interpolate all config data with resolved variables
    data = interpolate(data, variables)

    if args.get:
        try:
            val = get_nested(data, args.get)
            print(val)
            sys.exit(0)
        except KeyError as e:
            sys.exit(f"render_stage: {e}")

    # Stage rendering logic
    if args.stage not in data:
        sys.exit(f"render_stage: stage {args.stage!r} not in {args.config} "
                 f"(have: {sorted(k for k in data if isinstance(data[k], dict))})")

    section = dict(data[args.stage] or {})
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"render_stage: --set expects KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        key = key.strip()
        if "." not in key:
            section[key] = yaml.safe_load(value)

    Path(args.out).write_text(yaml.safe_dump(section, sort_keys=False))
    print(f"render_stage: {args.stage} -> {args.out} ({len(section)} keys)")


if __name__ == "__main__":
    main()
