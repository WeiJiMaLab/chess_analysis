#!/usr/bin/env python3
"""Slice one stage-section out of a merged per-rung config into a flat YAML that a
single `cts` entry point (each a pydantic model with extra="forbid") will accept.

    render_stage.py CONFIG STAGE --out OUT [--set KEY=VALUE ...]

CONFIG  a merged per-rung config (configs/elo{ELO}.yaml) whose top-level keys are
        stage names (treegen, split, gnn_pack, mc_pack, encoder, materialize,
        train, eval), each mapping to the exact flat fields that stage's entry
        point expects.
STAGE   which top-level section to emit.
--set   inject/override a key (YAML-parsed value). Used for the few values that
        are per-invocation rather than per-rung — e.g. the materialize stage's
        per-split output_dir / packed_data / final_cache, or mc_pack's worker
        count from $SLURM_CPUS_PER_TASK.

This is the *only* glue the merged-config design needs: the copied src/cts code is
untouched and still consumes one flat `--config FILE` per call.
"""
import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("stage")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="inject/override a field (repeatable); value is YAML-parsed")
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.stage not in data:
        sys.exit(f"render_stage: stage {args.stage!r} not in {args.config} "
                 f"(have: {sorted(k for k in data if isinstance(data[k], dict))})")
    section = dict(data[args.stage] or {})
    for kv in args.set:
        if "=" not in kv:
            sys.exit(f"render_stage: --set expects KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        section[key.strip()] = yaml.safe_load(value)

    Path(args.out).write_text(yaml.safe_dump(section, sort_keys=False))
    print(f"render_stage: {args.stage} -> {args.out} ({len(section)} keys)")


if __name__ == "__main__":
    main()
