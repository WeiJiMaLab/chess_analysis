#!/usr/bin/env python3
"""Time analysis I/O and a minimal controller-training step on ysagiv artifacts.

Usage::

    python3 analysis/smoke_benchmark.py
    python3 analysis/smoke_benchmark.py --train-epochs 1   # optional GPU train smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ANALYSIS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ANALYSIS_ROOT.parent
_LMCOS = _REPO_ROOT / "lmcos"
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from config import (  # noqa: E402
    ANALYSIS_ROOT,
    CACHE_SUBTREE_WEIGHTED_VAL,
    CONTROLLER_SUBTREE_ROOT_BUDGET,
    CONTROLLER_TRAIN_CONFIG_BEST,
    LMCOS_ROOT,
    PACKED_VALIDATION_MANIFEST,
)


def _fmt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def _timed(label: str, fn) -> None:
    t0 = time.perf_counter()
    fn()
    print(f"  {_fmt(time.perf_counter() - t0):>10}  {label}")


def bench_plot() -> None:
    from plot_controller_regret_curves import plot_runs, runs_from_config

    logs = ANALYSIS_ROOT / "logs"
    out = ANALYSIS_ROOT / "outputs" / "_smoke_regret.png"
    runs = runs_from_config(logs)
    if not any(r.path.exists() for r in runs):
        print("  (skip plot — no logs; run ./analysis/sync_logs.sh)")
        return
    _timed("regret curve PNG", lambda: plot_runs(runs, out))


def bench_cache_index() -> None:
    import torch

    def load():
        payload = torch.load(CACHE_SUBTREE_WEIGHTED_VAL, weights_only=False)
        assert payload.get("format") == "cts_materialized_advantage_cache_v2"

    _timed("load validation cache index (.pt)", load)


def bench_cache_shard() -> None:
    import torch

    def load_shard():
        payload = torch.load(CACHE_SUBTREE_WEIGHTED_VAL, weights_only=False)
        shard0 = Path(payload["shards"][0]["path"])
        torch.load(shard0, weights_only=False)

    _timed("load one validation cache shard", load_shard)


def bench_diagnostics_sample(n_lines: int = 500) -> None:
    path = CONTROLLER_SUBTREE_ROOT_BUDGET.diagnostics_jsonl
    if path is None or not path.exists():
        print("  (skip diagnostics sample — file missing)")
        return

    def read_lines():
        with path.open() as fh:
            for i, _ in enumerate(fh):
                if i >= n_lines - 1:
                    break

    _timed(f"read first {n_lines} diagnostics JSONL lines", read_lines)


def bench_manifest() -> None:
    def load():
        data = json.loads(PACKED_VALIDATION_MANIFEST.read_text())
        assert "total_episodes" in data

    _timed("read validation packed manifest JSON", load)


def bench_train_smoke(epochs: int) -> None:
    if not CONTROLLER_TRAIN_CONFIG_BEST.is_file():
        print("  (skip train — lmcos config missing)")
        return
    out_ckpt = ANALYSIS_ROOT / "outputs" / "_smoke_controller.pt"
    out_diag = ANALYSIS_ROOT / "outputs" / "_smoke_controller_diagnostics.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "cts.train.controller_train",
        "--config",
        str(CONTROLLER_TRAIN_CONFIG_BEST),
        "--override",
        f"epochs={epochs}",
        "--override",
        "log_interval=1000000",
        "--override",
        f"output_checkpoint={out_ckpt}",
        "--override",
        f"output_diagnostics={out_diag}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LMCOS_ROOT / "src") + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )

    print(f"  running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=LMCOS_ROOT, env=env, check=True)
    print(f"  {_fmt(time.perf_counter() - t0):>10}  controller_train ({epochs} epoch(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help="If >0, run controller_train for this many epochs (GPU; full train cache).",
    )
    args = parser.parse_args()

    print("Analysis smoke benchmarks\n")
    bench_manifest()
    bench_cache_index()
    bench_cache_shard()
    bench_diagnostics_sample()
    bench_plot()
    if args.train_epochs > 0:
        print()
        bench_train_smoke(args.train_epochs)
    else:
        print(
            "\nTrain smoke skipped (pass --train-epochs 1 to time one full epoch on "
            "~577k cached train snapshots; expect minutes on GPU, not seconds)."
        )


if __name__ == "__main__":
    main()
