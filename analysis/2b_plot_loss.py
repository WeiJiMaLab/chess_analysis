#!/usr/bin/env python3
"""Plot Stage 2b training loss curves from ``*_metrics.json`` files.

By default scans ``HL4291_2B_DIR`` for all ``*_metrics.json``, writes one two-panel
PNG per run under ``analysis/outputs/2b/``, plus an overlay of total loss.

Usage::

    python3 analysis/2b_train_controller.py --all --max-batches 1000 --save
    python3 analysis/2b_plot_loss.py

    python3 analysis/2b_plot_loss.py path/to/foo_metrics.json --output out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from config import (
    ANALYSIS_2B_PLOT_DIR,
    HL4291_2B_DIR,
    STAGE2B_VARIANTS,
)


def find_metrics_files(scan_dir: Path) -> list[Path]:
    """All ``*_metrics.json`` in ``scan_dir``, stable sort by variant name."""
    paths = sorted(scan_dir.glob("*_metrics.json"), key=lambda p: p.name)
    return paths


def load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "curves" not in payload:
        raise ValueError(f"{path}: expected top-level 'curves'")
    curves = payload["curves"]
    for key in ("batch", "total_loss", "advantage_mse", "sign_bce"):
        if key not in curves:
            raise ValueError(f"{path}: curves missing {key!r}")
    curves["_summary"] = payload.get("summary", {})
    curves["_run"] = payload.get("run", {})
    return curves


def label_for_metrics(path: Path, curves: dict[str, Any]) -> str:
    run = curves.get("_run", {})
    if run.get("variant_label"):
        return str(run["variant_label"])
    stem = path.name
    if stem.endswith("_metrics.json"):
        name = stem[: -len("_metrics.json")]
        if name in STAGE2B_VARIANTS:
            return STAGE2B_VARIANTS[name].plot_label
        return name.replace("_", " ")
    return path.stem


def variant_name_from_path(path: Path) -> str:
    stem = path.name
    if stem.endswith("_metrics.json"):
        return stem[: -len("_metrics.json")]
    return path.stem


def plot(
    curves: dict[str, Any],
    output: Path,
    *,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    batch = curves["batch"]
    total_loss = curves["total_loss"]
    mse = curves["advantage_mse"]
    bce = curves["sign_bce"]
    sign_loss_weight = float(curves.get("sign_loss_weight", 0.1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(batch, total_loss, linewidth=1.0, color="#4c72b0")
    axes[0].set_xlabel("Batch index")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss vs batch index")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(batch, mse, label="advantage_mse", linewidth=1.0)
    axes[1].plot(batch, bce, label="sign_bce", linewidth=1.0, alpha=0.85)
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Loss component")
    axes[1].set_title("Loss components vs batch index")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Stage 2b — {label} (total = mse + {sign_loss_weight}×bce)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_overlay(
    paths: list[Path],
    output: Path,
    *,
    labels: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, path in enumerate(paths):
        curves = load_metrics(path)
        name = labels[i] if labels and i < len(labels) else label_for_metrics(path, curves)
        ax.plot(curves["batch"], curves["total_loss"], linewidth=1.0, label=name)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Total loss")
    ax.set_title("Total loss vs batch index (all variants)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_scan_dir(
    scan_dir: Path,
    plot_dir: Path,
    *,
    do_overlay: bool = True,
) -> list[Path]:
    """Plot every ``*_metrics.json`` in ``scan_dir``; return paths plotted."""
    paths = find_metrics_files(scan_dir)
    if not paths:
        raise SystemExit(f"No *_metrics.json files in {scan_dir}")

    plot_dir.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    for path in paths:
        curves = load_metrics(path)
        label = label_for_metrics(path, curves)
        labels.append(label)
        out_name = f"{variant_name_from_path(path)}_loss.png"
        plot(curves, plot_dir / out_name, label=label)

    if do_overlay and len(paths) >= 1:
        plot_overlay(paths, plot_dir / "overlay_loss.png", labels=labels)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="*",
        type=Path,
        help="Optional explicit metrics.json path(s). Default: scan --scan-dir.",
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=HL4291_2B_DIR,
        help="Directory to scan for *_metrics.json when no paths given.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=ANALYSIS_2B_PLOT_DIR,
        help="Where to write PNGs when scanning.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single-file output (only with exactly one metrics path).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip combined overlay when scanning a directory.",
    )
    args = parser.parse_args()

    if args.metrics:
        paths = args.metrics
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"Missing: {path}")
        if len(paths) == 1:
            curves = load_metrics(paths[0])
            label = label_for_metrics(paths[0], curves)
            out = args.output or args.plot_dir / f"{variant_name_from_path(paths[0])}_loss.png"
            plot(curves, out, label=label)
        else:
            labels = [label_for_metrics(p, load_metrics(p)) for p in paths]
            out = args.output or args.plot_dir / "overlay_loss.png"
            plot_overlay(paths, out, labels=labels)
        return

    plot_scan_dir(args.scan_dir, args.plot_dir, do_overlay=not args.no_overlay)


if __name__ == "__main__":
    main()
