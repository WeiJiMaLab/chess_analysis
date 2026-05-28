#!/usr/bin/env python3
"""Plot Stage 2b training curves from ``*_metrics.json`` files.

Writes per-variant train-loss PNGs, validation/expected-regret eval PNGs (when present),
and combined overlays: ``overlay_loss.png``, ``overlay_regret.png``.

Usage::

    python3 analysis/2b_train_controller.py --all --max-batches 1000 --save --eval-interval 200
    python3 analysis/2b_plot_loss.py
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
    return sorted(scan_dir.glob("*_metrics.json"), key=lambda p: p.name)


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


def regret_series_key(curves: dict[str, Any]) -> str | None:
    """Metric key for validation regret overlay (expected preferred over legacy greedy)."""
    if curves.get("eval_expected_regret"):
        return "eval_expected_regret"
    if curves.get("eval_greedy_regret"):
        return "eval_greedy_regret"
    return None


def has_eval_curves(curves: dict[str, Any]) -> bool:
    return bool(curves.get("eval_batch")) and regret_series_key(curves) is not None


def regret_ylabel(curves: dict[str, Any]) -> str:
    key = regret_series_key(curves)
    if key == "eval_expected_regret":
        return "Expected validation regret"
    return "Greedy validation regret"


def regret_title_suffix(curves: dict[str, Any]) -> str:
    run = curves.get("_run", {})
    n_ep = run.get("eval_max_episodes")
    ep_note = ""
    if n_ep is not None and int(n_ep) > 0:
        ep_note = f" (first {int(n_ep):,} val episodes)"
    if regret_series_key(curves) == "eval_expected_regret":
        tau = run.get("eval_stop_temperature")
        tau_note = f", τ={tau:g}" if tau is not None else ""
        return f"Expected regret (prob. stop){tau_note}{ep_note}"
    return f"Greedy regret{ep_note}"


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
    axes[0].set_title("Train total loss vs batch index")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(batch, mse, label="advantage_mse", linewidth=1.0)
    axes[1].plot(batch, bce, label="sign_bce", linewidth=1.0, alpha=0.85)
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Loss component")
    axes[1].set_title("Train loss components vs batch index")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Stage 2b — {label} (train; total = mse + {sign_loss_weight}×bce)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_eval(
    curves: dict[str, Any],
    output: Path,
    *,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    if not has_eval_curves(curves):
        return

    regret_key = regret_series_key(curves)
    assert regret_key is not None

    eval_batch = curves["eval_batch"]
    val_mse = curves["eval_validation_mse"]
    regret = curves[regret_key]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(eval_batch, val_mse, marker="o", markersize=4, linewidth=1.2, color="#55a868")
    axes[0].set_xlabel("Training batch index")
    axes[0].set_ylabel("Validation advantage MSE")
    axes[0].set_title("Validation MSE vs batch index")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_batch, regret, marker="o", markersize=4, linewidth=1.2, color="#c44e52")
    axes[1].set_xlabel("Training batch index")
    axes[1].set_ylabel(regret_ylabel(curves))
    axes[1].set_title(f"{regret_title_suffix(curves)} vs batch index")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Stage 2b — {label} (periodic eval)", fontsize=10, y=1.02)
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
    y_key: str = "total_loss",
    batch_key: str = "batch",
    title: str = "Total loss vs batch index (all variants)",
    ylabel: str = "Total loss",
    y_key_resolver: Any | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, path in enumerate(paths):
        curves = load_metrics(path)
        key = y_key_resolver(curves) if y_key_resolver is not None else y_key
        if batch_key not in curves or key not in curves:
            continue
        if not curves[batch_key] or not curves[key]:
            continue
        name = labels[i] if labels and i < len(labels) else label_for_metrics(path, curves)
        ax.plot(curves[batch_key], curves[key], linewidth=1.0, label=name, marker="o", markersize=3)
    ax.set_xlabel("Batch index" if batch_key == "batch" else "Training batch index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    eval_paths: list[Path] = []
    eval_labels: list[str] = []

    for path in paths:
        curves = load_metrics(path)
        label = label_for_metrics(path, curves)
        labels.append(label)
        vname = variant_name_from_path(path)
        plot(curves, plot_dir / f"{vname}_loss.png", label=label)
        if has_eval_curves(curves):
            plot_eval(curves, plot_dir / f"{vname}_eval.png", label=label)
            eval_paths.append(path)
            eval_labels.append(label)

    if do_overlay and len(paths) >= 1:
        plot_overlay(paths, plot_dir / "overlay_loss.png", labels=labels)
    if do_overlay and len(eval_paths) >= 1:
        sample = load_metrics(eval_paths[0])
        regret_lbl = regret_ylabel(sample)
        regret_ttl = f"{regret_title_suffix(sample)} vs batch index (all variants)"
        plot_overlay(
            eval_paths,
            plot_dir / "overlay_regret.png",
            labels=eval_labels,
            batch_key="eval_batch",
            ylabel=regret_lbl,
            title=regret_ttl,
            y_key_resolver=regret_series_key,
        )
        plot_overlay(
            eval_paths,
            plot_dir / "overlay_val_mse.png",
            labels=eval_labels,
            y_key="eval_validation_mse",
            batch_key="eval_batch",
            title="Validation advantage MSE vs batch index (all variants)",
            ylabel="Validation advantage MSE",
        )
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
        help="Skip combined overlays when scanning a directory.",
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
            vname = variant_name_from_path(paths[0])
            out = args.output or args.plot_dir / f"{vname}_loss.png"
            plot(curves, out, label=label)
            if has_eval_curves(curves):
                plot_eval(curves, args.plot_dir / f"{vname}_eval.png", label=label)
        else:
            labels = [label_for_metrics(p, load_metrics(p)) for p in paths]
            out = args.output or args.plot_dir / "overlay_loss.png"
            plot_overlay(paths, out, labels=labels)
            eval_paths = [p for p in paths if has_eval_curves(load_metrics(p))]
            if eval_paths:
                eval_labels = [label_for_metrics(p, load_metrics(p)) for p in eval_paths]
                sample = load_metrics(eval_paths[0])
                plot_overlay(
                    eval_paths,
                    args.plot_dir / "overlay_regret.png",
                    labels=eval_labels,
                    batch_key="eval_batch",
                    ylabel=regret_ylabel(sample),
                    title=f"{regret_title_suffix(sample)} vs batch index (all variants)",
                    y_key_resolver=regret_series_key,
                )
        return

    plot_scan_dir(args.scan_dir, args.plot_dir, do_overlay=not args.no_overlay)


if __name__ == "__main__":
    main()
