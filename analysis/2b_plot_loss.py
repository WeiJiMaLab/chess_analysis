#!/usr/bin/env python3
"""Plot Stage 2b training curves from ``*_metrics.json`` files.

Writes per-variant train-loss PNGs, validation/expected-regret eval PNGs (when present),
and combined overlays: ``overlay_loss.png``, ``overlay_regret.png``, ``overlay_val_mse.png``.

Usage::

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
    STAGE2B_VARIANT_ORDER,
    STAGE2B_VARIANTS,
)

# Stable colors across all figures (Tableau-style, colorblind-friendly).
VARIANT_COLORS: dict[str, str] = {
    "subtree_weight_root+budget": "#4c72b0",
    "subtree_weight_root": "#55a868",
    "no_subtree_weight_root+budget": "#c44e52",
}

FIG_DPI = 150
TRAIN_FIGSIZE = (10.0, 3.6)
EVAL_FIGSIZE = (10.0, 3.6)
OVERLAY_FIGSIZE = (8.5, 4.5)


def apply_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "-",
            "lines.linewidth": 1.4,
        }
    )


def find_metrics_files(scan_dir: Path) -> list[Path]:
    paths = list(scan_dir.glob("*_metrics.json"))
    return order_metrics_paths(paths)


def order_metrics_paths(paths: list[Path]) -> list[Path]:
    rank = {name: i for i, name in enumerate(STAGE2B_VARIANT_ORDER)}
    return sorted(paths, key=lambda p: rank.get(variant_name_from_path(p), 999))


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
    curves["_path"] = path
    return curves


def regret_series_key(curves: dict[str, Any]) -> str | None:
    if curves.get("eval_expected_regret"):
        return "eval_expected_regret"
    if curves.get("eval_greedy_regret"):
        return "eval_greedy_regret"
    return None


def has_eval_curves(curves: dict[str, Any]) -> bool:
    return bool(curves.get("eval_batch")) and regret_series_key(curves) is not None


def variant_name_from_path(path: Path) -> str:
    stem = path.name
    if stem.endswith("_metrics.json"):
        return stem[: -len("_metrics.json")]
    return path.stem


def label_for_metrics(path: Path, curves: dict[str, Any]) -> str:
    run = curves.get("_run", {})
    if run.get("variant_label"):
        return str(run["variant_label"])
    name = variant_name_from_path(path)
    if name in STAGE2B_VARIANTS:
        return STAGE2B_VARIANTS[name].plot_label
    return name.replace("_", " ")


def color_for_variant(path: Path) -> str:
    name = variant_name_from_path(path)
    return VARIANT_COLORS.get(name, "#888888")


def eval_footnote(curves: dict[str, Any]) -> str:
    run = curves.get("_run", {})
    parts: list[str] = []
    n_batches = run.get("max_batches_requested")
    if n_batches is not None:
        parts.append(f"{int(n_batches):,}-batch proxy")
    interval = run.get("eval_interval")
    if interval:
        parts.append(f"eval every {int(interval)} batches")
    n_ep = run.get("eval_max_episodes")
    if n_ep is not None and int(n_ep) > 0:
        parts.append(f"{int(n_ep):,} val episodes")
    elif n_ep == 0:
        parts.append("full val set")
    if regret_series_key(curves) == "eval_expected_regret":
        tau = run.get("eval_stop_temperature")
        if tau is not None:
            parts.append(f"P(stop)=σ(−a/τ), τ={float(tau):g}")
        else:
            parts.append("probabilistic stop")
    else:
        parts.append("greedy stop (a≤0)")
    return " · ".join(parts)


def regret_ylabel(curves: dict[str, Any]) -> str:
    if regret_series_key(curves) == "eval_expected_regret":
        return "Expected regret"
    return "Greedy regret"


def _style_axes(ax: Any) -> None:
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_train(
    curves: dict[str, Any],
    output: Path,
    *,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    batch = curves["batch"]
    total_loss = curves["total_loss"]
    mse = curves["advantage_mse"]
    bce = curves["sign_bce"]
    color = color_for_variant(curves["_path"])
    sign_w = float(curves.get("sign_loss_weight", 0.1))

    fig, axes = plt.subplots(1, 2, figsize=TRAIN_FIGSIZE, sharex=True)

    axes[0].plot(batch, total_loss, color=color, linewidth=1.2, alpha=0.95)
    axes[0].set_ylabel("Train total loss")
    axes[0].set_title("Total loss")
    _style_axes(axes[0])

    axes[1].plot(batch, mse, color=color, linewidth=1.2, label="advantage MSE", alpha=0.95)
    axes[1].plot(batch, bce, color=color, linewidth=1.2, label="sign BCE", alpha=0.35)
    axes[1].set_ylabel("Train loss component")
    axes[1].set_title(f"Components (total = MSE + {sign_w:g}×BCE)")
    axes[1].legend(loc="upper right", frameon=False)
    _style_axes(axes[1])

    axes[1].set_xlabel("Training batch")
    fig.suptitle(label, fontsize=11, fontweight="medium", y=1.03)
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

    apply_plot_style()
    regret_key = regret_series_key(curves)
    assert regret_key is not None

    eval_batch = curves["eval_batch"]
    val_mse = curves["eval_validation_mse"]
    regret = curves[regret_key]
    color = color_for_variant(curves["_path"])

    fig, axes = plt.subplots(1, 2, figsize=EVAL_FIGSIZE, sharex=True)

    axes[0].plot(
        eval_batch,
        val_mse,
        color=color,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.5,
    )
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("Advantage MSE (full val cache)")
    _style_axes(axes[0])

    axes[1].plot(
        eval_batch,
        regret,
        color=color,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.5,
    )
    axes[1].set_ylabel(regret_ylabel(curves))
    axes[1].set_title("Policy regret (val subsample)")
    _style_axes(axes[1])

    axes[1].set_xlabel("Training batch")
    fig.suptitle(label, fontsize=11, fontweight="medium", y=1.03)
    fig.text(0.5, -0.02, eval_footnote(curves), ha="center", fontsize=8, color="#444444")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_overlay(
    paths: list[Path],
    output: Path,
    *,
    y_key: str,
    batch_key: str,
    title: str,
    ylabel: str,
    use_markers: bool,
    footnote: str | None = None,
    y_key_resolver: Any | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, ax = plt.subplots(figsize=OVERLAY_FIGSIZE)
    paths = order_metrics_paths(paths)

    for path in paths:
        curves = load_metrics(path)
        key = y_key_resolver(curves) if y_key_resolver is not None else y_key
        if batch_key not in curves or key not in curves:
            continue
        if not curves[batch_key] or not curves[key]:
            continue
        color = color_for_variant(path)
        label = label_for_metrics(path, curves)
        plot_kwargs: dict[str, Any] = {"color": color, "label": label, "linewidth": 1.6}
        if use_markers:
            plot_kwargs.update(
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgewidth=1.2,
            )
        else:
            plot_kwargs["alpha"] = 0.92
            plot_kwargs["linewidth"] = 1.3
        ax.plot(curves[batch_key], curves[key], **plot_kwargs)

    ax.set_xlabel("Training batch" if batch_key == "eval_batch" else "Batch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc="best", frameon=True, framealpha=0.92, edgecolor="#cccccc")
    _style_axes(ax)
    if footnote:
        fig.text(0.5, -0.01, footnote, ha="center", fontsize=8, color="#444444")
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
    paths = find_metrics_files(scan_dir)
    if not paths:
        raise SystemExit(f"No *_metrics.json files in {scan_dir}")

    plot_dir.mkdir(parents=True, exist_ok=True)
    eval_paths: list[Path] = []

    for path in paths:
        curves = load_metrics(path)
        label = label_for_metrics(path, curves)
        vname = variant_name_from_path(path)
        plot_train(curves, plot_dir / f"{vname}_loss.png", label=label)
        if has_eval_curves(curves):
            plot_eval(curves, plot_dir / f"{vname}_eval.png", label=label)
            eval_paths.append(path)

    if do_overlay and paths:
        plot_overlay(
            paths,
            plot_dir / "overlay_loss.png",
            y_key="total_loss",
            batch_key="batch",
            title="Train total loss (all variants)",
            ylabel="Train total loss",
            use_markers=False,
        )

    if do_overlay and eval_paths:
        sample = load_metrics(eval_paths[0])
        footnote = eval_footnote(sample)
        plot_overlay(
            eval_paths,
            plot_dir / "overlay_regret.png",
            y_key="eval_expected_regret",
            batch_key="eval_batch",
            title="Validation policy regret (all variants)",
            ylabel=regret_ylabel(sample),
            use_markers=True,
            footnote=footnote,
            y_key_resolver=regret_series_key,
        )
        plot_overlay(
            eval_paths,
            plot_dir / "overlay_val_mse.png",
            y_key="eval_validation_mse",
            batch_key="eval_batch",
            title="Validation advantage MSE (all variants)",
            ylabel="Validation MSE",
            use_markers=True,
            footnote=footnote,
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="*", type=Path)
    parser.add_argument("--scan-dir", type=Path, default=HL4291_2B_DIR)
    parser.add_argument("--plot-dir", type=Path, default=ANALYSIS_2B_PLOT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    if args.metrics:
        paths = order_metrics_paths(args.metrics)
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"Missing: {path}")
        if len(paths) == 1:
            curves = load_metrics(paths[0])
            label = label_for_metrics(paths[0], curves)
            vname = variant_name_from_path(paths[0])
            plot_train(curves, args.output or args.plot_dir / f"{vname}_loss.png", label=label)
            if has_eval_curves(curves):
                plot_eval(curves, args.plot_dir / f"{vname}_eval.png", label=label)
        else:
            plot_overlay(
                paths,
                args.output or args.plot_dir / "overlay_loss.png",
                y_key="total_loss",
                batch_key="batch",
                title="Train total loss (all variants)",
                ylabel="Train total loss",
                use_markers=False,
            )
            eval_paths = [p for p in paths if has_eval_curves(load_metrics(p))]
            if eval_paths:
                sample = load_metrics(eval_paths[0])
                plot_overlay(
                    eval_paths,
                    args.plot_dir / "overlay_regret.png",
                    y_key="eval_expected_regret",
                    batch_key="eval_batch",
                    title="Validation policy regret (all variants)",
                    ylabel=regret_ylabel(sample),
                    use_markers=True,
                    footnote=eval_footnote(sample),
                    y_key_resolver=regret_series_key,
                )
        return

    plot_scan_dir(args.scan_dir, args.plot_dir, do_overlay=not args.no_overlay)


if __name__ == "__main__":
    main()
