"""S1 diagnostic (plan.md): stats ablation for the Stats-Controller.

Standalone diagnostic script -- NOT a pipeline stage, does not touch/modify any pipeline
code. Reuses the exact fitting/regret machinery already used by ``analysis.evaluate``'s
frontier plot (``_load_split_episodes``, ``_split``, ``_return_curves``, ``_regret_at``,
``_train_readout``, ``fit_singlehalt_stop``) -- the SAME 70/30 tree-level fit/eval split,
SAME cost regime (``eval.time_mode``/``time_lambda``/``maintenance_scale``/
``maintenance_exponent``), SAME PG-readout training recipe (200 epochs, lr=1e-3) that
produced the real ``SingleHalt*=0.1204 / Stats-Controller=0.1137`` numbers already on
record (``diagnosis.md``, ``outputs/figures/.../normative/frontier_data.json``).

Candidate stat set (plan.md): ``{n_nodes, height, width, current_value}``, where
``current_value`` is the CURRENT step's own ``halt_reward`` (already-computed,
search-consolidated value at this snapshot -- distinct from R(t), which is about FUTURE
improvement). ``steps_taken`` is always included as the base feature (matches
``analysis.evaluate._steps_stats_tensor``'s convention: C-step is a budget-out feature,
not part of the ablation).

For each of ``{all, all-minus-one-per-stat}`` (5 cells total) we refit a fresh
Stats-Controller-shaped readout on the FIT split and report held-out regret (mean + 95%
bootstrap CI) on the EVAL split, alongside SingleHalt*'s regret as a reference line -- a
subset whose removal barely moves the bar shows that stat isn't pulling weight.

    python -m cts.analysis.stats_ablation --config config_minply15_maxply75.yaml
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)  # unconstrained BLAS threading measured ~5x wall-time inflation on
                           # this shared login node (see cts.data.preprocess_mc.filter_argmax) --
                           # this script does many small per-episode PG-training loops, same failure mode.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.utils.helpers import load_config_section
from analysis.evaluate import (
    _load_split_episodes,
    _oracle_config,
    _return_curves,
    _regret_at,
    _split,
    _train_readout,
    fit_singlehalt_stop,
    _mean_ci,
)

_ALL_STATS = ("n_nodes", "height", "width", "current_value")
_STAT_KEY = {"n_nodes": "tree_sizes", "height": "heights", "width": "widths", "current_value": "halt_rewards"}


def steps_subset_tensor(ep: dict, stats: tuple[str, ...]) -> torch.Tensor:
    """``[steps_taken] + selected stats`` per step. ``steps_taken`` is always included
    (matches ``analysis.evaluate._steps_stats_tensor``'s "C-step + C-maint, budget-out"
    convention) -- only the stats subset varies across S1's ablation cells."""
    n = len(ep["halt_rewards"])
    cols = [np.arange(n, dtype=np.float64)] + [np.asarray(ep[_STAT_KEY[s]], dtype=np.float64) for s in stats]
    return torch.tensor(np.column_stack(cols), dtype=torch.float32)


def _subset_regret(episodes, fit_idx, ev_idx, fit_curves, ev_curves, stats: tuple[str, ...], seed: int) -> np.ndarray:
    fit_feats = [steps_subset_tensor(episodes[i], stats) for i in fit_idx]
    ev_feats = [steps_subset_tensor(episodes[i], stats) for i in ev_idx]
    stops = _train_readout(fit_feats, ev_feats, fit_curves, in_dim=1 + len(stats), epochs=200, lr=1e-3, seed=seed)
    return _regret_at(ev_curves, stops)


def compute_stats_ablation(packed_root: Path, *, time_mode: str, time_lambda: float,
                            maintenance_scale: float, maintenance_exponent: float,
                            max_episodes: int | None, seed: int) -> tuple[list[dict], dict]:
    """Returns ``(rows, meta)``: one row per ablation cell + a SingleHalt* reference row."""
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes = _load_split_episodes(packed_root, "validation", max_episodes=max_episodes)
    fit_idx, ev_idx = _split([ep["trajectory_key"] for ep in episodes], seed)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev_curves = [curves[i] for i in ev_idx]

    kf = fit_singlehalt_stop(fit_curves)
    kf = min(kf, max(len(c) for c in ev_curves) - 1)
    sh_mean, sh_lo, sh_hi = _mean_ci(_regret_at(ev_curves, [kf] * len(ev_curves)))

    cells = [("all", _ALL_STATS)] + [
        (f"all − {drop}", tuple(s for s in _ALL_STATS if s != drop)) for drop in _ALL_STATS
    ]
    rows: list[dict] = []
    for label, stats in cells:
        regret = _subset_regret(episodes, fit_idx, ev_idx, fit_curves, ev_curves, stats, seed)
        mean, lo, hi = _mean_ci(regret)
        rows.append({
            "subset": label, "stats": ",".join(stats), "n_features": 1 + len(stats),
            "regret_mean": round(mean, 6), "regret_lo": round(lo, 6), "regret_hi": round(hi, 6),
        })
        print(f"[stats_ablation] subset={label:16s} stats={stats} regret={mean:.4f} [{lo:.4f},{hi:.4f}]", flush=True)

    meta = {"singlehalt_k": int(kf), "singlehalt_regret_mean": round(sh_mean, 6),
            "singlehalt_regret_lo": round(sh_lo, 6), "singlehalt_regret_hi": round(sh_hi, 6),
            "n_fit": int(len(fit_idx)), "n_eval": int(len(ev_idx))}
    print(f"[stats_ablation] SingleHalt*(k={kf}) regret={sh_mean:.4f} [{sh_lo:.4f},{sh_hi:.4f}]  "
          f"n_fit={meta['n_fit']} n_eval={meta['n_eval']}", flush=True)
    return rows, meta


def _plot(rows: list[dict], meta: dict, out_dir: Path) -> str:
    """Bar chart: x = feature subset, y = held-out regret (lower is better), SingleHalt*'s
    regret drawn as a horizontal reference line -- a subset whose removal barely moves the
    bar shows that stat isn't pulling weight for Stats-Controller."""
    labels = [r["subset"] for r in rows]
    means = np.array([r["regret_mean"] for r in rows])
    los = np.array([r["regret_lo"] for r in rows])
    his = np.array([r["regret_hi"] for r in rows])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#12A19A"] + ["#3F4DA0"] * (len(labels) - 1)
    ax.bar(x, means, yerr=[means - los, his - means], capsize=4, color=colors, alpha=0.9)
    ax.axhline(meta["singlehalt_regret_mean"], color="#C0392B", linestyle="--", linewidth=1.4,
               label=f"SingleHalt* (k={meta['singlehalt_k']}) regret={meta['singlehalt_regret_mean']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("held-out mean regret (lower better)")
    ax.set_title("S1: Stats-Controller feature ablation\n(all − X = drop stat X; steps_taken always kept)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    full_regret = rows[0]["regret_mean"]
    biggest_drop_row = max(rows[1:], key=lambda r: r["regret_mean"] - full_regret)
    fig.text(
        0.5, -0.05,
        f"Plain-English finding: removing stat set '{biggest_drop_row['subset']}' "
        f"hurts regret most ({biggest_drop_row['regret_mean']:.3f} vs {full_regret:.3f} for all stats); bars near the\n"
        "SingleHalt* dashed line show that stat subset isn't beating a fixed stop step at all.",
        ha="center", va="top", fontsize=8.5, style="italic",
    )
    fig.tight_layout()

    png_dir, pdf_dir = out_dir / "png", out_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = png_dir / "s1_stats_ablation.png", pdf_dir / "s1_stats_ablation.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return str(png_path)


def run(config_path: str, limit: int | None, seed: int, out_dir: str | None) -> None:
    eval_cfg = load_config_section("eval", config_path)
    packed_root = Path(eval_cfg["packed_root"])
    out = Path(out_dir) if out_dir else Path("outputs/figures/minply15_maxply75/diagnosis")
    out.mkdir(parents=True, exist_ok=True)

    time_mode = eval_cfg.get("time_mode", "linear")
    time_lambda = float(eval_cfg.get("time_lambda", 0.01))
    maintenance_scale = float(eval_cfg.get("maintenance_scale", 0.0))
    maintenance_exponent = float(eval_cfg.get("maintenance_exponent", 1.0))

    print(f"[stats_ablation] packed_root={packed_root} out_dir={out} limit={limit} "
          f"time_mode={time_mode} time_lambda={time_lambda} maintenance_scale={maintenance_scale} "
          f"maintenance_exponent={maintenance_exponent}", flush=True)

    rows, meta = compute_stats_ablation(packed_root, time_mode=time_mode, time_lambda=time_lambda,
                                         maintenance_scale=maintenance_scale,
                                         maintenance_exponent=maintenance_exponent,
                                         max_episodes=limit, seed=seed)

    csv_path = out / "s1_stats_ablation.csv"
    fieldnames = ["subset", "stats", "n_features", "regret_mean", "regret_lo", "regret_hi"]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"subset": "SingleHalt*", "stats": "", "n_features": 1,
                          "regret_mean": meta["singlehalt_regret_mean"],
                          "regret_lo": meta["singlehalt_regret_lo"], "regret_hi": meta["singlehalt_regret_hi"]})
    print(f"[stats_ablation] wrote {csv_path}", flush=True)

    png_path = _plot(rows, meta, out)
    print(f"[stats_ablation] wrote {png_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the run config YAML.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on validation episodes (smoke slice). Default: full split.")
    parser.add_argument("--seed", type=int, default=0, help="Split / init seed.")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: outputs/figures/minply15_maxply75/diagnosis).")
    args = parser.parse_args()
    run(args.config, args.limit, args.seed, args.out_dir)


if __name__ == "__main__":
    main()
