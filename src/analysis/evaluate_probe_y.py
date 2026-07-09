"""plan.md Probe Y -- apples-to-apples eval of Yotam Sagiv's subtree-size-weighted encoder
(``tree_encoder_child_wdl_async_k1_subtree_weighted.pt``) under THIS investigation's
current, calibrated methodology.

Context: Sagiv recalled a fitted-Q controller on top of this encoder ([z_t, T_t] inputs)
scoring "greedy regret" 0.024 -- recorded as "much better than the one trained on human
trees." That number comes from a different era's pipeline/scale (fitted-Q controller,
not the PG-trained advantage head this investigation now uses everywhere) and the same
lineage's own labnotebook shows "greedy regret" at wildly different scales across entries
over time. This script does NOT reuse the old fitted-Q controller checkpoints -- it takes
just the pretrained ENCODER, materializes z_t from it against a real, argmax-filtered
``human_trees`` validation split (Agent 1's, reused read-only), and fits a fresh
Stats-Controller / z_t-Controller pair on top using the exact SAME machinery every other
result in this investigation uses (``analysis.evaluate._fit_stop_controllers`` /
``_compute_frontier_data``, PG-trained advantage heads, paired-bootstrap significance via
``cts.stats.bootstrap_ci``, percentile not normal-theory).

The comparison this script resolves is WITHIN the human_trees population only: does the
subtree-weighted z_t-Controller significantly beat SingleHalt*/Stats-Controller THERE (all
three fit and scored on the identical held-out episodes)? It does NOT compare absolute
regret magnitudes against our own puctvalue_md36 corpus's ~0.11-0.12 numbers -- different
tree populations can have structurally different regret magnitudes for reasons unrelated
to methodology (branching, position-difficulty distribution), so an absolute cross-corpus
number is not informative on its own; only the within-population relative margin is.

Produces ONE combined figure (frontier-style comparison, both panels, annotated with the
paired-diff significance verdicts) at
``outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/probe_y_subtree_weighted_eval.*``.

Usage:
    python -m analysis.evaluate_probe_y \\
        --packed-root $MCP --cache $MAT/validation_cache.pt \\
        --out-dir outputs/figures/minply15_maxply75/diagnosis --d-embed 128
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.evaluate import (
    _C,
    _draw_zoom_indicator,
    _deconflict_points,
    _fit_stop_controllers,
    _frontier_panel,
    _load_assessment_data,
    _mean_ci,
    _oracle_config,
    _padded_range,
    _rcparams,
    _regret_at,
    _return_curves,
    _to_jsonable,
)
from analysis.utils.plots import save_pdf_png
from cts.stats import bootstrap_ci

_LABELS = {"singlehalt": "SingleHalt*", "stats": "Stats-Controller"}


def compute_probe_y(packed_root: Path, cache_path: str, *, d_embed: int = 128, max_episodes: int = 15000,
                     seed: int = 0, time_mode: str = "linear", time_lambda: float = 0.01,
                     maintenance_scale: float = 0.0, maintenance_exponent: float = 1.0,
                     n_boot: int = 2000) -> dict:
    """Fit SingleHalt* / Stats-Controller / subtree-weighted z_t-Controller once (shared fit/eval
    split + curves), on the human_trees argmax-filtered validation episodes. Returns everything the
    frontier plot AND the two paired-diff significance tests (SingleHalt* vs z_t, Stats vs z_t) need,
    built directly (not via two separate calls to ``_compute_frontier_data``, which would refit the
    200-epoch PG readouts twice for no reason)."""
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, d_embed, seed)

    # Sanity check (plan.md instruction): is the regime meaningful on THIS population, not just
    # assumed to transfer from the puctvalue_md36 grid it was originally selected on? A degenerate
    # SingleHalt*-fit k (0 = collapse to Always Stop, or the ceiling = Always Continue) would say no.
    kmax = max(len(c) for c in ev)
    k_singlehalt = ctrl["k_singlehalt"]
    regime_degenerate = k_singlehalt <= 0 or k_singlehalt >= kmax - 1

    fr = np.array([_regret_at(ev, [k] * len(ev)).mean() for k in range(kmax)])
    fr_x = np.array([np.mean([min(k, len(c) - 1) for c in ev]) for k in range(kmax)])

    def point(stops, color, lbl, **kw):
        xs = np.array([min(int(s), len(c) - 1) for s, c in zip(stops, ev)], dtype=float)
        xm, xlo, xhi = _mean_ci(xs)
        ym, ylo, yhi = _mean_ci(_regret_at(ev, stops))
        return dict(x=xm, xlo=xlo, xhi=xhi, y=ym, lo=ylo, hi=yhi, color=color, label=lbl, **kw)

    controllers = [point(ctrl["singlehalt"], _C["singlehalt"], "SingleHalt* (fixed stop)", zorder=7),
                  point(ctrl["stats"], _C["stats"], "Stats-Controller"),
                  point(ctrl["zt"], _C["zt"], "subtree-weighted $z_t$-Controller")]
    ends = [point(np.zeros(len(ev_idx), dtype=int), _C["always"], "Always Stop (k=0)"),
           point(np.full(len(ev_idx), kmax - 1, dtype=int), _C["never"], "Always Continue (k=max)")]
    all_points = controllers + ends

    zt_regret = _regret_at(ev, ctrl["zt"])
    paired_diffs = {}
    for name in ("singlehalt", "stats"):
        baseline_regret = _regret_at(ev, ctrl[name])
        # positive = baseline has HIGHER (worse) regret, i.e. subtree-weighted z_t wins
        paired_diffs[name] = bootstrap_ci(lambda d: float(d.mean()), baseline_regret - zt_regret, n_boot=n_boot)

    return {"fr_x": fr_x, "fr": fr, "controllers": controllers, "all_points": all_points,
            "k_singlehalt": k_singlehalt, "kmax": kmax, "regime_degenerate": regime_degenerate,
            "paired_diffs": paired_diffs, "n_eval": len(ev_idx), "n_fit": len(fit_idx),
            "time_mode": time_mode, "time_lambda": time_lambda,
            "maintenance_scale": maintenance_scale, "maintenance_exponent": maintenance_exponent}


def _render(data: dict, out_dir: Path) -> None:
    """Standard two-panel frontier plot (full range + zoom, same house style as
    ``analysis.evaluate._render_frontier``), annotated with the paired-diff significance verdicts
    as a suptitle finding -- same convention ``evaluate_e2e_z3._render_frontier_with_e2e`` uses.
    Single combined file (not split into a separate bar-panel figure) so the required output name
    (``probe_y_subtree_weighted_eval.{pdf,png}``) is one self-contained artifact."""
    fr_x, fr = np.asarray(data["fr_x"]), np.asarray(data["fr"])
    controllers, all_points = data["controllers"], data["all_points"]

    # Long-form finding (goes in the printed log + result JSON; the figure itself gets a much
    # more compact one-line-per-comparison version below -- an earlier version put this full
    # sentence into fig.suptitle(wrap=True) and, being long, it wrapped to enough lines to
    # visually overlap the right panel's axis (matplotlib doesn't reflow axes to make room for
    # a long suptitle; "tight" bbox_inches only expands the saved canvas, it doesn't prevent
    # overlap of already-positioned artists) -- confirmed by inspecting the rendered PNG.
    parts = []
    for name in ("singlehalt", "stats"):
        mean_d, lo_d, hi_d = data["paired_diffs"][name]
        label = _LABELS[name]
        if lo_d > 0:
            parts.append(f"CONFIRMED beats {label} (paired diff={mean_d:+.4f}, 95% CI [{lo_d:+.4f}, {hi_d:+.4f}])")
        elif hi_d < 0:
            parts.append(f"confirmed LOSES to {label} (paired diff={mean_d:+.4f}, 95% CI [{lo_d:+.4f}, {hi_d:+.4f}])")
        else:
            parts.append(f"NOT statistically distinguishable from {label} (paired diff={mean_d:+.4f}, "
                         f"95% CI [{lo_d:+.4f}, {hi_d:+.4f}] straddles 0)")
    regime_note = (" [REGIME WARNING: SingleHalt*-fit k is degenerate on this population]"
                   if data["regime_degenerate"] else "")
    finding = (f"subtree-weighted $z_t$-Controller (n={data['n_eval']} held-out human_trees episodes), "
              f"paired 95% bootstrap CI on per-episode regret diff: " + "; ".join(parts) + "." + regime_note)

    # Compact, figure-safe version: two short lines, one per baseline comparison.
    def _short(name):
        mean_d, lo_d, hi_d = data["paired_diffs"][name]
        verdict = "CONFIRMED" if lo_d > 0 else ("confirmed LOSS" if hi_d < 0 else "not significant")
        return f"vs {_LABELS[name]}: Δ={mean_d:+.3f} [{lo_d:+.3f},{hi_d:+.3f}] ({verdict})"
    title = (f"subtree-weighted $z_t$-Controller vs. baselines (n={data['n_eval']} human_trees episodes, "
            f"paired 95% bootstrap CI)\n" + "   ".join(_short(n) for n in ("singlehalt", "stats")))
    if data["regime_degenerate"]:
        title += "  [REGIME WARNING: degenerate SingleHalt* fit]"

    _rcparams()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})
    zoom_ylim = _padded_range(min(p["lo"] for p in controllers), max(p["hi"] for p in controllers), frac=0.25, min_pad=1e-3)
    zoom_xlim = _padded_range(min(p["x"] for p in controllers), max(p["x"] for p in controllers), frac=0.3, min_pad=1.0)
    zoom_xlim = (max(0, zoom_xlim[0]), zoom_xlim[1])
    full_lo, full_hi = _padded_range(min(p["lo"] for p in all_points), max(p["hi"] for p in all_points), frac=0.05, min_pad=1e-3)
    full_ylim = (min(0, full_lo), full_hi)
    all_points_dc = _deconflict_points(all_points, x_scale=fr_x[-1], y_scale=full_hi - full_ylim[0])
    _frontier_panel(axL, fr_x, fr, all_points_dc, xlim=(-1, fr_x[-1] + 1), ylim=full_ylim)
    _frontier_panel(axR, fr_x, fr, all_points_dc, ylim=zoom_ylim, xlim=zoom_xlim, label_line=True)
    _draw_zoom_indicator(fig, axL, axR, zoom_xlim, zoom_ylim)
    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9.5, frameon=False)
    fig.subplots_adjust(top=0.83)  # reserve headroom for the (short, fixed 2-3 line) title BEFORE
                                    # placing it -- tight_layout(rect=...) called after suptitle was
                                    # tried first and silently dropped the title from the saved
                                    # bbox_inches="tight" render; subplots_adjust does not have that
                                    # interaction and was confirmed to work by inspecting the PNG.
    title_obj = fig.suptitle(title, fontsize=10.5, y=0.97)
    save_pdf_png(fig, str(out_dir), "probe_y_subtree_weighted_eval", dpi=200,
                bbox_extra_artists=(fig.legends[0], title_obj))

    singlehalt_y, stats_y, zt_y = controllers[0]["y"], controllers[1]["y"], controllers[2]["y"]
    print(f"[probe-y] SingleHalt*={singlehalt_y:.4f} Stats={stats_y:.4f} subtree_zt={zt_y:.4f}  "
          f"k_singlehalt={data['k_singlehalt']}/{data['kmax']-1} degenerate={data['regime_degenerate']}  "
          f"-- {finding}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe Y -- subtree-weighted encoder vs SingleHalt*/Stats-Controller")
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--cache", required=True, help="materialized validation cache (.pt) from the subtree-weighted encoder, shuffle=False")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-embed", type=int, default=128)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-mode", default="linear")
    ap.add_argument("--time-lambda", type=float, default=0.01)
    ap.add_argument("--maintenance-scale", type=float, default=0.0)
    ap.add_argument("--maintenance-exponent", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = compute_probe_y(Path(args.packed_root), args.cache, d_embed=args.d_embed,
                           max_episodes=args.max_episodes, seed=args.seed, time_mode=args.time_mode,
                           time_lambda=args.time_lambda, maintenance_scale=args.maintenance_scale,
                           maintenance_exponent=args.maintenance_exponent)
    with open(out_dir / "probe_y_subtree_weighted_result.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)
    print(f"[probe-y] result -> {out_dir / 'probe_y_subtree_weighted_result.json'}", flush=True)
    _render(data, out_dir)

    confirmed_beats_singlehalt = data["paired_diffs"]["singlehalt"][1] > 0
    confirmed_beats_stats = data["paired_diffs"]["stats"][1] > 0
    print(f"[probe-y] SUCCESS CRITERION (CONFIRMED beats SingleHalt*, paired 95% CI excludes 0): {confirmed_beats_singlehalt}", flush=True)
    print(f"[probe-y] SUCCESS CRITERION (CONFIRMED beats Stats-Controller, paired 95% CI excludes 0): {confirmed_beats_stats}", flush=True)


if __name__ == "__main__":
    main()
