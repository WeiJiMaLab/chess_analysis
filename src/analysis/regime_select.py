"""REGIME — confirm which (time_lambda, maintenance_scale) cost-regime points are "meaningful"
before any more frontier/significance plots get generated at them.

Everything validated so far in the z_t investigation (S1's stats ablation, Z's e2e comparison) ran at
exactly ONE point in cost-regime space (`time_lambda=0.01, maintenance_scale=0.0`). Separately,
`maintenance_scale in {0.05, 0.1}` was observed to collapse ALL controllers (SingleHalt*, Stats, z_t)
to identical degenerate regret. This module grid-searches `time_lambda x maintenance_scale` and marks
each point "meaningful" by a single cheap, pure-arithmetic criterion applied to SingleHalt* alone (no
model fitting beyond `evaluate.fit_singlehalt_stop`, no GPU):

    meaningful  iff  1 < k* < ceiling - 1

where `k*` is SingleHalt*'s own fit-split-optimal fixed stop step (`fit_singlehalt_stop`, a pure
argmin over `_regret_at`) and `ceiling = max(len(curve) for curve in curves)` is the longest episode's
step count at that grid point (`fit_singlehalt_stop`'s own `kmax`; `k*` can never exceed `ceiling - 1`
by construction, since it's chosen by `argmin` over `range(kmax)`). The two exclusions:

  * `k* <= 1`  — SingleHalt* wants to stop essentially immediately (a `k*->0` collapse: continuing is
    never worth the cost at this regime, so every controller degenerates to "always stop").
  * `k* >= ceiling - 1` — SingleHalt* is pinned at the LAST available stop step, i.e. it always wants
    to keep going as far as the data allows: since `k*` can literally never exceed `ceiling - 1`, this
    is the operational meaning of "k* -> ceiling pileup / right-censoring" in a domain where the argmin
    can't range past `ceiling - 1` — the fit can't tell whether the true optimum is at `ceiling - 1` or
    somewhere further out that the data doesn't reach.

See `/home/hl4291/chess_analysis/regime_select.md` for the full grid, criterion discussion, confirmed
regime set, and the collapse root-cause finding this sweep produced as a side effect.

    python -m analysis.regime_select --packed-root <mc_packed> --out-dir outputs/figures/minply15_maxply75/diagnosis
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from analysis.evaluate import (_load_split_episodes, _oracle_config, _return_curves, fit_singlehalt_stop,
                               _rcparams, _MUTED)
from analysis.utils.plots import save_pdf_png

# Grids recalibrated 2026-07-08 from an initial coarse probe: the naive order-of-magnitude grid
# {0.003,0.01,0.03,0.1,0.3} x {0.0,0.01,0.02,...,0.3} found k* collapsed to 0 at EVERY nonzero
# maintenance_scale tested (the coarsest nonzero point, 0.01, was already past the boundary) and at
# every lambda >= 0.03 -- both boundaries are much sharper/closer to 0 than the historical
# pre-argmax-filter sweep (labnotebook "headroom sweep") suggested. A finer follow-up probe (still
# pure arithmetic, still seconds) resolved the true collapse points: maintenance collapses to k*=0
# between 0.002 (k*=4, still meaningful) and 0.003 (k*=0) at lambda=0.01; lambda collapses to k*=0
# between 0.017 (k*=6) and 0.02 (k*=0) at maintenance=0; and the LOW-lambda side pileup (right-
# censoring at k*=ceiling-1=95) sets in below ~lambda=0.0001. These grids are shaped around those
# empirical boundaries so the sweep doesn't skip the narrow meaningful bands the way the naive grid did.
LAMBDA_GRID: tuple[float, ...] = (0.00005, 0.0001, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003,
                                  0.0035, 0.004, 0.0045, 0.005, 0.007, 0.01, 0.011, 0.012, 0.013, 0.014,
                                  0.015, 0.017, 0.02, 0.03, 0.05, 0.1, 0.3)
MAINT_GRID: tuple[float, ...] = (0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003,
                                 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3)

# The heatmap PLOT (not the underlying sweep(), which still runs the full LAMBDA_GRID/MAINT_GRID
# above) renders its OWN dedicated, log-spaced grid over the narrower band a human actually reads:
# everything past ~lambda=0.015 or maintenance=0.003 is uniform instant-stop-collapse (k_frac->0)
# AND hatched-not-meaningful, i.e. wasted plot area. Rendered with a x10^4 axis-value scale (see
# `_render_heatmap`'s `scale` param) so tick labels read in units of 1e-4 (e.g. "5" not "0.0005").
# Fourth pass (2026-07-09): these grid points are hand-picked, not data-derived, so there's no
# reason to keep `np.geomspace`'s ugly fractional output (5.83265, 0.743254, ...) around just to
# round it for display -- the grid VALUES themselves are just clean round numbers in x1e-4 units
# (whole numbers at >=1, one decimal place below 1), still roughly log-spaced.
ZOOM_LAMBDA_GRID: tuple[float, ...] = tuple(x * 1e-4 for x in (5, 6, 7, 8, 9, 11, 13, 15, 17, 20))
ZOOM_MAINT_GRID: tuple[float, ...] = tuple(x * 1e-4 for x in (0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5))


def _sweep_grid(episodes, base_cfg, *, lambda_grid: tuple[float, ...],
                 maint_grid: tuple[float, ...]) -> list[dict[str, Any]]:
    """Grid-search `time_lambda x maintenance_scale` over ALREADY-loaded episodes (pure arithmetic,
    so this runs in seconds regardless of grid size) — the shared body behind `sweep()` and the
    zoomed-grid pass in `main()`, which both need the same episode load reused across two grids."""
    results = []
    for lam in lambda_grid:
        for mnt in maint_grid:
            cfg = replace(base_cfg, time_mode="linear", time_lambda=lam,
                          maintenance_scale=mnt, maintenance_exponent=1.0)
            curves = _return_curves(episodes, cfg)
            ceiling = max(len(c) for c in curves)
            k_star = fit_singlehalt_stop(curves)
            meaningful = bool(1 < k_star < ceiling - 1)
            results.append(dict(time_lambda=lam, maintenance_scale=mnt, k_star=k_star,
                                ceiling=ceiling, k_frac=k_star / (ceiling - 1), meaningful=meaningful))
    return results


def sweep(packed_root: Path, *, lambda_grid: tuple[float, ...] = LAMBDA_GRID,
          maint_grid: tuple[float, ...] = MAINT_GRID, split: str = "validation",
          max_episodes: int | None = None) -> list[dict[str, Any]]:
    """Grid-search `time_lambda x maintenance_scale`; one dict per grid point with `k_star`, `ceiling`,
    and the derived `meaningful` flag. Loads episodes ONCE (the expensive I/O step) and reuses them for
    every grid point (`_return_curves`/`fit_singlehalt_stop` are pure arithmetic on already-loaded data,
    so the whole grid after the initial load runs in seconds)."""
    episodes = _load_split_episodes(packed_root, split, max_episodes=max_episodes)
    base_cfg = _oracle_config(packed_root)
    return _sweep_grid(episodes, base_cfg, lambda_grid=lambda_grid, maint_grid=maint_grid)


def _render_heatmap(results: list[dict[str, Any]], out_dir: str | Path,
                     lambda_grid: tuple[float, ...] = LAMBDA_GRID,
                     maint_grid: tuple[float, ...] = MAINT_GRID, *, scale: float = 1.0,
                     scale_label: str = "") -> None:
    """One 2D grid: rows = time_lambda (log-spaced categorical), cols = maintenance_scale, cell color =
    k*/ceiling (0=instant-stop collapse, 1=always-continue pileup), cell text = raw k*, non-meaningful
    cells cross-hatched so the surviving region is visually obvious at a glance.

    ``scale`` multiplies the axis VALUES only (e.g. 1e4 turns "0.0005" into "5") for a legible zoomed
    view over a narrow, small-magnitude band; ``scale_label`` (e.g. r"$\times 10^{-4}$") is drawn as
    its own small unit tag anchored on each axis (not folded into the axis title) so the multiplied
    tick numbers still read as the true value."""
    by_key = {(r["time_lambda"], r["maintenance_scale"]): r for r in results}
    nrow, ncol = len(lambda_grid), len(maint_grid)
    frac = np.array([[by_key[(lam, mnt)]["k_frac"] for mnt in maint_grid] for lam in lambda_grid])

    # Small grids (e.g. the zoomed-in "meaningful band" view) get far fewer, much bigger cells, so
    # scale text/tick size up accordingly rather than using the dense-grid size everywhere — a tiny
    # grid rendered with the dense-grid's small fonts looks empty and cramped-looking-small, not
    # "much smaller and more legible". Re-tuned 2026-07-09 for the now-fixed 10x10 zoom grid: the
    # figure itself is much smaller (so text reads bigger relative to it) and text/tick sizes bumped
    # up accordingly -- the old >60-cell tier (6.3/8.0pt in a ~10x8in figure) was illegibly small.
    ncell = ncol * nrow
    if ncell <= 25:
        txt_fs, tick_fs = 17, 13.5
    elif ncell <= 60:
        txt_fs, tick_fs = 11, 10
    else:
        txt_fs, tick_fs = 9, 8.5

    # The colorbar label is fixed-length descriptive text, not per-cell content, so its font size is
    # capped independent of `tick_fs` — scaling it up with a narrow (few-column) zoomed grid the way
    # `tick_fs` scales makes the rotated colorbar label overrun its own axis.
    label_fs = min(tick_fs, 9.5)

    _rcparams()
    fig, ax = plt.subplots(figsize=(0.38 * ncol + 1.5, 0.32 * nrow + 1.2))
    cmap = plt.get_cmap("RdBu")
    norm = plt.Normalize(vmin=0, vmax=1)
    im = ax.imshow(frac, cmap=cmap, norm=norm, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Steps (% Max)", fontsize=label_fs)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    cbar.ax.tick_params(labelsize=label_fs)

    for i, lam in enumerate(lambda_grid):
        for j, mnt in enumerate(maint_grid):
            r = by_key[(lam, mnt)]
            if not r["meaningful"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor="#2C3E50", lw=0))
            # Text color follows the ACTUAL rendered cell color's luminance (not a band hardcoded
            # against one specific colormap) -- RdBu is dark at BOTH ends (deep red at 0, deep blue
            # at 1) and light only in the middle, the opposite of RdYlGn_r's profile this was
            # originally tuned against, so a fixed "mid-band is white" rule reads as inverted here.
            r_, g_, b_, _ = cmap(norm(r["k_frac"]))
            luminance = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            txt_color = "white" if luminance < 0.5 else "#2C3E50"
            ax.text(j, i, f"{r['k_star']}", ha="center", va="center", fontsize=txt_fs,
                    color=txt_color, fontweight="bold")

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{m * scale:g}" for m in maint_grid], fontsize=tick_fs, rotation=45, ha="right")
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"{l * scale:g}" for l in lambda_grid], fontsize=tick_fs)
    ax.set_xlabel("maintenance_scale", fontsize=tick_fs + 1.5)
    ax.set_ylabel("time_lambda", fontsize=tick_fs + 1.5)
    if scale_label:
        # A small unit tag anchored ON each axis (not folded into the axis title text, and not a
        # figure title) -- mirrors matplotlib's own scientific-notation offset-text convention: the
        # Y axis's multiplier sits directly ABOVE it (horizontal, top-left corner); the X axis's
        # sits flush with / just below the axis line, at its right end (horizontal) -- NOT at y=0
        # exactly, which collides with the colorbar's "0" tick (the colorbar occupies that exact
        # height immediately to the right of the main axes), and NOT pushed up to the top either.
        ax.text(1.01, -0.05, scale_label, transform=ax.transAxes, fontsize=tick_fs - 1,
                ha="left", va="top", color="black")
        ax.text(0.0, 1.01, scale_label, transform=ax.transAxes, fontsize=tick_fs - 1,
                ha="left", va="bottom", color="black")
    fig.tight_layout()
    save_pdf_png(fig, str(out_dir), "regime_select", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description="REGIME — confirm meaningful (time_lambda, maintenance_scale) points")
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--out-dir", default="outputs/figures/minply15_maxply75/diagnosis")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--results-json", default=None,
                    help="optional path to dump the raw per-grid-point results as JSON")
    args = ap.parse_args()

    episodes = _load_split_episodes(Path(args.packed_root), args.split, max_episodes=args.max_episodes)
    base_cfg = _oracle_config(Path(args.packed_root))

    results = _sweep_grid(episodes, base_cfg, lambda_grid=LAMBDA_GRID, maint_grid=MAINT_GRID)
    for r in results:
        flag = "MEANINGFUL" if r["meaningful"] else "degenerate"
        print(f"[regime] lambda={r['time_lambda']:<6g} maint={r['maintenance_scale']:<5g} "
              f"k*={r['k_star']:>3d}  ceiling={r['ceiling']:>3d}  k*/ceiling={r['k_frac']:.3f}  {flag}",
              flush=True)

    # The PLOT renders its OWN dedicated, log-spaced ZOOM_LAMBDA_GRID/ZOOM_MAINT_GRID (not a filtered
    # subset of the diagnostic grid above) — reuses the same loaded episodes, so this second grid
    # sweep is still just pure arithmetic (seconds).
    zoom_results = _sweep_grid(episodes, base_cfg, lambda_grid=ZOOM_LAMBDA_GRID, maint_grid=ZOOM_MAINT_GRID)
    _render_heatmap(zoom_results, args.out_dir, lambda_grid=ZOOM_LAMBDA_GRID, maint_grid=ZOOM_MAINT_GRID,
                    scale=1e4, scale_label=r"$\times 10^{-4}$")
    if args.results_json:
        Path(args.results_json).write_text(json.dumps(results, indent=2))
        print(f"[regime] wrote {args.results_json}", flush=True)
    # Cached separately from --results-json's coarse diagnostic grid: the zoom grid's points mostly
    # DON'T overlap it (ZOOM_LAMBDA_GRID/ZOOM_MAINT_GRID are their own hand-picked values), so a
    # cosmetic-only replot (font/size/color tweaks -- the kind this heatmap gets a lot of) can skip
    # the expensive episode-load + full sweep entirely via `replot_zoom` below.
    zoom_json = Path(args.out_dir) / "regime_select_zoom_results.json"
    zoom_json.parent.mkdir(parents=True, exist_ok=True)
    zoom_json.write_text(json.dumps(zoom_results, indent=2))
    print(f"[regime] wrote {zoom_json}", flush=True)


def replot_zoom(out_dir: str | Path) -> None:
    """Re-render ONLY the zoom heatmap from `<out_dir>/regime_select_zoom_results.json` (written by
    `main()`) -- for cosmetic-only tweaks (figsize, fonts, colorbar, tick formatting, ...) that don't
    need the episodes reloaded or the grids re-swept."""
    out_dir = Path(out_dir)
    zoom_results = json.loads((out_dir / "regime_select_zoom_results.json").read_text())
    _render_heatmap(zoom_results, out_dir, lambda_grid=ZOOM_LAMBDA_GRID, maint_grid=ZOOM_MAINT_GRID,
                    scale=1e4, scale_label=r"$\times 10^{-4}$")


if __name__ == "__main__":
    main()
