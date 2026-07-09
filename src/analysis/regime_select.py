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

from analysis.evaluate import _load_split_episodes, _oracle_config, _return_curves, fit_singlehalt_stop, _rcparams
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

# The heatmap PLOT (not the underlying sweep(), which still runs the full grids above) only
# renders this sub-range: everything past ~lambda=0.015 or maintenance=0.003 is uniform
# instant-stop-collapse (k_frac->0) AND hatched-not-meaningful, i.e. wasted plot area — the only
# band worth a human's eyes is where k* actually varies. Confirmed against the xaba20k sweep
# (2026-07-09): every point in this sub-range is "meaningful", so zooming here doesn't hide any
# non-degenerate cell. Densified (2026-07-09, second pass) with extra LAMBDA_GRID/MAINT_GRID
# points inside the range itself (0.00075; 0.0035/0.004/0.0045; 0.00025/0.00075) — the first zoom
# pass only filtered the ALREADY-existing coarse diagnostic grid down to 21 points, which read as
# too sparse once rendered on its own axes.
ZOOM_LAMBDA_RANGE: tuple[float, float] = (0.0005, 0.005)
ZOOM_MAINT_RANGE: tuple[float, float] = (0.0, 0.001)


def _zoom_grid(grid: tuple[float, ...], lo: float, hi: float) -> tuple[float, ...]:
    """Sub-tuple of `grid` within `[lo, hi]` inclusive, preserving order."""
    return tuple(v for v in grid if lo <= v <= hi)


def sweep(packed_root: Path, *, lambda_grid: tuple[float, ...] = LAMBDA_GRID,
          maint_grid: tuple[float, ...] = MAINT_GRID, split: str = "validation",
          max_episodes: int | None = None) -> list[dict[str, Any]]:
    """Grid-search `time_lambda x maintenance_scale`; one dict per grid point with `k_star`, `ceiling`,
    and the derived `meaningful` flag. Loads episodes ONCE (the expensive I/O step) and reuses them for
    every grid point (`_return_curves`/`fit_singlehalt_stop` are pure arithmetic on already-loaded data,
    so the whole grid after the initial load runs in seconds)."""
    episodes = _load_split_episodes(packed_root, split, max_episodes=max_episodes)
    base_cfg = _oracle_config(packed_root)
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


def _render_heatmap(results: list[dict[str, Any]], out_dir: str | Path,
                     lambda_grid: tuple[float, ...] = LAMBDA_GRID,
                     maint_grid: tuple[float, ...] = MAINT_GRID) -> None:
    """One 2D grid: rows = time_lambda (log-spaced categorical), cols = maintenance_scale, cell color =
    k*/ceiling (0=instant-stop collapse, 1=always-continue pileup), cell text = raw k*, non-meaningful
    cells cross-hatched so the surviving region is visually obvious at a glance."""
    by_key = {(r["time_lambda"], r["maintenance_scale"]): r for r in results}
    nrow, ncol = len(lambda_grid), len(maint_grid)
    frac = np.array([[by_key[(lam, mnt)]["k_frac"] for mnt in maint_grid] for lam in lambda_grid])

    # Small grids (e.g. the zoomed-in "meaningful band" view) get far fewer, much bigger cells, so
    # scale text/tick size up accordingly rather than using the dense-grid size everywhere — a tiny
    # grid rendered with the dense-grid's small fonts looks empty and cramped-looking-small, not
    # "much smaller and more legible".
    ncell = ncol * nrow
    if ncell <= 25:
        txt_fs, tick_fs = 17, 13.5
    elif ncell <= 60:
        txt_fs, tick_fs = 9.5, 9.5
    else:
        txt_fs, tick_fs = 6.3, 8.0

    # The colorbar label and title are fixed-length descriptive text, not per-cell content, so
    # their font size is capped independent of `tick_fs` — scaling them up with a narrow (few-
    # column) zoomed grid the way `tick_fs` scales makes the rotated colorbar label overrun its
    # own axis and collide with the title above it.
    label_fs = min(tick_fs, 9.5)

    _rcparams()
    fig, ax = plt.subplots(figsize=(0.72 * ncol + 2.8, 0.6 * nrow + 2.2))
    im = ax.imshow(frac, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("$k^*$ / ceiling  (0 = instant-stop collapse, 1 = always-continue pileup)",
                   fontsize=label_fs)
    cbar.ax.tick_params(labelsize=label_fs)

    for i, lam in enumerate(lambda_grid):
        for j, mnt in enumerate(maint_grid):
            r = by_key[(lam, mnt)]
            if not r["meaningful"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////",
                                           edgecolor="#2C3E50", lw=0))
            txt_color = "white" if 0.35 < r["k_frac"] < 0.75 else "#2C3E50"
            ax.text(j, i, f"{r['k_star']}", ha="center", va="center", fontsize=txt_fs,
                    color=txt_color, fontweight="bold")

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"{m:g}" for m in maint_grid], fontsize=tick_fs, rotation=45, ha="right")
    ax.set_yticks(range(nrow)); ax.set_yticklabels([f"{l:g}" for l in lambda_grid], fontsize=tick_fs)
    ax.set_xlabel("maintenance_scale", fontsize=tick_fs + 1.5)
    ax.set_ylabel("time_lambda", fontsize=tick_fs + 1.5)
    ax.set_title("REGIME sweep: SingleHalt*'s own $k^*$ (cell text) across cost regimes\n"
                 "hatched = NOT meaningful (k* collapsed to <=1 or pinned at ceiling-1)",
                 fontsize=label_fs + 1.5)
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

    results = sweep(Path(args.packed_root), split=args.split, max_episodes=args.max_episodes)
    for r in results:
        flag = "MEANINGFUL" if r["meaningful"] else "degenerate"
        print(f"[regime] lambda={r['time_lambda']:<6g} maint={r['maintenance_scale']:<5g} "
              f"k*={r['k_star']:>3d}  ceiling={r['ceiling']:>3d}  k*/ceiling={r['k_frac']:.3f}  {flag}",
              flush=True)
    # The PLOT only zooms into the meaningful sub-range (ZOOM_LAMBDA_RANGE/ZOOM_MAINT_RANGE) — the
    # full grids above are still swept and printed (and dumped to --results-json) in full, since
    # that full search is what pins down the collapse boundaries in the first place.
    zoom_lambda_grid = _zoom_grid(LAMBDA_GRID, *ZOOM_LAMBDA_RANGE)
    zoom_maint_grid = _zoom_grid(MAINT_GRID, *ZOOM_MAINT_RANGE)
    zoom_results = [r for r in results
                    if r["time_lambda"] in zoom_lambda_grid and r["maintenance_scale"] in zoom_maint_grid]
    _render_heatmap(zoom_results, args.out_dir, lambda_grid=zoom_lambda_grid, maint_grid=zoom_maint_grid)
    if args.results_json:
        Path(args.results_json).write_text(json.dumps(results, indent=2))
        print(f"[regime] wrote {args.results_json}", flush=True)


if __name__ == "__main__":
    main()
