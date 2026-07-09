"""plan.md Z3 -- does the Z2 end-to-end (encoder-unfrozen) MetaController beat
SingleHalt*/Stats-Controller/original-(frozen-encoder)-z_t?

This is deliberately NOT a change to ``analysis/evaluate.py`` -- that module's
``_compute_frontier_data``/``_compute_decodability_data`` are hardcoded to exactly
three series (SingleHalt*, Stats-Controller, z_t-Controller). Rather than bend that
module's shape around a one-off fourth series, this script imports its already-tested
private helpers directly (``_load_assessment_data``, ``_return_curves``, ``_fit_stop_controllers``,
``_train_readout``/``_steps_zt_tensor``, the ``point()`` closure pattern,
``_frontier_panel``/``_draw_zoom_indicator``/``_padded_range``/``_deconflict_points``,
``bootstrap_ci`` for paired-CI significance) and adds ONE new "e2e z_t" series on top of the SAME
already-computed baseline numbers (loaded from the existing
``outputs/figures/.../normative/{frontier,decodability}_data.json``, not recomputed --
they're the exact numbers plan.md's Context section cites: SingleHalt*=0.120,
Stats-Controller=0.114, z_t-Controller=0.121, z_t decodability MLP R^2=0.013). The decodability bar
chart is rendered by a LOCAL function (not ``analysis.evaluate._render_decodability`` -- that
function hardcodes exactly 4 y-tick labels for exactly 4 rows and breaks on a spliced-in 5th row;
see ``_render_decodability_with_e2e``'s docstring).

Usage (see slurm/z3_e2e_materialize_and_eval.slurm for the full pipeline this plugs
into -- materialize e2e z_t on validation -> this script):

    python -m analysis.evaluate_e2e_z3 \
        --packed-root $MCP --e2e-cache <e2e_materialized_validation_cache.pt> \
        --baseline-fig-dir outputs/figures/minply15_maxply75/normative \
        --out-dir outputs/figures/minply15_maxply75/diagnosis/z3_e2e_comparison \
        --d-embed 32 --maintenance-scales 0.0,0.05,0.1
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
    _compute_decodability_data,
    _draw_zoom_indicator,
    _deconflict_points,
    _fit_stop_controllers,
    _load_assessment_data,
    _load_json,
    _mean_ci,
    _oracle_config,
    _padded_range,
    _frontier_panel,
    _regret_at,
    _return_curves,
    _rcparams,
    _train_readout,
    _steps_zt_tensor,
    _to_jsonable,
)
from analysis.utils.plots import save_pdf_png
from cts.stats import bootstrap_ci

_BASELINE_LABELS = {"singlehalt": "SingleHalt*", "stats": "Stats-Controller", "zt": "orig $z_t$-Controller"}

_E2E_COLOR = "#8E44AD"  # purple -- distinct from every existing series color in analysis.evaluate._C
_E2E_LABEL = "e2e $z_t$-Controller (Z2, 1 epoch)"


def _e2e_point(packed_root: Path, e2e_cache_path: str, *, d_embed: int, max_episodes: int, seed: int,
               time_mode: str, time_lambda: float, maintenance_scale: float, maintenance_exponent: float) -> dict:
    """Fit the e2e-z_t readout (SAME recipe as evaluate.py's 'zt' arm: PG-trained advantage head,
    200 epochs, on steps+z_t features) and return its frontier point dict, at one cost regime.
    Mirrors ``_compute_frontier_data``'s ``point()`` closure + 'zt' controller arm exactly, just for
    a cache the base module doesn't know about."""
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, e2e_cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev = [curves[i] for i in ev_idx]

    stops = _train_readout(
        [_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in fit_idx],
        [_steps_zt_tensor(episodes[i], z_by_ep[i]) for i in ev_idx],
        fit_curves, in_dim=d_embed + 1, epochs=200, lr=1e-3, seed=seed,
    )

    xs = np.array([min(int(s), len(c) - 1) for s, c in zip(stops, ev)], dtype=float)
    xm, xlo, xhi = _mean_ci(xs)
    regret = _regret_at(ev, stops)
    ym, ylo, yhi = _mean_ci(regret)
    return dict(x=xm, xlo=xlo, xhi=xhi, y=ym, lo=ylo, hi=yhi, color=_E2E_COLOR, label=_E2E_LABEL,
                zorder=7, regret_per_episode=regret)


def _paired_baseline_diffs(packed_root: Path, original_cache_path: str, e2e_point: dict, *, d_embed: int,
                           max_episodes: int, seed: int, time_mode: str, time_lambda: float,
                           maintenance_scale: float, maintenance_exponent: float, n_boot: int = 2000) -> dict:
    """Paired-bootstrap 95% CI on the per-episode regret DIFFERENCE (baseline - e2e) for each of
    {SingleHalt*, Stats-Controller, orig z_t} vs e2e z_t, on the SAME held-out episodes -- NOT a
    point-estimate comparison and NOT two-independent-marginal-CIs eyeballed for overlap (both
    understate/overstate significance for paired data). Mirrors the exact pattern
    ``analysis.evaluate._compute_frontier_data`` already uses for its own zt-vs-stats comparison
    (``d_zs = bootstrap_ci(lambda d: float(d.mean()), _regret_at(ev, ctrl["zt"]) - _regret_at(ev, ctrl["stats"]), n_boot=2000)``,
    evaluate.py line ~415) -- a paired diff cancels the per-episode variance common to both methods
    (scored on the same episodes), which is far more sensitive than comparing two wide marginal CIs.

    A positive diff means the baseline has HIGHER (worse) regret than e2e z_t, i.e. e2e wins on that
    episode; CI entirely > 0 => e2e significantly beats that baseline. CI entirely < 0 => e2e
    significantly loses. CI straddling 0 => not statistically distinguishable at this sample size.

    Relies on ``_load_assessment_data``'s fit/eval split depending ONLY on episode trajectory_keys
    (not on which cache's z_t is loaded) -- same packed_root/seed/max_episodes therefore guarantees
    byte-identical fit_idx/ev_idx to whatever call produced ``e2e_point``, so the two regret arrays
    are safe to pair index-for-index (asserted below, not just assumed).
    """
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, original_cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, d_embed, seed)

    e2e_regret = e2e_point["regret_per_episode"]
    assert len(e2e_regret) == len(ev_idx), (
        f"e2e regret array (n={len(e2e_regret)}) doesn't match this baseline eval-split (n={len(ev_idx)}) "
        "-- fit/eval split must be identical (same packed_root/seed/max_episodes) for pairing to be valid"
    )

    diffs = {}
    for name in ("singlehalt", "stats", "zt"):
        baseline_regret = _regret_at(ev, ctrl[name])
        diff = baseline_regret - np.asarray(e2e_regret)
        diffs[name] = bootstrap_ci(lambda d: float(d.mean()), diff, n_boot=n_boot)  # (mean, lo, hi)
    return diffs


def _render_frontier_with_e2e(baseline_data: dict, e2e_point: dict, out_dir: Path, tag: str,
                              paired_diffs: dict | None = None) -> None:
    """Copy of ``analysis.evaluate._render_frontier``'s figure assembly, with the e2e point spliced
    into ``controllers``/``all_points`` and a one-sentence plain-English finding annotated on the
    figure (see plan.md's 'figures must explain themselves' constraint) -- not calling
    ``_render_frontier`` directly since that function has no series-injection hook and always saves
    to a fixed 'frontier' filename (would clobber the baseline plot)."""
    fr_x, fr = np.asarray(baseline_data["fr_x"]), np.asarray(baseline_data["fr"])
    controllers = list(baseline_data["controllers"]) + [e2e_point]
    all_points = list(baseline_data["all_points"]) + [e2e_point]

    singlehalt_y = baseline_data["controllers"][0]["y"]
    stats_y = baseline_data["controllers"][1]["y"]

    if paired_diffs is not None:
        # Honest, SIGNIFICANCE-based finding (paired bootstrap CI on the per-episode regret
        # difference, not a bare point-estimate comparison -- point estimates alone can't
        # distinguish a real effect from noise when marginal CIs overlap, see paired_diffs'
        # docstring). "confirmed" = CI on (baseline_regret - e2e_regret) excludes 0 in e2e's favor.
        parts = []
        any_confirmed = False
        for name in ("singlehalt", "stats"):
            mean_d, lo_d, hi_d = paired_diffs[name]
            label = _BASELINE_LABELS[name]
            if lo_d > 0:
                parts.append(f"CONFIRMED beats {label} (paired diff={mean_d:+.3f}, 95% CI [{lo_d:+.3f}, {hi_d:+.3f}], excludes 0)")
                any_confirmed = True
            elif hi_d < 0:
                parts.append(f"confirmed LOSES to {label} (paired diff={mean_d:+.3f}, 95% CI [{lo_d:+.3f}, {hi_d:+.3f}])")
            else:
                parts.append(f"directionally {'beats' if mean_d > 0 else 'loses to'} {label} but NOT statistically "
                             f"confirmed (paired diff={mean_d:+.3f}, 95% CI [{lo_d:+.3f}, {hi_d:+.3f}] straddles 0)")
        finding = ("e2e z_t (1 epoch, joint encoder+head), paired 95% bootstrap CI on per-episode regret diff: "
                  + "; ".join(parts) + ".")
    else:
        beats_singlehalt = e2e_point["y"] < singlehalt_y
        beats_stats = e2e_point["y"] < stats_y
        if beats_singlehalt or beats_stats:
            finding = (f"e2e z_t regret={e2e_point['y']:.3f} beats (point estimate only, NOT significance-tested) "
                      f"{'SingleHalt*' if beats_singlehalt else ''}"
                      f"{' and ' if beats_singlehalt and beats_stats else ''}"
                      f"{'Stats-Controller' if beats_stats else ''} "
                      f"(SingleHalt*={singlehalt_y:.3f}, Stats={stats_y:.3f}) -- lower is better.")
        else:
            finding = (f"e2e z_t regret={e2e_point['y']:.3f} does NOT beat SingleHalt*={singlehalt_y:.3f} "
                      f"or Stats-Controller={stats_y:.3f} at this cost regime -- lower is better.")

    _rcparams()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1.1]})
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
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle(finding, fontsize=10.5, y=1.02, wrap=True)
    save_pdf_png(fig, str(out_dir), f"z3_frontier_e2e_vs_baselines_{tag}", dpi=200, bbox_extra_artists=(fig.legends[0],))
    print(f"[z3] frontier[{tag}]: SingleHalt*={singlehalt_y:.4f} Stats={stats_y:.4f} "
          f"orig_z_t={baseline_data['controllers'][2]['y']:.4f} e2e_z_t={e2e_point['y']:.4f}  -- {finding}", flush=True)


def _render_decodability_with_e2e(baseline_decod: dict, e2e_decod_row: tuple, out_dir: Path) -> dict:
    """Splices the e2e z_t row into baseline's rows and renders a 5-row bar chart.

    NOT calling ``analysis.evaluate._render_decodability`` here (an earlier version of this script
    did, and it LOOKED reusable -- same ``rows``/``shuf_r2``/``orthogonality`` shape -- but it
    hardcodes exactly 4 ``set_yticklabels`` strings for exactly 4 rows; splicing a 5th row in throws
    ``ValueError: The number of FixedLocator locations (5) ... does not match ... (4)``, confirmed by
    a real failed run). So this duplicates ``_render_decodability``'s bar-drawing logic (same
    ``_C``/``_rcparams``/``save_pdf_png`` house style) but derives y-tick labels from ``rows[i][0]``
    directly instead of a hardcoded literal list, so it generalizes to N rows -- plus adds the
    plain-English finding as a figure suptitle (plan.md's 'figures must explain themselves' rule).
    """
    from analysis.evaluate import _C, _INK, _MUTED, _GRID

    rows = list(baseline_decod["rows"])
    frozen_z_r2 = next(r[2] for r in rows if r[0] == "z_t")
    rows.insert(3, e2e_decod_row)  # after steps/stats/frozen-z_t, before "all combined"
    shuf_r2 = baseline_decod["shuf_r2"]
    e2e_r2 = e2e_decod_row[2]
    beats = e2e_r2 > frozen_z_r2
    finding = (f"e2e z_t (1 epoch, joint encoder+head) alone decodes R(t) with MLP R^2={e2e_r2:.3f}, "
              f"vs frozen-encoder z_t's R^2={frozen_z_r2:.3f} ({'MORE' if beats else 'LESS'} decodable) "
              f"and the shuffle floor {shuf_r2:+.3f}.")

    _rcparams()
    mono = _C["mono"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    y, h = np.arange(len(rows)), 0.30
    ax.barh(y - h / 2, [r[1] for r in rows], h, color=mono, alpha=0.5, hatch="///",
           edgecolor=_INK, linewidth=0.3, label="linear (Ridge)")
    ax.barh(y + h / 2, [r[2] for r in rows], h, color=mono, edgecolor="none", linewidth=0, label="MLP (2x64)")
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.006, i + h / 2, f"{r[2]:.2f}", va="center", fontsize=10, fontweight="bold", color=_INK)
    ax.axvline(shuf_r2, color=_MUTED, ls="--", lw=1.3, label=f"shuffle floor ({shuf_r2:+.2f})")
    ax.axvline(0, color=_MUTED, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] if "\n" in r[0] or r[0].startswith("all") else f"{r[0]}\n(alone)" for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("held-out $R^2$ -- predicting R(t) = value of continuing")
    ax.grid(axis="x", color=_GRID, lw=1)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.suptitle(finding, fontsize=10, y=1.03, wrap=True)
    save_pdf_png(fig, str(out_dir), "z3_decodability_with_e2e", dpi=200)

    out = {r[0]: {"linear": r[1], "mlp": r[2]} for r in rows}
    print(f"[z3] decodability: frozen z_t MLP R^2={frozen_z_r2:.4f}  e2e z_t MLP R^2={e2e_r2:.4f}  "
          f"({'BEATS' if beats else 'does NOT beat'} the frozen encoder's decodability)  -- {finding}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Z3 -- e2e (encoder-unfrozen) z_t vs SingleHalt*/Stats/original-z_t")
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--e2e-cache", required=True, help="materialized validation cache built from the Z2 e2e encoder")
    ap.add_argument("--original-cache", required=True, help="materialized validation cache built from the FROZEN (production) encoder")
    ap.add_argument("--baseline-fig-dir", required=True, help="dir with existing frontier_data.json/decodability_data.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-mode", default="linear")
    ap.add_argument("--time-lambda", type=float, default=0.01)
    ap.add_argument("--maintenance-exponent", type=float, default=1.0)
    ap.add_argument("--maintenance-scales", default="0.0", help="comma-separated sweep grid; first entry is the PRIMARY regime the figures are rendered for")
    args = ap.parse_args()

    packed_root = Path(args.packed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_fig_dir)
    # Reuse the ALREADY-COMPUTED baseline decodability numbers (steps/stats/frozen-z_t/all rows +
    # shuffle floor) instead of recomputing them -- decodability has no maintenance_scale/time_lambda
    # dependency (R(t) is defined straight off halt_rewards, cost-independent), so the saved
    # normative/decodability_data.json is valid to reuse as-is regardless of the sweep grid below.
    baseline_decod = _load_json(baseline_dir / "decodability_data.json")

    from analysis.evaluate import _compute_frontier_data

    scales = [float(s) for s in args.maintenance_scales.split(",")]
    print(f"[z3] maintenance_scale sweep grid: {scales}", flush=True)

    sweep_rows = []
    primary_baseline_data = None
    primary_e2e_point = None
    for m in scales:
        baseline_data = _compute_frontier_data(
            packed_root, args.original_cache, d_embed=args.d_embed, time_mode=args.time_mode,
            time_lambda=args.time_lambda, maintenance_scale=m, maintenance_exponent=args.maintenance_exponent,
            max_episodes=args.max_episodes, seed=args.seed,
        )
        e2e_point = _e2e_point(
            packed_root, args.e2e_cache, d_embed=args.d_embed, max_episodes=args.max_episodes, seed=args.seed,
            time_mode=args.time_mode, time_lambda=args.time_lambda, maintenance_scale=m,
            maintenance_exponent=args.maintenance_exponent,
        )
        singlehalt_y, stats_y, origzt_y = (baseline_data["controllers"][0]["y"], baseline_data["controllers"][1]["y"],
                                           baseline_data["controllers"][2]["y"])
        row = dict(maintenance_scale=m, singlehalt=singlehalt_y, stats=stats_y, orig_zt=origzt_y,
                  e2e_zt=e2e_point["y"], beats_singlehalt=bool(e2e_point["y"] < singlehalt_y),
                  beats_stats=bool(e2e_point["y"] < stats_y))
        sweep_rows.append(row)
        print(f"[z3] sweep m={m}: SingleHalt*={singlehalt_y:.4f} Stats={stats_y:.4f} orig_z_t={origzt_y:.4f} "
              f"e2e_z_t={e2e_point['y']:.4f}  beats_singlehalt={row['beats_singlehalt']} beats_stats={row['beats_stats']}",
              flush=True)
        if m == scales[0]:
            primary_baseline_data = baseline_data
            primary_e2e_point = e2e_point

    with open(out_dir / "z3_maintenance_sweep.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(sweep_rows), f, indent=2)
    print(f"[z3] sweep table -> {out_dir / 'z3_maintenance_sweep.json'}", flush=True)

    # SIGNIFICANCE check at the PRIMARY regime (scales[0], the regime the headline point-estimate
    # win/loss claims above are about) -- paired bootstrap CI on the per-episode regret difference,
    # not a bare point-estimate/marginal-CI-overlap read. See _paired_baseline_diffs docstring.
    m0 = scales[0]
    paired_diffs = _paired_baseline_diffs(
        packed_root, args.original_cache, primary_e2e_point, d_embed=args.d_embed, max_episodes=args.max_episodes,
        seed=args.seed, time_mode=args.time_mode, time_lambda=args.time_lambda, maintenance_scale=m0,
        maintenance_exponent=args.maintenance_exponent, n_boot=2000,
    )
    for name, (mean_d, lo_d, hi_d) in paired_diffs.items():
        sig = "CONFIRMED (excludes 0)" if lo_d > 0 else ("confirmed loss (excludes 0)" if hi_d < 0 else "NOT significant (straddles 0)")
        print(f"[z3] paired diff @ m={m0} [{_BASELINE_LABELS[name]} regret] - [e2e z_t regret]: "
              f"mean={mean_d:+.4f} 95% CI=[{lo_d:+.4f}, {hi_d:+.4f}]  {sig}", flush=True)
    with open(out_dir / "z3_paired_diffs_primary.json", "w", encoding="utf-8") as f:
        json.dump({"maintenance_scale": m0, "diffs": _to_jsonable(paired_diffs)}, f, indent=2)
    print(f"[z3] paired diffs -> {out_dir / 'z3_paired_diffs_primary.json'}", flush=True)

    tag = f"m{scales[0]}".replace(".", "p")
    _render_frontier_with_e2e(primary_baseline_data, primary_e2e_point, out_dir, tag, paired_diffs=paired_diffs)

    e2e_decod = _compute_decodability_data(packed_root, args.e2e_cache, d_embed=args.d_embed,
                                           max_episodes=min(args.max_episodes, 12000), seed=args.seed)
    e2e_z_row = ("e2e $z_t$", e2e_decod["rows"][2][1], e2e_decod["rows"][2][2])  # ("z_t", linear, mlp) -> relabeled
    assert e2e_decod["rows"][2][0] == "z_t"
    _render_decodability_with_e2e(baseline_decod, e2e_z_row, out_dir)

    confirmed_win = any(lo_d > 0 for _mean_d, lo_d, _hi_d in paired_diffs.values())
    any_win_point_estimate = any(r["beats_singlehalt"] or r["beats_stats"] for r in sweep_rows)
    print(f"[z3] SUCCESS CRITERION (point-estimate, any swept regime): {any_win_point_estimate}", flush=True)
    print(f"[z3] SUCCESS CRITERION (STATISTICALLY CONFIRMED, paired 95% CI excludes 0, primary regime m={m0}): {confirmed_win}", flush=True)


if __name__ == "__main__":
    main()
