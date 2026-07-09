"""plan.md SIG-S -- is Stats-Controller's point-estimate win over SingleHalt* (S1) actually
statistically significant?

S1's reported result (job 10832187, m=0, time_lambda=0.01): Stats-Controller regret
0.1138 [0.1061, 0.1218] vs SingleHalt* 0.1204 [0.1129, 0.1286] -- the two 95% CIs nearly touch
(Stats' upper bound 0.1218 vs SingleHalt*'s lower bound 0.1129), but this was only ever eyeballed
as a point-estimate win, never run through a real paired-difference significance test.

This does the SAME paired-bootstrap-CI-on-the-per-episode-regret-difference test that
``analysis.evaluate._compute_frontier_data``'s own ``d_zs = bootstrap_ci(lambda d: float(d.mean()),
_regret_at(ev, ctrl["zt"]) - _regret_at(ev, ctrl["stats"]), n_boot=2000)`` (evaluate.py:415) already
does for zt-vs-stats, applied here to stats-vs-singlehalt instead, on the SAME held-out episodes and
the SAME 70/30 fit/eval tree-split (``_load_assessment_data`` / ``_fit_stop_controllers`` reused
directly, not reimplemented) -- a paired diff cancels the per-episode variance shared by both methods
(scored on the same episodes), which is far more sensitive than eyeballing two overlapping marginal
CIs computed independently.

Usage:
    python -m analysis.evaluate_sig_s \
        --packed-root $MCP --cache $MAT/validation_cache.pt \
        --out-dir outputs/figures/minply15_maxply75/diagnosis
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
    _load_assessment_data,
    _mean_ci,
    _oracle_config,
    _rcparams,
    _regret_at,
    _return_curves,
    _steps_stats_tensor,
    _to_jsonable,
    _train_readout,
    fit_singlehalt_stop,
)
from analysis.utils.plots import save_pdf_png
from cts.stats import bootstrap_ci


def compute_sig_s(packed_root: Path, cache_path: str, *, d_embed: int = 32, max_episodes: int = 15000,
                   seed: int = 0, time_mode: str = "linear", time_lambda: float = 0.01,
                   maintenance_scale: float = 0.0, maintenance_exponent: float = 1.0,
                   n_boot: int = 2000) -> dict:
    """Fit SingleHalt* + Stats-Controller on the fit split (same tree-split/regime as every other
    SIG-*/Z result in this investigation), score both on the SAME eval episodes, and paired-bootstrap
    the per-episode regret DIFFERENCE (SingleHalt* regret - Stats regret) -- positive means Stats has
    LOWER (better) regret on that episode, i.e. Stats wins. CI entirely > 0 => Stats significantly
    beats SingleHalt*. CI entirely < 0 => Stats significantly loses. CI straddling 0 => not
    statistically distinguishable at this sample size.

    Does NOT fit the zt readout (unlike ``_fit_stop_controllers``, which fits all three) -- SIG-S is
    scoped to stats-vs-singlehalt only, so skipping the unused 200-epoch zt PG fit saves real time.
    """
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, _z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev = [curves[i] for i in ev_idx]

    kf = fit_singlehalt_stop(fit_curves)
    singlehalt_stops = np.full(len(ev_idx), kf, dtype=int)
    stats_stops = _train_readout(
        [_steps_stats_tensor(episodes[i]) for i in fit_idx],
        [_steps_stats_tensor(episodes[i]) for i in ev_idx],
        fit_curves, in_dim=4, epochs=200, lr=1e-3, seed=seed,
    )

    singlehalt_regret = _regret_at(ev, singlehalt_stops)
    stats_regret = _regret_at(ev, stats_stops)
    sh_mean, sh_lo, sh_hi = _mean_ci(singlehalt_regret)
    st_mean, st_lo, st_hi = _mean_ci(stats_regret)

    diff = singlehalt_regret - stats_regret  # positive = Stats wins on that episode
    d_mean, d_lo, d_hi = bootstrap_ci(lambda d: float(d.mean()), diff, n_boot=n_boot)

    return dict(
        k_singlehalt=kf, n_eval=len(ev_idx),
        singlehalt=dict(mean=sh_mean, lo=sh_lo, hi=sh_hi),
        stats=dict(mean=st_mean, lo=st_lo, hi=st_hi),
        paired_diff=dict(mean=d_mean, lo=d_lo, hi=d_hi),
        confirmed=bool(d_lo > 0),
        time_mode=time_mode, time_lambda=time_lambda,
        maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent,
    )


def _render(result: dict, out_dir: Path) -> None:
    """Two-panel figure: (left) the two methods' marginal regret + 95% CI, same house style as the
    frontier plot's points; (right) the PAIRED diff with its own 95% CI -- the actual significance
    test the marginal CIs on the left cannot answer (see module docstring)."""
    _rcparams()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1, 1]})

    labels = ["SingleHalt*\n(fixed stop)", "Stats-Controller"]
    means = [result["singlehalt"]["mean"], result["stats"]["mean"]]
    los = [result["singlehalt"]["lo"], result["stats"]["lo"]]
    his = [result["singlehalt"]["hi"], result["stats"]["hi"]]
    colors = [_C["singlehalt"], _C["stats"]]
    x = np.arange(2)
    axL.bar(x, means, color=colors, width=0.5, alpha=0.85)
    axL.errorbar(x, means, yerr=[[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]],
                fmt="none", ecolor="#2C3E50", elinewidth=1.6, capsize=5)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=10.5)
    axL.set_ylabel("Regret  (marginal 95% CI)")
    axL.set_title("marginal (independent) CIs", fontsize=10.5)
    axL.grid(axis="y", color="#E3E7EB", lw=1)

    d = result["paired_diff"]
    color = "#12A19A" if result["confirmed"] else "#7A8894"
    axR.errorbar([0], [d["mean"]], yerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                fmt="o", ms=9, color=color, ecolor=color, elinewidth=2.2, capsize=6, zorder=5)
    axR.axhline(0, color="#7A8894", lw=1.2, ls="--")
    axR.set_xlim(-1, 1); axR.set_xticks([])
    axR.set_ylabel("paired $\\Delta$ regret  (SingleHalt$^*$ $-$ Stats)\npositive = Stats wins")
    axR.set_title("paired bootstrap CI (the actual test)", fontsize=10.5)
    axR.grid(axis="y", color="#E3E7EB", lw=1)

    verdict = ("CONFIRMED: Stats-Controller significantly beats SingleHalt*" if result["confirmed"]
              else "NOT statistically confirmed (paired 95% CI straddles 0)")
    fig.suptitle(f"SIG-S ({result['time_mode']} $\\lambda$={result['time_lambda']} m={result['maintenance_scale']}, "
                f"n={result['n_eval']} held-out episodes): SingleHalt*={result['singlehalt']['mean']:.4f} "
                f"[{result['singlehalt']['lo']:.4f}, {result['singlehalt']['hi']:.4f}]  "
                f"Stats={result['stats']['mean']:.4f} [{result['stats']['lo']:.4f}, {result['stats']['hi']:.4f}]\n"
                f"paired diff={d['mean']:+.4f}  95% CI=[{d['lo']:+.4f}, {d['hi']:+.4f}]  -- {verdict}",
                fontsize=10, y=1.08, wrap=True)
    save_pdf_png(fig, str(out_dir), "sig_s_significance", dpi=200)
    print(f"[sig-s] SingleHalt*={result['singlehalt']['mean']:.4f} Stats={result['stats']['mean']:.4f}  "
          f"paired_diff={d['mean']:+.4f} 95% CI=[{d['lo']:+.4f}, {d['hi']:+.4f}]  confirmed={result['confirmed']}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="SIG-S -- Stats-Controller vs SingleHalt*, paired significance")
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--cache", required=True, help="materialized validation cache (.pt), shuffle=False")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-mode", default="linear")
    ap.add_argument("--time-lambda", type=float, default=0.01)
    ap.add_argument("--maintenance-scale", type=float, default=0.0)
    ap.add_argument("--maintenance-exponent", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = compute_sig_s(Path(args.packed_root), args.cache, d_embed=args.d_embed,
                           max_episodes=args.max_episodes, seed=args.seed, time_mode=args.time_mode,
                           time_lambda=args.time_lambda, maintenance_scale=args.maintenance_scale,
                           maintenance_exponent=args.maintenance_exponent)
    with open(out_dir / "sig_s_result.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result), f, indent=2)
    print(f"[sig-s] result -> {out_dir / 'sig_s_result.json'}", flush=True)
    _render(result, out_dir)
    print(f"[sig-s] SUCCESS CRITERION (STATISTICALLY CONFIRMED, paired 95% CI excludes 0): {result['confirmed']}",
          flush=True)


if __name__ == "__main__":
    main()
