"""Probe X (plan.md "External expert input (Yotam Sagiv...)" section) -- does xaba-filtering
OUR OWN puctvalue_md36 corpus (never tried before -- exclude_xaba was previously only exercised
on ysagiv's human_trees, see config_ysagiv_xaba.yaml / ysagiv_xaba_filter.slurm) change the
SingleHalt*/Stats-Controller/original-frozen-z_t comparison? Three populations, all scored by the
SAME frozen encoder (byte-identical checkpoint reused everywhere -- exclude_xaba is a mc_pack-only
field, no encoder retrain anywhere in this probe) so differences are attributable to the
filter(s), not a different z_t representation:

  - ``argmax_only``      -- the PRODUCTION baseline: our existing argmax>2 "thinking helps" filter,
                             no xaba filter. This is the population every other number in this
                             investigation (SIG-S/SIG-Z/S1/Z) is expressed against.
  - ``argmax_and_xaba``  -- condition A: exclude_xaba STACKED on top of argmax>2 (config_probe_x_xaba.yaml).
  - ``xaba_only``        -- condition B: exclude_xaba ALONE, entirely replacing argmax>2 (no
                             argmax pre-filter at all -- config_probe_x_xaba_condb.yaml). Applied to
                             a random 30K-tree tractability subsample of the full 400K generated
                             corpus (uniform, no quality criterion) rather than the full population,
                             for compute tractability -- see that config's header comment.

Reuses ``analysis.ysagiv_sig.compute_pairwise_significance`` UNCHANGED (that script already
generalizes SIG-S's paired-bootstrap-on-the-regret-difference methodology to all three pairwise
diffs, parameterized over regime) for every population. At each of REGIME's confirmed meaningful
regime points, fits SingleHalt*/Stats/z_t independently on each population's own fit split and
scores on that population's own eval split -- a "paired" test in the SIG-S/SIG-Z sense means
same-episodes-both-methods, and the three populations are different episode populations by
construction, so the headline significance test is the WITHIN-population paired diff (does Stats
beat SingleHalt* on THIS population), with marginal regret levels placed side by side across
populations for a directional (not itself paired-tested) read on how each filter changes the gaps.

Usage:
    python -m analysis.evaluate_probe_x_xaba \
        --base-packed-root $BASE_MCP --base-cache $BASE_MAT/validation_cache.pt \
        --xaba-packed-root $CONDA_MCP --xaba-cache $CONDA_MAT/validation_cache.pt \
        --condb-packed-root $CONDB_MCP --condb-cache $CONDB_MAT/validation_cache.pt \
        --out-dir outputs/figures/minply15_maxply75/diagnosis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.evaluate import _C, _rcparams, _to_jsonable
from analysis.utils.plots import save_pdf_png
from analysis.ysagiv_sig import compute_pairwise_significance

# Three REGIME-confirmed meaningful points (outputs/figures/minply15_maxply75/diagnosis/
# regime_select_results.json, meaningful=true), spanning distinct k* regimes rather than just the
# single validated default -- per this investigation's "2-3 representative regime points" pattern:
#   R1 = the primary validated default (time_lambda=0.01, m=0.0, k*=10) -- the number every other
#        probe in this investigation (SIG-S/SIG-Z/Z) is expressed at.
#   R2 = a different time-cost weighting with NO maintenance term (time_lambda=0.0005, m=0.0,
#        k*=56) -- much looser budget pressure, tests whether the story holds far from R1.
#   R3 = R1's lambda with a small nonzero maintenance_scale (m=0.001, k*=6) -- the
#        "nonzero-maintenance-friendly" region REGIME identified as meaningful only at
#        maintenance_scale <~ 0.005, tighter budget pressure than R1.
REGIME_POINTS = [
    {"tag": "lambda0p01_m0", "label": "$\\lambda$=0.01, m=0", "time_lambda": 0.01, "maintenance_scale": 0.0},
    {"tag": "lambda0p0005_m0", "label": "$\\lambda$=5e-4, m=0", "time_lambda": 0.0005, "maintenance_scale": 0.0},
    {"tag": "lambda0p01_m0p001", "label": "$\\lambda$=0.01, m=0.001", "time_lambda": 0.01, "maintenance_scale": 0.001},
]

_METHODS = [("singlehalt", "SingleHalt*"), ("stats", "Stats-Controller"), ("zt", "z$_t$-Controller")]
_POP_ORDER = ["argmax_only", "argmax_and_xaba", "xaba_only"]
_POP_DISPLAY = {"argmax_only": "argmax>2 only\n(baseline)", "argmax_and_xaba": "argmax>2 + xaba\n(cond. A)",
                "xaba_only": "xaba only\n(cond. B)"}
_POP_STYLE = {
    "argmax_only": dict(hatch=None, alpha=0.90),
    "argmax_and_xaba": dict(hatch="////", alpha=0.65),
    "xaba_only": dict(hatch="....", alpha=0.65),
}


def _run_population(pop_key: str, packed_root: Path, cache: str, *, d_embed: int, max_episodes: int,
                    seed: int, n_boot: int) -> dict:
    pop_results: dict = {}
    for pt in REGIME_POINTS:
        print(f"[probe-x-xaba] {pop_key} @ {pt['tag']} ...", flush=True)
        r = compute_pairwise_significance(
            packed_root, cache, d_embed=d_embed, max_episodes=max_episodes, seed=seed,
            time_mode="linear", time_lambda=pt["time_lambda"], maintenance_scale=pt["maintenance_scale"],
            maintenance_exponent=1.0, n_boot=n_boot,
        )
        pop_results[pt["tag"]] = r
        m = r["marginal"]
        print(f"  SingleHalt*(k={r['k_singlehalt']})={m['singlehalt']['mean']:.4f} "
              f"Stats={m['stats']['mean']:.4f} zt={m['zt']['mean']:.4f}  n_eval={r['n_eval']}", flush=True)
        for name, d in r["paired"].items():
            print(f"    {name}: {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]", flush=True)
    return pop_results


def _render(results: dict, out_dir: Path) -> None:
    """Grouped bar chart: one panel per regime point. Within each panel, 3 method groups
    (SingleHalt*/Stats/z_t) x up-to-3 populations present in ``results`` -- color encodes METHOD
    identity (repo-house _C palette, unchanged across this whole investigation); population is a
    texture/alpha secondary encoding, not a second competing hue axis. Error bars are each bar's
    own marginal 95% CI. Below each panel, the SingleHalt*-vs-Stats PAIRED diff CI (the actual
    significance test, not just overlapping marginals) is annotated for every population present,
    so "does the verdict change under each filter" is a direct visual read.
    """
    _rcparams()
    pop_order = [p for p in _POP_ORDER if p in results]
    n_pop = len(pop_order)
    n_regimes = len(REGIME_POINTS)
    fig, axes = plt.subplots(2, n_regimes, figsize=(4.6 * n_regimes, 8.8),
                              gridspec_kw={"height_ratios": [2.4, 1]})
    if n_regimes == 1:
        axes = axes.reshape(2, 1)

    x = np.arange(len(_METHODS))
    width = 0.8 / n_pop

    for col, pt in enumerate(REGIME_POINTS):
        axB, axD = axes[0, col], axes[1, col]
        for pi, pop_key in enumerate(pop_order):
            r = results[pop_key][pt["tag"]]
            means = [r["marginal"][mk]["mean"] for mk, _ in _METHODS]
            los = [r["marginal"][mk]["lo"] for mk, _ in _METHODS]
            his = [r["marginal"][mk]["hi"] for mk, _ in _METHODS]
            colors = [_C[mk] for mk, _ in _METHODS]
            xpos = x + (pi - (n_pop - 1) / 2) * width
            style = _POP_STYLE[pop_key]
            axB.bar(xpos, means, width=width * 0.92, color=colors, alpha=style["alpha"],
                    hatch=style["hatch"], edgecolor="#2C3E50", linewidth=0.8,
                    label=_POP_DISPLAY[pop_key] if col == 0 else None)
            axB.errorbar(xpos, means, yerr=[[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]],
                        fmt="none", ecolor="#2C3E50", elinewidth=1.3, capsize=4)
        axB.set_xticks(x)
        axB.set_xticklabels([lbl for _, lbl in _METHODS], fontsize=9.5)
        if col == 0:
            axB.set_ylabel("Regret (marginal 95% CI)")
        axB.set_title(pt["label"], fontsize=10.5)
        axB.grid(axis="y", color="#E3E7EB", lw=1)

        # Paired-diff panel: SingleHalt* minus Stats, for every population, side by side.
        for pi, pop_key in enumerate(pop_order):
            d = results[pop_key][pt["tag"]]["paired"]["singlehalt_minus_stats"]
            confirmed = d["b_significantly_beats_a"]
            color = _C["stats"] if confirmed else "#7A8894"
            axD.errorbar([pi], [d["mean"]], yerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]],
                        fmt="o", ms=8, color=color, ecolor=color, elinewidth=2.0, capsize=5,
                        mfc=color if pop_key == "argmax_only" else "white", mew=1.8, zorder=5)
        axD.axhline(0, color="#7A8894", lw=1.1, ls="--")
        axD.set_xlim(-0.6, n_pop - 0.4)
        axD.set_xticks(range(n_pop))
        axD.set_xticklabels([_POP_DISPLAY[p].replace("\n", " ") for p in pop_order], fontsize=7.6, rotation=12)
        if col == 0:
            axD.set_ylabel("paired $\\Delta$ (SingleHalt$^*$ $-$ Stats)\n+ = Stats wins")
        axD.grid(axis="y", color="#E3E7EB", lw=1)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#7A8894", alpha=_POP_STYLE[p]["alpha"],
                             hatch=_POP_STYLE[p]["hatch"], edgecolor="#2C3E50", linewidth=0.8)
              for p in pop_order]
    fig.legend(handles, [_POP_DISPLAY[p].replace("\n", " ") for p in pop_order], loc="upper center",
              ncol=n_pop, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=10)
    fig.suptitle("Probe X — exclude_xaba on our own puctvalue_md36 corpus\n"
                "top: marginal regret per method (95% CI)   bottom: SingleHalt*$-$Stats paired-diff 95% CI "
                "(filled marker = confirmed win)",
                fontsize=10.5, y=1.11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_pdf_png(fig, str(out_dir), "probe_x_xaba_on_own_corpus", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-packed-root", required=True, help="argmax_only (production baseline)")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--xaba-packed-root", required=True, help="argmax_and_xaba (condition A)")
    ap.add_argument("--xaba-cache", required=True)
    ap.add_argument("--condb-packed-root", default=None, help="xaba_only (condition B); omit to skip")
    ap.add_argument("--condb-cache", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    populations = [
        ("argmax_only", Path(args.base_packed_root), args.base_cache),
        ("argmax_and_xaba", Path(args.xaba_packed_root), args.xaba_cache),
    ]
    if args.condb_packed_root and args.condb_cache:
        populations.append(("xaba_only", Path(args.condb_packed_root), args.condb_cache))

    results: dict[str, dict] = {}
    for pop_key, packed_root, cache in populations:
        results[pop_key] = _run_population(pop_key, packed_root, cache, d_embed=args.d_embed,
                                           max_episodes=args.max_episodes, seed=args.seed, n_boot=args.n_boot)

    out_path = out_dir / "probe_x_xaba_results.json"
    out_path.write_text(json.dumps(_to_jsonable(results), indent=2))
    print(f"[probe-x-xaba] results -> {out_path}", flush=True)

    _render(results, out_dir)
    print("[probe-x-xaba] DONE", flush=True)


if __name__ == "__main__":
    main()
