"""CP node (plan.md's Next-phase DAG) -- value-scale recalibration probe, no retrain.

T3 (see plan.md T3 "Related discovery") found severe value-scale saturation in the tree
corpus: 56% of all nodes / 89% of leaf nodes have ``|value| >= 0.99`` (WDL win_prob -
loss_prob saturates like a step function around ~300cp). The raw ``cp_order`` column is
NOT saturated (continuous, median 268). This module answers: if node values are relabeled
via ``tanh(cp_order / T)`` (a gentler saturation curve) and PUCT's backup is REPLAYED over
the EXISTING tree topology (shape/expansion order held fixed -- see
``analysis.relabel_replay`` module docstring for the exact backup rule and its one
approximation), does the regret/headroom story change?

This is a first-order approximation, NOT a full re-generation: real PUCT selection during
generation was made under the OLD saturating value, so this only tests "what would the
halt-reward trajectory have looked like with different labels on the SAME shape" -- not
"what would search look like if it explored differently under better-calibrated values."
A null result here does not rule out the fix helping if search itself were rerun; a
positive result is still evidence worth acting on (motivates ``CP-RETRAIN``).

Reuses ``analysis.relabel_replay`` (tested, do-not-modify) for the replay itself and
``analysis.evaluate``'s existing regret machinery (``_return_curves``/``_regret_at``/
``fit_singlehalt_stop``) so results are directly comparable to every other regret number
in this investigation (S1, T3, Z).
"""
from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.evaluate import _mean_ci, _regret_at, _return_curves, fit_singlehalt_stop
from analysis.relabel_replay import build_trajectory, saturation_fraction, tanh_cp_value
from analysis.utils.plots import save_pdf_png
from cts.data.preprocess_gnn.teacher_targets import RawPretrainExampleRecord
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.stats import bootstrap_ci

# Same non-degenerate regime used throughout T2/T3/S1 (labnotebook 2026-07-07's
# validated, cost-regime-independent baseline) -- keeps this diagnostic's regret numbers
# directly comparable to the rest of the investigation without introducing a second
# free variable (cost regime) on top of the one this probe actually targets (value scale).
ORACLE_CONFIG = BudgetedOracleConfig(
    time_mode="linear", time_lambda=0.01, maintenance_scale=0.0, maintenance_exponent=1.0
)


# ===========================================================================
# Tree sampling / loading
# ===========================================================================
def sample_tree_paths(corpus_dir: str | Path, n_trees: int, seed: int = 0) -> list[str]:
    """Uniform random sample (without replacement) of ``*.pt`` paths under ``corpus_dir``.

    Random (not first-N) so the sample isn't biased toward whatever ordering the corpus's
    filenames happen to have (root index order tracks generation order, which could
    correlate with e.g. opening-book position).
    """
    paths = sorted(glob.glob(str(Path(corpus_dir) / "*.pt")))
    if len(paths) <= n_trees:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, n_trees)


# ===========================================================================
# Pure per-tree / per-grid-point computation (testable without touching disk)
# ===========================================================================
def leaf_mask(child_ptr: np.ndarray) -> np.ndarray:
    """Boolean mask over nodes: True iff the node has zero children (a leaf of the
    generated tree -- matches T3's "leaf node" definition: childless, whether or not it
    was ever flagged terminal)."""
    child_ptr = np.asarray(child_ptr)
    return (child_ptr[1:] - child_ptr[:-1]) == 0


def build_variant_trajectory(record: RawPretrainExampleRecord, node_values: np.ndarray) -> dict | None:
    """One tree's trajectory dict (``analysis.relabel_replay.build_trajectory`` shape)
    under an arbitrary per-node value labeling, holding topology fixed."""
    return build_trajectory(
        record.parent_index.numpy(), record.child_ptr.numpy(), record.children_index.numpy(),
        record.is_expanded.numpy(), record.is_terminal.numpy(), record.depth.numpy(), node_values,
    )


def noise_proxy(trajectory: dict) -> float:
    """Mean |consecutive-step delta| of ``halt_rewards`` -- T3's leaf-eval noise proxy,
    reused here so CP's numbers are directly comparable to T3's quiescence/pruning sweep."""
    hr = np.asarray(trajectory["halt_rewards"], dtype=np.float64)
    if hr.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(hr))))


def headroom_summary(eval_curves: list[np.ndarray], k_singlehalt: int) -> dict[str, Any]:
    """SingleHalt*'s regret vs the true per-episode oracle, and the fraction of
    "achievable" regret (relative to the AlwaysStop k=0 floor) it already recovers --
    the exact "retraining-free headroom" framing from labnotebook.md (search
    "Retraining-free \"headroom\" sweep"). Pure arithmetic, bootstrapped CIs throughout
    (repo convention: percentile-bootstrap only, never normal-theory -- ``cts.stats.bootstrap_ci``).
    """
    regret_sh = _regret_at(eval_curves, [k_singlehalt] * len(eval_curves))
    regret_as = _regret_at(eval_curves, [0] * len(eval_curves))  # AlwaysStop (k=0) floor
    mean_sh, lo_sh, hi_sh = _mean_ci(regret_sh)
    def _frac_recovered(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(b.mean())
        return 1.0 - float(a.mean()) / denom if denom > 0 else 1.0  # both floors hit oracle -> nothing to recover
    frac_point, frac_lo, frac_hi = bootstrap_ci(_frac_recovered, regret_sh, regret_as, n_boot=2000)
    return {
        "k_singlehalt": int(k_singlehalt),
        "regret_singlehalt_mean": mean_sh, "regret_singlehalt_lo": lo_sh, "regret_singlehalt_hi": hi_sh,
        "regret_always_stop_mean": float(regret_as.mean()),
        "fraction_recovered_mean": frac_point, "fraction_recovered_lo": frac_lo, "fraction_recovered_hi": frac_hi,
        # Raw per-episode arrays (SAME eval-episode order/population across every value-labeling
        # variant for a given sweep -- see run_sweep's shared `filtered_records`/`fit_mask`) so a
        # caller can run a PAIRED significance test between two variants directly, not just eyeball
        # whether two independently-computed CIs happen to overlap (a strictly weaker check).
        "regret_singlehalt_per_episode": regret_sh.tolist(),
        "regret_always_stop_per_episode": regret_as.tolist(),
    }


def paired_fraction_recovered_delta(variant: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    """Paired-bootstrap significance test of the SCALE-INVARIANT "fraction of achievable
    regret recovered" metric, ``variant`` minus ``baseline``, on the SAME held-out episode
    population (shared index resampling across all four raw arrays keeps the pairing
    intact). Positive = ``variant`` gives SingleHalt* MORE headroom already recovered
    (i.e. LESS room left for a smarter controller); negative = MORE room left.

    Absolute regret is NOT directly comparable across different value-scale relabelings
    (e.g. a large temperature ``T`` uniformly compresses reward magnitude toward 0, which
    mechanically shrinks absolute regret regardless of any real change in headroom) --
    "fraction recovered" is a same-scale RATIO (numerator/denominator both computed under
    the SAME labeling) so it cancels that confound to first order, making this the
    correct like-for-like comparison across temperatures.
    """
    def _frac(a, b):
        denom = float(b.mean())
        return 1.0 - float(a.mean()) / denom if denom > 0 else 1.0

    def _delta(sh_v, as_v, sh_b, as_b):
        return _frac(sh_v, as_v) - _frac(sh_b, as_b)

    v_sh = np.asarray(variant["regret_singlehalt_per_episode"])
    v_as = np.asarray(variant["regret_always_stop_per_episode"])
    b_sh = np.asarray(baseline["regret_singlehalt_per_episode"])
    b_as = np.asarray(baseline["regret_always_stop_per_episode"])
    point, lo, hi = bootstrap_ci(_delta, v_sh, v_as, b_sh, b_as, n_boot=2000)
    return {"delta_fraction_recovered_mean": point, "delta_fraction_recovered_lo": lo,
            "delta_fraction_recovered_hi": hi, "significant": bool(lo > 0 or hi < 0)}


# ===========================================================================
# Grid-point driver (I/O)
# ===========================================================================
def _wdl_values(record: RawPretrainExampleRecord) -> np.ndarray:
    col = record.feature_names.index("value")
    return record.node_features[:, col].to(torch.float64).numpy()


def _cp_values(record: RawPretrainExampleRecord, temperature: float) -> np.ndarray:
    col = record.feature_names.index("cp_order")
    cp = record.node_features[:, col].to(torch.float64).numpy()
    return tanh_cp_value(cp, temperature)


def saturation_over_sample(records: Sequence[RawPretrainExampleRecord],
                            value_fn: Callable[[RawPretrainExampleRecord], np.ndarray]) -> dict[str, float]:
    """Saturation fraction (all nodes + leaf-only) over the RAW, unfiltered tree sample --
    matches T3's own methodology/population exactly (T3's 56%/89% numbers were computed
    over raw generated trees, not the argmax-filtered training population), so this is the
    directly-comparable number for "did desaturating values reduce saturation."""
    all_values, leaf_values = [], []
    for record in records:
        values = value_fn(record)
        all_values.append(values)
        leaf_values.append(values[leaf_mask(record.child_ptr.numpy())])
    return {
        "saturation_all": saturation_fraction(np.concatenate(all_values)),
        "saturation_leaf": saturation_fraction(np.concatenate([v for v in leaf_values if v.size])),
    }


def evaluate_variant(
    filtered_records: Sequence[RawPretrainExampleRecord],
    value_fn: Callable[[RawPretrainExampleRecord], np.ndarray],
    fit_mask: np.ndarray,
    *,
    oracle_config: BudgetedOracleConfig = ORACLE_CONFIG,
) -> dict[str, Any]:
    """One labeling variant (original WDL, or ``tanh(cp_order/T)`` at some T)'s
    headroom/regret + noise proxy, computed over ``filtered_records`` -- the SAME
    "thinking helps" (baseline-argmax > 2) population every variant is scored on, matching
    the population production's real pipeline (``filter_argmax.py`` -> ``pack_trees`` ->
    S1/Z's validation split) actually trains/evaluates controllers on. ``fit_mask`` is a
    fixed boolean split shared across every variant so the comparison is apples-to-apples
    on the same held-out trees.
    """
    trajectories = [build_variant_trajectory(record, value_fn(record)) for record in filtered_records]

    kept_fit = [t for t, keep in zip(trajectories, fit_mask) if keep and t is not None]
    kept_eval = [t for t, keep in zip(trajectories, fit_mask) if not keep and t is not None]
    n_skipped = sum(1 for t in trajectories if t is None)

    fit_curves = _return_curves(kept_fit, oracle_config)
    eval_curves = _return_curves(kept_eval, oracle_config)
    k_singlehalt = fit_singlehalt_stop(fit_curves)
    headroom = headroom_summary(eval_curves, k_singlehalt)

    per_tree_noise = np.array([noise_proxy(t) for t in trajectories if t is not None])
    noise_mean, noise_lo, noise_hi = _mean_ci(per_tree_noise)

    return {
        "n_trees": len(filtered_records), "n_skipped": n_skipped,
        "n_fit_episodes": len(kept_fit), "n_eval_episodes": len(kept_eval),
        "noise_proxy_mean": noise_mean, "noise_proxy_lo": noise_lo, "noise_proxy_hi": noise_hi,
        **headroom,
    }


def run_sweep(
    corpus_dir: str | Path, n_trees_raw: int, temperatures: Sequence[float], *,
    min_argmax: int = 2, seed: int = 0, oracle_config: BudgetedOracleConfig = ORACLE_CONFIG,
) -> dict[str, Any]:
    """Load a raw sample of real trees once. Saturation is reported over the FULL raw
    sample (T3-comparable). Headroom/regret/noise is reported over the subset that
    survives the SAME "thinking helps" filter production applies (baseline-WDL oracle
    argmax step > ``min_argmax``, ``filter_argmax.py``'s exact criterion) -- so those
    numbers are directly comparable to S1/Z/T2's validated regret numbers rather than
    diluted by the ~75% of raw trees where any stop step is already near-optimal
    regardless of value labeling (see T3's ~24.7% "thinking helps" yield).
    """
    paths = sample_tree_paths(corpus_dir, n_trees_raw, seed=seed)
    records = [RawPretrainExampleRecord.load(p) for p in paths]

    baseline_trajs = [build_variant_trajectory(r, _wdl_values(r)) for r in records]
    keep = []
    for r, traj in zip(records, baseline_trajs):
        if traj is None:
            continue
        curve = _return_curves([traj], oracle_config)[0]
        if int(curve.argmax()) > min_argmax:
            keep.append(r)
    filtered_records = keep

    rng = np.random.default_rng(seed)
    fit_mask = rng.random(len(filtered_records)) < 0.7  # 70/30, same convention as evaluate._split

    saturation = {
        "wdl_baseline": saturation_over_sample(records, _wdl_values),
        **{f"T={T}": saturation_over_sample(records, lambda r, T=T: _cp_values(r, T)) for T in temperatures},
    }

    results = {
        "wdl_baseline": {**saturation["wdl_baseline"],
                          **evaluate_variant(filtered_records, _wdl_values, fit_mask, oracle_config=oracle_config)},
    }
    for T in temperatures:
        results[f"T={T}"] = {
            **saturation[f"T={T}"],
            **evaluate_variant(filtered_records, lambda r, T=T: _cp_values(r, T), fit_mask, oracle_config=oracle_config),
        }

    # Paired significance test (SAME held-out episodes) of the scale-invariant "fraction
    # recovered" metric, each T vs the WDL baseline -- see paired_fraction_recovered_delta
    # docstring for why this, not the raw regret delta, is the right like-for-like test.
    paired = {f"T={T}": paired_fraction_recovered_delta(results[f"T={T}"], results["wdl_baseline"])
              for T in temperatures}

    return {
        "n_trees_raw_requested": n_trees_raw, "n_trees_raw_loaded": len(records),
        "n_trees_filtered": len(filtered_records), "min_argmax": min_argmax, "seed": seed,
        "temperatures": list(temperatures), "results": results, "paired_vs_baseline": paired,
    }


# ===========================================================================
# Plot
# ===========================================================================
def plot_cp_recalibration(sweep: dict[str, Any], *, out_dir: str | Path, conclusion: str,
                           base: str = "cp_recalibration") -> str:
    """2x2 panel vs temperature T: saturation (all + leaf), SingleHalt* regret, fraction of
    achievable regret recovered, noise proxy -- each with the WDL baseline as a horizontal
    reference line. House style matches T3's sweep figures (``t3_quiescence_sweep.png``).
    """
    results = sweep["results"]
    temps = sweep["temperatures"]
    xs = [float(t) for t in temps]
    base_r = results["wdl_baseline"]
    variant = lambda key: [results[f"T={t}"][key] for t in temps]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(conclusion, fontsize=10.5, y=1.02, wrap=True)

    ax = axes[0, 0]
    ax.plot(xs, variant("saturation_all"), marker="o", color="#3F4DA0", label="all nodes")
    ax.plot(xs, variant("saturation_leaf"), marker="s", color="#C0392B", label="leaf nodes")
    ax.axhline(base_r["saturation_all"], color="#3F4DA0", ls="--", lw=1.2, alpha=0.6, label="WDL baseline (all)")
    ax.axhline(base_r["saturation_leaf"], color="#C0392B", ls="--", lw=1.2, alpha=0.6, label="WDL baseline (leaf)")
    ax.set_title("Saturation fraction (|value| >= 0.99)")
    ax.set_xlabel("temperature T"); ax.set_ylabel("fraction saturated"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    means = variant("regret_singlehalt_mean")
    los = variant("regret_singlehalt_lo"); his = variant("regret_singlehalt_hi")
    err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    ax.errorbar(xs, means, yerr=err, marker="o", color="#12A19A", capsize=3)
    ax.axhline(base_r["regret_singlehalt_mean"], color="#8B97A3", ls="--", lw=1.2,
               label=f"WDL baseline ({base_r['regret_singlehalt_mean']:.4f})")
    ax.set_title("SingleHalt* mean regret\n(NOT scale-comparable across T -- see panel below)")
    ax.set_xlabel("temperature T"); ax.set_ylabel("regret (T-dependent reward scale)"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    means = variant("fraction_recovered_mean")
    los = variant("fraction_recovered_lo"); his = variant("fraction_recovered_hi")
    err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    ax.errorbar(xs, means, yerr=err, marker="o", color="#E4A11B", capsize=3)
    ax.axhline(base_r["fraction_recovered_mean"], color="#8B97A3", ls="--", lw=1.2,
               label=f"WDL baseline ({base_r['fraction_recovered_mean']:.3f})")
    paired = sweep.get("paired_vs_baseline", {})
    for x, t in zip(xs, temps):
        if paired.get(f"T={t}", {}).get("significant"):
            y_top = results[f"T={t}"]["fraction_recovered_hi"]
            ax.annotate("*", (x, y_top), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=16, color="#C0392B", fontweight="bold")
    ax.set_title("Fraction of achievable regret recovered by SingleHalt*\n(scale-invariant; * = paired-bootstrap significant vs WDL baseline)")
    ax.set_xlabel("temperature T"); ax.set_ylabel("fraction recovered"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    means = variant("noise_proxy_mean")
    los = variant("noise_proxy_lo"); his = variant("noise_proxy_hi")
    err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    ax.errorbar(xs, means, yerr=err, marker="o", color="#8E44AD", capsize=3)
    ax.axhline(base_r["noise_proxy_mean"], color="#8B97A3", ls="--", lw=1.2,
               label=f"WDL baseline ({base_r['noise_proxy_mean']:.4f})")
    ax.set_title("Leaf-eval noise proxy: mean |Δhalt_reward|")
    ax.set_xlabel("temperature T"); ax.set_ylabel("mean |Δhalt_reward|"); ax.legend(fontsize=8)

    for row in axes:
        for a in row:
            a.grid(color="#E3E7EB", lw=1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_pdf_png(fig, str(out_dir), base, dpi=200)
