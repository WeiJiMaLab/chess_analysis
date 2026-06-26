"""Train-once / sweep-many: one cost-free value-gain head across all 7 regimes.

Trains a SINGLE :class:`cts.models.readout.ValueGainReadout` on the cost-free
``value_gain(t)`` target (zero-cost DP; regime-independent), then evaluates it
against EVERY relabeled cost regime by subtracting that regime's analytic cost
at decision time (``swept_advantage_trace``) and computing regret vs that
regime's relabeled oracle.  No retrain per regime -- that is the whole point.

Per regime it reports the cost-sweep row: Always / Never / Fraction(theta* fit
on that regime's TRAIN regret) / value-gain-MC.  The value-gain-MC vs Fraction
gap is the headline.

    PYTHONPATH=lmcos/src python -m cts.analysis._budgeted.value_gain_sweep \
        --costsweep-root /scratch/gpfs/GRIFFITHS/hl4291/packed/mc_costsweep \
        --out-dir figures --max-episodes 3000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch

from cts.analysis._budgeted.alt_models_eval import (
    _load_split_episodes,
    _oracle_config,
    _stats_features,
    _fit_fraction,
    _always_stop,
    _never_stop,
    _fraction_rule,
)
from cts.analysis._budgeted.baselines import _evaluate_baseline
from cts.data.preprocess_mc.oracle import BudgetedOracleConfig
from cts.data.preprocess_mc.value_gain import (
    value_gain_trace,
    swept_advantage_trace,
    predicted_halt_curve,
    dp_stop_over_curve,
)
from cts.models.readout import ValueGainReadout, HaltCurveReadout, stop_step_from_advantages


# Regime display order: cheap -> expensive (matches the P1 table).
REGIME_ORDER = [
    "linear_0.0", "lambda_0.1", "lambda_1.0", "linear_0.003",
    "lambda_5.0", "lambda_18.537", "linear_0.02",
]


def _fit_value_gain_head(
    train: list[dict[str, Any]],
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
) -> ValueGainReadout:
    """Fit a ValueGainReadout via MSE on the cost-free value_gain target.

    The target is a rich non-flat landscape (not the flat boundary), so plain
    MSE is appropriate.  Normalization stats are baked onto the model for the
    deploy wrapper.  Single-threaded, full-batch -- a few seconds.
    """
    torch.manual_seed(seed)
    model = ValueGainReadout()

    path_lengths = [len(ep["heights"]) for ep in train]
    feats = torch.cat([_stats_features(ep) for ep in train], dim=0)
    budgets = torch.cat(
        [torch.full((n, 1), float(ep["starting_budget"])) for n, ep in zip(path_lengths, train)], dim=0
    )
    full = torch.cat([feats, budgets], dim=-1)
    mean = full.mean(dim=0)
    std = full.std(dim=0).clamp_min(1e-6)
    normed = (full - mean) / std

    target = torch.cat(
        [torch.tensor(value_gain_trace(ep["halt_rewards"], ep["tree_sizes"], ep["starting_budget"]),
                      dtype=torch.float32)
         for ep in train],
        dim=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model.head(normed).squeeze(-1).clamp_min(0.0)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()

    model.eval()
    model._norm_mean = mean  # type: ignore[attr-defined]
    model._norm_std = std  # type: ignore[attr-defined]
    return model


def _predict_value_gain(model: ValueGainReadout, episode: dict[str, Any]) -> list[float]:
    """Cost-free predicted value gain (>=0) for one episode (deploy wrapper)."""
    mean = model._norm_mean  # type: ignore[attr-defined]
    std = model._norm_std  # type: ignore[attr-defined]
    f = _stats_features(episode)
    n = f.shape[0]
    b = torch.full((n, 1), float(episode["starting_budget"]))
    x = (torch.cat([f, b], dim=-1) - mean) / std
    with torch.no_grad():
        pred = model.head(x).squeeze(-1).clamp_min(0.0)
    return pred.tolist()


def _value_gain_rule(model: ValueGainReadout, config: BudgetedOracleConfig) -> Callable[[dict[str, Any]], int]:
    """[GREEDY BASELINE] Deploy rule: predicted total gain - per-step cost, greedy stop.

    Kept only to SHOW the myopic/horizon misalignment vs the DP rule.
    """
    def rule(episode: dict[str, Any]) -> int:
        gains = _predict_value_gain(model, episode)
        adv = swept_advantage_trace(gains, episode["tree_sizes"], episode["starting_budget"], config)
        return stop_step_from_advantages(torch.tensor(adv, dtype=torch.float32))
    return rule


def _fit_halt_curve_head(
    train: list[dict[str, Any]],
    *,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
) -> HaltCurveReadout:
    """Fit a HaltCurveReadout via MSE on the cost-free per-step halt-reward curve.

    Same backbone/normalization as the value-gain head, but the target is the
    per-step VALUE CURVE (halt_reward at each step), so the eval can run the true
    DP over it for any regime.  Single-threaded, full-batch -- a few seconds.
    """
    torch.manual_seed(seed)
    model = HaltCurveReadout()

    path_lengths = [len(ep["heights"]) for ep in train]
    feats = torch.cat([_stats_features(ep) for ep in train], dim=0)
    budgets = torch.cat(
        [torch.full((n, 1), float(ep["starting_budget"])) for n, ep in zip(path_lengths, train)], dim=0
    )
    full = torch.cat([feats, budgets], dim=-1)
    mean = full.mean(dim=0)
    std = full.std(dim=0).clamp_min(1e-6)
    normed = (full - mean) / std

    target = torch.cat(
        [torch.tensor(predicted_halt_curve(ep["halt_rewards"], ep["tree_sizes"], ep["starting_budget"]),
                      dtype=torch.float32)
         for ep in train],
        dim=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model.head(normed).squeeze(-1)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()

    model.eval()
    model._norm_mean = mean  # type: ignore[attr-defined]
    model._norm_std = std  # type: ignore[attr-defined]
    return model


def _predict_halt_curve(model: HaltCurveReadout, episode: dict[str, Any]) -> list[float]:
    """Cost-free predicted per-step halt-reward curve for one episode."""
    mean = model._norm_mean  # type: ignore[attr-defined]
    std = model._norm_std  # type: ignore[attr-defined]
    f = _stats_features(episode)
    n = f.shape[0]
    b = torch.full((n, 1), float(episode["starting_budget"]))
    x = (torch.cat([f, b], dim=-1) - mean) / std
    with torch.no_grad():
        return model.head(x).squeeze(-1).tolist()


def _halt_curve_rule(model: HaltCurveReadout, config: BudgetedOracleConfig) -> Callable[[dict[str, Any]], int]:
    """[PRODUCTION] Deploy rule: DP backward-induction over the predicted curve."""
    def rule(episode: dict[str, Any]) -> int:
        curve = _predict_halt_curve(model, episode)
        return dp_stop_over_curve(curve, episode["tree_sizes"], episode["starting_budget"], config)
    return rule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--costsweep-root", required=True)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--max-episodes", type=int, default=3000)
    args = ap.parse_args()
    torch.set_num_threads(1)

    sweep_root = Path(args.costsweep_root)
    regimes = [r for r in REGIME_ORDER if (sweep_root / r).is_dir()]

    # Load every regime's train/val ONCE; cache. Raw traces are identical across
    # regimes, so train the value-gain head once on the first regime's TRAIN.
    print("loading regimes (once each)...", flush=True)
    cache: dict[str, dict[str, Any]] = {}
    for r in regimes:
        root = sweep_root / r
        tr = _load_split_episodes(root, "train", args.max_episodes)
        va = _load_split_episodes(root, "validation", args.max_episodes)
        cache[r] = {"train": tr, "val": va, "config": _oracle_config(root)}
        print(f"  {r}: train={len(tr)} val={len(va)}", flush=True)

    # ---- train ONCE (cost-free targets are regime-independent) --------------
    base_train = cache[regimes[0]]["train"]
    curve_model = _fit_halt_curve_head(base_train)   # PRODUCTION: trajectory + DP
    vg_model = _fit_value_gain_head(base_train)       # BASELINE: greedy gain-scalar
    print("trained halt-curve head (DP) + value-gain head (greedy baseline), once each.", flush=True)

    rows: list[dict[str, Any]] = []
    for r in regimes:
        cfg = cache[r]["config"]
        tr, va = cache[r]["train"], cache[r]["val"]
        best_f, _ = _fit_fraction(tr, cfg)  # regime-specific Fraction theta*

        m_always = _evaluate_baseline(va, cfg, _always_stop)
        m_never = _evaluate_baseline(va, cfg, _never_stop)
        m_frac = _evaluate_baseline(va, cfg, _fraction_rule(best_f))
        m_dp = _evaluate_baseline(va, cfg, _halt_curve_rule(curve_model, cfg))     # production
        m_vg = _evaluate_baseline(va, cfg, _value_gain_rule(vg_model, cfg))         # greedy baseline
        rows.append({
            "regime": r,
            "always": m_always["average_regret"],
            "never": m_never["average_regret"],
            "fraction": m_frac["average_regret"],
            "vg_dp": m_dp["average_regret"],         # trajectory + real DP (PRODUCTION)
            "vg_greedy": m_vg["average_regret"],     # myopic gain-scalar (BASELINE)
            "frac_f_star": best_f,
            "dp_stop_oss": m_dp["exact_stop_step_accuracy"],
            "dp_avg_expansions": m_dp["average_expansions"],
        })

    # ---- table --------------------------------------------------------------
    hdr = (f"{'regime':<16s}{'Always':>9s}{'Never':>9s}{'Fraction':>10s}"
           f"{'VG-DP(prod)':>13s}{'VG-greedy':>11s}{'gap(DP-Frac)':>13s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for row in rows:
        gap = row["vg_dp"] - row["fraction"]
        print(f"{row['regime']:<16s}{row['always']:>9.3f}{row['never']:>9.3f}"
              f"{row['fraction']:>10.3f}{row['vg_dp']:>13.3f}{row['vg_greedy']:>11.3f}{gap:>+13.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "value_gain_sweep.json").write_text(json.dumps(rows, indent=2))

    # ---- figure: Fraction vs value-gain-MC per regime -----------------------
    labels = [row["regime"] for row in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.27
    ax.bar([i - 1.5 * w for i in x], [r["always"] for r in rows], w, label="Always", color="#bbbbbb")
    ax.bar([i - 0.5 * w for i in x], [r["fraction"] for r in rows], w, label="Fraction (budget-only)", color="#4063A3")
    ax.bar([i + 0.5 * w for i in x], [r["vg_dp"] for r in rows], w, label="VG-DP train-once (prod)", color="#C84B31")
    ax.bar([i + 1.5 * w for i in x], [r["vg_greedy"] for r in rows], w, label="VG-greedy (baseline)", color="#E8A87C")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean regret (lower = better)")
    ax.set_title("Train-once value-gain head swept across cost regimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "value_gain_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out_dir / 'value_gain_sweep.png'} and value_gain_sweep.json")


if __name__ == "__main__":
    main()
