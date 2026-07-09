"""plan.md Phase 2, Agent 2 (our_trees_continued) -- paired-bootstrap significance for BOTH
continued sub-lines against AlwaysStop (low bar) and SingleHalt* (real bar), across a small
grid of REGIME-confirmed cost regimes, at one or more checkpoints/epochs per sub-line.

Two sub-lines, two modes, ONE shared statistical methodology (paired bootstrap 95% CI on the
per-episode regret difference, ``cts.stats.bootstrap_ci``, same pattern SIG-S/SIG-Z already
used -- see ``evaluate_sig_s.py``/``evaluate_e2e_z3.py``):

``--mode pg``  -- sub-line B, the production PG controller (frozen encoder, head-only PG
    training, ``cts.train.pg_controller_train``). Cheap: no re-materialization needed, since
    the encoder never changes -- the checkpoint's head is applied directly on top of the
    EXISTING frozen materialized z_t cache (``$MAT/validation_cache.pt``), reusing
    ``cts.train.controller_train._build_model_and_optimizer`` (with the new
    ``resume_checkpoint`` hook) to reconstruct the exact trained architecture from
    ``config_minply15_maxply75.yaml``'s ``train:`` stage.

``--mode e2e`` -- sub-line A, the e2e (encoder-unfrozen) Z2/Z-EXT checkpoints. Expensive: the
    encoder changes every epoch, so each checkpoint needs its OWN materialized z_t cache (built
    by the existing ``slurm/z3_e2e_materialize_and_eval.slurm`` pattern before this script runs).
    Reuses ``analysis.evaluate_e2e_z3``'s already-tested ``_e2e_point``/``_paired_baseline_diffs``
    (imported, NOT modified -- that module belongs to the already-closed SIG-Z track) for the
    e2e-vs-{SingleHalt*,Stats,orig-z_t} diffs, and adds ONE thing that track never computed:
    the AlwaysStop/AlwaysContinue paired diffs (the LOW bar plan.md explicitly asks this agent to
    check and report honestly, separate from the real SingleHalt* bar).

Both modes write one JSON per (checkpoint, regime) to ``--out-dir`` and print a one-line summary
so results can be scraped or gathered afterward for the required plots.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)  # unconstrained BLAS threading measured ~5x wall-time inflation on this
                          # shared login node (cts.analysis.stats_ablation's fix, same failure mode:
                          # many small per-episode forward passes).

from analysis.evaluate import (
    _fit_stop_controllers,
    _load_assessment_data,
    _mean_ci,
    _oracle_config,
    _regret_at,
    _return_curves,
    _to_jsonable,
)
from cts._config import load_config
from cts.data.preprocess_mc.oracle import predicted_stop_from_advantages
from cts.stats import bootstrap_ci
from cts.train.controller_train import ControllerTrainConfig, _build_model_and_optimizer, _feature_schema


# ===========================================================================
# mode=pg -- frozen-encoder production PG controller, evaluated straight off the
# existing materialized cache (no re-materialization needed).
# ===========================================================================
def _pg_features(z_ep: np.ndarray, d_embed: int) -> torch.Tensor:
    """Build the ``[z_t, N_t(unused-dummy), T_t=steps]`` feature layout
    ``MetaController.predict_from_features`` expects for ``controller_inputs=('z_t','T_t')`` --
    SAME transform ``pg_controller_train.py``'s ``_use_steps_not_budget`` applies at train time
    (budget slot overwritten with the within-episode step index; the N_t slot is sliced out by
    ``controller_inputs`` not including it, so its value is irrelevant -- zero-filled here).
    """
    n = z_ep.shape[0]
    z = torch.as_tensor(z_ep, dtype=torch.float32)
    dummy_n_t = torch.zeros(n, 1, dtype=torch.float32)
    steps = torch.arange(n, dtype=torch.float32).unsqueeze(-1)
    return torch.cat([z, dummy_n_t, steps], dim=-1)


def build_pg_model(config_path: str, checkpoint_path: str, *, d_embed: int, unfreeze_encoder: bool):
    """Reconstruct the exact production-controller architecture from ``config_minply15_maxply75.yaml``'s
    ``train:`` stage and warm-start it from ``checkpoint_path`` via the ``resume_checkpoint`` hook
    (plan.md Agent 2's addition to ``controller_train._build_model_and_optimizer``) -- no training,
    inference only."""
    config = load_config(
        ControllerTrainConfig, config_path, stage="train",
        sets=[f"resume_checkpoint={checkpoint_path}", "device=cpu", f"unfreeze_encoder={str(unfreeze_encoder).lower()}"],
    )
    schema = _feature_schema()
    model, _optimizer, _scheduler = _build_model_and_optimizer(config, schema)
    model.eval()
    return model


def pg_checkpoint_stops(model, episodes, z_by_ep, idx, d_embed: int) -> np.ndarray:
    """Per-episode GREEDY STOP STEP of the checkpoint's REAL policy (predicted_stop_from_advantages
    on its own advantage trace) -- not a re-fit readout, the actual trained weights doing inference.
    Caller converts to regret via ``_regret_at(ev, stops)`` (this function does NOT -- it has no
    access to the return curves, only the model + z_t)."""
    stops = []
    with torch.no_grad():
        for i in idx:
            feats = _pg_features(z_by_ep[i], d_embed)
            adv, _sign = model.predict_from_features(feats)
            stops.append(predicted_stop_from_advantages(adv.reshape(-1).tolist()))
    return np.array(stops)


def pg_regime_baselines(episodes, z_by_ep, fit_idx, ev_idx, packed_root: Path, *, d_embed: int, seed: int,
                        time_mode: str, time_lambda: float, maintenance_scale: float, maintenance_exponent: float):
    """Everything that depends on the REGIME but NOT the checkpoint: the return curves, and the
    SingleHalt*/Stats-Controller readouts fit on the fit split. Factored out so a sweep over MANY
    checkpoints at a FIXED regime (the common case here: sub-line B has 150 cheap per-epoch
    checkpoints, all scored against the SAME 3 regimes) doesn't redundantly re-fit these per
    checkpoint -- ``_fit_stop_controllers`` alone does a 200-epoch PG training pass per call."""
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, d_embed, seed)
    baselines = {
        "always_stop": _regret_at(ev, np.zeros(len(ev_idx), dtype=int)),
        "always_continue": _regret_at(ev, np.array([len(c) - 1 for c in ev], dtype=int)),
        "singlehalt": _regret_at(ev, ctrl["singlehalt"]),
        "stats": _regret_at(ev, ctrl["stats"]),
    }
    return ev, baselines


def score_pg_checkpoint_against_baselines(model, episodes, z_by_ep, ev_idx, d_embed: int, ev, baselines,
                                          checkpoint_path: str, epoch: int, *, time_mode: str, time_lambda: float,
                                          maintenance_scale: float, maintenance_exponent: float, n_boot: int = 2000) -> dict:
    ckpt_stops = pg_checkpoint_stops(model, episodes, z_by_ep, ev_idx, d_embed)
    ckpt_regret = _regret_at(ev, ckpt_stops)

    diffs = {}
    for name, baseline_regret in baselines.items():
        diff = baseline_regret - ckpt_regret
        diffs[name] = dict(zip(("mean", "lo", "hi"), bootstrap_ci(lambda d: float(d.mean()), diff, n_boot=n_boot)))
        diffs[name]["confirmed"] = bool(diffs[name]["lo"] > 0)

    ckpt_mean, ckpt_lo, ckpt_hi = _mean_ci(ckpt_regret)
    return dict(
        checkpoint=str(checkpoint_path), epoch=epoch, n_eval=len(ev_idx),
        checkpoint_regret=dict(mean=ckpt_mean, lo=ckpt_lo, hi=ckpt_hi),
        baseline_regret={k: dict(zip(("mean", "lo", "hi"), _mean_ci(v))) for k, v in baselines.items()},
        paired_diff=diffs,
        time_mode=time_mode, time_lambda=time_lambda,
        maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent,
    )


# ===========================================================================
# mode=e2e -- encoder-unfrozen e2e checkpoints, one materialized cache per epoch.
# ===========================================================================
def evaluate_e2e_checkpoint(packed_root: Path, original_cache_path: str, e2e_cache_path: str, *,
                            d_embed: int = 32, max_episodes: int = 15000, seed: int = 0,
                            time_mode: str = "linear", time_lambda: float = 0.01,
                            maintenance_scale: float = 0.0, maintenance_exponent: float = 1.0,
                            n_boot: int = 2000) -> dict:
    from analysis.evaluate_e2e_z3 import _e2e_point, _paired_baseline_diffs

    e2e_point = _e2e_point(packed_root, e2e_cache_path, d_embed=d_embed, max_episodes=max_episodes, seed=seed,
                           time_mode=time_mode, time_lambda=time_lambda, maintenance_scale=maintenance_scale,
                           maintenance_exponent=maintenance_exponent)
    diffs = _paired_baseline_diffs(packed_root, original_cache_path, e2e_point, d_embed=d_embed,
                                   max_episodes=max_episodes, seed=seed, time_mode=time_mode,
                                   time_lambda=time_lambda, maintenance_scale=maintenance_scale,
                                   maintenance_exponent=maintenance_exponent, n_boot=n_boot)

    # AlwaysStop/AlwaysContinue -- the LOW bar SIG-Z never checked (it only compared against the
    # three LEARNED baselines). Recomputed independently here on the SAME original-cache episode
    # split (fit/eval split depends only on trajectory_keys, identical across caches -- same
    # invariant ``_paired_baseline_diffs`` itself relies on, see its docstring).
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, _z_by_ep, _fit_idx, ev_idx = _load_assessment_data(packed_root, original_cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    ev = [curves[i] for i in ev_idx]
    e2e_regret = np.asarray(e2e_point["regret_per_episode"])
    assert len(e2e_regret) == len(ev_idx)

    always_stop_regret = _regret_at(ev, np.zeros(len(ev_idx), dtype=int))
    always_continue_regret = _regret_at(ev, np.array([len(c) - 1 for c in ev], dtype=int))
    diffs = dict(diffs)  # (mean, lo, hi) tuples for singlehalt/stats/zt, from _paired_baseline_diffs
    diffs["always_stop"] = bootstrap_ci(lambda d: float(d.mean()), always_stop_regret - e2e_regret, n_boot=n_boot)
    diffs["always_continue"] = bootstrap_ci(lambda d: float(d.mean()), always_continue_regret - e2e_regret, n_boot=n_boot)

    paired_diff = {name: dict(mean=m, lo=lo, hi=hi, confirmed=bool(lo > 0)) for name, (m, lo, hi) in diffs.items()}
    ckpt_mean, ckpt_lo, ckpt_hi = _mean_ci(e2e_regret)
    return dict(
        checkpoint=str(e2e_cache_path), n_eval=len(ev_idx),
        checkpoint_regret=dict(mean=ckpt_mean, lo=ckpt_lo, hi=ckpt_hi),
        baseline_regret=dict(
            always_stop=dict(zip(("mean", "lo", "hi"), _mean_ci(always_stop_regret))),
            always_continue=dict(zip(("mean", "lo", "hi"), _mean_ci(always_continue_regret))),
        ),
        paired_diff=paired_diff,
        time_mode=time_mode, time_lambda=time_lambda,
        maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent,
    )


def _parse_regimes(s: str) -> list[tuple[float, float]]:
    """``"0.01:0.0,0.01:0.001"`` -> ``[(0.01, 0.0), (0.01, 0.001)]`` (time_lambda:maintenance_scale)."""
    out = []
    for chunk in s.split(","):
        lam, m = chunk.split(":")
        out.append((float(lam), float(m)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pg", "e2e"], required=True)
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--cache", help="[pg] frozen materialized validation cache (unchanged across checkpoints)")
    ap.add_argument("--original-cache", help="[e2e] frozen-encoder baseline cache (for SingleHalt*/Stats/orig-z_t)")
    ap.add_argument("--config", help="[pg] config yaml (to rebuild the exact train-stage architecture)")
    ap.add_argument("--checkpoints", required=True, help="comma-separated checkpoint paths "
                    "([pg]: full model_state_dict .pt files; [e2e]: materialized e2e validation caches)")
    ap.add_argument("--epochs", required=True, help="comma-separated epoch labels, aligned with --checkpoints")
    ap.add_argument("--regimes", default="0.01:0.0", help="comma-separated time_lambda:maintenance_scale pairs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-embed", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-mode", default="linear")
    ap.add_argument("--unfreeze-encoder", action="store_true", help="[pg only] set if the checkpoint being "
                    "scored is itself encoder-unfrozen (not the normal pg lineage; included for completeness)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.checkpoints.split(",")
    epochs = [int(e) for e in args.epochs.split(",")]
    assert len(checkpoints) == len(epochs), "--checkpoints and --epochs must be aligned 1:1"
    regimes = _parse_regimes(args.regimes)

    all_results = []
    if args.mode == "pg":
        # Load episodes/z_t ONCE (same frozen cache for every checkpoint), then loop regimes
        # OUTER (each regime's SingleHalt*/Stats readout fit ONCE, reused across every
        # checkpoint) and checkpoints INNER -- sub-line B has up to 150 cheap per-epoch
        # checkpoints, so avoiding a redundant 200-epoch readout re-fit per checkpoint matters.
        episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(
            Path(args.packed_root), args.cache, args.d_embed, args.max_episodes, args.seed)
        for time_lambda, maintenance_scale in regimes:
            ev, baselines = pg_regime_baselines(
                episodes, z_by_ep, fit_idx, ev_idx, Path(args.packed_root), d_embed=args.d_embed, seed=args.seed,
                time_mode=args.time_mode, time_lambda=time_lambda, maintenance_scale=maintenance_scale,
                maintenance_exponent=1.0)
            for ckpt, epoch in zip(checkpoints, epochs):
                model = build_pg_model(args.config, ckpt, d_embed=args.d_embed, unfreeze_encoder=args.unfreeze_encoder)
                result = score_pg_checkpoint_against_baselines(
                    model, episodes, z_by_ep, ev_idx, args.d_embed, ev, baselines, ckpt, epoch,
                    time_mode=args.time_mode, time_lambda=time_lambda, maintenance_scale=maintenance_scale,
                    maintenance_exponent=1.0)
                all_results.append(result)
                pd = result["paired_diff"]
                print(f"[our-trees-continued mode=pg] epoch={epoch} lambda={time_lambda} m={maintenance_scale} "
                      f"regret={result['checkpoint_regret']['mean']:.4f} "
                      f"vs_always_stop={pd['always_stop']['mean']:+.4f}[{pd['always_stop']['lo']:+.4f},{pd['always_stop']['hi']:+.4f}] "
                      f"confirmed={pd['always_stop']['confirmed']} "
                      f"vs_singlehalt={pd['singlehalt']['mean']:+.4f}[{pd['singlehalt']['lo']:+.4f},{pd['singlehalt']['hi']:+.4f}] "
                      f"confirmed={pd['singlehalt']['confirmed']}", flush=True)
    else:
        for ckpt, epoch in zip(checkpoints, epochs):
            for time_lambda, maintenance_scale in regimes:
                result = evaluate_e2e_checkpoint(
                    Path(args.packed_root), args.original_cache, ckpt,
                    d_embed=args.d_embed, max_episodes=args.max_episodes, seed=args.seed,
                    time_mode=args.time_mode, time_lambda=time_lambda, maintenance_scale=maintenance_scale,
                )
                result["epoch"] = epoch
                all_results.append(result)
                pd = result["paired_diff"]
                print(f"[our-trees-continued mode=e2e] epoch={epoch} lambda={time_lambda} m={maintenance_scale} "
                      f"regret={result['checkpoint_regret']['mean']:.4f} "
                      f"vs_always_stop={pd['always_stop']['mean']:+.4f}[{pd['always_stop']['lo']:+.4f},{pd['always_stop']['hi']:+.4f}] "
                      f"confirmed={pd['always_stop']['confirmed']} "
                      f"vs_singlehalt={pd['singlehalt']['mean']:+.4f}[{pd['singlehalt']['lo']:+.4f},{pd['singlehalt']['hi']:+.4f}] "
                      f"confirmed={pd['singlehalt']['confirmed']}", flush=True)

    # MERGE with any pre-existing results at this path rather than overwriting -- sub-line A's
    # chained-job pattern (agent2_e2e_materialize_eval_1step.slurm) invokes this script once per
    # step, each with a DIFFERENT single checkpoint/epoch but the SAME --out-dir/--mode (so the
    # same fixed filename): step 2 finishing after step 1 would otherwise silently clobber step
    # 1's results. Dedup key is (checkpoint, epoch, time_lambda, maintenance_scale) -- a rerun of
    # the SAME step overwrites its own prior entry (last-write-wins), but never drops a DIFFERENT
    # step's entries. (Caught live during this investigation: step 2's job was about to overwrite
    # step 1's already-written results when this fix landed.)
    out_path = out_dir / f"our_trees_continued_{args.mode}_results.json"
    existing = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[our-trees-continued] WARNING: could not read/parse existing {out_path} "
                  f"({exc}) -- starting fresh, not merging", flush=True)
    def _key(r):
        return (r["checkpoint"], r["epoch"], r["time_lambda"], r["maintenance_scale"])
    merged = {_key(r): r for r in existing}
    merged.update({_key(r): r for r in all_results})
    merged_results = sorted(merged.values(), key=lambda r: (r["epoch"], r["time_lambda"], r["maintenance_scale"]))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(merged_results), f, indent=2)
    print(f"[our-trees-continued] -> {out_path} ({len(existing)} pre-existing + {len(all_results)} new "
          f"-> {len(merged_results)} merged entries)", flush=True)


if __name__ == "__main__":
    main()
