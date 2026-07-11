from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from analysis.evaluate import (
    _fit_stop_controllers,
    _load_assessment_data,
    _mean_ci,
    _oracle_config,
    _regret_at,
    _return_curves,
    _to_jsonable,
)
from cts.stats import bootstrap_ci


def compute_pairwise_significance(packed_root: Path, cache_path: str, *, d_embed: int = 32,
                                  max_episodes: int = 15000, seed: int = 0, time_mode: str = "linear",
                                  time_lambda: float = 0.01, maintenance_scale: float = 0.0,
                                  maintenance_exponent: float = 1.0, n_boot: int = 2000) -> dict:
    config = replace(_oracle_config(packed_root), time_mode=time_mode, time_lambda=time_lambda,
                     maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent)
    episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache_path, d_embed, max_episodes, seed)
    curves = _return_curves(episodes, config)
    ev = [curves[i] for i in ev_idx]
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, [curves[i] for i in fit_idx], d_embed, seed)

    regret = {
        "singlehalt": _regret_at(ev, ctrl["singlehalt"]),
        "stats": _regret_at(ev, ctrl["stats"]),
        "zt": _regret_at(ev, ctrl["zt"]),
    }
    marginal = {name: dict(zip(("mean", "lo", "hi"), _mean_ci(arr))) for name, arr in regret.items()}

    pairs = [("singlehalt", "stats"), ("singlehalt", "zt"), ("stats", "zt")]
    paired = {}
    for a, b in pairs:
        diff = regret[a] - regret[b]  # positive = b has LOWER (better) regret, i.e. b wins
        mean, lo, hi = bootstrap_ci(lambda d: float(d.mean()), diff, n_boot=n_boot)
        paired[f"{a}_minus_{b}"] = dict(mean=mean, lo=lo, hi=hi, b_significantly_beats_a=bool(lo > 0),
                                        a_significantly_beats_b=bool(hi < 0))

    return dict(
        k_singlehalt=int(ctrl["k_singlehalt"]), n_fit=len(fit_idx), n_eval=len(ev_idx),
        marginal=marginal, paired=paired,
        time_mode=time_mode, time_lambda=time_lambda,
        maintenance_scale=maintenance_scale, maintenance_exponent=maintenance_exponent,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="regime", help="filename tag, e.g. 'lambda0p01_m0'")
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
    result = compute_pairwise_significance(
        Path(args.packed_root), args.cache, d_embed=args.d_embed, max_episodes=args.max_episodes,
        seed=args.seed, time_mode=args.time_mode, time_lambda=args.time_lambda,
        maintenance_scale=args.maintenance_scale, maintenance_exponent=args.maintenance_exponent,
    )
    out_path = out_dir / f"ysagiv_sig_{args.tag}.json"
    out_path.write_text(json.dumps(_to_jsonable(result), indent=2))
    m = result["marginal"]
    print(f"[ysagiv-sig] {args.tag}: SingleHalt*(k={result['k_singlehalt']})={m['singlehalt']['mean']:.4f} "
          f"Stats={m['stats']['mean']:.4f} zt={m['zt']['mean']:.4f}", flush=True)
    for name, d in result["paired"].items():
        print(f"  {name}: {d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]", flush=True)
    print(f"[ysagiv-sig] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
