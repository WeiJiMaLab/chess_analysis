"""HEAVY oracle-side step (run via sbatch CPU).

For each FILTERED elo2000 tree: build the canonical (halt_rewards, tree_sizes)
exactly as pack._build_compact_trajectory does, then run the budgeted-oracle DP
at starting_budget=96 across a 12-point cost grid, extracting the normative
"value of thinking" per-position scalars. Also computes the softmax (VOC) halt
rewards from oracle_root_q_trace. Writes one parquet keyed by 4-field FEN; the
RT join + correlations + plots happen in a light follow-up.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")

from cts.data.preprocess_mc.oracle import (  # noqa: E402
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    return_for_stop_step,
    time_cost,
)
from cts.data.preprocess_mc.pack import build_compact_trajectory_from_payload  # noqa: E402

# Tree set + filtered/unfiltered are env-driven so the 2d_data_analysis stage can run
# this on any set (e.g. n1md36, n100md36) over the full unfiltered population by default.
SET = os.environ.get("VOC_SET", "n1md36")          # dir under sf_trees/
FILTERED = os.environ.get("VOC_FILTERED", "0") == "1"  # default: unfiltered (all trees)
TREES_DIR = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/{SET}"
OUTDIR = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}"
FILTER_TXT = f"{OUTDIR}/clean_trees.txt"           # only consulted when FILTERED=1
OUT = f"{OUTDIR}/voc_signals.parquet"
STARTING_BUDGET = 96
BASE_LAMBDA = 18.537

# Cost grid: time_mode x time_lambda multiplier. 12 configs total.
LAMBDA_MULTS = [0.25, 1.0, 4.0]
MODES = [
    ("power_law", {"time_p": 1.5}),
    ("power_law", {"time_p": 2.8}),
    ("linear", {}),
    ("quadratic", {"time_budget_ref": STARTING_BUDGET}),
]


def _cfg_label(mode: str, extra: dict, mult: float) -> str:
    if mode == "power_law":
        return f"power_law_p{extra['time_p']}_x{mult}"
    return f"{mode}_x{mult}"


def _unit_total(mode: str, extra: dict) -> float:
    """Total time-cost of a full B-step search at time_lambda=1 (tree-independent since
    maintenance=0). Used to put every cost SHAPE on a comparable total-budget scale."""
    c = BudgetedOracleConfig(maintenance_scale=0.0, time_lambda=1.0, time_mode=mode, **extra)
    return sum(time_cost(b, c) for b in range(2, STARTING_BUDGET + 1))


# Reference total = cost of a full search under the CURRENT pipeline cost (power_law p2.8,
# lambda=18.537) ≈ 1.94 in win-prob units. Every (mode, mult) is calibrated so a full
# search costs mult * T_REF — so mult sweeps the TOTAL thinking budget while the mode
# changes only WHERE the cost falls (early/uniform/late). This fixes the earlier bug where
# linear/quadratic reused the raw power_law lambda (~900x too costly -> degenerate).
T_REF = _unit_total("power_law", {"time_p": 2.8}) * BASE_LAMBDA


def _make_cfg(mode: str, extra: dict, mult: float) -> BudgetedOracleConfig:
    lam = (mult * T_REF) / _unit_total(mode, extra)
    return BudgetedOracleConfig(maintenance_scale=0.0, time_lambda=lam, time_mode=mode, **extra)


COST_CONFIGS = {
    _cfg_label(mode, extra, mult): _make_cfg(mode, extra, mult)
    for mode, extra in MODES
    for mult in LAMBDA_MULTS
}
# Cost-free reference config (no time cost) for gain_costfree.
COST_FREE = BudgetedOracleConfig(maintenance_scale=0.0, time_lambda=0.0, time_mode="linear")


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / tau
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def _worker(path: str):
    try:
        torch.set_num_threads(1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        fen = " ".join(payload["root_position_spec"].split()[:4])
        traj = build_compact_trajectory_from_payload(payload, source_path=path)
        if traj is None or traj["num_steps"] <= 0:
            return None
        halt_rewards = [float(v) for v in traj["halt_rewards"].tolist()]
        tree_sizes = [int(v) for v in traj["tree_sizes"].tolist()]
        n_steps = len(halt_rewards)
        if n_steps == 0:
            return None

        # --- softmax (VOC) halt rewards, tau=1 -----------------------------
        # halt_rewards_softmax[s] = sum_c softmax(q_trace[s,:]/1) * final_q[c]
        # The trace is over the FULL search (root_rank trimming applies to the
        # trajectory steps); align the softmax trace to the trimmed steps.
        qt = np.asarray(payload["oracle_root_q_trace"], dtype=float)
        qt = qt.reshape(qt.shape[0], -1)
        final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
        root_rank = int(traj["first_decision_expansion_count"]) - 1
        softmax_rewards = None
        if qt.shape[1] == final_q.size and final_q.size >= 1:
            qt_trim = qt[root_rank:root_rank + n_steps]
            if qt_trim.shape[0] == n_steps:
                sr = []
                for s in range(n_steps):
                    row = qt_trim[s]
                    if not np.isfinite(row).all():
                        sr = None
                        break
                    p = _softmax(row, 1.0)
                    sr.append(float(np.dot(p, final_q)))
                softmax_rewards = sr

        row: dict = {"fen": fen, "n_steps": n_steps, "legal_moves": int(final_q.size)}

        # decision-difficulty signals (sign-flip test): # good moves within eps of best,
        # and the top-1 minus top-2 action gap.
        mxq = float(final_q.max())
        row["action_gap"] = float(mxq - np.sort(final_q)[-2]) if final_q.size >= 2 else float("nan")
        for _e in (0.02, 0.05, 0.1, 0.2, 0.5):
            row[f"n_good_{_e}"] = int((final_q >= mxq - _e).sum())

        # gain_costfree = oracle_value(cost=0) - halt_rewards[0]
        pol0 = compute_budgeted_oracle(halt_rewards, tree_sizes, STARTING_BUDGET, COST_FREE)
        row["gain_costfree"] = float(pol0.oracle_value - halt_rewards[0])

        # theta grid for regret_fraction (fit globally later; here store
        # return_for_stop_step at a grid of fractions per cost so the fit is cheap).
        theta_grid = np.linspace(0.0, 1.0, 21)

        for label, cfg in COST_CONFIGS.items():
            pol = compute_budgeted_oracle(halt_rewards, tree_sizes, STARTING_BUDGET, cfg)
            ov = float(pol.oracle_value)
            tb = pol.time_budgets
            hr = pol.halt_rewards
            ts = pol.tree_sizes
            m = len(hr)
            ret0 = return_for_stop_step(hr, ts, tb, 0, cfg)
            row[f"regret_alwaysstop__{label}"] = ov - ret0
            row[f"oss__{label}"] = int(pol.optimal_stop_step)
            row[f"oracle_value__{label}"] = ov
            # per-theta returns (clamped index) for the fraction fit
            for th in theta_grid:
                idx = min(int(round(th * STARTING_BUDGET)), m - 1)
                row[f"retfrac__{label}__{th:.2f}"] = return_for_stop_step(hr, ts, tb, idx, cfg)

            # softmax VOC: same DP but on softmax halt rewards
            if softmax_rewards is not None:
                pols = compute_budgeted_oracle(softmax_rewards, tree_sizes, STARTING_BUDGET, cfg)
                tbs = pols.time_budgets
                hrs = pols.halt_rewards
                tss = pols.tree_sizes
                rets0 = return_for_stop_step(hrs, tss, tbs, 0, cfg)
                row[f"softmax_voc__{label}"] = float(pols.oracle_value) - rets0
            else:
                row[f"softmax_voc__{label}"] = float("nan")
        return row
    except Exception as e:  # noqa: BLE001
        return ("ERR", path, repr(e))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if FILTERED and os.path.exists(FILTER_TXT):
        with open(FILTER_TXT) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        paths = [os.path.join(TREES_DIR, n) for n in names]
        paths = [p for p in paths if os.path.exists(p)]
        print(f"SET={SET} FILTERED: {len(names)} listed; existing: {len(paths)}", flush=True)
    else:
        import glob as _glob
        paths = sorted(_glob.glob(os.path.join(TREES_DIR, "*.pt")))
        print(f"SET={SET} UNFILTERED: {len(paths)} trees", flush=True)

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    print(f"workers={n_workers}", flush=True)
    rows = []
    errs = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_worker, p) for p in paths]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is None:
                continue
            if isinstance(r, tuple) and r and r[0] == "ERR":
                errs += 1
                if errs <= 5:
                    print("ERR", r[1], r[2], flush=True)
                continue
            rows.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(paths)} done, kept={len(rows)}, errs={errs}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}  rows={len(df)}  errs={errs}", flush=True)
    print("cost configs:", list(COST_CONFIGS.keys()), flush=True)


if __name__ == "__main__":
    main()
