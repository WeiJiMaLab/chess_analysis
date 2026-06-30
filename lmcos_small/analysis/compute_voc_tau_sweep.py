"""HEAVY oracle-side step (run via sbatch CPU): softmax-VOC TEMPERATURE sweep.

For each FILTERED elo2000 tree, build the canonical (halt_rewards, tree_sizes) via
pack.build_compact_trajectory_from_payload, then for each tau in TAUS build the
softmax halt rewards
    halt_softmax[s] = sum_c softmax(oracle_root_q_trace[s,:]/tau) * oracle_final_root_q_values[c]
and run the budgeted-oracle DP to get the softmax-VOC = oracle_value - return_for_stop_step(0),
at TWO costs:
  - cur : current pipeline cost (power_law, time_p=2.8, time_lambda=18.537, maintenance_scale=0)
  - free: cost-free (time_lambda=0)

Motivation: the first pass used tau=1, too HOT for win-prob-scale child values (gaps ~0.01-0.1
=> softmax(v/1) ~ uniform => policy barely sharpens => VOC degenerated, negatively correlated
with RT and legal moves). Sweep tau in the win-prob scale to see if it flips/strengthens.

Writes one parquet keyed by 4-field FEN with columns
  softmax_voc_tau{tau}__{cur|free}  +  legal_moves (= oracle_final_root_q_values size).

python /home/hl4291/venv/bin/python ; sys.path => lmcos_small/src
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_small/src")

from cts.data.preprocess_mc.oracle import (  # noqa: E402
    BudgetedOracleConfig,
    compute_budgeted_oracle,
    return_for_stop_step,
)
from cts.data.preprocess_mc.pack import build_compact_trajectory_from_payload  # noqa: E402

SET = os.environ.get("VOC_SET", "n1md36")
FILTERED = os.environ.get("VOC_FILTERED", "0") == "1"
TREES_DIR = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/{SET}"
OUTDIR = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}"
FILTER_TXT = f"{OUTDIR}/clean_trees.txt"
OUT = f"{OUTDIR}/voc_tau_sweep.parquet"
STARTING_BUDGET = 96

TAUS = [0.001, 0.01, 0.05, 0.1, 0.2]

# Current pipeline cost (matches power_law_p2.8 with the raw base lambda).
CUR = BudgetedOracleConfig(
    maintenance_scale=0.0, time_lambda=18.537, time_mode="power_law", time_p=2.8
)
# Cost-free.
FREE = BudgetedOracleConfig(maintenance_scale=0.0, time_lambda=0.0, time_mode="linear")
COSTS = {"cur": CUR, "free": FREE}


def _tau_label(tau: float) -> str:
    return f"{tau:g}"


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
        tree_sizes = [int(v) for v in traj["tree_sizes"].tolist()]
        n_steps = len(tree_sizes)
        if n_steps == 0:
            return None

        qt = np.asarray(payload["oracle_root_q_trace"], dtype=float)
        qt = qt.reshape(qt.shape[0], -1)
        final_q = np.asarray(payload["oracle_final_root_q_values"], dtype=float).ravel()
        root_rank = int(traj["first_decision_expansion_count"]) - 1

        row: dict = {"fen": fen, "n_steps": n_steps, "legal_moves": int(final_q.size)}

        ok = qt.shape[1] == final_q.size and final_q.size >= 1
        qt_trim = qt[root_rank:root_rank + n_steps] if ok else None
        if ok and (qt_trim.shape[0] != n_steps or not np.isfinite(qt_trim).all()):
            ok = False

        for tau in TAUS:
            lab = _tau_label(tau)
            if not ok:
                for ck in COSTS:
                    row[f"softmax_voc_tau{lab}__{ck}"] = float("nan")
                continue
            softmax_rewards = [
                float(np.dot(_softmax(qt_trim[s], tau), final_q)) for s in range(n_steps)
            ]
            for ck, cfg in COSTS.items():
                pol = compute_budgeted_oracle(
                    softmax_rewards, tree_sizes, STARTING_BUDGET, cfg
                )
                ret0 = return_for_stop_step(
                    pol.halt_rewards, pol.tree_sizes, pol.time_budgets, 0, cfg
                )
                row[f"softmax_voc_tau{lab}__{ck}"] = float(pol.oracle_value) - ret0
        return row
    except Exception as e:  # noqa: BLE001
        return ("ERR", path, repr(e))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if FILTERED and os.path.exists(FILTER_TXT):
        with open(FILTER_TXT) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        paths = [os.path.join(TREES_DIR, n) for n in names if os.path.exists(os.path.join(TREES_DIR, n))]
        print(f"SET={SET} FILTERED: {len(paths)} trees", flush=True)
    else:
        import glob as _glob
        paths = sorted(_glob.glob(os.path.join(TREES_DIR, "*.pt")))
        print(f"SET={SET} UNFILTERED: {len(paths)} trees", flush=True)

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    print(f"workers={n_workers}; taus={TAUS}; costs={list(COSTS)}", flush=True)
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


if __name__ == "__main__":
    main()
