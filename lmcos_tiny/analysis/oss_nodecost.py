"""OSS with a NODE cost — the dynamic-stop test (R-TREESEARCH follow-up).

The cost we always zeroed (`maintenance_scale=0`) was the per-NODE cost. Here we turn it on
and make it THE cost: a hindsight oracle that stops at

    OSS(c) = argmax_t [ halt_reward(t) − c · n_total(t) ]

where halt_reward(t) = V_deep[argmax_t] (value of the move you'd commit to at step t, already
in the trajectory) and n_total(t) = cumulative tree size at step t (the total operations paid).
This is a DYNAMIC stop whose tree-size n_total(OSS) varies position-to-position — the variable
we never measured (prior trees froze the budget at 96, so size ∝ branching ≈ legal-moves).

Question: does n_total(OSS) (or OSS itself) track RT BEYOND the +0.34 legal-moves floor, or does
the optimal stop saturate so that size collapses back to branching?

  python oss_nodecost.py            # compute -> parquet (existing n1md36 trees, no regen)
  python oss_nodecost.py analyze    # RT join + Spearman(OSS / n_total(OSS), RT) + partial|legal
"""
from __future__ import annotations
import os, sys, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, torch

sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")
from cts.data.preprocess_mc.pack import build_compact_trajectory_from_payload  # noqa: E402

SET = os.environ.get("VOC_SET", "n1md36")
TREES = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/{SET}"
OUT = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/oss_nodecost.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
# c sweeps from "almost no node cost" (stop near the budget) to "heavy" (stop early).
# halt_reward ~ win-prob [-1,1]; n_total up to ~2800, so c·n_total spans ~0.03..8 across the grid.
CGRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]


def _worker(path):
    try:
        torch.set_num_threads(1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        fen = " ".join(payload["root_position_spec"].split()[:4])
        traj = build_compact_trajectory_from_payload(payload, source_path=path)
        if traj is None or traj["num_steps"] <= 0:
            return None
        hr = np.asarray(traj["halt_rewards"], dtype=float)
        ts = np.asarray(traj["tree_sizes"], dtype=float)
        if hr.size < 2:
            return None
        legal = int(np.asarray(payload["oracle_final_root_q_values"]).ravel().size)
        row = {"fen": fen, "legal_moves": legal, "n_total": int(ts[-1])}
        for c in CGRID:
            oss = int(np.argmax(hr - c * ts))     # the dynamic optimal stop under a node cost
            row[f"oss_c{c:g}"] = oss
            row[f"ntot_c{c:g}"] = int(ts[oss])    # the tree size at that stop = the RT predictor
        return row
    except Exception:  # noqa: BLE001
        return None


def compute():
    paths = sorted(glob.glob(TREES + "/*.pt"))
    print(f"SET={SET}  trees={len(paths)}  cgrid={CGRID}", flush=True)
    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    rows = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for i, r in enumerate(ex.map(_worker, paths, chunksize=64), 1):
            if r is not None:
                rows.append(r)
            if i % 50000 == 0:
                print(f"  {i}/{len(paths)} kept={len(rows)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"wrote {OUT}  rows={len(rows)}", flush=True)


def _spear(a, b):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _partial(y, x, z):
    from scipy.stats import rankdata
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def _boot(fn, *c, B=400, seed=0):
    rng = np.random.default_rng(seed); n = len(c[0]); v = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); v[b] = fn(*[a[i] for a in c])
    return float(fn(*c)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def analyze():
    import duckdb
    df = pd.read_parquet(OUT)
    con = duckdb.connect(DB, read_only=True); con.register("t", df)
    m = con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x "
                    "ON t.fen=x.fen WHERE x.move_time>0").df()
    con.close()
    y = m["move_time"].to_numpy(); leg = m["legal_moves"].to_numpy().astype(float)
    r, lo, hi = _boot(_spear, leg, y)
    print(f"n={len(m):,}\nlegal_moves (floor)   rho={r:+.3f}[{lo:+.3f},{hi:+.3f}]\n", flush=True)
    print(f"{'c':>8} {'med OSS':>8} {'med ntot(OSS)':>14} "
          f"{'rho(OSS,RT)':>20} {'rho(ntot_oss,RT)':>22} {'partial(ntot_oss|legal)':>24}", flush=True)
    for c in CGRID:
        oss = m[f"oss_c{c:g}"].to_numpy().astype(float)
        nto = m[f"ntot_c{c:g}"].to_numpy().astype(float)
        ro, _, _ = _boot(_spear, oss, y)
        rn, lnn, hnn = _boot(_spear, nto, y)
        pn, lpp, hpp = _boot(_partial, y, nto, leg)
        print(f"{c:>8g} {int(np.median(oss)):>8} {int(np.median(nto)):>14} "
              f"{ro:>+8.3f} {rn:>+8.3f}[{lnn:+.3f},{hnn:+.3f}] {pn:>+8.3f}[{lpp:+.3f},{hpp:+.3f}]", flush=True)


if __name__ == "__main__":
    (analyze if len(sys.argv) > 1 and sys.argv[1] == "analyze" else compute)()
