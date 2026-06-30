"""OSS under a NODE cost vs a STEP cost — the dynamic-stop test (R-TREESEARCH follow-up).

Two hindsight-oracle stops, on the SAME trajectory:
    node-cost:  OSS = argmax_t [ halt_reward(t) − c · n_total(t) ]   (cost ∝ tree size / operations)
    step-cost:  OSS = argmax_t [ halt_reward(t) − c · n_steps(t) ]   (cost ∝ expansions, the old step*)
halt_reward(t) = V_deep[argmax_t]; n_total(t) = cumulative nodes; n_steps(t) = t (expansions).
For each: record OSS (steps) and n_total(OSS) (operations at the stop) and correlate both with RT.
The contrast: a per-STEP cost ignores branching → n_total(OSS) ∝ branching ≈ legal-moves; a per-NODE
cost charges wide positions faster → de-branches n_total(OSS). Existing n1md36 trees, no regen.

  python oss_nodecost.py            # compute -> parquet
  python oss_nodecost.py analyze    # RT join + correlations for both cost metrics
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
CGRID_NODE = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]   # cost = c·n_total (n_total up to ~2800)
CGRID_STEP = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]   # cost = c·n_steps (n_steps up to ~96)


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
        steps = np.arange(1, hr.size + 1, dtype=float)   # n_steps (expansions) at each stop
        legal = int(np.asarray(payload["oracle_final_root_q_values"]).ravel().size)
        row = {"fen": fen, "legal_moves": legal, "n_total": int(ts[-1])}
        for c in CGRID_NODE:
            oss = int(np.argmax(hr - c * ts))
            row[f"node_oss_c{c:g}"] = oss
            row[f"node_ntot_c{c:g}"] = int(ts[oss])
        for c in CGRID_STEP:
            oss = int(np.argmax(hr - c * steps))
            row[f"step_oss_c{c:g}"] = oss
            row[f"step_ntot_c{c:g}"] = int(ts[oss])
        return row
    except Exception:  # noqa: BLE001
        return None


def compute():
    paths = sorted(glob.glob(TREES + "/*.pt"))
    print(f"SET={SET}  trees={len(paths)}", flush=True)
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


def analyze():
    import duckdb
    df = pd.read_parquet(OUT)
    con = duckdb.connect(DB, read_only=True); con.register("t", df)
    m = con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x "
                    "ON t.fen=x.fen WHERE x.move_time>0").df()
    con.close()
    y = m["move_time"].to_numpy(); leg = m["legal_moves"].to_numpy().astype(float)
    print(f"n={len(m):,}   legal_moves floor: rho={_spear(leg, y):+.3f}", flush=True)
    for tag, grid in (("NODE-cost (c*n_total)", CGRID_NODE), ("STEP-cost (c*n_steps)", CGRID_STEP)):
        pre = "node" if tag.startswith("NODE") else "step"
        print(f"\n=== {tag} ===", flush=True)
        print(f"{'c':>8} {'medOSS':>7} {'med ntot(OSS)':>13} {'rho(OSS,RT)':>12} "
              f"{'rho(ntot,RT)':>13} {'partial(ntot|legal)':>20}", flush=True)
        for c in grid:
            oss = m[f"{pre}_oss_c{c:g}"].to_numpy().astype(float)
            nto = m[f"{pre}_ntot_c{c:g}"].to_numpy().astype(float)
            print(f"{c:>8g} {int(np.median(oss)):>7} {int(np.median(nto)):>13} "
                  f"{_spear(oss, y):>+8.3f} {_spear(nto, y):>+8.3f} {_partial(y, nto, leg):>+12.3f}", flush=True)


if __name__ == "__main__":
    (analyze if len(sys.argv) > 1 and sys.argv[1] == "analyze" else compute)()
