"""Settle the action-gap sign: SHALLOW (myopic n=1 leaf-eval) gap vs DEEP (converged final_Q) gap,
both top1−top2, on the SAME n1md36 trees, each vs RT."""
import os, glob, numpy as np, pandas as pd, duckdb, torch
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import rankdata
TREES="/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/n1md36"; DB="/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
def _w(p):
    try:
        d=torch.load(p,map_location="cpu",weights_only=False)
        fen=" ".join(d["root_position_spec"].split()[:4])
        nf=np.asarray(d["node_features"],float); pi=np.asarray(d["parent_index"]); fn=list(d["feature_names"]); vi=fn.index("value")
        fq=np.asarray(d["oracle_final_root_q_values"],float).ravel()
        kids=np.where(pi==0)[0]
        if kids.size<2 or fq.size<2: return None
        myo=np.sort(-nf[kids,vi])[::-1]          # mover-perspective shallow values, desc
        shallow_gap=float(myo[0]-myo[1])
        fqs=np.sort(fq)[::-1]
        deep_gap=float(fqs[0]-fqs[1])
        return {"fen":fen,"legal_moves":int(kids.size),"shallow_gap":shallow_gap,"deep_gap":deep_gap}
    except Exception: return None
paths=sorted(glob.glob(TREES+"/*.pt"))
rows=[]
with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK",32))) as ex:
    for r in ex.map(_w,paths,chunksize=64):
        if r: rows.append(r)
df=pd.DataFrame(rows); con=duckdb.connect(DB,read_only=True); con.register("t",df)
m=con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x ON t.fen=x.fen WHERE x.move_time>0").df(); con.close()
def sp(a,b): return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
y=m.move_time.to_numpy()
print(f"n={len(m):,}")
print(f"rho(SHALLOW gap, RT) = {sp(m.shallow_gap.to_numpy(),y):+.3f}   (engine.py's action_gap)")
print(f"rho(DEEP gap,    RT) = {sp(m.deep_gap.to_numpy(),y):+.3f}   (voc_signals / decomposition 'sharpness')")
print(f"rho(shallow, deep)   = {sp(m.shallow_gap.to_numpy(),m.deep_gap.to_numpy()):+.3f}   (are they even the same thing?)")
print(f"rho(legal, RT)       = {sp(m.legal_moves.to_numpy().astype(float),y):+.3f}   (floor)")
