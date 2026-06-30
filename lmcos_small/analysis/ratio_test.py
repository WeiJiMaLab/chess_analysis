"""Does RT ∝ #legal/#good (satisficing-search length) beat the bare #legal floor?
One-number test on the existing voc_signals parquet. Spearman ρ vs log-RT, with
partial|legal where relevant. ratio = legal/n_good = inverse density of good moves."""
import os, numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
SET=os.environ.get("VOC_SET","n1md36")
PQ=f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/voc_signals.parquet"
DB="/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
df=pd.read_parquet(PQ); con=duckdb.connect(DB,read_only=True); con.register("t",df)
m=con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x ON t.fen=x.fen WHERE x.move_time>0").df(); con.close()
y=m["move_time"].to_numpy(); leg=m["legal_moves"].to_numpy().astype(float)
def sp(a,b): return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def pa(yy,x,z):
    ry,rx,rz=rankdata(yy),rankdata(x),rankdata(z); Z=np.c_[np.ones_like(rz),rz]
    ex=rx-Z@np.linalg.lstsq(Z,rx,rcond=None)[0]; ey=ry-Z@np.linalg.lstsq(Z,ry,rcond=None)[0]
    return float(np.corrcoef(ex,ey)[0,1])
def bo(fn,*c,B=400,seed=0):
    rng=np.random.default_rng(seed); n=len(c[0]); v=np.array([fn(*[a[rng.integers(0,n,n)] for a in [c[0]]]+[a for a in c[1:]]) for _ in range(B)]) if False else None
    v=np.empty(B)
    for b in range(B):
        i=rng.integers(0,n,n); v[b]=fn(*[a[i] for a in c])
    return float(fn(*c)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))
r,lo,hi=bo(sp,leg,y); print(f"n={len(m):,}\nlegal_moves (FLOOR)        rho={r:+.3f}[{lo:+.3f},{hi:+.3f}]\n")
print(f"{'eps':>5} {'rho(ngood,RT)':>16} {'rho(fracgood,RT)':>18} {'rho(legal/ngood,RT)':>22} {'partial(ratio|legal)':>22}")
for e in (0.02,0.05,0.1,0.2,0.5):
    ng=m[f"n_good_{e}"].to_numpy().astype(float); ng=np.maximum(ng,1.0)
    ratio=leg/ng; frac=ng/leg
    rg,_,_=bo(sp,ng,y); rf,_,_=bo(sp,frac,y); rr,lr,hr=bo(sp,ratio,y); pr,lp,hp=bo(pa,y,ratio,leg)
    print(f"{e:>5} {rg:>+8.3f} {rf:>+12.3f} {rr:>+12.3f}[{lr:+.3f},{hr:+.3f}] {pr:>+10.3f}[{lp:+.3f},{hp:+.3f}]")
