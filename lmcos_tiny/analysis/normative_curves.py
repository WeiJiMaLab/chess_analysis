"""No-prune normative-curve test (the reclaimed criterion).

Does a resource-rational stop's PREDICTED think-time reproduce the human RT decomposition?
Model RT = n_total(OSS), OSS = argmax_t[halt_reward(t) − c·n_total(t)] (node cost, no pruning;
from oss_nodecost.parquet). For each cost c we compare the MODEL's correlations with the two
PRE-SEARCH causal features — {n_moves (size), fraction_good (satisfaction)} — against the HUMAN
RT's correlations with the same. Success = the model REPRODUCES the signs (not 'beats a floor');
legal-moves is the explanandum.

NOTE: action-gap is deliberately EXCLUDED. The decomposition's "sharpness" used the *deep*
(converged final_Q) gap, a POST-search OUTCOME (large because you searched) — confounded with
think-time, not a pre-search cause you can read a priori. And the *myopic* (n=1, pre-search) gap
is ~null and sign-unstable (Spearman +0.06 / Pearson −0.07; coarse n=1 values make it lumpy). So
sharpness is not a pre-search cause; the myopic gap stays only in the engine-derived plot
(human_analytics/engine.py / figures/engine/action_gap). gap_reconcile confirms shallow⊥deep (ρ=0.07).
"""
import os, numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
SET = os.environ.get("VOC_SET", "n1md36")
OSS = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/oss_nodecost.parquet"
VOC = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
CGRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
oss = pd.read_parquet(OSS)
voc = pd.read_parquet(VOC)[["fen", "n_good_0.1"]]
df = oss.merge(voc, on="fen")
con = duckdb.connect(DB, read_only=True); con.register("t", df)
m = con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x "
                "ON t.fen=x.fen WHERE x.move_time>0").df(); con.close()
def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
human = m["move_time"].to_numpy()
legal = m["legal_moves"].to_numpy().astype(float)
frac = (m["n_good_0.1"].to_numpy() / np.maximum(m["legal_moves"].to_numpy(), 1)).astype(float)
print(f"n={len(m):,}\n")
print(f"{'feature':>14} {'HUMAN rho(.,RT)':>16}")
print(f"{'n_moves (size)':>14} {sp(legal, human):>+12.3f}")
print(f"{'fraction_good':>14} {sp(frac, human):>+12.3f}   (satisfaction)")
print("\n--- MODEL n_total(OSS): does it reproduce those two signs? ---")
print(f"{'c':>8} {'rho(model,RT)':>14} {'model~n_moves':>13} {'model~frac_good':>15}")
for c in CGRID:
    mr = m[f"node_ntot_c{c:g}"].to_numpy().astype(float)
    print(f"{c:>8g} {sp(mr, human):>+12.3f} {sp(legal, mr):>+11.3f} {sp(frac, mr):>+13.3f}")
