"""No-prune normative-curve test (the reclaimed criterion).

Does a resource-rational stop's PREDICTED think-time reproduce the human RT decomposition?
Model RT = n_total(OSS), OSS = argmax_t[halt_reward(t) − c·n_total(t)] (node cost, no pruning;
from oss_nodecost.parquet). For each cost c we compare the MODEL's correlations with
{n_moves, fraction_good, action_gap} against the HUMAN RT's correlations with the same — i.e.
does the model curve +size, −satisfaction, +sharpness the way people do? Success = the model
REPRODUCES the signs/shape (not 'beats a floor'). Legal-moves is the explanandum.
"""
import os, numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
SET = os.environ.get("VOC_SET", "n1md36")
OSS = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/oss_nodecost.parquet"
VOC = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
CGRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
oss = pd.read_parquet(OSS)
voc = pd.read_parquet(VOC)[["fen", "action_gap", "n_good_0.1"]]
df = oss.merge(voc, on="fen")
con = duckdb.connect(DB, read_only=True); con.register("t", df)
m = con.execute("SELECT t.*, x.move_time FROM t JOIN processed_moves_nonzero x "
                "ON t.fen=x.fen WHERE x.move_time>0").df(); con.close()
def sp(a, b): return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
human = m["move_time"].to_numpy()
legal = m["legal_moves"].to_numpy().astype(float)
gap = m["action_gap"].to_numpy().astype(float)
frac = (m["n_good_0.1"].to_numpy() / np.maximum(m["legal_moves"].to_numpy(), 1)).astype(float)
mh = np.isfinite(gap)
print(f"n={len(m):,}\n")
print(f"{'feature':>14} {'HUMAN rho(.,RT)':>16}")
print(f"{'n_moves (size)':>14} {sp(legal, human):>+12.3f}")
print(f"{'fraction_good':>14} {sp(frac[mh], human[mh]):>+12.3f}   (satisfaction)")
print(f"{'action_gap':>14} {sp(gap[mh], human[mh]):>+12.3f}   (sharpness)")
print("\n--- MODEL n_total(OSS): does it reproduce those signs? ---")
print(f"{'c':>8} {'rho(model,RT)':>14} {'model~n_moves':>13} {'model~frac_good':>15} {'model~action_gap':>16}")
for c in CGRID:
    mr = m[f"node_ntot_c{c:g}"].to_numpy().astype(float)
    print(f"{c:>8g} {sp(mr, human):>+12.3f} {sp(legal, mr):>+11.3f} {sp(frac[mh], mr[mh]):>+13.3f} {sp(gap[mh], mr[mh]):>+14.3f}")
