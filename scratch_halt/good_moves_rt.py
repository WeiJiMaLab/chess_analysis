"""Sign-flip test: # ALL legal moves should track RT positively (enumeration/branching),
but # GOOD moves (within eps of best) should track RT NEGATIVELY (many acceptable options
=> easy decision => faster). Sweep the quality threshold; look for the flip. Per-child deep
values from the trees; correlate raw and partialled-on n_legal (the clean 'given the action
space, how forgiving is it' test)."""
import sys, os, random
sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")
import torch, numpy as np, pandas as pd, duckdb

TREES = "/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/elo2000"
FILT = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/clean_trees.txt"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
EPS = [0.02, 0.05, 0.1, 0.2, 0.5]   # win-prob below best to still count as "good"
N = 6000


def rsp(a, b):
    m = np.isfinite(a) & np.isfinite(b); a = pd.Series(a[m]).rank().to_numpy(); b = pd.Series(b[m]).rank().to_numpy()
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def partial(x, y, z):
    rxy, rxz, ryz = rsp(x, y), rsp(x, z), rsp(y, z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    names = [l.strip() for l in open(FILT) if l.strip()]
    random.seed(0); random.shuffle(names); names = names[:N]
    rows = []
    for nm in names:
        try:
            p = torch.load(os.path.join(TREES, nm), map_location="cpu", weights_only=False)
        except Exception:
            continue
        fq = np.asarray(p["oracle_final_root_q_values"], dtype=float).ravel()
        if fq.size < 1:
            continue
        mx = fq.max()
        row = {"fen": " ".join(p["root_position_spec"].split()[:4]), "n_legal": int(fq.size),
               "action_gap": float(mx - np.sort(fq)[-2]) if fq.size >= 2 else float("nan")}
        for e in EPS:
            row[f"n_good_{e}"] = int((fq >= mx - e).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", df[["fen"]])
    mv = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt FROM processed_moves_nonzero m
        JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen WHERE m.move_time>0""").df(); conn.close()
    j = mv.merge(df, on="fen", how="inner")
    rt = j["log_rt"].to_numpy(); nl = j["n_legal"].to_numpy(float)
    print(f"trees loaded: {len(df):,}  matched moves: {len(j):,}", flush=True)
    print(f"\n{'signal':22s} {'raw vs RT':>10s} {'vs RT | n_legal':>16s}")
    print(f"{'n_legal (ALL)':22s} {rsp(nl, rt):+10.4f} {'—':>16s}")
    print(f"{'action_gap (top-2)':22s} {rsp(j['action_gap'].to_numpy(float), rt):+10.4f} {partial(j['action_gap'].to_numpy(float), rt, nl):+16.4f}")
    for e in EPS:
        s = j[f"n_good_{e}"].to_numpy(float)
        print(f"{'n_good(<= '+str(e)+')':22s} {rsp(s, rt):+10.4f} {partial(s, rt, nl):+16.4f}")
    # fraction good (within 0.05) — 'how forgiving is the position'
    fg = (j["n_good_0.05"] / j["n_legal"]).to_numpy(float)
    print(f"{'frac_good(0.05)':22s} {rsp(fg, rt):+10.4f} {partial(fg, rt, nl):+16.4f}")


if __name__ == "__main__":
    main()
