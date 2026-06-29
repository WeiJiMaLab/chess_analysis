"""Generate the proof figures for metareasoning.md (interim, filtered elo2000).
Produces:
  rt_headline.png        — what predicts human RT (ρ landscape, bootstrap CIs)
  rt_partials.png        — value/VOC/step* signals: raw↔RT vs ↔RT|legal-moves (proxy proof)
  good_moves_signflip.png — # all moves (+) vs # good moves (−): the sign flip
All Spearman with percentile-bootstrap 95% CIs; partials via the rank formula.
"""
import sys, os, random
sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")
import torch, numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOC = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
TAU = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_tau_sweep.parquet"
TREES = "/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/elo2000"
FILT = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/clean_trees.txt"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIG = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
MAIN, ACC, GRN, GREY = "#2E86C1", "#C0392B", "#27AE60", "#7F8C8D"


def _ranks(a):
    return pd.Series(a).rank().to_numpy()


def _pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def sb(x, y, nb=500, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx, ry = _ranks(x), _ranks(y); pt = _pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([_pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(nb))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def pb(x, y, z, nb=400, seed=0):  # partial spearman rho_xy.z + bootstrap CI
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); x, y, z = x[m], y[m], z[m]; n = x.size
    rx, ry, rz = _ranks(x), _ranks(y), _ranks(z)
    def part(ix):
        a, b, c = rx[ix], ry[ix], rz[ix]
        rxy, rxz, ryz = _pear(a, b), _pear(a, c), _pear(b, c)
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    pt = part(np.arange(n)); rng = np.random.default_rng(seed)
    bs = np.array([part(rng.integers(0, n, n)) for _ in range(nb)])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def hbar(ax, labels, pts, los, his, colors, title, xlabel="Spearman ρ vs human log-RT"):
    y = np.arange(len(labels))
    ax.barh(y, pts, color=colors, height=0.62)
    ax.errorbar(pts, y, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)],
                fmt="none", ecolor="#222", capsize=3, lw=1)
    ax.axvline(0, color="#222", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=10); ax.set_title(title, fontsize=12)
    ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    for yi, p in zip(y, pts):
        ax.text(p + (0.006 if p >= 0 else -0.006), yi, f"{p:+.2f}", va="center",
                ha="left" if p >= 0 else "right", fontsize=8)


def main():
    voc = pd.read_parquet(VOC)
    tau = pd.read_parquet(TAU)[["fen", "softmax_voc_tau0.1__free"]]
    df = voc.merge(tau, on="fen", how="left").drop(columns=[c for c in ["legal_moves"] if c in voc.columns])
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", df[["fen"]])
    mv = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); conn.close()
    j = mv.merge(df, on="fen", how="inner")
    rt = j["log_rt"].to_numpy(); lm = j["legal_moves"].to_numpy(float)
    print(f"parquet join: {len(j):,} moves", flush=True)

    # signal columns in the parquet join
    P = {"cost-free Gain": "gain_costfree", "regret-alwaysstop": "regret_alwaysstop__power_law_p2.8_x1.0",
         "step* (best cost)": "oss__quadratic_x0.25", "softmax-VOC (τ0.1)": "softmax_voc_tau0.1__free"}

    # ---- tree subsample for # good / # all / action_gap ----
    names = [l.strip() for l in open(FILT) if l.strip()]; random.seed(0); random.shuffle(names); names = names[:6000]
    rows = []
    for nm in names:
        try:
            p = torch.load(os.path.join(TREES, nm), map_location="cpu", weights_only=False)
        except Exception:
            continue
        fq = np.asarray(p["oracle_final_root_q_values"], dtype=float).ravel()
        if fq.size < 1: continue
        mx = fq.max()
        rows.append({"fen": " ".join(p["root_position_spec"].split()[:4]), "n_legal": int(fq.size),
                     "action_gap": float(mx - np.sort(fq)[-2]) if fq.size >= 2 else np.nan,
                     "n_good_0.02": int((fq >= mx - 0.02).sum()), "n_good_0.05": int((fq >= mx - 0.05).sum()),
                     "n_good_0.1": int((fq >= mx - 0.1).sum()), "n_good_0.2": int((fq >= mx - 0.2).sum()),
                     "n_good_0.5": int((fq >= mx - 0.5).sum())})
    gd = pd.DataFrame(rows); gd["frac_good_0.05"] = gd["n_good_0.05"] / gd["n_legal"]
    jg = mv.merge(gd, on="fen", how="inner"); rtg = jg["log_rt"].to_numpy(); lmg = jg["n_legal"].to_numpy(float)
    print(f"good-moves join: {len(jg):,} moves", flush=True)

    # ===== FIG 1: headline RT landscape =====
    items = [("# legal moves (size)", lm, MAIN),
             ("action gap (sharpness)", jg["action_gap"].to_numpy(float), MAIN),
             ("softmax-VOC (τ0.1)", j[P["softmax-VOC (τ0.1)"]].to_numpy(float), GREY),
             ("step* (best cost)", j[P["step* (best cost)"]].to_numpy(float), GREY),
             ("regret / cost-free Gain", j["gain_costfree"].to_numpy(float), GREY),
             ("fraction good (forgiveness)", jg["frac_good_0.05"].to_numpy(float), ACC),
             ("# good moves (≤0.1)", jg["n_good_0.1"].to_numpy(float), ACC)]
    L, Pt, Lo, Hi, C = [], [], [], [], []
    for lab, col, c in items:
        r = sb(col, rtg if col is jg["action_gap"].to_numpy(float) else rt) if False else sb(col, rtg if len(col) == len(rtg) else rt)
        L.append(lab); Pt.append(r[0]); Lo.append(r[1]); Hi.append(r[2]); C.append(c)
    # halter (from the held-out rollout run, n≈2.5k) — hardcoded point
    L.append("causal halter (stop step)"); Pt.append(0.025); Lo.append(-0.015); Hi.append(0.065); C.append("#BDC3C7")
    order = np.argsort(Pt)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    hbar(ax, [L[i] for i in order], [Pt[i] for i in order], [Lo[i] for i in order], [Hi[i] for i in order],
         [C[i] for i in order], "What predicts human response time? (filtered elo2000)")
    ax.text(0.99, 0.02, "blue=problem size · red=forgiveness · grey=value-of-computation",
            transform=ax.transAxes, ha="right", fontsize=8, color="#555")
    fig.tight_layout(); [fig.savefig(f"{FIG}/rt_headline.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== FIG 2: partials (proxy proof) =====
    fig, ax = plt.subplots(figsize=(9, 5))
    labs = list(P.keys()); ybase = np.arange(len(labs))
    raw = [sb(j[P[k]].to_numpy(float), rt) for k in labs]
    par = [pb(j[P[k]].to_numpy(float), rt, lm) for k in labs]
    ax.barh(ybase - 0.2, [r[0] for r in raw], height=0.38, color=MAIN, label="raw ρ vs RT")
    ax.barh(ybase + 0.2, [r[0] for r in par], height=0.38, color=ACC, label="ρ vs RT | legal-moves (partial)")
    ax.errorbar([r[0] for r in raw], ybase - 0.2, xerr=[[r[0]-r[1] for r in raw], [r[2]-r[0] for r in raw]], fmt="none", ecolor="#222", capsize=2, lw=1)
    ax.errorbar([r[0] for r in par], ybase + 0.2, xerr=[[r[0]-r[1] for r in par], [r[2]-r[0] for r in par]], fmt="none", ecolor="#222", capsize=2, lw=1)
    ax.axvline(0, color="#222", lw=0.9); ax.set_yticks(ybase); ax.set_yticklabels(labs, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ vs human log-RT"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Value-of-computation signals are proxies for legal-moves\n(every one collapses when legal-moves is partialled out)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    # reference: legal-moves controlling for the signal stays high
    lm_par = pb(lm, rt, j[P["regret-alwaysstop"]].to_numpy(float))
    ax.text(0.99, 0.02, f"by contrast: legal-moves ρ|regret = {lm_par[0]:+.2f} (survives)", transform=ax.transAxes, ha="right", fontsize=8, color=GRN)
    fig.tight_layout(); [fig.savefig(f"{FIG}/rt_partials.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== FIG 3: sign flip =====
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    eps = [0.02, 0.05, 0.1, 0.2, 0.5]
    raw_all = sb(lmg, rtg)
    raw_good = [sb(jg[f"n_good_{e}"].to_numpy(float), rtg) for e in eps]
    par_good = [pb(jg[f"n_good_{e}"].to_numpy(float), rtg, lmg) for e in eps]
    ax.axhline(raw_all[0], color=MAIN, lw=2, label=f"# ALL legal moves ({raw_all[0]:+.2f})")
    ax.fill_between([eps[0], eps[-1]], raw_all[1], raw_all[2], color=MAIN, alpha=0.12)
    ax.errorbar(eps, [r[0] for r in raw_good], yerr=[[r[0]-r[1] for r in raw_good], [r[2]-r[0] for r in raw_good]],
                fmt="o-", color=ACC, capsize=3, label="# GOOD moves (raw)")
    ax.errorbar(eps, [r[0] for r in par_good], yerr=[[r[0]-r[1] for r in par_good], [r[2]-r[0] for r in par_good]],
                fmt="s--", color="#E67E22", capsize=3, label="# GOOD moves | legal (partial)")
    ax.axhline(0, color="#222", lw=0.9); ax.set_xscale("log")
    ax.set_xlabel("'good' threshold ε  (win-prob below best move)"); ax.set_ylabel("Spearman ρ vs human log-RT")
    ax.set_title("The sign flip: more OPTIONS ⇒ slower, more GOOD options ⇒ faster", fontsize=12)
    ax.legend(fontsize=9, loc="center left"); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); [fig.savefig(f"{FIG}/good_moves_signflip.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)
    print("saved rt_headline, rt_partials, good_moves_signflip (png+pdf)", flush=True)


if __name__ == "__main__":
    main()
