"""Single source for every ρ-vs-RT comparison figure in metareasoning.md, in ONE visual
language: horizontal bars, ρ on the x-axis, names on the y-axis, shared palette.
Produces: rt_headline (3), cost_sweep_rt (4a), voc_tau_sweep (4b), rt_partials (4d),
good_moves_signflip (5), oss_dist_elo2000 (line, no smoothing). Interim filtered elo2000.
Spearman with percentile-bootstrap 95% CIs; partials via the rank formula.
"""
import sys, os, random
sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")
import torch, numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SET = os.environ.get("VOC_SET", "n1md36")
VOCPQ = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/voc_signals.parquet"
TAU = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/voc_tau_sweep.parquet"
TREES = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/{SET}"
FILT = f"/scratch/gpfs/GRIFFITHS/hl4291/sf_analysis/{SET}/clean_trees.txt"  # may be absent (unfiltered)
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIG = "/home/hl4291/chess_analysis/figures/lmcos_tiny"

# ---- shared palette (semantic; mapped to the deck theme) ----
SIZE = "#475569"   # structural / problem size (legal moves, action gap) — slate
SAT = "#e11d48"    # satisfaction / good-moves (negative term) — rose
VOC = "#6366f1"    # value-of-computation family (gain, regret, step*, softmax-VOC) — indigo
VOC2 = "#a5b4fc"   # second VOC series (step* vs regret) in grouped plots — indigo-light
HALT = "#94a3b8"   # causal halter — light slate
REF = "#475569"    # reference line (legal-moves) — slate (thin dashed)
XLIM = (-0.4, 0.4)  # fixed x-axis for every ρ-vs-RT bar figure
CONFIGS = ["power_law_p1.5_x0.25", "power_law_p1.5_x1.0", "power_law_p1.5_x4.0",
           "power_law_p2.8_x0.25", "power_law_p2.8_x1.0", "power_law_p2.8_x4.0",
           "linear_x0.25", "linear_x1.0", "linear_x4.0",
           "quadratic_x0.25", "quadratic_x1.0", "quadratic_x4.0"]
TAUS = [0.001, 0.01, 0.05, 0.1, 0.2]


def _ranks(a): return pd.Series(a).rank().to_numpy()
def _pear(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan
def sb(x, y, nb=400, seed=0):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]; n = x.size
    if n < 50 or np.std(x) == 0: return (np.nan, np.nan, np.nan, n)
    rx, ry = _ranks(x), _ranks(y); pt = _pear(rx, ry); rng = np.random.default_rng(seed)
    bs = np.array([_pear(rx[i], ry[i]) for i in (rng.integers(0, n, n) for _ in range(nb))])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)
def pb(x, y, z, nb=300, seed=0):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); x, y, z = x[m], y[m], z[m]; n = x.size
    rx, ry, rz = _ranks(x), _ranks(y), _ranks(z)
    def part(ix):
        a, b, c = rx[ix], ry[ix], rz[ix]; rxy, rxz, ryz = _pear(a, b), _pear(a, c), _pear(b, c)
        return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    pt = part(np.arange(n)); rng = np.random.default_rng(seed)
    bs = np.array([part(rng.integers(0, n, n)) for _ in range(nb)])
    return (pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), n)


def _style(ax, title, n=None):
    ax.axvline(0, color="#222", lw=0.9); ax.invert_yaxis()
    ax.set_xlim(*XLIM)
    ax.set_xlabel("Spearman ρ vs human RT", fontsize=13)
    ax.set_title(title + (f"  (n={n:,})" if n else ""), fontsize=15)
    ax.tick_params(axis="x", labelsize=11); ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)


def hbar(ax, labels, rs, colors, title, n=None, ref=None):
    y = np.arange(len(labels)); pts = [r[0] for r in rs]
    ax.barh(y, pts, color=colors, height=0.66)
    ax.errorbar(pts, y, xerr=[[r[0] - r[1] for r in rs], [r[2] - r[0] for r in rs]],
                fmt="none", ecolor="#222", capsize=3, lw=1)
    if ref is not None:
        ax.axvline(ref, color=REF, lw=1.0, ls="--", label=f"legal moves ({ref:+.2f})")
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10); _style(ax, title, n)


def hbar_grouped(ax, labels, A, B, cA, cB, lA, lB, title, n=None, ref=None, hatchB=True):
    y = np.arange(len(labels))
    ax.barh(y - 0.2, [r[0] for r in A], height=0.36, color=cA, label=lA)
    ax.barh(y + 0.2, [r[0] for r in B], height=0.36, color=cB, label=lB,
            hatch="//" if hatchB else None, edgecolor="white" if hatchB else None)
    ax.errorbar([r[0] for r in A], y - 0.2, xerr=[[r[0]-r[1] for r in A], [r[2]-r[0] for r in A]], fmt="none", ecolor="#222", capsize=2, lw=1)
    ax.errorbar([r[0] for r in B], y + 0.2, xerr=[[r[0]-r[1] for r in B], [r[2]-r[0] for r in B]], fmt="none", ecolor="#222", capsize=2, lw=1)
    if ref is not None:
        ax.axvline(ref, color=REF, lw=1.0, ls="--", label=f"legal moves ({ref:+.2f})")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10); _style(ax, title, n)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)


def main():
    voc = pd.read_parquet(VOCPQ)
    tau = pd.read_parquet(TAU)[["fen"] + [f"softmax_voc_tau{t:g}__free" for t in TAUS]]
    df = voc.merge(tau, on="fen", how="left").drop(columns=[c for c in ["legal_moves"] if c in voc.columns])
    conn = duckdb.connect(DB, read_only=True); conn.register("_s", df[["fen"]])
    mv = conn.execute("""SELECT m.fen, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM processed_moves_nonzero m JOIN (SELECT DISTINCT fen FROM _s) f ON m.fen=f.fen
        WHERE m.move_time>0""").df(); conn.close()
    j = mv.merge(df, on="fen", how="inner")
    rt = j["log_rt"].to_numpy(); lm = j["legal_moves"].to_numpy(float); N = len(j)
    lm_ref = sb(lm, rt)[0]
    print(f"parquet join: {N:,} moves", flush=True)

    # good-moves / action_gap: prefer the parquet columns (full set) if present, else tree subsample
    if "n_good_0.1" in voc.columns:
        cols = ["fen", "legal_moves", "action_gap"] + [f"n_good_{e}" for e in (0.02, 0.05, 0.1, 0.2, 0.5)]
        gd = voc[cols].rename(columns={"legal_moves": "n_legal"}).copy()
    else:
        import glob as _glob
        names = ([l.strip() for l in open(FILT) if l.strip()] if os.path.exists(FILT)
                 else [os.path.basename(p) for p in _glob.glob(os.path.join(TREES, "*.pt"))])
        random.seed(0); random.shuffle(names); names = names[:6000]
        rows = []
        for nm in names:
            try: p = torch.load(os.path.join(TREES, nm), map_location="cpu", weights_only=False)
            except Exception: continue
            fq = np.asarray(p["oracle_final_root_q_values"], dtype=float).ravel()
            if fq.size < 1: continue
            mx = fq.max()
            r = {"fen": " ".join(p["root_position_spec"].split()[:4]), "n_legal": int(fq.size),
                 "action_gap": float(mx - np.sort(fq)[-2]) if fq.size >= 2 else np.nan}
            for e in (0.02, 0.05, 0.1, 0.2, 0.5): r[f"n_good_{e}"] = int((fq >= mx - e).sum())
            rows.append(r)
        gd = pd.DataFrame(rows)
    gd["frac_good_0.05"] = gd["n_good_0.05"] / gd["n_legal"]
    jg = mv.merge(gd, on="fen", how="inner"); rtg = jg["log_rt"].to_numpy(); lmg = jg["n_legal"].to_numpy(float); Ng = len(jg)
    print(f"good-moves join: {Ng:,} moves", flush=True)

    # ===== FIG 3: headline =====
    items = [("# legal moves (size)", sb(lm, rt), SIZE),
             ("action gap (sharpness)", sb(jg["action_gap"].to_numpy(float), rtg), SIZE),
             ("softmax-VOC (τ=0.1)", sb(j["softmax_voc_tau0.1__free"].to_numpy(float), rt), VOC),
             ("step* (best cost)", sb(j["oss__quadratic_x0.25"].to_numpy(float), rt), VOC),
             ("regret / cost-free Gain", sb(j["gain_costfree"].to_numpy(float), rt), VOC),
             ("causal halter (stop step)", (0.025, -0.015, 0.065, 0), HALT),
             ("# good moves (≤0.1)", sb(jg["n_good_0.1"].to_numpy(float), rtg), SAT),
             ("satisfaction (fraction good)", sb(jg["frac_good_0.05"].to_numpy(float), rtg), SAT)]
    order = np.argsort([it[1][0] for it in items])[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    hbar(ax, [items[i][0] for i in order], [items[i][1] for i in order], [items[i][2] for i in order],
         "What predicts human response time? (filtered elo2000)", n=N)
    ax.text(0.99, 0.02, "slate = problem size · rose = satisfaction · indigo = value-of-computation",
            transform=ax.transAxes, ha="right", fontsize=9, color="#555")
    fig.tight_layout(); [fig.savefig(f"{FIG}/rt_headline.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== FIG 4a: cost-function sweep (regret flat, step* varies) =====
    A = [sb(j[f"regret_alwaysstop__{c}"].to_numpy(float), rt) for c in CONFIGS]
    B = [sb(j[f"oss__{c}"].to_numpy(float), rt) for c in CONFIGS]
    fig, ax = plt.subplots(figsize=(9, 8))
    hbar_grouped(ax, CONFIGS, A, B, VOC, VOC2, "regret-alwaysstop", "step* (OSS)",
                 "Normative signals vs human RT, by cost regime", n=N, ref=lm_ref, hatchB=False)
    fig.tight_layout(); [fig.savefig(f"{FIG}/cost_sweep_rt.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== FIG 4b: softmax-VOC by temperature =====
    rs = [sb(j[f"softmax_voc_tau{t:g}__free"].to_numpy(float), rt) for t in TAUS]
    best_i = int(np.nanargmax([r[0] for r in rs]))
    cols = [REF if i == best_i else VOC for i in range(len(TAUS))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hbar(ax, [f"τ = {t:g}" + ("  (best)" if i == best_i else "") for i, t in enumerate(TAUS)], rs, cols,
         "Softmax-VOC vs human RT, by temperature (deep-tree supervisor)", n=N, ref=lm_ref)
    fig.tight_layout(); [fig.savefig(f"{FIG}/voc_tau_sweep.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== FIG 4d: partials (proxy proof) =====
    P = {"cost-free Gain": "gain_costfree", "regret-alwaysstop": "regret_alwaysstop__power_law_p2.8_x1.0",
         "step* (best cost)": "oss__quadratic_x0.25", "softmax-VOC (τ=0.1)": "softmax_voc_tau0.1__free"}
    labs = list(P.keys())
    raw = [sb(j[P[k]].to_numpy(float), rt) for k in labs]
    par = [pb(j[P[k]].to_numpy(float), rt, lm) for k in labs]
    fig, ax = plt.subplots(figsize=(9, 5))
    hbar_grouped(ax, labs, raw, par, VOC, VOC, "raw ρ vs RT", "ρ vs RT | legal-moves",
                 "Value-of-computation signals are proxies for legal-moves", n=N)
    fig.tight_layout(); [fig.savefig(f"{FIG}/rt_partials.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== FIG 5: sign flip =====
    eps = [0.02, 0.05, 0.1, 0.2, 0.5]
    raw_g = [sb(jg[f"n_good_{e}"].to_numpy(float), rtg) for e in eps]
    par_g = [pb(jg[f"n_good_{e}"].to_numpy(float), rtg, lmg) for e in eps]
    fig, ax = plt.subplots(figsize=(9, 5))
    hbar_grouped(ax, [f"# good ≤ {e}" for e in eps], raw_g, par_g, SAT, SAT,
                 "raw ρ vs RT", "ρ vs RT | legal-moves",
                 "The sign flip: more OPTIONS slower (slate), more GOOD options faster (rose)",
                 n=Ng, ref=sb(lmg, rtg)[0])
    fig.tight_layout(); [fig.savefig(f"{FIG}/good_moves_signflip.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)

    # ===== OSS distribution (line, no smoothing) =====
    sig = voc
    fig, ax = plt.subplots(figsize=(7.5, 5))
    edges = np.arange(0, 99, 3); ctr = (edges[:-1] + edges[1:]) / 2; maxoss = 0
    for cfg, c, lab in [("power_law_p2.8_x1.0", VOC, "power-law"),
                        ("linear_x1.0", SIZE, "linear"), ("quadratic_x1.0", SAT, "quadratic")]:
        v = sig[f"oss__{cfg}"].to_numpy(float); maxoss = max(maxoss, np.nanpercentile(v, 99.5))
        d, _ = np.histogram(v, bins=edges, density=True)
        ax.plot(ctr, d, lw=2.2, color=c, label=lab)
    ax.axvline(maxoss, color="#555", ls="--", lw=1)
    ax.annotate("forced choice —\nOSS truncated near budget 96", xy=(maxoss, ax.get_ylim()[1] * 0.6),
                xytext=(maxoss - 38, ax.get_ylim()[1] * 0.8), fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    ax.set_xlabel("oracle stop step (OSS)", fontsize=13); ax.set_ylabel("density", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, title="cost shape"); ax.set_title("OSS distribution by cost shape (mult=1, filtered elo2000)", fontsize=15)
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); [fig.savefig(f"{FIG}/oss_dist_elo2000.{e}", dpi=200, bbox_inches="tight") for e in ("png", "pdf")]
    plt.close(fig)
    print("saved: rt_headline, cost_sweep_rt, voc_tau_sweep, rt_partials, good_moves_signflip, oss_dist_elo2000", flush=True)


if __name__ == "__main__":
    main()
