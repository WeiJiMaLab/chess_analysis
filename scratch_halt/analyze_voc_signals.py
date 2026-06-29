"""LIGHT step: RT join + Spearman correlations (bootstrap CIs) + plots.

Reads the per-tree VOC signals parquet, joins per-FEN scalars to human RT
(processed_moves_nonzero, move_time>0) in personal.db, fits theta* for
regret_fraction per cost, computes RAW Spearman(signal, log RT) with
percentile-bootstrap 95% CIs across the 12-cost grid, and emits the 4 figures.
"""
from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hl4291/chess_analysis/lmcos_tiny/src")

SIGNALS = "/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo2000/voc_signals.parquet"
DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
FIGDIR = "/home/hl4291/chess_analysis/figures/lmcos_tiny"
STARTING_BUDGET = 96

LAMBDA_MULTS = [0.25, 1.0, 4.0]
MODES = [("power_law", "p1.5"), ("power_law", "p2.8"), ("linear", None), ("quadratic", None)]


def _labels():
    labs = []
    for mode, p in MODES:
        for mult in LAMBDA_MULTS:
            labs.append(f"power_law_{p}_x{mult}" if mode == "power_law" else f"{mode}_x{mult}")
    return labs


COST_LABELS = _labels()

_MAIN = "#2E86C1"
_STAR = "#F1C40F"
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("  saved", os.path.join(FIGDIR, f"{name}.png"), "(+ .pdf)")


def _spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def _spearman_boot_ci(x, y, n_boot=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    rho = _spearman(x, y)
    boots = np.empty(n_boot)
    rx_full = pd.Series(x); ry_full = pd.Series(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = pd.Series(x[idx]).rank().corr(pd.Series(y[idx]).rank())
    return rho, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def main():
    sig = pd.read_parquet(SIGNALS)
    print(f"signals rows (trees): {len(sig)}; FENs: {sig['fen'].nunique()}", flush=True)

    # ---- fit theta* per cost on the tree set (regret-minimizing fraction) ----
    theta_grid = [f"{t:.2f}" for t in np.linspace(0, 1, 21)]
    theta_star = {}
    for lab in COST_LABELS:
        ov = sig[f"oracle_value__{lab}"].to_numpy()
        best_th, best_regret = 0.0, np.inf
        for ts in theta_grid:
            col = f"retfrac__{lab}__{ts}"
            mean_regret = float(np.nanmean(ov - sig[col].to_numpy()))
            if mean_regret < best_regret:
                best_regret, best_th = mean_regret, float(ts)
        theta_star[lab] = best_th
        # materialize regret_fraction signal at theta*
        idx_ts = f"{min(theta_grid, key=lambda t: abs(float(t)-best_th))}"
        sig[f"regret_fraction__{lab}"] = ov - sig[f"retfrac__{lab}__{idx_ts}"].to_numpy()
    print("theta* per cost:", {k: round(v, 2) for k, v in theta_star.items()}, flush=True)

    # ---- RT join: per-FEN scalars -> all human moves at that FEN ----
    signal_cols = (
        [f"regret_alwaysstop__{l}" for l in COST_LABELS]
        + [f"regret_fraction__{l}" for l in COST_LABELS]
        + [f"softmax_voc__{l}" for l in COST_LABELS]
        + [f"oss__{l}" for l in COST_LABELS]
        + ["gain_costfree"]
    )
    sig_join = sig[["fen"] + signal_cols].copy()
    conn = duckdb.connect(DB, read_only=True)
    conn.register("_sig", sig_join)
    df = conn.execute("""
        SELECT s.*, ln(m.move_time) AS log_rt, m.n_possible_moves AS legal_moves
        FROM _sig s
        JOIN processed_moves_nonzero m ON m.fen = s.fen AND m.move_time > 0
    """).df()
    conn.close()
    print(f"joined human moves: {len(df):,} across {df['fen'].nunique():,} FENs", flush=True)

    # legal_moves from the tree (final_q size) is per-FEN; DB n_possible_moves is the
    # human driver. Use the DB one for the RT/legal correlations.
    n = len(df)

    # ---- correlations across the 12-cost grid for the 3 main signals ----
    families = {
        "regret_alwaysstop": "regret_alwaysstop",
        "regret_fraction": "regret_fraction",
        "softmax_voc": "softmax_voc",
    }
    results = {}  # fam -> list of (label, rho, lo, hi)
    for fam, pref in families.items():
        rows = []
        for lab in COST_LABELS:
            col = f"{pref}__{lab}"
            sub = df[[col, "log_rt"]].dropna()
            rho, lo, hi = _spearman_boot_ci(sub[col].to_numpy(), sub["log_rt"].to_numpy(), seed=hash(lab) % 1000)
            rows.append((lab, rho, lo, hi, len(sub)))
        results[fam] = rows

    # headline: best |corr| cost per family
    best = {}
    for fam, rows in results.items():
        b = max(rows, key=lambda r: abs(r[1]))
        best[fam] = b
        print(f"[{fam}] best |rho| cost = {b[0]}: rho={b[1]:+.4f} CI[{b[2]:+.4f},{b[3]:+.4f}] n={b[4]:,}", flush=True)

    # gain_costfree reference + legal_moves vs RT
    sub = df[["gain_costfree", "log_rt"]].dropna()
    r_gain = _spearman_boot_ci(sub["gain_costfree"].to_numpy(), sub["log_rt"].to_numpy(), seed=1)
    sub = df[["legal_moves", "log_rt"]].dropna()
    r_legal = _spearman_boot_ci(sub["legal_moves"].to_numpy(), sub["log_rt"].to_numpy(), seed=2)
    print(f"[ref] gain_costfree vs logRT: rho={r_gain[0]:+.4f} CI[{r_gain[1]:+.4f},{r_gain[2]:+.4f}]", flush=True)
    print(f"[ref] legal_moves   vs logRT: rho={r_legal[0]:+.4f} CI[{r_legal[1]:+.4f},{r_legal[2]:+.4f}]", flush=True)

    # =====================================================================
    # FIG 1: correlation-vs-cost-config bars (95% CI) for the 3 families
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, (fam, rows) in zip(axes, results.items()):
        labs = [r[0] for r in rows]
        rhos = np.array([r[1] for r in rows])
        los = np.array([r[2] for r in rows])
        his = np.array([r[3] for r in rows])
        x = np.arange(len(labs))
        bmax = max(range(len(rows)), key=lambda i: abs(rhos[i]))
        colors = [_STAR if i == bmax else _MAIN for i in range(len(rows))]
        ax.bar(x, rhos, color=colors, yerr=[rhos - los, his - rhos], capsize=3, error_kw={"lw": 1})
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"{fam}\nmax |ρ| @ {labs[bmax]} (ρ={rhos[bmax]:+.3f})", fontsize=11)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Spearman ρ(signal, human log-RT)")
    fig.suptitle(f"Normative value-of-thinking vs human log-RT across 12 cost configs "
                 f"(filtered elo2000, n={n:,} moves)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "voc_correlation_vs_cost_config")

    # =====================================================================
    # FIG 2: best-cost signal vs log-RT, binned scatter (per family)
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (fam, b) in zip(axes, best.items()):
        col = f"{families[fam]}__{b[0]}"
        sub = df[[col, "log_rt"]].dropna()
        xv = sub[col].to_numpy(); yv = sub["log_rt"].to_numpy()
        k = 20
        edges = np.quantile(xv, np.linspace(0, 1, k + 1))
        edges = np.unique(edges)
        idx = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, len(edges) - 2)
        xs, ys, es = [], [], []
        for bi in range(len(edges) - 1):
            mask = idx == bi
            if mask.sum() < 50:
                continue
            xs.append(xv[mask].mean()); ys.append(yv[mask].mean())
            es.append(1.96 * yv[mask].std() / np.sqrt(mask.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=_MAIN, capsize=2, ms=4, lw=1.2)
        ax.set_xlabel(f"{fam} @ {b[0]}")
        ax.set_ylabel("mean human log-RT")
        ax.set_title(f"{fam}: ρ={b[1]:+.3f}", fontsize=11)
        ax.set_axisbelow(True)
    fig.suptitle(f"Best-cost normative signal vs human log-RT (binned, filtered elo2000, n={n:,})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "voc_bestcost_signal_vs_logrt")

    # =====================================================================
    # FIG 3: OSS distribution (cost-aware, best cost for regret_alwaysstop)
    # =====================================================================
    oss_lab = best["regret_alwaysstop"][0]
    oss_tree = sig[f"oss__{oss_lab}"].to_numpy()  # per-tree, not RT-weighted
    frac0 = float((oss_tree == 0).mean())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.arange(0, min(int(oss_tree.max()) + 2, STARTING_BUDGET + 1)) - 0.5
    ax.hist(oss_tree, bins=bins, color=_MAIN, edgecolor="white", lw=0.4)
    ax.set_xlabel("oracle optimal stop step (OSS), cost-aware")
    ax.set_ylabel("# filtered trees")
    ax.set_title(f"Cost-aware OSS distribution @ {oss_lab}\nP(OSS=0) = {frac0:.3f}  (n={len(oss_tree):,} trees)")
    ax.set_axisbelow(True)
    _save(fig, "voc_oss_distribution")

    # =====================================================================
    # FIG 4: softmax_voc vs legal_moves (binned) + corr
    # =====================================================================
    voc_lab = best["softmax_voc"][0]
    col = f"softmax_voc__{voc_lab}"
    sub = df[[col, "legal_moves"]].dropna()
    r_voc_legal = _spearman_boot_ci(sub[col].to_numpy(), sub["legal_moves"].to_numpy(), seed=3)
    print(f"[bridge] softmax_voc@{voc_lab} vs legal_moves: rho={r_voc_legal[0]:+.4f} "
          f"CI[{r_voc_legal[1]:+.4f},{r_voc_legal[2]:+.4f}]", flush=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    lm = sub["legal_moves"].to_numpy(); vv = sub[col].to_numpy()
    xs, ys, es = [], [], []
    for lv in sorted(np.unique(lm)):
        mask = lm == lv
        if mask.sum() < 50:
            continue
        xs.append(lv); ys.append(vv[mask].mean())
        es.append(1.96 * vv[mask].std() / np.sqrt(mask.sum()))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=_MAIN, capsize=2, ms=4, lw=1.2)
    ax.set_xlabel("legal moves at root (human width driver)")
    ax.set_ylabel(f"softmax_voc @ {voc_lab}")
    ax.set_title(f"Does softmax-VOC track the human width driver?\nρ={r_voc_legal[0]:+.3f} "
                 f"CI[{r_voc_legal[1]:+.3f},{r_voc_legal[2]:+.3f}] (n={len(sub):,})")
    ax.set_axisbelow(True)
    _save(fig, "voc_softmax_vs_legalmoves")

    # ---- dump a small results json for the report prose ----
    import json
    out = {
        "n_moves": int(n), "n_fens": int(df["fen"].nunique()), "n_trees": int(len(sig)),
        "theta_star": theta_star,
        "best": {fam: {"cost": b[0], "rho": b[1], "ci": [b[2], b[3]], "n": b[4]} for fam, b in best.items()},
        "all": {fam: [{"cost": r[0], "rho": r[1], "ci": [r[2], r[3]], "n": r[4]} for r in rows]
                for fam, rows in results.items()},
        "gain_costfree_vs_rt": {"rho": r_gain[0], "ci": [r_gain[1], r_gain[2]]},
        "legal_moves_vs_rt": {"rho": r_legal[0], "ci": [r_legal[1], r_legal[2]]},
        "oss_best_cost": oss_lab, "p_oss0": frac0,
        "softmax_voc_vs_legal": {"cost": voc_lab, "rho": r_voc_legal[0], "ci": [r_voc_legal[1], r_voc_legal[2]]},
    }
    with open(os.path.join(FIGDIR, "voc_firstpass_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", os.path.join(FIGDIR, "voc_firstpass_results.json"), flush=True)


if __name__ == "__main__":
    main()
