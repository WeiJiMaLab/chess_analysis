"""One-off driver (plan.md Phase 2, Agent 2) -- builds sub-line B's two required plots from the
original 20-epoch production training curve + the continued run's curve + the paired-sig eval
JSON. Not part of the reusable src/ package (this is a per-agent report-assembly script, same
spirit as other one-off `scripts_*` drivers elsewhere in this investigation's history) -- run
directly with `python scripts_agent2_make_pg_plots.py`.
"""
import csv
import json
from pathlib import Path

from analysis.our_trees_continued_plots import plot_paired_diff_vs_epoch, plot_regret_vs_epoch

ORIG_CSV = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller_training_curve.csv"
CONT_CSV = "/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_pg_continued/mchalt_controller_continued_training_curve.csv"
RESULTS_JSON = "/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_pg_continued/our_trees_continued_pg_results.json"
OUT_DIR = "/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_pg_continued"
ORIG_FINAL_EPOCH = 20


def _load_csv(path):
    with open(path, newline="") as f:
        return [{k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    orig = _load_csv(ORIG_CSV)
    cont = _load_csv(CONT_CSV)
    combined = orig + [{**r, "epoch": r["epoch"] + ORIG_FINAL_EPOCH} for r in cont]

    # Always-Stop/Always-Continue reference lines -- same values SIG-S/SIG-Z's regime already
    # established at the primary regime (time_lambda=0.01, m=0.0): computed once here directly
    # from the paired-sig eval JSON's own baseline_regret block (epoch=20 row, m=0.0), so the
    # reference lines are guaranteed consistent with the SAME held-out split/regime the paired
    # panel uses, not a separately-recomputed number that could silently drift.
    results = json.loads(Path(RESULTS_JSON).read_text())
    primary = [r for r in results if r["time_lambda"] == 0.01 and r["maintenance_scale"] == 0.0]
    anchor = next(r for r in primary if r["epoch"] == 20)
    always_stop = anchor["baseline_regret"]["always_stop"]["mean"]
    always_continue = anchor["baseline_regret"]["always_continue"]["mean"]

    p1 = plot_regret_vs_epoch(
        combined, always_stop=always_stop, always_continue=always_continue,
        out_dir=OUT_DIR, base="pg_continued_regret_vs_epoch",
        title="Sub-line B: production PG controller (frozen encoder), regret vs epoch\n"
              "(epochs 1-20 original run + 21-170 this agent's continuation)",
        boundary_epoch=ORIG_FINAL_EPOCH,
    )
    print("wrote", p1)

    p2 = plot_paired_diff_vs_epoch(
        primary, out_dir=OUT_DIR, base="pg_continued_paired_diff_vs_epoch",
        title="Sub-line B: paired 95% CI, regret diff vs AlwaysStop / SingleHalt*\n"
              "(time_lambda=0.01, maintenance_scale=0.0 -- primary regime)",
        boundary_epoch=ORIG_FINAL_EPOCH,
    )
    print("wrote", p2)


if __name__ == "__main__":
    main()
