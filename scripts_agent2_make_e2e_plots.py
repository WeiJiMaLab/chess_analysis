"""One-off driver (plan.md Phase 2, Agent 2) -- builds sub-line A's two required plots from the
original Z2 epoch + the chained steps' training curves + the paired-sig eval JSON. Re-run this
after each new step's materialize+eval lands (it just re-reads whatever CSVs/JSON exist on disk,
so it naturally picks up more points as the chain progresses -- no code change needed between
runs). Not part of the reusable src/ package, same spirit as scripts_agent2_make_pg_plots.py.
"""
import csv
import json
from pathlib import Path

from analysis.our_trees_continued_plots import plot_paired_diff_vs_epoch, plot_regret_vs_epoch

E2E_DIR = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller")
CONT_DIR = Path("/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_e2e_continued")
RESULTS_JSON = "/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_e2e_continued/our_trees_continued_e2e_results.json"
OUT_DIR = "/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_e2e_continued"
ORIG_FINAL_EPOCH = 1  # the original Z2 run was exactly 1 epoch


def _load_csv_row(path):
    with open(path, newline="") as f:
        row = next(csv.DictReader(f))
        return {k: float(v) for k, v in row.items()}


def main():
    # epoch 1 = original Z2 run; epoch 2/3/4 = this agent's chained steps 1/2/3 (whichever exist).
    # NB: _load_csv_row's own dict already has an "epoch" key (the source CSV's own per-file
    # epoch=1, since each chained job only ever runs one epoch) -- it MUST be spread first and
    # the global epoch index applied second, or dict-literal key order silently lets the source
    # file's local epoch=1 win over the intended global label (caught live: an earlier version of
    # this script did it backwards and every point plotted as epoch 1).
    combined = [{**_load_csv_row(E2E_DIR / "e2e_controller.csv"), "epoch": 1}]
    for i, stem in enumerate(("e2e_step1", "e2e_step2", "e2e_step3"), start=2):
        csv_path = CONT_DIR / f"{stem}.csv"
        if csv_path.exists():
            combined.append({**_load_csv_row(csv_path), "epoch": i})
    rows = [{"epoch": r["epoch"], "train_E_regret": r["train_loss"], "val_greedy_regret": r["val_loss"]}
            for r in combined]
    print(f"[plots] {len(rows)} epoch(s) of training-curve data found: epochs={[r['epoch'] for r in rows]}")

    results = json.loads(Path(RESULTS_JSON).read_text())
    primary = [r for r in results if r["time_lambda"] == 0.01 and r["maintenance_scale"] == 0.0]
    print(f"[plots] {len(primary)} epoch(s) of primary-regime paired-eval data found: "
          f"epochs={[r['epoch'] for r in primary]}")
    anchor = primary[0]
    always_stop = anchor["baseline_regret"]["always_stop"]["mean"]
    always_continue = anchor["baseline_regret"]["always_continue"]["mean"]

    p1 = plot_regret_vs_epoch(
        rows, always_stop=always_stop, always_continue=always_continue,
        out_dir=OUT_DIR, base="e2e_continued_regret_vs_epoch",
        title="Sub-line A: e2e (encoder-unfrozen) controller, soft-loss vs epoch\n"
              "(epoch 1 = original Z2 run; 2+ = this agent's chained 1-epoch-per-job continuation)",
        boundary_epoch=ORIG_FINAL_EPOCH,
    )
    print("wrote", p1)

    p2 = plot_paired_diff_vs_epoch(
        primary, out_dir=OUT_DIR, base="e2e_continued_paired_diff_vs_epoch",
        title="Sub-line A: paired 95% CI, regret diff vs AlwaysStop / SingleHalt*\n"
              "(time_lambda=0.01, maintenance_scale=0.0 -- primary regime)",
        boundary_epoch=ORIG_FINAL_EPOCH,
    )
    print("wrote", p2)

    # The nonzero-maintenance regime is where the headline finding actually lives (a confirmed
    # win at epoch 2 that does NOT replicate at epoch 3) -- worth its own plot, not just the
    # primary regime's (where nothing ever crosses into significance either way).
    maint = [r for r in results if r["time_lambda"] == 0.01 and r["maintenance_scale"] == 0.001]
    if len(maint) >= 2:
        p3 = plot_paired_diff_vs_epoch(
            maint, out_dir=OUT_DIR, base="e2e_continued_paired_diff_vs_epoch_maintenance",
            title="Sub-line A: paired 95% CI, regret diff vs AlwaysStop / SingleHalt*\n"
                  "(time_lambda=0.01, maintenance_scale=0.001 -- confirmed win at epoch 2 only, "
                  "not replicated at epochs 3-4)",
            boundary_epoch=ORIG_FINAL_EPOCH,
        )
        print("wrote", p3)


if __name__ == "__main__":
    main()
