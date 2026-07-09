"""One-off recovery script: step 2's materialize+eval job (10844785) started running BEFORE the
merge-not-overwrite fix landed in analysis.evaluate_our_trees_continued.main(), so it will
overwrite (not merge with) step 1's already-written results when it finishes. Backed up step 1's
results to our_trees_continued_e2e_results.step1_backup.json beforehand; this script merges that
backup with whatever step 2 (and later, if needed, any other pre-fix step) wrote, using the SAME
dedup-by-(checkpoint,epoch,time_lambda,maintenance_scale) logic as the real fix, so the final
JSON ends up identical to what the fixed code would have produced natively.
"""
import json
from pathlib import Path

OUT_DIR = Path("/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_e2e_continued")
LIVE = OUT_DIR / "our_trees_continued_e2e_results.json"
BACKUP = OUT_DIR / "our_trees_continued_e2e_results.step1_backup.json"


def _key(r):
    return (r["checkpoint"], r["epoch"], r["time_lambda"], r["maintenance_scale"])


def main():
    backup = json.loads(BACKUP.read_text())
    live = json.loads(LIVE.read_text())
    merged = {_key(r): r for r in backup}
    merged.update({_key(r): r for r in live})
    merged_results = sorted(merged.values(), key=lambda r: (r["epoch"], r["time_lambda"], r["maintenance_scale"]))
    LIVE.write_text(json.dumps(merged_results, indent=2))
    print(f"backup={len(backup)} live={len(live)} -> merged={len(merged_results)} entries -> {LIVE}")
    for r in merged_results:
        print(r["epoch"], r["time_lambda"], r["maintenance_scale"])


if __name__ == "__main__":
    main()
