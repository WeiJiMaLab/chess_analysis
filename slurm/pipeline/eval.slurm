#!/bin/bash
# Pipeline step — eval (CPU). Writes the meta-controller ASSESSMENT figures
# (frontier / R-decodability / delta-mean-regret lambda+maintenance sweeps) via
# `analysis.evaluate` — the mechanism behind outputs/reports/normative.md. Reads
# paths from the `eval:` section of config.yaml. Refits the tree-stats + z_t
# readouts (budget-out / steps-in) on the validation 70/30 split, so it needs the
# z_t cache in RAM.
#
# The delta-regret separation sweep (independent cost regimes, see
# analysis.evaluate._regime_grid) used to run sequentially in-process here: each
# regime does 3 separate 200-epoch PG trainings on tiny MLPs (Python-loop-bound, not
# BLAS-bound, so this job's cpus mostly sat idle during it) and was the dominant share
# of the runtime. It now fans out as its own `sbatch --array` (eval_regime_array.slurm,
# one task per regime), chained via a plain --dependency=afterok to a small collector
# job (eval_collect.slurm, runs `--which regime-collect`) once the array finishes --
# the same array-then-afterok-chained-externally convention submit_all.sh already uses
# everywhere else in this pipeline.
#
# NOT `sbatch --wait` (tried first, see labnotebook 2026-07-11): blocking here held one
# of qos=test's MaxJobsPU=2 slots for the array's ENTIRE runtime, which both serialized
# the array down to ~1 concurrent task (the blocked parent occupied the other slot) AND
# put the array's now ~2x-inflated wall-clock on THIS job's own --time budget -- job
# 10987146 (8 regimes) hit its 45min limit and was killed by SLURM mid-wait, before ever
# reaching regime-collect, even though the array itself kept running fine independently
# (child array jobs aren't tied to the parent's allocation/QOS slot). Submit-and-exit +
# afterok avoids both problems: this job finishes in ~2min once the array is submitted,
# freeing its QOS slot so the array gets full concurrency, and the collector's own
# --time budget is sized to ITS job, not entangled with the frontier/decodability fit.
#   sbatch --dependency=afterok:<pack_root_merge_jobid> slurm/pipeline/eval.slurm
#SBATCH --job-name=eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
# Now covers only the frontier fit + decodability probe + submitting the array/collector
# (no longer blocks on either) -- the historical 10-regime in-process run needed 01:30:00
# total, almost all of which was the sequential regime sweep now split out entirely. Was
# 00:30:00 (2026-07-11 first cut, "generous headroom" for these two fits) but job 10989543
# hit that limit mid-decodability: frontier+decodability took ~24min on della-h14n7 but
# >30min on della-h12n16 -- node-speed variance on top of an already-thin margin.
# 55min keeps real headroom while staying under qos=test's <=1h cutoff (della-qos-time-routing).

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
TIME_MODE=$(field time_mode)
TIME_LAMBDA=$(field time_lambda)
MAINTENANCE_SCALE=$(field maintenance_scale)
MAINTENANCE_EXPONENT=$(field maintenance_exponent)
MAINT_LAMBDA=$(field maint_lambda)
MAX_EPISODES=$(field max_episodes)
mkdir -p "$OUT"

echo "eval host=$(hostname)"; date

echo "=== frontier ==="
python -m analysis.evaluate --which frontier \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" \
  --time-mode "$TIME_MODE" --time-lambda "$TIME_LAMBDA" \
  --maintenance-scale "$MAINTENANCE_SCALE" --maintenance-exponent "$MAINTENANCE_EXPONENT" \
  --max-episodes "$MAX_EPISODES"

echo "=== decodability ==="
python -m analysis.evaluate --which decodability \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" \
  --max-episodes "$MAX_EPISODES"

echo "=== separation: submitting regime-fit array + collector ==="
# Grid size read straight off analysis.evaluate's own constants so the array bound can never drift
# out of sync with what compute_regime_task/collect_regime_results actually index against.
NREGIMES=$(python -c "from analysis.evaluate import DEFAULT_LAMBDA_GRID, DEFAULT_MAINT_GRID; \
print(len(DEFAULT_LAMBDA_GRID) + len(DEFAULT_MAINT_GRID))")
echo "NREGIMES=$NREGIMES"
ARRAY_JOBID=$(sbatch --parsable --array="0-$((NREGIMES - 1))" --export=ALL,CONFIG="$CONFIG" \
  "$TINY/slurm/pipeline/eval_regime_array.slurm")
echo "regime array job: $ARRAY_JOBID"
COLLECT_JOBID=$(sbatch --parsable --dependency=afterok:$ARRAY_JOBID --export=ALL,CONFIG="$CONFIG" \
  "$TINY/slurm/pipeline/eval_collect.slurm")
echo "regime collect job (afterok:$ARRAY_JOBID): $COLLECT_JOBID"

echo "EVAL SUBMIT DONE (frontier+decodability done here; separation continues as $ARRAY_JOBID -> $COLLECT_JOBID) -> $OUT"; date
