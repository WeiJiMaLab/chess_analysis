#!/bin/bash
# Pipeline step — eval (CPU). Writes the meta-controller ASSESSMENT figures
# (frontier / R-decodability / delta-mean-regret lambda+maintenance sweeps) via
# `analysis.evaluate` — the mechanism behind outputs/reports/normative.md. Reads
# paths from the `eval:` section of config.yaml. Refits the tree-stats + z_t
# readouts (budget-out / steps-in) on the validation 70/30 split, so it needs the
# z_t cache in RAM.
#
# The delta-regret separation sweep (10 independent cost regimes -- 3 lambda points
# + 7 maintenance points, see analysis.evaluate._regime_grid) used to run
# sequentially in-process here: each regime does 3 separate 200-epoch PG trainings
# on tiny MLPs (Python-loop-bound, not BLAS-bound, so this job's 8 cpus mostly sat
# idle during it) and was the dominant share of the runtime the --time history below
# was chasing. It now fans out as its own `sbatch --array` (eval_regime_array.slurm,
# one task per regime) submitted from here and waited on (`sbatch --wait`, which
# blocks without consuming this job's own cpu/mem) before collecting the results --
# see labnotebook 2026-07-11. First use of a job submitting a sub-array job from
# inside a running allocation in this repo (everywhere else, submit_all.sh wires
# array+afterok chains externally) -- watch the first real run for anything Della-
# specific (compute-node sbatch access, --wait exit-code semantics on arrays).
#   sbatch --dependency=afterok:<train_readout_pg_jobid> slurm/pipeline/eval.slurm
#SBATCH --job-name=eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
# was 01:30:00 (qos=short): that budget existed almost entirely to cover the 10-regime
# separation sweep now split out into eval_regime_array.slurm (own --time, own qos). What's left
# here is one frontier regime fit + the decodability probe, plus blocking on the array via
# `sbatch --wait` + the cheap regime-collect render -- 45min is generous headroom for that; dropped
# to qos=test since --time is now <=1h (della-qos-time-routing).

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

echo "=== separation: submitting regime-fit array ==="
# Grid size read straight off analysis.evaluate's own constants so the array bound can never drift
# out of sync with what compute_regime_task/collect_regime_results actually index against.
NREGIMES=$(python -c "from analysis.evaluate import DEFAULT_LAMBDA_GRID, DEFAULT_MAINT_GRID; \
print(len(DEFAULT_LAMBDA_GRID) + len(DEFAULT_MAINT_GRID))")
echo "NREGIMES=$NREGIMES"
sbatch --wait --array="0-$((NREGIMES - 1))" --export=ALL,CONFIG="$CONFIG" \
  "$TINY/slurm/pipeline/eval_regime_array.slurm"

echo "=== separation: regime array complete, collecting ==="
python -m analysis.evaluate --which regime-collect \
  --out-dir "$OUT" --maint-lambda "$MAINT_LAMBDA"

echo "EVAL DONE -> $OUT"; date
