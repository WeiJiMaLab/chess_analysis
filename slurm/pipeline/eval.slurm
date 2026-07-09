#!/bin/bash
# Pipeline step — eval (CPU). Writes the meta-controller ASSESSMENT figures
# (frontier / R-decodability / delta-mean-regret lambda+maintenance sweeps) via
# `analysis.evaluate` — the mechanism behind outputs/reports/normative.md. Reads
# paths from the `eval:` section of config.yaml. Refits the tree-stats + z_t
# readouts (budget-out / steps-in) on the validation 70/30 split, so it needs the
# z_t cache in RAM.
#   sbatch --dependency=afterok:<train_readout_pg_jobid> slurm/pipeline/eval.slurm
#SBATCH --job-name=eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00  # was 00:40:00 -> 00:55:00 -> 01:30:00; a correctly-calibrated (non-degenerate)
                          # lambda does real readout-fitting work per regime instead of instantly
                          # collapsing, and the maint_grid sweep grew 5->7 points, so this run legitimately
                          # takes longer each time we've made the results less degenerate (see labnotebook
                          # 2026-07-07) -- generous margin here since job 10785971 was still running past
                          # 54min with no sign of being stuck, just slow

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
python -m analysis.evaluate --which all \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" \
  --time-mode "$TIME_MODE" --time-lambda "$TIME_LAMBDA" \
  --maintenance-scale "$MAINTENANCE_SCALE" --maintenance-exponent "$MAINTENANCE_EXPONENT" \
  --maint-lambda "$MAINT_LAMBDA" --max-episodes "$MAX_EPISODES"
echo "EVAL DONE -> $OUT"; date
