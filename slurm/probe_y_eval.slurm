#!/bin/bash
# Probe Y (plan.md "External expert input" section) -- eval half. Fits SingleHalt* /
# Stats-Controller / subtree-weighted z_t-Controller on the human_trees argmax-filtered
# validation split (Agent 1's mc_packed, reused read-only) and paired-bootstraps both
# significance tests. Chain after probe_y_materialize.slurm:
#   sbatch --dependency=afterok:<materialize_jobid> slurm/probe_y_eval.slurm
#SBATCH --job-name=probe-y-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="${CONFIG:-$TINY/config_probe_y_subtree.yaml}"
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
MAX_EPISODES=$(field max_episodes)
DEMBED=$(field d_embed)
mkdir -p "$OUT"

echo "probe_y_eval host=$(hostname) PACKED=$PACKED MATVAL=$MATVAL OUT=$OUT d_embed=$DEMBED"; date
test -f "$MATVAL" || { echo "ERROR: materialized cache missing: $MATVAL (did probe_y_materialize.slurm finish?)" >&2; exit 1; }

python -m analysis.evaluate_probe_y \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" --d-embed "$DEMBED" \
  --time-mode "$TIME_MODE" --time-lambda "$TIME_LAMBDA" \
  --maintenance-scale "$MAINTENANCE_SCALE" --maintenance-exponent "$MAINTENANCE_EXPONENT" \
  --max-episodes "$MAX_EPISODES"

echo "PROBE Y EVAL DONE -> $OUT/probe_y_subtree_weighted_eval.{pdf,png}"; date
