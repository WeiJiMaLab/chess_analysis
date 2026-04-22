#!/bin/bash
#SBATCH --job-name=ld-moves
#SBATCH --output=/home/hl4291/chess_analysis/src/slurm/logs/ld-moves_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/src/slurm/logs/ld-moves_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --array=0-99

# One array task per stride index; must match TOTAL_SHARDS / --total-shards in src/utils/load_data.py.
# Raise --time if any task is killed mid-shard (see .out for last parquet).

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

mkdir -p src/slurm/logs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "Warning: .venv not found; using PATH python."
fi

echo "Job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname) at $(date)"

python3 src/utils/load_data.py process --total-shards 100

echo "Finished at $(date)"
