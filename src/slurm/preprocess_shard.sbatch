#!/bin/bash
#SBATCH --job-name=ld-moves
#SBATCH --output=/home/hl4291/chess_analysis/src/slurm/logs/ld-moves_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/src/slurm/logs/ld-moves_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --array=0-1

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

if [[ -z "${LOAD_MOVES_TMPDIR:-}" ]]; then
  echo "ERROR: set LOAD_MOVES_TMPDIR (_preprocess.sh does this)." >&2
  exit 1
fi

export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-8GB}"

python3 src/slurm/scripts/preprocess_data.py process_shard --total-shards 2 --tmpdir "${LOAD_MOVES_TMPDIR}"

echo "Finished at $(date)"
