#!/bin/bash
#SBATCH --job-name=chess-eval-shard
#SBATCH --output=/home/hl4291/chess_analysis/human_analytics/slurm/logs/eval_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/human_analytics/slurm/logs/eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-116%50

# One Slurm **shard** (array task): part of a full eval; see engine_eval.sh for merge + join.
# Engine: Default is Stockfish. Use --export=ALL,ENGINE=lc0 for lc0.
# Output: one parquet per task under $OUTPUT_DIR.

set -euo pipefail

ENGINE="${ENGINE:-stockfish}"
LIMIT="${LIMIT:-1000000}"
PROJECT_DIR="/home/hl4291/chess_analysis"
OUTPUT_DIR="/scratch/gpfs/GRIFFITHS/hl4291/tmp/eval_results/${ENGINE}"
cd "${PROJECT_DIR}"

mkdir -p human_analytics/slurm/logs
mkdir -p "${OUTPUT_DIR}"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "Warning: .venv not found; using PATH python."
fi

# Scaling analysis showed 16 workers is the sweet spot for throughput on Della.
N_WORKERS=16

echo "Starting evaluation with engine: ${ENGINE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-?} on $(hostname) at $(date)"

srun python3 human_analytics/slurm/scripts/script_engine_eval.py eval \
    --engine "${ENGINE}" \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    --total_shards 117 \
    --limit "${LIMIT}" \
    --n_workers "${N_WORKERS}" \
    --output_dir "${OUTPUT_DIR}" \
    --depth 5
