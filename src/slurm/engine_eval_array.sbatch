#!/bin/bash
#SBATCH --job-name=chess-eval-full
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-116%20

# Optimized Evaluation Script
# Engine: Default is Stockfish. Use --export=ALL,ENGINE=lc0 for lc0.
# Sharding: Based on FEN hash to ensure mutual exclusivity and zero repeat work.

set -euo pipefail

ENGINE="${ENGINE:-stockfish}"
PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

source .venv/bin/activate

# Scaling analysis showed 16 workers is the sweet spot for throughput on Della.
N_WORKERS=16

echo "Starting evaluation with engine: ${ENGINE}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

srun python src/script_engine_eval.py \
    --engine "${ENGINE}" \
    --shard_id "${SLURM_ARRAY_TASK_ID}" \
    --total_shards 117 \
    --limit 1000000 \
    --n_workers "${N_WORKERS}" \
    --depth 5
