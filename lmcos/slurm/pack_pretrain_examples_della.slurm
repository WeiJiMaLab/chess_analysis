#!/bin/bash

#SBATCH --job-name=cts-pack
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed}"
SHARD_SIZE="${SHARD_SIZE:-2000}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
CLEAR_FLAG="${CLEAR_FLAG:-1}"
SINGLE_SHARD="${SINGLE_SHARD:-0}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_ROOT}"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "split_root=${SPLIT_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "shard_size=${SHARD_SIZE}"
echo "num_workers=${NUM_WORKERS}"
echo "log_interval=${LOG_INTERVAL}"
echo "hostname=$(hostname)"
date

ARGS=(
  --split-root "${SPLIT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --shard-size "${SHARD_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
)

if [[ "${CLEAR_FLAG}" == "1" ]]; then
  ARGS+=(--clear)
fi

if [[ "${SINGLE_SHARD}" == "1" ]]; then
  ARGS+=(--single-shard)
fi

python3 scripts/pack_pretrain_examples.py "${ARGS[@]}"

date
