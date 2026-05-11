#!/bin/bash

#SBATCH --job-name=cts-materialize
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"
PACKED_DATA="${PACKED_DATA:-}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
WORKER_INDEX="${WORKER_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-3}"
DEVICE="${DEVICE:-cuda}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${PROJECT_DIR}/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "packed_data=${PACKED_DATA}"
echo "encoder_checkpoint=${ENCODER_CHECKPOINT}"
echo "output_dir=${OUTPUT_DIR}"
echo "worker_index=${WORKER_INDEX}"
echo "num_workers=${NUM_WORKERS}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 scripts/materialize_controller_cache.py materialize \
  --packed-data "${PACKED_DATA}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --worker-index "${WORKER_INDEX}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --episode-batch-size "${EPISODE_BATCH_SIZE}" \
  --log-interval "${LOG_INTERVAL}"

date
