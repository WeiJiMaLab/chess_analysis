#!/bin/bash

#SBATCH --job-name=cts-pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS}"
DEFAULT_PACKED_ROOT="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_packed"
DEFAULT_SPLIT_ROOT="/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split"
if [[ -z "${TRAIN_DIR:-}" ]]; then
  if [[ -f "${DEFAULT_PACKED_ROOT}/train_manifest.json" ]]; then
    TRAIN_DIR="${DEFAULT_PACKED_ROOT}/train_manifest.json"
  else
    TRAIN_DIR="${DEFAULT_SPLIT_ROOT}/train_manifest.txt"
  fi
fi
if [[ -z "${VALIDATION_DIR:-}" ]]; then
  if [[ -f "${DEFAULT_PACKED_ROOT}/validation_manifest.json" ]]; then
    VALIDATION_DIR="${DEFAULT_PACKED_ROOT}/validation_manifest.json"
  else
    VALIDATION_DIR="${DEFAULT_SPLIT_ROOT}/validation_manifest.txt"
  fi
fi
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain.pt}"

DEVICE="${DEVICE:-cuda}"
K="${K:-2}"
NODE_EMBED_HIDDEN="${NODE_EMBED_HIDDEN:-128}"
D_EMBED="${D_EMBED:-128}"
D_MESSAGE="${D_MESSAGE:-128}"
N_HEADS="${N_HEADS:-4}"
D_ATT="${D_ATT:-32}"
VALUE_HIDDEN="${VALUE_HIDDEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ROOT_LOSS_WEIGHT="${ROOT_LOSS_WEIGHT:-1.0}"
EPOCHS="${EPOCHS:-10}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "$(dirname "${OUTPUT_CHECKPOINT}")"

set +u
module purge
module load anaconda3/2023.3
module load cudatoolkit/12.8
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

# OpenMP/BLAS: one thread per process avoids oversubscription (main + NUM_WORKERS dataloader
# processes each on the same SLURM allocation) and fork+OpenMP deadlocks. This is unrelated to
# --cpus-per-task; those cores are for parallel worker processes, not multiplying threads inside each.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "train_dir=${TRAIN_DIR}"
echo "validation_dir=${VALIDATION_DIR}"
echo "output_checkpoint=${OUTPUT_CHECKPOINT}"
echo "device=${DEVICE}"
echo "num_workers=${NUM_WORKERS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 supervised_branch_cli.py pretrain-encoder \
  --train-dir "${TRAIN_DIR}" \
  --validation-dir "${VALIDATION_DIR}" \
  --output-checkpoint "${OUTPUT_CHECKPOINT}" \
  --device "${DEVICE}" \
  --k "${K}" \
  --node-embed-hidden "${NODE_EMBED_HIDDEN}" \
  --d-embed "${D_EMBED}" \
  --d-message "${D_MESSAGE}" \
  --n-heads "${N_HEADS}" \
  --d-att "${D_ATT}" \
  --value-hidden "${VALUE_HIDDEN}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --root-loss-weight "${ROOT_LOSS_WEIGHT}" \
  --epochs "${EPOCHS}"

date
