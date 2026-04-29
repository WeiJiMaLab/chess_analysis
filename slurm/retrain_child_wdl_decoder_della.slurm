#!/bin/bash

#SBATCH --job-name=cts-wdl-decoder
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1.pt}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/retrained_child_wdl_decoder.pt}"

EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MATERIALIZE_BATCH_SIZE="${MATERIALIZE_BATCH_SIZE:-4}"
DECODER_HIDDEN="${DECODER_HIDDEN:-128}"
K="${K:-1}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "$(dirname "${OUTPUT_CHECKPOINT}")"

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
echo "encoder_checkpoint=${ENCODER_CHECKPOINT}"
echo "train_data=${TRAIN_DATA}"
echo "val_data=${VAL_DATA}"
echo "output_checkpoint=${OUTPUT_CHECKPOINT}"
echo "epochs=${EPOCHS} lr=${LR} batch_size=${BATCH_SIZE}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

EXTRA_ARGS=()
if [[ -n "${VAL_DATA}" ]]; then
  EXTRA_ARGS+=(--val-data "${VAL_DATA}")
fi

python3 scripts/retrain_child_wdl_decoder.py \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --train-data "${TRAIN_DATA}" \
  --output-checkpoint "${OUTPUT_CHECKPOINT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH_SIZE}" \
  --materialize-batch-size "${MATERIALIZE_BATCH_SIZE}" \
  --decoder-hidden "${DECODER_HIDDEN}" \
  --k "${K}" \
  --device "${DEVICE}" \
  --num-workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"

date
