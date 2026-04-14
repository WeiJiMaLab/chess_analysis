#!/bin/bash

#SBATCH --job-name=cts-adv-probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split}"
TRAIN_DATA="${TRAIN_DATA:-${SPLIT_ROOT}/train_manifest.txt}"
VALIDATION_DATA="${VALIDATION_DATA:-${SPLIT_ROOT}/validation_manifest.txt}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain_interactive.pt}"
CACHE_DIR="${CACHE_DIR:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/cache/probe_controller_representation}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
CONTINUE_COST="${CONTINUE_COST:-0.001}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
MAX_DEPTH="${MAX_DEPTH:-10}"
C_PUCT="${C_PUCT:-1.0}"
TRAIN_SAMPLE_SIZE="${TRAIN_SAMPLE_SIZE:-0}"
VALIDATION_SAMPLE_SIZE="${VALIDATION_SAMPLE_SIZE:-0}"

K="${K:-2}"
NODE_EMBED_HIDDEN="${NODE_EMBED_HIDDEN:-128}"
D_EMBED="${D_EMBED:-128}"
D_MESSAGE="${D_MESSAGE:-128}"
N_HEADS="${N_HEADS:-4}"
D_ATT="${D_ATT:-32}"
CONTROLLER_HIDDEN="${CONTROLLER_HIDDEN:-128}"
VALUE_HIDDEN="${VALUE_HIDDEN:-128}"

PROBE_TARGET="${PROBE_TARGET:-advantage}"
PROBE_TYPE="${PROBE_TYPE:-mlp}"
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-128}"
PROBE_LEARNING_RATE="${PROBE_LEARNING_RATE:-1e-3}"
PROBE_EPOCHS="${PROBE_EPOCHS:-50}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-256}"
CACHE_SAVE_INTERVAL_EXAMPLES="${CACHE_SAVE_INTERVAL_EXAMPLES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
LOAD_LOG_INTERVAL="${LOAD_LOG_INTERVAL:-250}"
EMBEDDING_LOG_INTERVAL="${EMBEDDING_LOG_INTERVAL:-5000}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${CACHE_DIR}"

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

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "train_data=${TRAIN_DATA}"
echo "validation_data=${VALIDATION_DATA}"
echo "encoder_checkpoint=${ENCODER_CHECKPOINT}"
echo "cache_dir=${CACHE_DIR}"
echo "probe_target=${PROBE_TARGET}"
echo "probe_type=${PROBE_TYPE}"
echo "probe_epochs=${PROBE_EPOCHS}"
echo "probe_batch_size=${PROBE_BATCH_SIZE}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 scripts/probe_controller_representation.py \
  --train-data "${TRAIN_DATA}" \
  --validation-data "${VALIDATION_DATA}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --k "${K}" \
  --node-embed-hidden "${NODE_EMBED_HIDDEN}" \
  --d-embed "${D_EMBED}" \
  --d-message "${D_MESSAGE}" \
  --n-heads "${N_HEADS}" \
  --d-att "${D_ATT}" \
  --controller-hidden "${CONTROLLER_HIDDEN}" \
  --value-hidden "${VALUE_HIDDEN}" \
  --continue-cost "${CONTINUE_COST}" \
  --search-budget "${SEARCH_BUDGET}" \
  --max-depth "${MAX_DEPTH}" \
  --c-puct "${C_PUCT}" \
  --train-sample-size "${TRAIN_SAMPLE_SIZE}" \
  --validation-sample-size "${VALIDATION_SAMPLE_SIZE}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
  --cache-dir "${CACHE_DIR}" \
  --cache-save-interval-examples "${CACHE_SAVE_INTERVAL_EXAMPLES}" \
  --load-log-interval "${LOAD_LOG_INTERVAL}" \
  --embedding-log-interval "${EMBEDDING_LOG_INTERVAL}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --probe-target "${PROBE_TARGET}" \
  --probe-type "${PROBE_TYPE}" \
  --probe-hidden-dim "${PROBE_HIDDEN_DIM}" \
  --probe-learning-rate "${PROBE_LEARNING_RATE}" \
  --probe-epochs "${PROBE_EPOCHS}" \
  --log-interval "${LOG_INTERVAL}"

date
