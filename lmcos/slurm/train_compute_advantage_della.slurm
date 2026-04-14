#!/bin/bash

#SBATCH --job-name=cts-compute-adv
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split}"
TRAIN_DATA="${TRAIN_DATA:-${SPLIT_ROOT}/train_manifest.txt}"
VALIDATION_DATA="${VALIDATION_DATA:-${SPLIT_ROOT}/validation_manifest.txt}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain_interactive.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_controller_compute_adv_full.pt}"

REPRESENTATION="${REPRESENTATION:-tree}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
CONTINUE_COST="${CONTINUE_COST:-0.001}"
REWARD_SCALE="${REWARD_SCALE:-1.0}"
MIN_DECISION_MARGIN="${MIN_DECISION_MARGIN:-0.0}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
MAX_DEPTH="${MAX_DEPTH:-10}"
C_PUCT="${C_PUCT:-1.0}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
MAX_VALIDATION_EXAMPLES="${MAX_VALIDATION_EXAMPLES:-0}"

K="${K:-2}"
NODE_EMBED_HIDDEN="${NODE_EMBED_HIDDEN:-128}"
D_EMBED="${D_EMBED:-128}"
D_MESSAGE="${D_MESSAGE:-128}"
N_HEADS="${N_HEADS:-4}"
D_ATT="${D_ATT:-32}"
Q_HIDDEN="${Q_HIDDEN:-128}"

BATCH_SIZE="${BATCH_SIZE:-2048}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-250}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5}"
GREEDY_EVAL_INTERVAL="${GREEDY_EVAL_INTERVAL:-5}"
UNFREEZE_ENCODER="${UNFREEZE_ENCODER:-0}"

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

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "train_data=${TRAIN_DATA}"
echo "validation_data=${VALIDATION_DATA}"
echo "encoder_checkpoint=${ENCODER_CHECKPOINT}"
echo "output_checkpoint=${OUTPUT_CHECKPOINT}"
echo "representation=${REPRESENTATION}"
echo "device=${DEVICE}"
echo "epochs=${EPOCHS}"
echo "batch_size=${BATCH_SIZE}"
echo "episode_batch_size=${EPISODE_BATCH_SIZE}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

EXTRA_ARGS=()
if [[ "${UNFREEZE_ENCODER}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze-encoder)
fi

if [[ "${REPRESENTATION}" == "tree" ]]; then
  EXTRA_ARGS+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
fi

python3 scripts/train_fitted_q_controller.py \
  --train-data "${TRAIN_DATA}" \
  --validation-data "${VALIDATION_DATA}" \
  --output-checkpoint "${OUTPUT_CHECKPOINT}" \
  --representation "${REPRESENTATION}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --continue-cost "${CONTINUE_COST}" \
  --reward-scale "${REWARD_SCALE}" \
  --min-decision-margin "${MIN_DECISION_MARGIN}" \
  --search-budget "${SEARCH_BUDGET}" \
  --max-depth "${MAX_DEPTH}" \
  --c-puct "${C_PUCT}" \
  --max-train-examples "${MAX_TRAIN_EXAMPLES}" \
  --max-validation-examples "${MAX_VALIDATION_EXAMPLES}" \
  --k "${K}" \
  --node-embed-hidden "${NODE_EMBED_HIDDEN}" \
  --d-embed "${D_EMBED}" \
  --d-message "${D_MESSAGE}" \
  --n-heads "${N_HEADS}" \
  --d-att "${D_ATT}" \
  --q-hidden "${Q_HIDDEN}" \
  --batch-size "${BATCH_SIZE}" \
  --episode-batch-size "${EPISODE_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --num-workers "${NUM_WORKERS}" \
  --log-interval "${LOG_INTERVAL}" \
  --validation-interval "${VALIDATION_INTERVAL}" \
  --greedy-eval-interval "${GREEDY_EVAL_INTERVAL}" \
  "${EXTRA_ARGS[@]}"

date
