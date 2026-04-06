#!/bin/bash

#SBATCH --job-name=cts-controller
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
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_controller.pt}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
K="${K:-2}"
NODE_EMBED_HIDDEN="${NODE_EMBED_HIDDEN:-128}"
D_EMBED="${D_EMBED:-128}"
D_MESSAGE="${D_MESSAGE:-128}"
N_HEADS="${N_HEADS:-4}"
D_ATT="${D_ATT:-32}"
CONTROLLER_HIDDEN="${CONTROLLER_HIDDEN:-128}"
VALUE_HIDDEN="${VALUE_HIDDEN:-128}"
NUM_ENVS="${NUM_ENVS:-16}"
CONTINUE_COST="${CONTINUE_COST:-0.05}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
MAX_DEPTH="${MAX_DEPTH:-10}"
C_PUCT="${C_PUCT:-1.0}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-256}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
VALUE_LOSS_COEF="${VALUE_LOSS_COEF:-0.5}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-128}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_UPDATES="${NUM_UPDATES:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-50}"
MAX_CACHE_SIZE="${MAX_CACHE_SIZE:-32}"
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
echo "device=${DEVICE}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

EXTRA_ARGS=()
if [[ "${UNFREEZE_ENCODER}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze-encoder)
fi

python3 supervised_branch_cli.py frozen-rl-generated \
  --train-data "${TRAIN_DATA}" \
  --validation-data "${VALIDATION_DATA}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --output-checkpoint "${OUTPUT_CHECKPOINT}" \
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
  --num-envs "${NUM_ENVS}" \
  --continue-cost "${CONTINUE_COST}" \
  --search-budget "${SEARCH_BUDGET}" \
  --max-depth "${MAX_DEPTH}" \
  --c-puct "${C_PUCT}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --gae-lambda "${GAE_LAMBDA}" \
  --clip-epsilon "${CLIP_EPSILON}" \
  --entropy-coef "${ENTROPY_COEF}" \
  --value-loss-coef "${VALUE_LOSS_COEF}" \
  --ppo-epochs "${PPO_EPOCHS}" \
  --minibatch-size "${MINIBATCH_SIZE}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --num-updates "${NUM_UPDATES}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --log-interval "${LOG_INTERVAL}" \
  --validation-interval "${VALIDATION_INTERVAL}" \
  --max-cache-size "${MAX_CACHE_SIZE}" \
  "${EXTRA_ARGS[@]}"

date
