#!/bin/bash

#SBATCH --job-name=cts-fittedq
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
PACKED_ROOT="${PACKED_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed}"
PACKED_TRAIN_DATA="${PACKED_TRAIN_DATA:-${PACKED_ROOT}/train_manifest.json}"
PACKED_VALIDATION_DATA="${PACKED_VALIDATION_DATA:-${PACKED_ROOT}/validation_manifest.json}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_pretrain.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/fittedq_controller.pt}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
K="${K:-1}"
NODE_EMBED_HIDDEN="${NODE_EMBED_HIDDEN:-128}"
D_EMBED="${D_EMBED:-128}"
D_MESSAGE="${D_MESSAGE:-128}"
N_HEADS="${N_HEADS:-4}"
D_ATT="${D_ATT:-32}"
Q_HIDDEN="${Q_HIDDEN:-256}"
Q_HIDDEN_LAYERS="${Q_HIDDEN_LAYERS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MIN_LR="${MIN_LR:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SIGN_LOSS_WEIGHT="${SIGN_LOSS_WEIGHT:-1.0}"
NONTRIVIAL_LOSS_WEIGHT="${NONTRIVIAL_LOSS_WEIGHT:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-1}"
GREEDY_EVAL_INTERVAL="${GREEDY_EVAL_INTERVAL:-10}"
OUTPUT_DIAGNOSTICS="${OUTPUT_DIAGNOSTICS:-${OUTPUT_CHECKPOINT%.pt}_diagnostics.jsonl}"
UNFREEZE_ENCODER="${UNFREEZE_ENCODER:-0}"
SEPARATE_SIGN_HEAD="${SEPARATE_SIGN_HEAD:-0}"
ACTIVATION="${ACTIVATION:-relu}"
INVERSE_FREQ_WEIGHTS="${INVERSE_FREQ_WEIGHTS:-0}"
MAINTENANCE_SCALE="${MAINTENANCE_SCALE:-0.0}"
MAINTENANCE_REF_NODES="${MAINTENANCE_REF_NODES:-30.0}"
MAINTENANCE_EXPONENT="${MAINTENANCE_EXPONENT:-1.1}"
TIME_LAMBDA="${TIME_LAMBDA:-18.537}"
TIME_P="${TIME_P:-2.8}"
TIME_TAU="${TIME_TAU:-2.5}"
TIME_DELTA="${TIME_DELTA:-1}"
TIMEOUT_VALUE="${TIMEOUT_VALUE:--1.0}"
SAMPLES_PER_BUCKET="${SAMPLES_PER_BUCKET:-2}"

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
echo "packed_train_data=${PACKED_TRAIN_DATA}"
echo "packed_validation_data=${PACKED_VALIDATION_DATA}"
echo "encoder_checkpoint=${ENCODER_CHECKPOINT}"
echo "output_checkpoint=${OUTPUT_CHECKPOINT}"
echo "output_diagnostics=${OUTPUT_DIAGNOSTICS}"
echo "device=${DEVICE}"
echo "k=${K}"
echo "epochs=${EPOCHS}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

MATERIALIZED_TRAIN_CACHE="${MATERIALIZED_TRAIN_CACHE:-}"
MATERIALIZED_VALIDATION_CACHE="${MATERIALIZED_VALIDATION_CACHE:-}"

EXTRA_ARGS=()
if [[ "${UNFREEZE_ENCODER}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze-encoder)
fi
if [[ "${SEPARATE_SIGN_HEAD}" == "1" ]]; then
  EXTRA_ARGS+=(--separate-sign-head)
fi
if [[ "${INVERSE_FREQ_WEIGHTS}" == "1" ]]; then
  EXTRA_ARGS+=(--inverse-freq-weights)
fi
if [[ -n "${MATERIALIZED_TRAIN_CACHE}" ]]; then
  EXTRA_ARGS+=(--materialized-train-cache "${MATERIALIZED_TRAIN_CACHE}")
fi
if [[ -n "${MATERIALIZED_VALIDATION_CACHE}" ]]; then
  EXTRA_ARGS+=(--materialized-validation-cache "${MATERIALIZED_VALIDATION_CACHE}")
fi

python3 scripts/train_fitted_q_controller.py \
  --packed-train-data "${PACKED_TRAIN_DATA}" \
  --packed-validation-data "${PACKED_VALIDATION_DATA}" \
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
  --q-hidden "${Q_HIDDEN}" \
  --q-hidden-layers "${Q_HIDDEN_LAYERS}" \
  --batch-size "${BATCH_SIZE}" \
  --episode-batch-size "${EPISODE_BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --sign-loss-weight "${SIGN_LOSS_WEIGHT}" \
  --nontrivial-loss-weight "${NONTRIVIAL_LOSS_WEIGHT}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --num-workers "${NUM_WORKERS}" \
  --log-interval "${LOG_INTERVAL}" \
  --validation-interval "${VALIDATION_INTERVAL}" \
  --greedy-eval-interval "${GREEDY_EVAL_INTERVAL}" \
  --output-diagnostics "${OUTPUT_DIAGNOSTICS}" \
  --maintenance-scale "${MAINTENANCE_SCALE}" \
  --maintenance-ref-nodes "${MAINTENANCE_REF_NODES}" \
  --maintenance-exponent "${MAINTENANCE_EXPONENT}" \
  --time-lambda "${TIME_LAMBDA}" \
  --time-p "${TIME_P}" \
  --time-tau "${TIME_TAU}" \
  --time-delta "${TIME_DELTA}" \
  --timeout-value "${TIMEOUT_VALUE}" \
  --samples-per-bucket "${SAMPLES_PER_BUCKET}" \
  --activation "${ACTIVATION}" \
  "${EXTRA_ARGS[@]}"

date
