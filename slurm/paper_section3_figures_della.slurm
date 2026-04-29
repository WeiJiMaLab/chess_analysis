#!/bin/bash

#SBATCH --job-name=cts-s3-figs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"
MODEL_CONFIGS="${MODEL_CONFIGS:-}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/tree_encoder_child_wdl_async_k1.pt}"
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/retrained_child_wdl_decoder.pt}"
PACKED_CONTROLLER_EPISODES="${PACKED_CONTROLLER_EPISODES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/paper_figures/section3}"

DEVICE="${DEVICE:-cuda}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-8}"
ABLATION_MAX_K="${ABLATION_MAX_K:-64}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
CACHE="${CACHE:-${OUTPUT_DIR}/materialized_cache.pt}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

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
echo "model_configs=${MODEL_CONFIGS}"
echo "encoder_checkpoint=${ENCODER_CHECKPOINT}"
echo "decoder_checkpoint=${DECODER_CHECKPOINT}"
echo "packed_controller_episodes=${PACKED_CONTROLLER_EPISODES}"
echo "output_dir=${OUTPUT_DIR}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 scripts/paper_section3_figures.py \
  --model-configs "${MODEL_CONFIGS}" \
  --encoder-checkpoint "${ENCODER_CHECKPOINT}" \
  --decoder-checkpoint "${DECODER_CHECKPOINT}" \
  --packed-controller-episodes "${PACKED_CONTROLLER_EPISODES}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --episode-batch-size "${EPISODE_BATCH_SIZE}" \
  --ablation-max-k "${ABLATION_MAX_K}" \
  --num-workers "${NUM_WORKERS}" \
  --cache "${CACHE}"

date
