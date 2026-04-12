#!/bin/bash

#SBATCH --job-name=cts-pack-ctrl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_split}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed}"
SHARD_SIZE="${SHARD_SIZE:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CLEAR_FLAG="${CLEAR_FLAG:-1}"

CONTINUE_COST="${CONTINUE_COST:-0.001}"
REWARD_SCALE="${REWARD_SCALE:-1.0}"
MIN_DECISION_MARGIN="${MIN_DECISION_MARGIN:-0.0}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
MAX_DEPTH="${MAX_DEPTH:-10}"
C_PUCT="${C_PUCT:-1.0}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

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
echo "log_interval=${LOG_INTERVAL}"
echo "continue_cost=${CONTINUE_COST}"
echo "reward_scale=${REWARD_SCALE}"
echo "min_decision_margin=${MIN_DECISION_MARGIN}"
echo "hostname=$(hostname)"
date

ARGS=(
  --split-root "${SPLIT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --shard-size "${SHARD_SIZE}"
  --log-interval "${LOG_INTERVAL}"
  --continue-cost "${CONTINUE_COST}"
  --reward-scale "${REWARD_SCALE}"
  --min-decision-margin "${MIN_DECISION_MARGIN}"
  --search-budget "${SEARCH_BUDGET}"
  --max-depth "${MAX_DEPTH}"
  --c-puct "${C_PUCT}"
)

if [[ "${CLEAR_FLAG}" == "1" ]]; then
  ARGS+=(--clear)
fi

python3 scripts/pack_controller_episodes.py "${ARGS[@]}"

date
