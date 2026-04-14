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
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CLEAR_FLAG="${CLEAR_FLAG:-1}"

REWARD_SCALE="${REWARD_SCALE:-1.0}"
MIN_HALT_REWARD_RANGE="${MIN_HALT_REWARD_RANGE:-0.10}"
MIN_DECISION_MARGIN="${MIN_DECISION_MARGIN:-0.0}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
MAX_DEPTH="${MAX_DEPTH:-10}"
C_PUCT="${C_PUCT:-1.0}"
MAINTENANCE_SCALE="${MAINTENANCE_SCALE:-0.01}"
MAINTENANCE_REF_NODES="${MAINTENANCE_REF_NODES:-30.0}"
MAINTENANCE_EXPONENT="${MAINTENANCE_EXPONENT:-1.1}"
TIME_LAMBDA="${TIME_LAMBDA:-18.537}"
TIME_P="${TIME_P:-2.8}"
TIME_TAU="${TIME_TAU:-2.5}"
TIME_DELTA="${TIME_DELTA:-1}"
TIMEOUT_VALUE="${TIMEOUT_VALUE:--1.0}"
SAMPLES_PER_BUCKET="${SAMPLES_PER_BUCKET:-2}"
SEED="${SEED:-0}"

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
echo "num_workers=${NUM_WORKERS}"
echo "log_interval=${LOG_INTERVAL}"
echo "reward_scale=${REWARD_SCALE}"
echo "min_halt_reward_range=${MIN_HALT_REWARD_RANGE}"
echo "min_decision_margin=${MIN_DECISION_MARGIN}"
echo "maintenance_scale=${MAINTENANCE_SCALE}"
echo "maintenance_ref_nodes=${MAINTENANCE_REF_NODES}"
echo "maintenance_exponent=${MAINTENANCE_EXPONENT}"
echo "time_lambda=${TIME_LAMBDA}"
echo "time_p=${TIME_P}"
echo "time_tau=${TIME_TAU}"
echo "time_delta=${TIME_DELTA}"
echo "timeout_value=${TIMEOUT_VALUE}"
echo "samples_per_bucket=${SAMPLES_PER_BUCKET}"
echo "hostname=$(hostname)"
date

ARGS=(
  --split-root "${SPLIT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --shard-size "${SHARD_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
  --reward-scale "${REWARD_SCALE}"
  --min-halt-reward-range "${MIN_HALT_REWARD_RANGE}"
  --min-decision-margin "${MIN_DECISION_MARGIN}"
  --search-budget "${SEARCH_BUDGET}"
  --max-depth "${MAX_DEPTH}"
  --c-puct "${C_PUCT}"
  --maintenance-scale "${MAINTENANCE_SCALE}"
  --maintenance-ref-nodes "${MAINTENANCE_REF_NODES}"
  --maintenance-exponent "${MAINTENANCE_EXPONENT}"
  --time-lambda "${TIME_LAMBDA}"
  --time-p "${TIME_P}"
  --time-tau "${TIME_TAU}"
  --time-delta "${TIME_DELTA}"
  --timeout-value "${TIMEOUT_VALUE}"
  --samples-per-bucket "${SAMPLES_PER_BUCKET}"
  --seed "${SEED}"
)

if [[ "${CLEAR_FLAG}" == "1" ]]; then
  ARGS+=(--clear)
fi

python3 scripts/pack_controller_episodes.py "${ARGS[@]}"

date
