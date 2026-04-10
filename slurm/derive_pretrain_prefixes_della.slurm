#!/bin/bash
# Usage:
# sbatch \
#   --export=ALL,PROJECT_DIR=/home/ysagiv/chess/CTS/uncertainty,INPUT_DATA=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/voc_teacher_trees_96_10x,OUTPUT_DIR=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/pretrain_prefixes_uncertainty/train \
#   slurm/derive_pretrain_prefixes_della.slurm

#SBATCH --job-name=cts-prefix
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -z "${INPUT_DATA:-}" ]]; then
  echo "INPUT_DATA is required" >&2
  exit 1
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is required" >&2
  exit 1
fi
PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
START_INDEX="${START_INDEX:-0}"
MIN_NODES="${MIN_NODES:-8}"
MAX_NODES="${MAX_NODES:-48}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
C_PUCT="${C_PUCT:-1.0}"
MAX_DEPTH="${MAX_DEPTH:-10}"
SEED="${SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
RESUME_FLAG="${RESUME_FLAG:-1}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "input_data=${INPUT_DATA}"
echo "output_dir=${OUTPUT_DIR}"
echo "start_index=${START_INDEX}"
echo "end_index=${END_INDEX:-all}"
echo "min_expanded_nodes=${MIN_NODES}"
echo "max_expanded_nodes=${MAX_NODES}"
echo "search_budget=${SEARCH_BUDGET}"
echo "c_puct=${C_PUCT}"
echo "max_depth=${MAX_DEPTH}"
echo "seed=${SEED}"
echo "num_workers=${NUM_WORKERS}"
echo "hostname=$(hostname)"
date

ARGS=(
  --input-data "${INPUT_DATA}"
  --output-dir "${OUTPUT_DIR}"
  --min-nodes "${MIN_NODES}"
  --max-nodes "${MAX_NODES}"
  --search-budget "${SEARCH_BUDGET}"
  --c-puct "${C_PUCT}"
  --max-depth "${MAX_DEPTH}"
  --seed "${SEED}"
  --start-index "${START_INDEX}"
  --log-interval "${LOG_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -n "${END_INDEX:-}" ]]; then
  ARGS+=(--end-index "${END_INDEX}")
fi

if [[ "${RESUME_FLAG}" == "1" ]]; then
  ARGS+=(--resume)
fi

"${PYTHON_BIN}" -u scripts/derive_pretrain_prefixes.py "${ARGS[@]}"

date
