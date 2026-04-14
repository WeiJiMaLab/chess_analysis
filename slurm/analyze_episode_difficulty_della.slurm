#!/bin/bash

#SBATCH --job-name=cts-difficulty
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
PACKED_ROOT="${PACKED_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/controller_packed}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PACKED_ROOT}/train_manifest.json}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-${PACKED_ROOT}/validation_manifest.json}"
CONTINUE_COST="${CONTINUE_COST:-0.001}"
PLANNING_COST_KIND="${PLANNING_COST_KIND:-linear}"
PLANNING_COST_EXPONENT="${PLANNING_COST_EXPONENT:-1.0}"
OUTPUT="${OUTPUT:-${PACKED_ROOT}/episode_difficulty.jsonl}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${PROJECT_DIR}/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "train_manifest=${TRAIN_MANIFEST}"
echo "validation_manifest=${VALIDATION_MANIFEST}"
echo "continue_cost=${CONTINUE_COST}"
echo "planning_cost_kind=${PLANNING_COST_KIND}"
echo "planning_cost_exponent=${PLANNING_COST_EXPONENT}"
echo "output=${OUTPUT}"
date

python3 scripts/analyze_episode_difficulty.py \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --continue-cost "${CONTINUE_COST}" \
  --planning-cost-kind "${PLANNING_COST_KIND}" \
  --planning-cost-exponent "${PLANNING_COST_EXPONENT}" \
  --output "${OUTPUT}"

date
