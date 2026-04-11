#!/bin/bash

#SBATCH --job-name=cts-split
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

if [[ -z "${SOURCE_ROOT:-}" ]]; then
  echo "SOURCE_ROOT is required" >&2
  exit 1
fi
if [[ -z "${SPLIT_ROOT:-}" ]]; then
  echo "SPLIT_ROOT is required" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS/uncertainty}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.05}"
SEED="${SEED:-0}"
CLEAR_FLAG="${CLEAR_FLAG:-1}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${SPLIT_ROOT}"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

cd "${PROJECT_DIR}"

echo "project_dir=${PROJECT_DIR}"
echo "source_root=${SOURCE_ROOT}"
echo "split_root=${SPLIT_ROOT}"
echo "validation_fraction=${VALIDATION_FRACTION}"
echo "seed=${SEED}"
echo "hostname=$(hostname)"
date

ARGS=(
  --source-root "${SOURCE_ROOT}"
  --split-root "${SPLIT_ROOT}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --seed "${SEED}"
)

if [[ "${CLEAR_FLAG}" == "1" ]]; then
  ARGS+=(--clear)
fi

python3 scripts/prepare_pretrain_split.py "${ARGS[@]}"

date
