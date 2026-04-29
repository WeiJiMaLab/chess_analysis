#!/bin/bash

#SBATCH --job-name=cts-s1-figs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/paper_figures/section1}"

# Diagnostics paths (label:path format, space-separated)
DIAGNOSTICS="${DIAGNOSTICS:-}"

# Oracle config source (optional)
ORACLE_LOG="${ORACLE_LOG:-}"
ORACLE_CONFIG_JSON="${ORACLE_CONFIG_JSON:-}"

# Source path rewriting for cluster path remapping
REWRITE_FROM="${REWRITE_FROM:-}"
REWRITE_TO="${REWRITE_TO:-}"

SKIP_VALUE_GAP="${SKIP_VALUE_GAP:-0}"

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
echo "output_dir=${OUTPUT_DIR}"
echo "diagnostics=${DIAGNOSTICS}"
echo "hostname=$(hostname)"
date

EXTRA_ARGS=()
if [[ -n "${ORACLE_LOG}" ]]; then
  EXTRA_ARGS+=(--oracle-log "${ORACLE_LOG}")
fi
if [[ -n "${ORACLE_CONFIG_JSON}" ]]; then
  EXTRA_ARGS+=(--oracle-config-json "${ORACLE_CONFIG_JSON}")
fi
if [[ -n "${REWRITE_FROM}" && -n "${REWRITE_TO}" ]]; then
  EXTRA_ARGS+=(--rewrite-from "${REWRITE_FROM}" --rewrite-to "${REWRITE_TO}")
fi
if [[ "${SKIP_VALUE_GAP}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-value-gap)
fi

# shellcheck disable=SC2086
python3 scripts/paper_section1_figures.py \
  --diagnostics ${DIAGNOSTICS} \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"

date
