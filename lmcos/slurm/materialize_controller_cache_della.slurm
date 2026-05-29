#!/bin/bash

#SBATCH --job-name=cts-materialize
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to YAML config)}"

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"

mkdir -p "${PROJECT_DIR}/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "config=${CONFIG}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

# When WORKER_INDEX is set in the env (e.g. submitting 3 parallel workers
# per split), thread it through as a config override so the YAML's
# default worker_index can stay 0 and submission just varies the env var.
# Unset / empty WORKER_INDEX leaves the YAML value (e.g. merge stage
# doesn't need this).
EXTRA_ARGS=()
if [[ -n "${WORKER_INDEX:-}" ]]; then
  EXTRA_ARGS+=("--override" "worker_index=${WORKER_INDEX}")
fi

python3 -m cts.data.preprocess_mc.materialize --config "${CONFIG}" "${EXTRA_ARGS[@]}"

date
