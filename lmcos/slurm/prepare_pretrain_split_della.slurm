#!/bin/bash

#SBATCH --job-name=cts-split
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to YAML config)}"

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/cts/async_soph}"

mkdir -p "${PROJECT_DIR}/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
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

python3 -m cts.data.preprocess_gnn.split --config "${CONFIG}"

date
