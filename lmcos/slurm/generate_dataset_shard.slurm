#!/bin/bash

#SBATCH --job-name=cts-gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

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

python3 -m cts.data.build_tree --config "${CONFIG}"

date
