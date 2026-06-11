#!/bin/bash

#SBATCH --job-name=cts-gen
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to YAML config)}"

PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"
CONDA_ENV="${CONDA_ENV:-}"

mkdir -p "${PROJECT_DIR}/slurm/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
export PS1="${PS1:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  echo "python_env=conda:${CONDA_ENV}"
elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
  echo "python_env=venv:${VENV_DIR}"
else
  echo "Set CONDA_ENV or VENV_DIR to a valid Python environment." >&2
  exit 1
fi
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "config=${CONFIG}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 -m cts.data.build_tree --config "${CONFIG}"

date
