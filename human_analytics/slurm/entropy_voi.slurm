#!/bin/bash
#SBATCH --job-name=a4-smoke
#SBATCH --output=human_analytics/slurm/logs/%x_%j.out
#SBATCH --error=human_analytics/slurm/logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"

mkdir -p "${PROJECT_DIR}/human_analytics/slurm/logs"
cd "${PROJECT_DIR}"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
export PS1="${PS1:-}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
  echo "python_env=venv:${VENV_DIR}"
elif [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
  echo "python_env=venv:.venv"
else
  echo "VENV_DIR not found. Please setup python virtual environment." >&2
  exit 1
fi
set -u

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "hostname=$(hostname)"
date

python3 human_analytics/entropy_voi_analysis.py generate \
    --n-moves 1000 \
    --n-workers 16 \
    --output /scratch/gpfs/GRIFFITHS/hl4291/tmp/sf_multidepth_traces_1k.pkl

date
