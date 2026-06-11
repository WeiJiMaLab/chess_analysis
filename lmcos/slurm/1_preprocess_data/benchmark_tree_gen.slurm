#!/bin/bash
#SBATCH --job-name=bench-gen
#SBATCH --output=slurm/logs/%x_%j_%a.out
#SBATCH --error=slurm/logs/%x_%j_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to YAML config)}"

PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"

mkdir -p "${PROJECT_DIR}/slurm/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
# ysagiv lc0 binary is CUDA-linked; libcublas is required even for Backend=blas.
if ! module load cudatoolkit/12.8 2>/dev/null; then
  export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PS1="${PS1:-}"
source "${VENV_DIR}/bin/activate"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "config=${CONFIG}"
echo "hostname=$(hostname)"
echo "cpus=${SLURM_CPUS_PER_TASK}"
nvidia-smi -L 2>/dev/null || echo "no_gpu"
date

/usr/bin/time -p python3 -u -m cts.data.build_tree --config "${CONFIG}"

date
