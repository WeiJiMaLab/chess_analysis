#!/bin/bash
# U1.1 CPU lane: array tree-gen on pure-CPU nodes (Lc0 blas backend).
#
# Each array task owns a contiguous FEN index range computed from its task id:
#   start = BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE
#   end   = min(start + SHARD_SIZE, LANE_END)
# Tasks whose start has run off the end of the lane no-op cleanly.
#
# The ysagiv lc0 binary is CUDA-linked and needs libcublas/libcudart EVEN for
# the blas backend; those nodes have no CUDA install, so we satisfy the dynamic
# link from the venv's nvidia-* wheels (loaded, never used — backend=blas).
#SBATCH --job-name=gen50k-cpu
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00

set -euo pipefail
: "${CONFIG:?CONFIG (path to base YAML) required}"
: "${SHARD_SIZE:?SHARD_SIZE required}"
: "${BASE_START:?BASE_START required}"
: "${LANE_END:?LANE_END required}"

PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"

start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE ))
if (( end > LANE_END )); then end=${LANE_END}; fi
if (( start >= LANE_END )); then
  echo "no-op: start=${start} >= LANE_END=${LANE_END}"
  exit 0
fi

mkdir -p "${PROJECT_DIR}/slurm/logs"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
export PS1="${PS1:-}"
source "${VENV_DIR}/bin/activate"
# Satisfy lc0's CUDA link from the venv nvidia wheels (no GPU on these nodes).
NVLIB="${VENV_DIR}/lib/python3.11/site-packages/nvidia/cublas/lib:${VENV_DIR}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVLIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} host=$(hostname)"
nvidia-smi -L 2>/dev/null || echo "no_gpu"
date
python3 -m cts.data.build_tree --config "${CONFIG}" \
    --override start_index=${start} --override end_index=${end}
date
