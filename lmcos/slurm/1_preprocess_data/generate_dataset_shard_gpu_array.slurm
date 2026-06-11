#!/bin/bash
# U1.1 GPU lane: array tree-gen on A100 nodes (Lc0 cuda backend).
#
# Each array task owns a contiguous FEN index range computed from its task id:
#   start = BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE
#   end   = min(start + SHARD_SIZE, LANE_END)
# Tasks whose start has run off the end of the lane no-op cleanly.
#SBATCH --job-name=gen50k-gpu
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

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
module load cudatoolkit/12.8
export PS1="${PS1:-}"
source "${VENV_DIR}/bin/activate"
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} host=$(hostname)"
nvidia-smi -L || true
date
python3 -m cts.data.build_tree --config "${CONFIG}" \
    --override start_index=${start} --override end_index=${end}
date
