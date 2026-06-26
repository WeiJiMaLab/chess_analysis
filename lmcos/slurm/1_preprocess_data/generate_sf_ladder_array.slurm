#!/bin/bash
# P2 Stockfish strength-ladder tree generation (CPU array).
#
# Stockfish is a native CPU engine (no CUDA link, unlike the ysagiv lc0 binary),
# so this lane is simpler than the lc0 CPU array: just activate the venv and run.
#
# Each array task owns a contiguous FEN index range:
#   start = BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE
#   end   = min(start + SHARD_SIZE, LANE_END)
# Output is keyed by global FEN index so disjoint tasks never collide.
#
# Submit one array per rung, e.g.:
#   CONFIG=slurm/configs/1_preprocess_data/sf_ladder_1800.yaml \
#   SHARD_SIZE=2000 BASE_START=0 LANE_END=20000 \
#   sbatch --array=0-9 slurm/1_preprocess_data/generate_sf_ladder_array.slurm
#SBATCH --job-name=sf-ladder
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00

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
set -u

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} host=$(hostname)"
date
python3 -m cts.data.build_tree --config "${CONFIG}" \
    --override start_index=${start} --override end_index=${end}
date
