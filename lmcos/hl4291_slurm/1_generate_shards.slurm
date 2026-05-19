#!/bin/bash
# Step 1 — GPU array: lc0 teacher trees (slice CONFIG by SLURM_ARRAY_TASK_ID × CHUNK_SIZE).
#
#   export CONFIG="$PWD/configs/data/hl4291_build_tree_50k.yaml"
#   sbatch --array=0-9 --export=ALL,CHUNK_SIZE=5000 hl4291_slurm/1_generate_shards.slurm

#SBATCH --job-name=cts-gen-50k
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to hl4291 build_tree YAML)}"

CHUNK_SIZE="${CHUNK_SIZE:-5000}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
start_index=$((TASK_ID * CHUNK_SIZE))
end_index=$((start_index + CHUNK_SIZE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
hl4291_slurm_setup_gpu

if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG file not found: ${CONFIG}"
  exit 1
fi

echo "config=${CONFIG}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-} SLURM_ARRAY_TASK_ID=${TASK_ID}"
echo "chunk_size=${CHUNK_SIZE} start_index=${start_index} end_index=${end_index}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 -u -m cts.data.build_tree \
  --config "${CONFIG}" \
  --override "start_index=${start_index}" \
  --override "end_index=${end_index}"

date
