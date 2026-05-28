#!/bin/bash
# Step 2 — Pack raw cts_raw_pretrain_example_v3 trees into tensorized shards (serial).
# Split, topology preset, and paths come from the YAML CONFIG.
#
#   export CONFIG="$PWD/configs/data/preprocess_gnn/hl4291_pack_50k_topology_full.yaml"
#   sbatch hl4291_slurm/2_pack_shards.slurm

#SBATCH --job-name=cts-pack-pretrain
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=0

set -euo pipefail
: "${CONFIG:?CONFIG env var required (YAML for cts.data.preprocess_gnn.pack)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/_env.sh" ]]; then
  source "${SCRIPT_DIR}/_env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/hl4291_slurm/_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/hl4291_slurm/_env.sh"
else
  echo "Error: Could not locate _env.sh" >&2
  exit 1
fi
hl4291_slurm_setup
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG file not found: ${CONFIG}"
  exit 1
fi

echo "config=${CONFIG}"
echo "PYTHON=$(command -v python3)"
python3 -V
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
date

python3 -u -m cts.data.preprocess_gnn.pack --config "${CONFIG}"

date
