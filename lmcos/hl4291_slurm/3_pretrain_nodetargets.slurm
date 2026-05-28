#!/bin/bash
# Step 3 — Node targets encoder pretrain (TreeEncoder + NodeTargetsHead, Huber on packed node targets).
#
#   export CONFIG="$PWD/configs/data/hl4291_pretrain_nodetargets_shard_slurm_smoke.yaml"
#   sbatch hl4291_slurm/3_pretrain_nodetargets.slurm
#
# Full 50k corpus:
#   export CONFIG="$PWD/configs/data/hl4291_pretrain_nodetargets_50k.yaml"

#SBATCH --job-name=cts-nodetargets-pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to hl4291_pretrain_nodetargets_*.yaml)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/_env.sh" ]]; then
  source "${SCRIPT_DIR}/_env.sh"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/hl4291_slurm/_env.sh" ]]; then
  source "${SLURM_SUBMIT_DIR}/hl4291_slurm/_env.sh"
else
  echo "Error: Could not locate _env.sh" >&2
  exit 1
fi
hl4291_slurm_setup_gpu

if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG file not found: ${CONFIG}"
  exit 1
fi

echo "config=${CONFIG}"
echo "PYTHON=$(command -v python3)"
python3 -V
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

python3 -u -m cts.data.build_tree --config "${CONFIG}"

date
