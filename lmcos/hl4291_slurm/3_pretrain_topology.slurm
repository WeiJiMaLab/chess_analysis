#!/bin/bash
# Step 3 — Topology encoder pretrain (TreeEncoder + TopologyHead, Huber on packed topology targets).
#
#   export CONFIG="$PWD/configs/data/hl4291_pretrain_topology_shard_slurm_smoke.yaml"
#   sbatch hl4291_slurm/3_pretrain_topology.slurm
#
# Full 50k corpus:
#   export CONFIG="$PWD/configs/data/hl4291_pretrain_topology_50k.yaml"

#SBATCH --job-name=cts-topo-pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail
: "${CONFIG:?CONFIG env var required (path to hl4291_pretrain_topology_*.yaml)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
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
