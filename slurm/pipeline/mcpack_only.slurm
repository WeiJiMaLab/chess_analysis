#!/bin/bash
# Budget-sweep helper: run ONLY mc_pack (repack episodes at a new fixed_budget) into
# the isolated mc_packed_dir named by $CONFIG. Does NOT redo split/gnn_pack (those are
# budget-independent and already built in the shared run).
#SBATCH --job-name=mcpack-only
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
NW=${SLURM_CPUS_PER_TASK:-16}
echo "mcpack-only CONFIG=$CONFIG host=$(hostname) cpus=$NW"; date
echo "mc_packed_dir=$MCP"
python -m cts.data.preprocess_mc.pack --config "$CONFIG" --stage mc_pack --set num_workers="$NW"
echo "MCPACK DONE -> $MCP"; date
