#!/bin/bash
# Phase 1b — pack trees (CPU). Three sub-stages, each reads its section of
# configs/core.yaml directly via --stage (no intermediary YAML):
#   split    -> cts.data.preprocess_gnn.split   (train/val partition)
#   gnn_pack -> cts.data.preprocess_gnn.pack    (encoder training examples)
#   mc_pack  -> cts.data.preprocess_mc.pack     (budgeted-DP oracle episodes)
#
#   ELO=1800 sbatch --dependency=afterok:<gentrees_jobid> \
#     pipeline/1_treegen_pack_trees/pack_trees.slurm
#SBATCH --job-name=tiny-packtrees
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/0_helpers/setup_env.sh"
NW=${SLURM_CPUS_PER_TASK:-32}

echo "elo=$ELO host=$(hostname) cpus=$NW"; date

echo "=== split ==="
python -m cts.data.preprocess_gnn.split --config "$CONFIG" --stage split --set globals.sf_elo="$ELO"

echo "=== gnn_pack (encoder examples) ==="
python -m cts.data.preprocess_gnn.pack --config "$CONFIG" --stage gnn_pack --set globals.sf_elo="$ELO"

echo "=== mc_pack (oracle episodes) ==="
python -m cts.data.preprocess_mc.pack --config "$CONFIG" --stage mc_pack --set globals.sf_elo="$ELO" --set num_workers="$NW"

echo "PACK TREES DONE elo=$ELO"; date
