#!/bin/bash
# Pipeline step — pack_trees (CPU). Turns the generated trees into encoder +
# oracle training data. Depends on gen_trees. Sequential sub-stages:
#   split    -> cts.data.preprocess_gnn.split   (train/val partition; all trees)
#   gnn_pack -> cts.data.preprocess_gnn.pack    (encoder training examples)
#   mc_pack  -> cts.data.preprocess_mc.pack     (budgeted-DP oracle episodes)
# split.include_list is null, so every generated tree is packed (no pre-filter).
#   sbatch --dependency=afterok:<gen_trees_jobid> slurm/pipeline/pack_trees.slurm
#SBATCH --job-name=pack-trees
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
NW=${SLURM_CPUS_PER_TASK:-16}

echo "pack_trees host=$(hostname) cpus=$NW"; date

echo "=== split (train/val partition; all trees) ==="
python -m cts.data.preprocess_gnn.split --config "$CONFIG" --stage split

echo "=== gnn_pack (encoder examples) ==="
python -m cts.data.preprocess_gnn.pack --config "$CONFIG" --stage gnn_pack

echo "=== mc_pack (oracle episodes) ==="
python -m cts.data.preprocess_mc.pack --config "$CONFIG" --stage mc_pack --set num_workers="$NW"

echo "PACK DONE"; date
