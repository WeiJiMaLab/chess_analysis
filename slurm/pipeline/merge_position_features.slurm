#!/bin/bash
# Join the captures/checks shards from the featurize array into processed_moves
# [_nonzero]. Single DB writer (opens personal.db read_write).
#   sbatch --dependency=afterok:<featurize_positions_array> slurm/pipeline/merge_position_features.slurm
#SBATCH --job-name=merge-posfeat
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK:-16}" DUCKDB_MEMORY_LIMIT=100GB
echo "merge_position_features host=$(hostname) config=$CONFIG"; date
python -m analysis.preprocess merge_position_features --config "$CONFIG"
echo "MERGE_POSITION_FEATURES DONE"; date
