#!/bin/bash
# Board precondition — merge the featurize shards → the board_features table
# (single writer; opens personal.db read_write). Depends on the board_featurize
# array (afterok); board_analysis (--plot) depends on this.
#   sbatch --dependency=afterok:<board_featurize_array> slurm/pipeline/board_merge.slurm
#SBATCH --job-name=board-merge
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"
source "$TINY/slurm/helpers/setup_env.sh"
export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK:-8}" DUCKDB_MEMORY_LIMIT=32GB
echo "board merge on $(hostname) config=$CONFIG at $(date)"
python -m analysis.board --merge
echo "Finished at $(date)"
