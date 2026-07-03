#!/bin/bash
# Pipeline step — filter_and_sample. Builds the run's canonical windowed table
# (filtered_moves_<run> = processed_moves_nonzero in [min_ply,max_ply] + game_fraction)
# AND samples N distinct FENs from it as the tree-generation roots. All params from
# the active config; opens personal.db read_write. Depends on preprocess.
#   sbatch slurm/pipeline/filter_and_sample.slurm
#SBATCH --job-name=filter-and-sample
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"
source "$TINY/slurm/helpers/setup_env.sh"
export DUCKDB_THREADS="${SLURM_CPUS_PER_TASK:-16}" DUCKDB_MEMORY_LIMIT=40GB
echo "filter_and_sample on $(hostname) config=$CONFIG at $(date)"
python -m analysis.filter_and_sample
echo "Finished at $(date)"
