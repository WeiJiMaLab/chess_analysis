#!/bin/bash
# Rebuild processed_moves / processed_moves_nonzero from `moves` with the SQL board
# features (material / in_check / prev_move_was_capture) and dump the DISTINCT FENs
# to a parquet for the featurize array. Paths from the config `preprocess:` section
# (personal_db unified with analysis). Opens personal.db read_write — no other
# process may hold it. Chain: process_moves -> featurize_positions[] -> merge.
#   sbatch slurm/pipeline/process_moves.slurm
#SBATCH --job-name=process-moves
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
echo "process_moves host=$(hostname) config=$CONFIG"; date
python -m analysis.preprocess process_moves
echo "PROCESS_MOVES DONE"; date
