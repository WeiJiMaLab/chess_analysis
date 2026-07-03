#!/bin/bash
# Massively-parallel captures/checks featurize of the distinct FENs (python-chess
# legal-move enumeration). Each array task featurizes its round-robin slice of the
# distinct-FEN parquet's row groups, using its node's cores, into one shard. No DB
# access here (the merge step writes the DB).
#   sbatch --array=0-31 --dependency=afterok:<process_moves> slurm/pipeline/featurize_positions.slurm
#SBATCH --job-name=featurize-positions
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
echo "featurize_positions task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"; date
python -m analysis.preprocess featurize_positions
echo "FEATURIZE_POSITIONS TASK ${SLURM_ARRAY_TASK_ID} DONE"; date
