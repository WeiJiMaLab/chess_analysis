#!/bin/bash
#SBATCH --job-name=ld-moves-merge
#SBATCH --output=/home/hl4291/chess_analysis/src/slurm/logs/ld-moves-merge_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/src/slurm/logs/ld-moves-merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Single writer: rebuild selected_moves from staging/*.parquet (run after array completes).
# Raise --time if merge reads many large parquet shards.

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

mkdir -p src/slurm/logs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "Warning: .venv not found; using PATH python."
fi

echo "Merge job ${SLURM_JOB_ID:-?} on $(hostname) at $(date)"

python3 src/utils/load_data.py merge

echo "Finished at $(date)"
