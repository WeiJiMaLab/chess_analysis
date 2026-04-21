#!/bin/bash
#SBATCH --job-name=voc-compute
#SBATCH --output=src/slurm/logs/%x_%j.out
#SBATCH --error=src/slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail

# Project directory
PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p src/slurm/logs

# Active virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv/bin/activate not found."
    exit 1
fi

echo "Job started on $(hostname) at $(date)"
echo "Processing 500 games dataset..."

# Run the computation script
# Using 40 workers to match cpus-per-task
python3 src/slurm/script_compute_voc.py --n_games 500 --n_jobs 40 --retries 2 --batch_size 100

echo "Job completed at $(date)"
