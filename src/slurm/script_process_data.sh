#!/bin/bash
#SBATCH --job-name=chess-voc-process
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

# Create logs directory
mkdir -p src/slurm/logs

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv/bin/activate not found. Using system python."
fi

echo "Job started at $(date)"
echo "Running parallel VOC data processing (Full Dataset, 40 workers)..."

python3 src/script_process_data.py

echo "Job completed at $(date)"
