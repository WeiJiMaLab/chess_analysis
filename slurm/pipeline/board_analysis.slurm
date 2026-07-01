#!/bin/bash
# Analysis pipeline step 1a — board-level (no-model) RT analysis on the full
# dataset. Runs board.py over personal.db (the processed_moves tables built by
# step 0), emitting the ply / legal-moves / clock RT dashboards + the board
# feature correlation matrix. Reads only the DB; no trees, no model.
#   sbatch slurm/pipeline/board_analysis.slurm
# Figures are written under human_analysis.figures_dir/board (see config.yaml).
#SBATCH --job-name=board-analysis
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"

source "${TINY}/slurm/helpers/setup_env.sh"

echo "board analysis on $(hostname) at $(date)"
python3 src/analysis/board.py --all
echo "Finished at $(date)"
