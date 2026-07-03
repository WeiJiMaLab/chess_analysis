#!/bin/bash
# Board step (--plot): the four RT analyses over filtered_moves ⋈ board_features.
# Depends on board_merge (built board_features from the featurize array), which
# depends on filter_and_sample (filtered_moves). Reads only the DB; no trees/model.
#   sbatch --dependency=afterok:<board_merge> slurm/pipeline/board_analysis.slurm
# Figures land under human_analysis.figures_dir/board.
#SBATCH --job-name=board-analysis
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"

source "${TINY}/slurm/helpers/setup_env.sh"

echo "board analysis (--plot) on $(hostname) config=$CONFIG at $(date)"
python -m analysis.board --plot
echo "Finished at $(date)"
