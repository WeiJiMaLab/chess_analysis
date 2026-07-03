#!/bin/bash
# P0 board-correlate battery — featurize DISTINCT windowed FENs (engine-free,
# python-chess only; plan_final_push.md P0). ~84M FENs at ~4.7k/s/core ->
# ~10 min on 32 workers. Independent of tree generation.
#   sbatch slurm/pipeline/featurize_board.slurm
#SBATCH --job-name=featurize-board
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:59:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # CPU: no cudatoolkit needed

echo "featurize_board host=$(hostname) workers=${SLURM_CPUS_PER_TASK}"; date
python -m analysis.featurize_board --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"
echo "FEATURIZE BOARD DONE"; date
