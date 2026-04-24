#!/bin/bash
# End-to-end analysis script. Runs all standardized plots.
#
# Usage:
#   bash /home/hl4291/chess_analysis/src/slurm/script_analysis.sh

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "Running analysis pipeline at $(date)"

echo "1. Move Time Summary..."
python3 src/move_time_summary.py

echo "2. Clock Movetime (Player)..."
python3 src/clock_movetime.py

echo "3. Clock Movetime (Opponent)..."
python3 src/clock_movetime.py --opp

echo "4. Branching Factor (n_possible_moves)..."
python3 src/npossiblemoves_movetime.py

echo "5. Game Stage (Ply Movetime)..."
python3 src/ply_movetime.py

echo "6. Game Stage (Ply Pre-move Probability)..."
python3 src/ply_premove.py

echo "Analysis complete at $(date)"
