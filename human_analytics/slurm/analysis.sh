#!/bin/bash
# End-to-end analysis script. Runs all standardized plots.
#
# Engine-backed workflows (e.g. selected_moves_with_engine) are not run here.
#
# Usage:
#   bash human_analytics/slurm/analysis.sh

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "Running analysis pipeline at $(date)"

echo "1. Move Time Summary..."
python3 human_analytics/move_time_summary.py

echo "2. Move-time dashboards (clock, branching, material, ply)..."
python3 human_analytics/movetime_analysis.py --only clock clock_opp npossiblemoves pieces_exc self_pieces_exc ply

echo "3. Game Stage (Ply Pre-move Probability)..."
python3 human_analytics/ply_premove.py

echo "Analysis complete at $(date)"
