#!/bin/bash
# End-to-end FULL-dataset analysis. Runs the kept standardized plots only.
#
# The tree-derived "generated values" (OSS / VOC / Action Gap on the lc0-tree
# subset) are NOT run here — see slurm/tree_values.slurm.
#
# Usage:
#   bash human_analytics/slurm/analysis.sh

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="${PYTHONPATH:-}:human_analytics"

echo "Running full-dataset analysis pipeline at $(date)"

echo "1. Move time: log(MT) histogram + normal QQ..."
python3 human_analytics/move_time_summary.py

echo "2. Move-time dashboards (clock, branching, own non-pawn material, ply)..."
python3 human_analytics/movetime_analysis.py --only clock npossiblemoves self_pieces_exc ply

echo "3. Game stage (ply vs instant-move probability)..."
python3 human_analytics/ply_premove.py

echo "4. Move quality (MQ) vs move time..."
python3 human_analytics/mq_analysis.py

echo "Analysis complete at $(date)"
