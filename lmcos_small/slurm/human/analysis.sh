#!/bin/bash
# End-to-end FULL-dataset analysis. Runs the kept standardized plots only.
#
# The tree-derived "generated values" (OSS / VOC / Action Gap / MQ on the lc0-tree
# subset) are NOT run here — see slurm/tree_values.slurm. MQ in particular is now
# the Lc0 tree definition (final_Q loss of the played move), not a Stockfish plot.
#
# Usage:
#   bash lmcos_small/slurm/human/analysis.sh

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="${PYTHONPATH:-}:lmcos_small/src/human"

echo "Running full-dataset board analysis pipeline..."
python3 lmcos_small/src/human/board.py --all

echo "Analysis complete at $(date)"
