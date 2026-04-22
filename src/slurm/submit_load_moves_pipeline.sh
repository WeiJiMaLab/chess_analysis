#!/usr/bin/env bash
# Submit 100-way shard export, then merge into personal.db when all array tasks succeed.
#
# Usage (from anywhere):
#   bash /home/hl4291/chess_analysis/src/slurm/submit_load_moves_pipeline.sh
#
# Requires: staging dir empty or merge-safe; merge uses CREATE OR REPLACE TABLE selected_moves.

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

mkdir -p src/slurm/logs

array_id="$(sbatch "${PROJECT_DIR}/src/slurm/load_moves_shards.sbatch" | awk '{print $NF}')"
echo "Submitted load_moves array: ${array_id}"

merge_id="$(sbatch --dependency=afterok:"${array_id}" "${PROJECT_DIR}/src/slurm/load_moves_merge.sbatch" | awk '{print $NF}')"
echo "Submitted merge (runs after array ${array_id} succeeds): ${merge_id}"
