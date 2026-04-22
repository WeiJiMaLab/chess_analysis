#!/usr/bin/env bash
# Fresh tmpdir → single-shard export → merge into personal.db (same tmpdir on all jobs).
#
# Usage:
#   bash /home/hl4291/chess_analysis/src/slurm/script_load_moves.sh

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

mkdir -p src/slurm/logs

TMPDIR="/scratch/gpfs/GRIFFITHS/hl4291/tmp/load_moves"
rm -rf "${TMPDIR}"
mkdir -p "${TMPDIR}"
export LOAD_MOVES_TMPDIR="${TMPDIR}"
echo "Fresh LOAD_MOVES_TMPDIR=${LOAD_MOVES_TMPDIR}"

array_id="$(sbatch --export=ALL "${PROJECT_DIR}/src/slurm/load_moves_shards.sbatch" | awk '{print $NF}')"
echo "Submitted load_moves array: ${array_id}"

merge_id="$(sbatch --dependency=afterok:"${array_id}" --export=ALL "${PROJECT_DIR}/src/slurm/load_moves_merge.sbatch" | awk '{print $NF}')"
echo "Submitted merge (runs after array ${array_id} succeeds): ${merge_id}"
