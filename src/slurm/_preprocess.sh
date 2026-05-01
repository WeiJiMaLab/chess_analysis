#!/usr/bin/env bash
# Fresh tmpdir → single-shard export → merge into personal.db (same tmpdir on all jobs).
#
# Usage:
#   bash /home/hl4291/chess_analysis/src/slurm/_preprocess.sh

set -euo pipefail

PROJECT_DIR="/home/hl4291/chess_analysis"
cd "${PROJECT_DIR}"

mkdir -p src/slurm/logs

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  echo "Warning: .venv not found; using PATH python."
fi

TMPDIR="/scratch/gpfs/GRIFFITHS/hl4291/tmp/load_moves"
rm -rf "${TMPDIR}"
mkdir -p "${TMPDIR}"
export LOAD_MOVES_TMPDIR="${TMPDIR}"
echo "Fresh LOAD_MOVES_TMPDIR=${LOAD_MOVES_TMPDIR}"

echo "Selecting games on $(hostname) at $(date)"
python3 src/slurm/scripts/preprocess_data.py select_games --tmpdir "${LOAD_MOVES_TMPDIR}" --threads 40 --memory 64GB

echo "Submitting load_moves array job..."
sbatch --wait --export=ALL "${PROJECT_DIR}/src/slurm/preprocess_shard.sbatch"

echo "Merge job on $(hostname) at $(date)"
python3 src/slurm/scripts/preprocess_data.py merge --tmpdir "${LOAD_MOVES_TMPDIR}" --threads 40 --memory 64GB
python3 src/slurm/scripts/preprocess_data.py berserk --tmpdir "${LOAD_MOVES_TMPDIR}" --threads 40 --memory 64GB
python3 src/slurm/scripts/preprocess_data.py grant_more_time --tmpdir "${LOAD_MOVES_TMPDIR}" --threads 40 --memory 64GB

echo "Preprocess job on $(hostname) at $(date)"
python3 src/slurm/scripts/preprocess_data.py preprocess --tmpdir "${LOAD_MOVES_TMPDIR}" --threads 40 --memory 128GB

echo "Finished pipeline at $(date)"
