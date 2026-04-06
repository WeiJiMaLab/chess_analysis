#!/bin/bash

#SBATCH --job-name=cts-roots
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PERSONAL_DB="${PERSONAL_DB:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS_personal.db}"
CORE_DB="${CORE_DB:-/scratch/gpfs/GRIFFITHS/chess-db/lichess.db}"
OUT_DIR="${OUT_DIR:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/roots}"
YEAR="${YEAR:-2023}"
SAMPLED_GAMES="${SAMPLED_GAMES:-200000}"
FINAL_FENS="${FINAL_FENS:-100000}"
SEED="${SEED:-2023}"
THREADS="${THREADS:-10}"
MEMORY_LIMIT="${MEMORY_LIMIT:-60GB}"
TEMP_DIRECTORY="${TEMP_DIRECTORY:-.}"
MIN_ELO="${MIN_ELO:-1800}"
MAX_ELO="${MAX_ELO:-2600}"
MIN_GAME_HALFMOVES="${MIN_GAME_HALFMOVES:-20}"
MIN_MOVE_PLY="${MIN_MOVE_PLY:-8}"
MAX_MOVE_PLY="${MAX_MOVE_PLY:-120}"
MIN_LEGAL_MOVES="${MIN_LEGAL_MOVES:-2}"
MAX_LEGAL_MOVES="${MAX_LEGAL_MOVES:-60}"
MIN_PIECES="${MIN_PIECES:-8}"
MAX_PIECES="${MAX_PIECES:-32}"
PREFIX="${PREFIX:-sampled_root_fens}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUT_DIR}"

set +u
module purge
module load anaconda3/2023.3
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

echo "project_dir=${PROJECT_DIR}"
echo "personal_db=${PERSONAL_DB}"
echo "core_db=${CORE_DB}"
echo "out_dir=${OUT_DIR}"
echo "year=${YEAR}"
echo "sampled_games=${SAMPLED_GAMES}"
echo "final_fens=${FINAL_FENS}"
echo "threads=${THREADS}"
echo "memory_limit=${MEMORY_LIMIT}"
echo "hostname=$(hostname)"
date

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/sample_root_fens_from_della.py \
  --personal-db "${PERSONAL_DB}" \
  --core-db "${CORE_DB}" \
  --out-dir "${OUT_DIR}" \
  --year "${YEAR}" \
  --sampled-games "${SAMPLED_GAMES}" \
  --final-fens "${FINAL_FENS}" \
  --seed "${SEED}" \
  --threads "${THREADS}" \
  --memory-limit "${MEMORY_LIMIT}" \
  --temp-directory "${TEMP_DIRECTORY}" \
  --min-elo "${MIN_ELO}" \
  --max-elo "${MAX_ELO}" \
  --min-game-halfmoves "${MIN_GAME_HALFMOVES}" \
  --min-move-ply "${MIN_MOVE_PLY}" \
  --max-move-ply "${MAX_MOVE_PLY}" \
  --min-legal-moves "${MIN_LEGAL_MOVES}" \
  --max-legal-moves "${MAX_LEGAL_MOVES}" \
  --min-pieces "${MIN_PIECES}" \
  --max-pieces "${MAX_PIECES}" \
  --prefix "${PREFIX}"

date
