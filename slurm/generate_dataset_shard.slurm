#!/bin/bash
# Usage example:
# sbatch \
#   --export=ALL,PROJECT_DIR=/scratch/gpfs/GRIFFITHS/ysagiv/chess/Project3CTS,FENS_PATH=/scratch/gpfs/GRIFFITHS/ysagiv/chess/root_position_samples/sampled_root_fens_2023.txt,OUTPUT_ROOT=/scratch/gpfs/GRIFFITHS/ysagiv/chess/generated_examples_2023,SHARD_ID=0,START_INDEX=0,END_INDEX=500 \
#   slurm/generate_dataset_shard.slurm

#SBATCH --job-name=cts-gen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

set -euo pipefail

if [[ -z "${SHARD_ID:-}" ]]; then
  echo "SHARD_ID is required" >&2
  exit 1
fi
if [[ -z "${START_INDEX:-}" ]]; then
  echo "START_INDEX is required" >&2
  exit 1
fi
if [[ -z "${END_INDEX:-}" ]]; then
  echo "END_INDEX is required" >&2
  exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-/home/ysagiv/chess/CTS}"
FENS_PATH="${FENS_PATH:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/roots/sampled_root_fens_2023.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/generated_trees}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENGINE_PATH="${ENGINE_PATH:-/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz}"
BACKEND="${BACKEND:-}"

MULTIPV="${MULTIPV:-8}"
MAX_DEPTH="${MAX_DEPTH:-10}"
SEARCH_BUDGET="${SEARCH_BUDGET:-64}"
MIN_NODES="${MIN_NODES:-5}"
MAX_NODES="${MAX_NODES:-30}"
SEED="${SEED:-0}"
CACHE_CLEAR_INTERVAL="${CACHE_CLEAR_INTERVAL:-128}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"

SHARD_OUTPUT_DIR="${OUTPUT_ROOT}/shard_$(printf "%05d" "${SHARD_ID}")"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${SHARD_OUTPUT_DIR}"

set +u
module purge
module load cudatoolkit/12.8
module load anaconda3/2023.3
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CTS
set -u

echo "project_dir=${PROJECT_DIR}"
echo "fens_path=${FENS_PATH}"
echo "shard_id=${SHARD_ID}"
echo "start_index=${START_INDEX}"
echo "end_index=${END_INDEX}"
echo "shard_output_dir=${SHARD_OUTPUT_DIR}"
echo "python_bin=${PYTHON_BIN}"
echo "engine_path=${ENGINE_PATH}"
echo "weights_path=${WEIGHTS_PATH}"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "hostname=$(hostname)"
nvidia-smi -L || true
date

cd "${PROJECT_DIR}"

if [[ -n "${BACKEND}" ]]; then
  EXTRA_ARGS=(--backend "${BACKEND}")
else
  EXTRA_ARGS=()
fi

"${PYTHON_BIN}" -u supervised_branch_cli.py generate-dataset \
  --engine-path "${ENGINE_PATH}" \
  --weights-path "${WEIGHTS_PATH}" \
  "${EXTRA_ARGS[@]}" \
  --fens "${FENS_PATH}" \
  --output-dir "${SHARD_OUTPUT_DIR}" \
  --multipv "${MULTIPV}" \
  --max-depth "${MAX_DEPTH}" \
  --search-budget "${SEARCH_BUDGET}" \
  --min-nodes "${MIN_NODES}" \
  --max-nodes "${MAX_NODES}" \
  --seed "${SEED}" \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --cache-clear-interval "${CACHE_CLEAR_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --resume

date
