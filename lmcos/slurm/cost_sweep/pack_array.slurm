#!/bin/bash
#SBATCH --job-name=cs-pack
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-19

: "${PACK_CONFIG:?PACK_CONFIG env var required (path to pack YAML config)}"
: "${SWEEP_NAME:?SWEEP_NAME env var required (e.g. lambda_1.0)}"

PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv
ROOT=/scratch/gpfs/GRIFFITHS/hl4291

module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3 >/dev/null 2>&1 || true
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

k=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
SPLIT_ROOT="$ROOT/packed/array/chunk_$k"
OUTPUT_ROOT="$ROOT/cost_sweep/$SWEEP_NAME/mc_chunks/chunk_$k"

echo "task=$k sweep=$SWEEP_NAME host=$(hostname) start=$(date)"

python -u -m cts.data.preprocess_mc.pack \
  --config "$PACK_CONFIG" \
  --override split_root="$SPLIT_ROOT" \
  --override output_root="$OUTPUT_ROOT"

rc=$?
echo "task=$k done rc=$rc $(date)"
exit $rc
