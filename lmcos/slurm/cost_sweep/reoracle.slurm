#!/bin/bash
#SBATCH --job-name=cs-reoracle
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

: "${REORACLE_CONFIG:?REORACLE_CONFIG env var required}"
: "${SWEEP_NAME:?SWEEP_NAME env var required}"

PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv

module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3 >/dev/null 2>&1 || true
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "sweep=$SWEEP_NAME config=$REORACLE_CONFIG host=$(hostname) start=$(date)"
python -u -m cts.data.preprocess_mc.reoracle --config "$REORACLE_CONFIG"
echo "done rc=$? $(date)"
