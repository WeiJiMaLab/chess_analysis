#!/bin/bash
#SBATCH --job-name=cs-train
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/cost_sweep/logs/%x_%j.err
#SBATCH --qos=gpu-medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

: "${SWEEP_NAME:?SWEEP_NAME env var required}"
: "${TRAIN_CONFIG:?TRAIN_CONFIG env var required}"

PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv

module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3 >/dev/null 2>&1 || true
module load cudatoolkit/12.8 >/dev/null 2>&1 || true
source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

echo "sweep=$SWEEP_NAME config=$TRAIN_CONFIG host=$(hostname) start=$(date)"
nvidia-smi -L || true

python -u -m cts.train.controller_train --config "$TRAIN_CONFIG"

echo "done rc=$? $(date)"
