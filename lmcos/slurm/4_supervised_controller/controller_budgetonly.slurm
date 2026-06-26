#!/bin/bash
#SBATCH --job-name=ctrl-budgetonly
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/GNN/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/GNN/logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# NOTE: no `set -e` — inherited active venv makes a harmless 127 in module/activate
# abort before the first line (matches working mc_materialize_array.slurm).
PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3 >/dev/null 2>&1 || true
module load cudatoolkit/12.8 >/dev/null 2>&1 || true
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"; export PYTHONPATH="$PROJECT_DIR/src"
echo "host=$(hostname) start=$(date)"; nvidia-smi -L 2>/dev/null || true
python3 -u -m cts.train.controller_train \
  --config /scratch/gpfs/GRIFFITHS/hl4291/GNN/configs/controller_budgetonly.yaml
echo "done rc=$? $(date)"
