#!/bin/bash
# Phase 3 — train the readout (MCHalt meta-controller), GPU. Renders the `train`
# section of configs/elo${ELO}.yaml -> cts.train.controller_train. Frozen tiny
# encoder + ReLU advantage MLP on [z_t, T_t]; checkpoint selected by val regret.
#
#   ELO=1800 sbatch --dependency=afterok:<rootreps_jobid> \
#     pipeline/3_train_readout/train_readout.slurm
#SBATCH --job-name=tiny-readout
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
VENV_DIR=/home/hl4291/venv
CONFIG=$TINY/configs/elo${ELO}.yaml
WORK=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_configs/elo${ELO}
mkdir -p "$TINY/slurm/logs" "$WORK"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
source "$VENV_DIR/bin/activate"
set -u
export PYTHONPATH="$TINY/src"; export PYTHONUNBUFFERED=1

echo "elo=$ELO host=$(hostname)"; nvidia-smi -L || true; date
python "$TINY/configs/render_stage.py" "$CONFIG" train --out "$WORK/train.yaml"
python -m cts.train.controller_train --config "$WORK/train.yaml"
date
