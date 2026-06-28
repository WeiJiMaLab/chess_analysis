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
source "$TINY/pipeline/0_helpers/setup_env.sh" gpu

echo "elo=$ELO host=$(hostname)"; nvidia-smi -L || true; date
# controller_train reads the `train` section of core.yaml directly.
python -m cts.train.controller_train --config "$CONFIG" --stage train --set globals.sf_elo="$ELO"
date
