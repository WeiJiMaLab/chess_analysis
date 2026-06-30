#!/bin/bash
# Phase 3 — train the readout (MCHalt meta-controller), CPU. Renders the `train`
# section of configs/core.yaml -> cts.train.controller_train. Frozen tiny encoder
# + ReLU advantage MLP on [z_t, T_t]; checkpoint selected by val regret.
# It is a tiny MLP over the PRECOMPUTED z_t cache (~seconds of compute), so it runs
# on cpu/short (device=cpu) to skip the throttled GPU queue rather than reserve a
# whole A100 for a 30-second job.
#
#   ELO=1800 sbatch --dependency=afterok:<merge_rootreps_jobid> \
#     slurm/pipeline/3_train_readout.slurm
#SBATCH --job-name=tiny-readout
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:15:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # CPU: no cudatoolkit needed

echo "elo=$ELO host=$(hostname) device=cpu"; date
# controller_train reads the `train` section of core.yaml directly.
python -m cts.train.controller_train --config "$CONFIG" --stage train \
  --set globals.sf_elo="$ELO" --set device=cpu
date
