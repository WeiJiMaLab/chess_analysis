#!/bin/bash
# Phase 3 (PG) — train the readout (MCHalt meta-controller) by POLICY GRADIENT, CPU.
# Optimizes E[regret] directly via the exact closed-form expected return over stop
# steps (cts.train.pg_controller_train), instead of the advantage-MSE surrogate.
# Writes the SAME output_checkpoint core.yaml configures (sf_mchalt_controller.pt),
# so 4_eval scores it as the "mchalt" tier with no eval-side change. Head-only on the
# frozen-encoder z_t cache -> CPU-cheap; rides cpu/short (no GPU contention).
#
#   ELO=2000 sbatch --dependency=afterok:<merge_rootreps_jobid> \
#     pipeline/3_train_readout_pg.slurm
#SBATCH --job-name=tiny-readout-pg
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh"   # CPU: no cudatoolkit needed

echo "elo=$ELO host=$(hostname) device=cpu objective=policy_gradient"; date
python -u -m cts.train.pg_controller_train --config "$CONFIG" --stage train \
  --set globals.sf_elo="$ELO" --set device=cpu \
  --set epochs=20 --set learning_rate=2.0e-3 --set pg_episode_batch=1024
date
