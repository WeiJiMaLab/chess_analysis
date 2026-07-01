#!/bin/bash
# Pipeline step — train_readout_pg (CPU). Trains the readout (MCHalt
# meta-controller) by POLICY GRADIENT, optimizing E[regret] directly via the
# exact closed-form expected return over stop steps (cts.train.pg_controller_train).
# Head-only on the frozen-encoder z_t cache -> CPU-cheap. Depends on pack_root.
#   sbatch --dependency=afterok:<pack_root_jobid> slurm/pipeline/train_readout_pg.slurm
#SBATCH --job-name=train-readout-pg
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # CPU: no cudatoolkit needed

echo "train_readout_pg host=$(hostname) device=cpu objective=policy_gradient"; date
python -u -m cts.train.pg_controller_train --config "$CONFIG" --stage train \
  --set device=cpu \
  --set epochs=20 --set learning_rate=2.0e-3 --set pg_episode_batch=1024
date
