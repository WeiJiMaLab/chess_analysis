#!/bin/bash
# Phase 2a — train encoder (GPU).
# Renders the `encoder` section of configs/core.yaml -> cts.data.build_tree
#
#   ELO=1800 sbatch --dependency=afterok:<packtrees_jobid> \
#     pipeline/2a_train_encoder.slurm
#SBATCH --job-name=tiny-encoder
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:15:00
#SBATCH --gres=gpu:1

set -euo pipefail
: "${ELO:?ELO required (1800|2000|2200)}"

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh" gpu

echo "elo=$ELO host=$(hostname)"; nvidia-smi -L || true; date

# ── train encoder (tiny, child-WDL) ───────────────────────────────────────────
echo "=== encoder ==="
python -m cts.data.build_tree --config "$CONFIG" --stage encoder --set globals.sf_elo="$ELO"

echo "ENCODER DONE elo=$ELO"; date
