#!/bin/bash
# Pipeline step — train_encoder (GPU). Pretrains the tiny child-WDL encoder on the
# packed examples (the `encoder` stage of config.yaml). Depends on pack_trees.
#   sbatch --dependency=afterok:<pack_trees_jobid> slurm/pipeline/train_encoder.slurm
#SBATCH --job-name=train-encoder
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:15:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

echo "train_encoder host=$(hostname)"; nvidia-smi -L || true; date
python -m cts.data.build_tree --config "$CONFIG" --stage encoder
echo "ENCODER DONE"; date
