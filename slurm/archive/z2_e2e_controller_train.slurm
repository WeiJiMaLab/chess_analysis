#!/bin/bash
# plan.md Z2 -- real end-to-end (encoder-unfrozen) MetaController training run.
# NOT a pipeline stage (does not touch gen_trees/pack_trees/materialize/train_readout_pg/
# eval or their production outputs/checkpoints). Trains cts.train.e2e_controller_train,
# reusing the real `train` config section (same packed data / architecture pg_controller_train.py
# uses) with unfreeze_encoder=true so the encoder trains jointly with the advantage head
# against the real expected-regret objective, instead of the frozen-transfer path.
#
# Writes to a SEPARATE scratch checkpoint path -- does NOT touch or overwrite the real
# production controller checkpoint (${packed_dir}/mchalt_controller.pt).
#
# REVISED 2026-07-08 (resuming after job 10829588 CANCELLED at 0:00 elapsed -- the
# original 18-epoch/~12h version never actually ran): new hard constraint from the user
# is NO job in this effort may exceed 1h wall-clock, and a single epoch is an acceptable
# proof-of-concept (not a converged model). epochs=18 -> epochs=1, gpu-short -> gpu-test,
# time budget cut to well under 1h. pg_max_episodes=15000 caps the (already-full, 22338-
# episode) train pass so one epoch's compute comfortably clears 1h with buffer left for
# validation + checkpointing + plotting (see the throughput math below).
#
#   sbatch slurm/z2_e2e_controller_train.slurm
#SBATCH --job-name=z2-e2e-controller
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-test
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:58:00
#SBATCH --gres=gpu:1
# NOTE (from a live pulse-check job, 10829540): Z0's GPU timing script only timed compute
# on ONE pre-loaded batch and never measured ControllerEpisodeDataset's disk I/O (shard
# torch.load + tree reconstruction) over a real epoch. Real measured throughput here is
# ~0.65-0.7s/batch (vs Z0's 0.186s/batch compute-only number). At episode_batch_size=8:
#   train (pg_max_episodes=15000 cap): ceil(15000/8)=1875 batches * 0.7s ~= 22 min
#   val   (uncapped, 5599 episodes < cap so stays full):  700 batches * 0.7s ~=  8 min
#   -> ~30 min compute for 1 epoch, leaving ~25 min of the 58 min budget for env setup /
#      model build / per-epoch checkpoint saves (incl. the new encoder-only checkpoint) /
#      loss-curve plotting.
# Routed to gpu-test (della QOS time-routing convention: <=1h -> test/gpu-test, never
# --partition=gpu directly), not gpu-short (that was for the abandoned 12h version).

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh" gpu

Z2_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller
Z2_CKPT="$Z2_DIR/e2e_controller.pt"
mkdir -p "$Z2_DIR"

echo "z2_e2e_controller host=$(hostname)"; nvidia-smi -L || true; date
echo "Z2_CKPT=$Z2_CKPT (scratch path, real controller checkpoint ${MCP}/../packed/mchalt_controller.pt untouched)"

python -m cts.train.e2e_controller_train --config "$CONFIG" --stage train \
    --set unfreeze_encoder=true \
    --set output_checkpoint="$Z2_CKPT" \
    --set epochs=1 \
    --set pg_max_episodes=15000

echo "Z2 E2E CONTROLLER TRAIN DONE"; date
