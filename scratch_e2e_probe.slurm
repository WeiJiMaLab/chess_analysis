#!/bin/bash
# End-to-end (encoder-unfrozen) controller training probe (2026-07-09): does a jointly-trained
# encoder+readout avoid the AlwaysStop collapse both the frozen-z_t and AG readouts show under
# high per-node maintenance cost (verified: SingleHalt* k*->1, both readouts' stop distributions
# collapse toward step 0, starting around maintenance>=0.1)? Targets maintenance=0.2 (a clear,
# already-measured collapse point, +0.0412 z_t-vs-ag delta there) at the standard lambda=0.0015
# regime, on the 200K/xaba+ply-window population (most episodes, cleanest signal so far).
# Initializes from the already-pretrained child-WDL encoder checkpoint (not from scratch) --
# output_checkpoint points at a scratch probe path, never the production checkpoint.
#   sbatch scratch_e2e_probe.slurm
#SBATCH --job-name=e2e-probe-maint0p2
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=gpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"
source "$TINY/slurm/helpers/setup_env.sh" gpu

CFG=config_ysagiv_xaba100k_minply15_maxply75.yaml
OUT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba100k_minply15_maxply75/e2e_probe/e2e_controller_maint0p2.pt

echo "e2e_probe host=$(hostname) config=$CFG out=$OUT"; date
python -m cts.train.e2e_controller_train --config "$CFG" --stage train \
  --set unfreeze_encoder=true \
  --set output_checkpoint="$OUT" \
  --set maintenance_scale=0.2 --set maintenance_exponent=1.0 \
  --set time_mode=linear --set time_lambda=0.0015 \
  --set epochs=12 --set pg_max_episodes=4000 --set episode_batch_size=8 \
  --set learning_rate=1e-4
echo "E2E PROBE DONE"; date
