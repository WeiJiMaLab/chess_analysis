#!/bin/bash
# plan.md Phase 2, Agent 2, sub-line B -- continue the PRODUCTION 20-epoch PG controller
# (frozen encoder, head-only policy-gradient training, cts.train.pg_controller_train --
# lineage CONFIRMED separate from the e2e Z2/Z-EXT joint-encoder line: pg_controller_train.py
# never unfreezes the encoder, reads z_t straight from the pre-materialized cache, and its
# checkpoint at ${packed_dir}/mchalt_controller.pt was produced by
# slurm/pipeline/train_readout_pg.slurm, epochs=20, job history in slurm/logs/train-readout-pg_*).
#
# Warm-starts from that checkpoint via the new resume_checkpoint hook (controller_train.py,
# _build_model_and_optimizer) and continues for many MORE epochs -- this lineage is CPU-cheap
# (~4s/epoch measured: 20-epoch prod run = job 10824168, 1m16s total), so a much longer
# continuation than the e2e line is affordable in the same wall-clock budget.
#
# stop_temperature/stop_temperature_final BOTH pinned to 1.0 (not the usual 4.0->1.0 anneal):
# the original 20-epoch run's OWN schedule already annealed down to tau=1.0 by its final epoch,
# so a continuation should pick up AT that temperature, not restart a fresh high-temperature
# anneal (which would be a discontinuity, not a real continuation).
#
# Writes to a SEPARATE scratch path -- does NOT touch or overwrite the real production
# checkpoint (${packed_dir}/mchalt_controller.pt).
#
#   sbatch slurm/agent2_pg_continue_train.slurm
#SBATCH --job-name=agent2-pg-continue
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh"

PROD_CKPT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt
OUT_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_pg_continued
mkdir -p "$OUT_DIR"
OUT_CKPT="$OUT_DIR/mchalt_controller_continued.pt"

echo "agent2_pg_continue host=$(hostname) resume_from=$PROD_CKPT"; date
test -f "$PROD_CKPT" || { echo "ERROR: production checkpoint missing: $PROD_CKPT" >&2; exit 1; }

python -u -m cts.train.pg_controller_train --config "$CONFIG" --stage train \
  --set device=cpu \
  --set resume_checkpoint="$PROD_CKPT" \
  --set output_checkpoint="$OUT_CKPT" \
  --set save_every_epoch=true \
  --set epochs=150 \
  --set stop_temperature=1.0 \
  --set stop_temperature_final=1.0 \
  --set pg_episode_batch=1024

echo "AGENT2 PG CONTINUE DONE -> $OUT_CKPT"; date
