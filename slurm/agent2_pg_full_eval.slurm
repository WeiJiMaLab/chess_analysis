#!/bin/bash
# plan.md Phase 2, Agent 2, sub-line B -- FULL-scale paired-significance eval across the
# continued PG controller's training curve (11 checkpoints: the original 20-epoch production
# checkpoint + 10 more spread across the 150-epoch continuation, up to overall epoch 170) at
# the 3 REGIME-confirmed cost regimes. Uncapped validation split (--max-episodes default
# comfortably exceeds the real ~5599-episode validation set).
#
#   sbatch slurm/agent2_pg_full_eval.slurm
#SBATCH --job-name=agent2-pg-full-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=short
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS=4

D=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_pg_continued
PROD=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/mchalt_controller.pt
OUT_DIR=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative_pg_continued
mkdir -p "$OUT_DIR"

CKPTS="$PROD"
EPOCHS="20"
for N in 001 010 020 030 050 070 090 110 130 150; do
    f="$D/mchalt_controller_continued_epoch${N}.pt"
    test -f "$f" || { echo "WARNING: missing $f, skipping" >&2; continue; }
    overall=$((20 + 10#$N))
    CKPTS="$CKPTS,$f"
    EPOCHS="$EPOCHS,$overall"
done
echo "checkpoints=$CKPTS"
echo "epochs=$EPOCHS"

echo "agent2_pg_full_eval host=$(hostname)"; date
python -u -m analysis.evaluate_our_trees_continued \
    --mode pg \
    --packed-root "$MCP" \
    --cache "$MAT/validation_cache.pt" \
    --config "$CONFIG" \
    --checkpoints "$CKPTS" \
    --epochs "$EPOCHS" \
    --regimes "0.01:0.0,0.01:0.001,0.001:0.0005" \
    --d-embed 32 \
    --out-dir "$OUT_DIR"

echo "AGENT2 PG FULL EVAL DONE -> $OUT_DIR"; date
