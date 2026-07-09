#!/bin/bash
# Smoke test for analysis.evaluate_our_trees_continued --mode e2e, using the ALREADY-MATERIALIZED
# Z2 (epoch-1) cache from SIG-Z's own run -- validates the e2e code path independently of and
# before the (still-queued) continued-training job finishes.
#SBATCH --job-name=agent2-eval-smoke-e2e
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --qos=test
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_minply15_maxply75.yaml"
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS=4

Z2_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller
OUT_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/agent2_eval_smoke_e2e
mkdir -p "$OUT_DIR"

echo "agent2_eval_smoke_e2e host=$(hostname)"; date
python -u -m analysis.evaluate_our_trees_continued \
  --mode e2e \
  --packed-root "$MCP" \
  --original-cache "$MAT/validation_cache.pt" \
  --checkpoints "$Z2_DIR/materialized_validation_cache.pt" \
  --epochs 1 \
  --regimes 0.01:0.0 \
  --max-episodes 200 \
  --d-embed 32 \
  --out-dir "$OUT_DIR"

echo "AGENT2 EVAL SMOKE E2E DONE"; date
