#!/bin/bash
# Phase 2 Agent 1 (ysagiv human_trees) -- multi-regime eval. Runs analysis.evaluate's
# frontier plot at 2-3 REGIME-confirmed meaningful cost-regime points (not just the single
# m=0/lambda=0.01 point every other eval in this investigation defaults to -- plan.md's
# instruction), plus ONE regime-independent decodability plot. Reuses analysis.evaluate
# unmodified, same as eval.slurm, just looped and with per-regime output subdirectories.
#   CONFIG=config_ysagiv_argmax.yaml sbatch slurm/pipeline/ysagiv_eval_multi.slurm
#SBATCH --job-name=ysagiv-eval-multi
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
# qos=test (<=1h cap), not short: this job only runs frontier+paired-sig (no separation/
# maintenance-sweep branch) on a SMOKE-SCALE population (745-2628 eval episodes vs.
# production's full validation split), so the 1:30:00 budget eval.slurm's own comment
# justifies for the FULL corpus is generous overkill here. Coordinator resource-sizing
# check (2026-07-08): short QOS had ~20x more queued jobs than test; switched down once
# reasonably confident of finishing under test's 1h cap. If this ever times out, split
# per-regime into 3 separate qos=test jobs rather than reverting to qos=short.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
MAX_EPISODES=$(field max_episodes)
mkdir -p "$OUT"

echo "ysagiv_eval_multi host=$(hostname) config=$CONFIG"; date

# REGIME's 3 chosen meaningful points (outputs/figures/minply15_maxply75/diagnosis/regime_select_results.json):
#   1) time_lambda=0.01,   maintenance_scale=0.0    (k*=10, k_frac=0.105 on OUR corpus -- the point used everywhere else)
#   2) time_lambda=0.001,  maintenance_scale=0.001  (k*=8,  k_frac=0.084 -- nonzero-maintenance regime)
#   3) time_lambda=0.0005, maintenance_scale=0.0    (k*=56, k_frac=0.589 -- a much less aggressive stop, different regime shape)
REGIMES=(
  "0.01:0.0:regime_lambda0p01_m0"
  "0.001:0.001:regime_lambda0p001_m0p001"
  "0.0005:0.0:regime_lambda0p0005_m0"
)

for spec in "${REGIMES[@]}"; do
  IFS=':' read -r LAM MSCALE TAG <<< "$spec"
  RDIR="$OUT/$TAG"
  mkdir -p "$RDIR"
  echo "=== frontier @ time_lambda=$LAM maintenance_scale=$MSCALE -> $RDIR ==="
  python -m analysis.evaluate --which frontier \
    --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$RDIR" \
    --time-mode linear --time-lambda "$LAM" \
    --maintenance-scale "$MSCALE" --maintenance-exponent 1.0 \
    --max-episodes "$MAX_EPISODES"
  echo "=== paired significance @ time_lambda=$LAM maintenance_scale=$MSCALE -> $OUT ==="
  python -m analysis.ysagiv_sig \
    --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" --tag "$TAG" \
    --time-mode linear --time-lambda "$LAM" \
    --maintenance-scale "$MSCALE" --maintenance-exponent 1.0 \
    --max-episodes "$MAX_EPISODES"
done

echo "=== decodability (regime-independent) -> $OUT ==="
python -m analysis.evaluate --which decodability \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" \
  --max-episodes "$MAX_EPISODES"

echo "YSAGIV EVAL MULTI DONE -> $OUT"; date
