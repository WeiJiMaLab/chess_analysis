#!/bin/bash
# Phase 2 capacity sweep (Agent 5) -- multi-regime eval for the wide64/deep_k2 encoder-capacity
# arms. Copy of slurm/pipeline/ysagiv_eval_multi.slurm (Agent 1's original, NOT modified here)
# with ONE difference: threads eval.d_embed through as an explicit --d-embed flag to both
# analysis.evaluate and analysis.ysagiv_sig. Needed because BOTH scripts' --d-embed CLI arg
# defaults to 32 -- correct for the baseline and the deep_k2 arm (d_embed unchanged at 32), but
# WRONG for the wide64 arm (d_embed=64): a silent mismatch there would misread the T_t column
# out of the wrong index of the raw feature array (analysis/evaluate.py _load_zt_by_episode
# does allf[:, :d_embed] / allf[:, d_embed]) rather than crash -- see
# config_ysagiv_xaba_wide64.yaml's eval: section comment. See ysagiv_xaba_capacity_sweep.md.
#   CONFIG=config_ysagiv_xaba_wide64.yaml sbatch slurm/pipeline/ysagiv_eval_multi_capacity.slurm
#   CONFIG=config_ysagiv_xaba_deep_k2.yaml sbatch slurm/pipeline/ysagiv_eval_multi_capacity.slurm
#SBATCH --job-name=ysagiv-eval-multi-cap
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00
# qos=test (<=1h cap): same resource sizing as ysagiv_eval_multi.slurm's own justification --
# smoke-scale population (~2093 xaba validation episodes), no separation/maintenance-sweep
# branch. The baseline run of this exact eval shape on the xaba population completed in 3-38
# min (job 10845263/10845264), so 55 min is comfortable headroom, not a guess.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
MAX_EPISODES=$(field max_episodes)
D_EMBED=$(field d_embed)
mkdir -p "$OUT"

echo "ysagiv_eval_multi_capacity host=$(hostname) config=$CONFIG d_embed=$D_EMBED"; date

# Same 3 REGIME-confirmed regime points as ysagiv_eval_multi.slurm / ysagiv_trees.md, so this
# track's numbers are directly row-for-row comparable to the baseline table:
#   1) time_lambda=0.01,   maintenance_scale=0.0    (the point used everywhere else)
#   2) time_lambda=0.001,  maintenance_scale=0.001  (nonzero-maintenance regime)
#   3) time_lambda=0.0005, maintenance_scale=0.0    (the xaba branch's winning regime)
REGIMES=(
  "0.01:0.0:regime_lambda0p01_m0"
  "0.001:0.001:regime_lambda0p001_m0p001"
  "0.0005:0.0:regime_lambda0p0005_m0"
)

for spec in "${REGIMES[@]}"; do
  IFS=':' read -r LAM MSCALE TAG <<< "$spec"
  RDIR="$OUT/$TAG"
  mkdir -p "$RDIR"
  echo "=== frontier @ time_lambda=$LAM maintenance_scale=$MSCALE d_embed=$D_EMBED -> $RDIR ==="
  python -m analysis.evaluate --which frontier \
    --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$RDIR" \
    --d-embed "$D_EMBED" \
    --time-mode linear --time-lambda "$LAM" \
    --maintenance-scale "$MSCALE" --maintenance-exponent 1.0 \
    --max-episodes "$MAX_EPISODES"
  echo "=== paired significance @ time_lambda=$LAM maintenance_scale=$MSCALE d_embed=$D_EMBED -> $OUT ==="
  python -m analysis.ysagiv_sig \
    --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" --tag "$TAG" \
    --d-embed "$D_EMBED" \
    --time-mode linear --time-lambda "$LAM" \
    --maintenance-scale "$MSCALE" --maintenance-exponent 1.0 \
    --max-episodes "$MAX_EPISODES"
done

echo "=== decodability (regime-independent) d_embed=$D_EMBED -> $OUT ==="
python -m analysis.evaluate --which decodability \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" \
  --d-embed "$D_EMBED" \
  --max-episodes "$MAX_EPISODES"

echo "YSAGIV EVAL MULTI CAPACITY DONE -> $OUT"; date
