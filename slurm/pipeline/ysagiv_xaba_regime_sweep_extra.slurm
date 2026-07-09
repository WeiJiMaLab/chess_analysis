#!/bin/bash
# Agent 4 follow-up: two extra time_lambda points (0.005, 0.007) bracketing the transition
# between the "z_t wins" band (lambda<=0.002) and the "ties" band (lambda>=0.01) found by
# ysagiv_xaba_regime_sweep.slurm (job 10846482). Same population, same methodology, see
# ysagiv_xaba_regime_sweep.md.
#   CONFIG=config_ysagiv_xaba.yaml sbatch slurm/pipeline/ysagiv_xaba_regime_sweep_extra.slurm
#SBATCH --job-name=ysagiv-xaba-regime-sweep-extra
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
MAX_EPISODES=$(field max_episodes)

for spec in "0.005:xswp_lambda0p005_m0" "0.007:xswp_lambda0p007_m0"; do
  IFS=':' read -r LAM TAG <<< "$spec"
  echo "=== $TAG ==="
  python -m analysis.ysagiv_sig \
    --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" --tag "$TAG" \
    --time-mode linear --time-lambda "$LAM" \
    --maintenance-scale 0.0 --maintenance-exponent 1.0 \
    --max-episodes "$MAX_EPISODES"
done
echo "EXTRA SWEEP DONE"; date
