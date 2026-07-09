#!/bin/bash
# plan.md SIG-S -- is Stats-Controller's point-estimate win over SingleHalt* (S1) statistically
# significant? Paired-bootstrap CI on the per-episode regret difference, same regime
# (linear, time_lambda=0.01, maintenance_scale=0.0) and same 70/30 tree-split as everything else
# in this investigation. See src/analysis/evaluate_sig_s.py for the methodology.
#SBATCH --job-name=sig-s-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

OUT=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/diagnosis
mkdir -p "$OUT"

echo "sig-s-eval host=$(hostname)"; date
python -m analysis.evaluate_sig_s \
    --packed-root "$MCP" --cache "$MAT/validation_cache.pt" \
    --out-dir "$OUT" \
    --d-embed 32 --time-mode linear --time-lambda 0.01 --maintenance-scale 0.0
echo "SIG-S EVAL DONE -> $OUT"; date
