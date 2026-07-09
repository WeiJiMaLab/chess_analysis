#!/bin/bash
# plan.md SIG-Z -- CPU-only eval step of Z3 (materialize already done, caches exist on disk).
# Runs analysis.evaluate_e2e_z3 directly (paired significance + fixed decodability plot).
# See slurm/z3_e2e_materialize_and_eval.slurm for the full pipeline this is a subset of --
# skips the (already-complete) materialize/merge steps and just reruns the comparison script,
# which needed to run on CPU anyway (no GPU inference inside evaluate_e2e_z3.py, just reads
# already-materialized caches + fits stop-controller readouts). Submitted directly (not via GPU
# qos) because an earlier foreground attempt on the login node got silently killed (no OOM/error
# in the log, just truncated output) after ~2min wall time / ~44min CPU-time -- almost certainly
# a login-node resource-governor kill, not a real bug in the script itself.
#SBATCH --job-name=sig-z-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

Z2_DIR=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/diagnosis/z2_e2e_controller
E2E_CACHE="$Z2_DIR/materialized_validation_cache.pt"
Z3_OUT=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/diagnosis/z3_e2e_comparison
mkdir -p "$Z3_OUT"

test -f "$E2E_CACHE" || { echo "ERROR: e2e materialized cache missing: $E2E_CACHE" >&2; exit 1; }
test -f "$MAT/validation_cache.pt" || { echo "ERROR: original materialized cache missing: $MAT/validation_cache.pt" >&2; exit 1; }

echo "sig-z-eval host=$(hostname)"; date
python -m analysis.evaluate_e2e_z3 \
    --packed-root "$MCP" \
    --e2e-cache "$E2E_CACHE" \
    --original-cache "$MAT/validation_cache.pt" \
    --baseline-fig-dir /home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/normative \
    --out-dir "$Z3_OUT" \
    --d-embed 32 \
    --maintenance-scales 0.0,0.05,0.1
echo "SIG-Z EVAL DONE -> $Z3_OUT"; date
