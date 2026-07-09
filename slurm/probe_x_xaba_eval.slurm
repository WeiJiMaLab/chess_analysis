#!/bin/bash
# Probe X (plan.md "External expert input" -- xaba filter on our own corpus). Paired-bootstrap
# significance of SingleHalt*/Stats-Controller/original-frozen-z_t, run across THREE populations
# (argmax_only baseline, argmax_and_xaba = condition A, xaba_only = condition B) at 3
# REGIME-confirmed regime points, via src/analysis/evaluate_probe_x_xaba.py (reuses
# analysis.ysagiv_sig.compute_pairwise_significance unchanged). CPU-only -- readout heads are tiny
# MLPs (same footprint as sig_s_eval.slurm, which took 56s for one population/one regime; this
# does 9 such fits).
#   sbatch --dependency=afterok:<condA_materialize_jobid>:<condB_materialize_jobid> slurm/probe_x_xaba_eval.slurm
#SBATCH --job-name=probe-x-xaba-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:55:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"   # default CONFIG=config_minply15_maxply75.yaml -> base run paths (MCP/MAT)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_MCP="$MCP"
BASE_MAT="$MAT"
XABA_MCP=$(python "$TINY/render_stage.py" "$TINY/config_probe_x_xaba.yaml" --get globals.mc_packed_dir)
XABA_MAT=$(python "$TINY/render_stage.py" "$TINY/config_probe_x_xaba.yaml" --get globals.materialized_dir)
CONDB_MCP=$(python "$TINY/render_stage.py" "$TINY/config_probe_x_xaba_condb.yaml" --get globals.mc_packed_dir)
CONDB_MAT=$(python "$TINY/render_stage.py" "$TINY/config_probe_x_xaba_condb.yaml" --get globals.materialized_dir)

OUT=/home/hl4291/chess_analysis/outputs/figures/minply15_maxply75/diagnosis
mkdir -p "$OUT"

echo "probe-x-xaba-eval host=$(hostname)"; date
echo "BASE_MCP=$BASE_MCP BASE_MAT=$BASE_MAT"
echo "XABA_MCP(condA)=$XABA_MCP XABA_MAT(condA)=$XABA_MAT"
echo "CONDB_MCP=$CONDB_MCP CONDB_MAT=$CONDB_MAT"
test -f "$BASE_MAT/validation_cache.pt" || { echo "ERROR: base cache missing" >&2; exit 1; }
test -f "$XABA_MAT/validation_cache.pt" || { echo "ERROR: condA (xaba) cache missing" >&2; exit 1; }
test -f "$CONDB_MAT/validation_cache.pt" || { echo "ERROR: condB (xaba-only) cache missing" >&2; exit 1; }

python -m analysis.evaluate_probe_x_xaba \
    --base-packed-root "$BASE_MCP" --base-cache "$BASE_MAT/validation_cache.pt" \
    --xaba-packed-root "$XABA_MCP" --xaba-cache "$XABA_MAT/validation_cache.pt" \
    --condb-packed-root "$CONDB_MCP" --condb-cache "$CONDB_MAT/validation_cache.pt" \
    --out-dir "$OUT" \
    --d-embed 32

echo "PROBE X XABA EVAL DONE -> $OUT"; date
