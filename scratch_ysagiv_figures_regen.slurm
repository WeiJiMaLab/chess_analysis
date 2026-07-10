#!/bin/bash
# Regenerate the ysagiv headline figures (frontier / decodability / delta-regret) for all
# three populations (xaba20k, xaba100k, xaba100k_minply15_maxply75), now that:
#   (1) AG-Controller is included in the decodability probe (frontier + delta-regret already
#       had it, per the 2026-07-09 AGController investigation);
#   (2) tree-gen is reverted to factory PUCT (no FPU) -- NOTE: this run still reads the
#       EXISTING packed/materialized data (generated under the old FPU-including code,
#       2026-07-08/09), since regenerating the underlying trees is a separate, much larger
#       effort. These figures reflect the AG-Controller addition, not a fresh factory-PUCT
#       corpus.
# Uses each config's own eval.* regime (identical across all three): time_lambda=0.0015,
# maintenance_scale=0.0, maintenance_exponent=1.0, max_episodes=15000, d_embed=32.
#   sbatch scratch_ysagiv_figures_regen.slurm
#SBATCH --job-name=ysagiv-figures-regen
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

declare -A POPS=(
  [xaba20k]=ysagiv_xaba20k
  [xaba100k]=ysagiv_xaba100k
  [xaba100k_minply15_maxply75]=ysagiv_xaba100k_minply15_maxply75
)

for tag in "${!POPS[@]}"; do
  pop="${POPS[$tag]}"
  PACKED="/scratch/gpfs/GRIFFITHS/hl4291/lmcos/${pop}/mc_packed"
  CACHE="/scratch/gpfs/GRIFFITHS/hl4291/lmcos/${pop}/materialized/validation_cache.pt"
  OUT="outputs/figures/ysagiv/${tag}"
  echo "=== $tag -> $OUT ==="; date
  python -m analysis.evaluate --which all \
    --packed-root "$PACKED" --cache "$CACHE" --out-dir "$OUT" \
    --d-embed 32 --time-mode linear --time-lambda 0.0015 \
    --maintenance-scale 0.0 --maintenance-exponent 1.0 --maint-lambda 0.0015 \
    --max-episodes 15000
  echo "=== $tag DONE ==="; date
done

echo "YSAGIV FIGURES REGEN DONE"; date
