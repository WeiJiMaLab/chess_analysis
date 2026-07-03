#!/bin/bash
# Pipeline step — eval (CPU). Writes the three meta-controller ASSESSMENT figures
# (frontier / R-decodability / tier-separation) via `analysis.evaluate` — the
# mechanism behind outputs/reports/evaluate.md. Reads paths from the `eval:`
# section of config.yaml. Refits the tree-stats + z_t readouts (budget-out /
# steps-in) on the validation 70/30 split, so it needs the z_t cache in RAM.
#   sbatch --dependency=afterok:<train_readout_pg_jobid> slurm/pipeline/eval.slurm
#SBATCH --job-name=eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
mkdir -p "$OUT"

echo "eval host=$(hostname)"; date
python -m analysis.evaluate --which all \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT"
echo "EVAL DONE -> $OUT"; date
