#!/bin/bash
# Pipeline step — eval regime-collect (CPU), the COLLECTOR half of the delta-regret
# separation sweep. Runs after eval_regime_array.slurm's array finishes (chained via
# --dependency=afterok from eval.slurm, not `sbatch --wait` -- see eval.slurm's own
# comment for why blocking there caused a real timeout on 2026-07-11). Reassembles
# every <out_dir>/regime_results/<NN>.json into delta_regret_data.json and renders the
# delta-mean-regret (lambda/maintenance) figures -- the "cheap half" of the sweep, no
# data loading of the z_t cache, just JSON + matplotlib.
#   sbatch --dependency=afterok:<eval_regime_array_jobid> --export=ALL,CONFIG=<...> slurm/pipeline/eval_collect.slurm
#SBATCH --job-name=eval-collect
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
# Cheap by construction (JSON load + render, no z_t cache reload) -- 10min is generous headroom.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

field() { $RENDER --get "eval.$1"; }
OUT=$(field out_dir)
MAINT_LAMBDA=$(field maint_lambda)

echo "eval_collect host=$(hostname) config=$CONFIG"; date
python -m analysis.evaluate --which regime-collect \
  --out-dir "$OUT" --maint-lambda "$MAINT_LAMBDA"
echo "EVAL COLLECT DONE -> $OUT"; date
