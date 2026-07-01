#!/bin/bash
# Pipeline step — eval (CPU). Scores the 4 halt models against the oracle and
# writes the results JSON + figures. The `eval` section of config.yaml supplies
# the paths. Depends on train_readout_pg.
#   sbatch --dependency=afterok:<train_readout_pg_jobid> slurm/pipeline/eval.slurm
#SBATCH --job-name=eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

field() { $RENDER --get "eval.$1"; }

PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
CTRL=$(field controller_checkpoint)
RJSON=$(field results_json)
OUT=$(field out_dir)
mkdir -p "$OUT"

echo "eval host=$(hostname)"; date
python -m analysis.evaluate \
  --packed-root "$PACKED" --out-dir "$OUT" \
  --controller-checkpoint "$CTRL" \
  --materialized-validation-cache "$MATVAL" \
  --results-json "$RJSON"
echo "EVAL DONE -> $OUT, $RJSON"; date
