#!/bin/bash
# Phase 4 — eval 4 halt models per rung (CPU). Minutes.
# Runs afterok on EVERY rung's 3_train_readout. The `eval` section of core.yaml
# supplies the paths per rung (argv-driven, not --config). RUNGS (comma-separated)
# selects which rungs to eval; it MUST match the rungs whose controllers were
# trained this session.
#
#   RUNGS=2000 sbatch --dependency=afterok:<readout_jobid> slurm/pipeline/4_eval.slurm
#   RUNGS=1800,2000,2200 sbatch --dependency=afterok:<id1>:<id2>:<id3> slurm/pipeline/4_eval.slurm
#
#SBATCH --job-name=tiny-eval
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

set -euo pipefail

TINY=/home/hl4291/chess_analysis
IFS=',' read -ra RUNGS_ARR <<< "${RUNGS:-1800,2000,2200}"
# setup_env needs ELO (loads modules + venv + PYTHONPATH); eval is multi-rung and
# renders paths itself, so any rung works — use the first.
export ELO=${ELO:-${RUNGS_ARR[0]}}
source "$TINY/slurm/helpers/setup_env.sh"

rung_field() {  # rung_field <ELO> <key>
  python "$TINY/configs/render_stage.py" "$TINY/configs/core.yaml" --set globals.sf_elo="$1" --get "eval.$2"
}

echo "elo_rungs=${RUNGS_ARR[*]} host=$(hostname)"; date

for E in "${RUNGS_ARR[@]}"; do
  PACKED=$(rung_field "$E" packed_root)
  MATVAL=$(rung_field "$E" materialized_validation_cache)
  CTRL=$(rung_field "$E" controller_checkpoint)
  RJSON=$(rung_field "$E" results_json)
  OUT=$(rung_field "$E" out_dir)
  mkdir -p "$OUT"
  echo "=== eval elo${E} ==="
  python -m cts.analysis._budgeted.evaluate \
    --packed-root "$PACKED" --out-dir "$OUT" \
    --controller-checkpoint "$CTRL" \
    --materialized-validation-cache "$MATVAL" \
    --results-json "$RJSON"
done

echo "EVAL DONE rungs=${RUNGS_ARR[*]} -> figures/normative/{regret_by_model,oss_by_model,regret_vs_compute,regret_distribution,regret_decomposition}.png + sf_elo*_results.json"; date
