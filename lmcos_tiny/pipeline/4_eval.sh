#!/bin/bash
# Phase 4 — eval 4 halt models per rung + cross-Elo ladder plot. Local CPU, minutes.
# The `eval` section of each configs/elo${ELO}.yaml supplies the paths; this entry
# point is argv-driven (not --config), so we read the section's fields here.
#
#   bash pipeline/4_eval.sh            # all three rungs + ladder
set -euo pipefail

TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/env.sh"

rung_field() {  # rung_field <ELO> <key>
  python "$TINY/configs/render_stage.py" "$TINY/configs/core.yaml" --set globals.sf_elo="$1" --get "eval.$2"
}

RUNGS=(1800 2000 2200)
LADDER_ARGS=()
for ELO in "${RUNGS[@]}"; do
  PACKED=$(rung_field "$ELO" packed_root)
  MATVAL=$(rung_field "$ELO" materialized_validation_cache)
  CTRL=$(rung_field "$ELO" controller_checkpoint)
  RJSON=$(rung_field "$ELO" results_json)
  OUT=$(rung_field "$ELO" out_dir)
  mkdir -p "$OUT"
  echo "=== eval elo${ELO} ==="
  python -m cts.analysis._budgeted.alt_models_eval \
    --packed-root "$PACKED" --out-dir "$OUT" \
    --controller-checkpoint "$CTRL" \
    --materialized-validation-cache "$MATVAL" \
    --results-json "$RJSON"
  LADDER_ARGS+=( --rung ${ELO} "$RJSON" )
done

echo "=== ladder plot ==="
python -m cts.analysis._budgeted.ladder_plot "${LADDER_ARGS[@]}" --out-dir "$(rung_field 1800 out_dir)"
echo "done -> figures/lmcos_tiny/{regret_by_model,oss_by_model,ladder_by_model}.png + sf_elo*_results.json"
