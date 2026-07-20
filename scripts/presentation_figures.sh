#!/usr/bin/env bash
# (Re)generate the 4 presentation-facing figures (ply-vs-RT, captures-avail-vs-RT,
# metacontroller regret scatter, delta-regret-vs-regime forest) as PDF/PNG/JSON trios,
# then collect copies for hand-off to the sister `defense-talk` deck. Plain bash, no
# sbatch -- board.py --plot and evaluate.py --replot are both cheap enough to run
# interactively.
#
# Env vars:
#   CONFIG       active run config yaml (same var slurm/helpers/setup_env.sh exports);
#                drives $DB and board.py's figures_dir. Defaults to repo default config.
#   DB           db path for board.py --db. Defaults to human_analysis.selected_db_default.
#   OUT_DIR      evaluate.py --out-dir (must already contain regret_scatter_data.json /
#                delta_regret_data.json for the default --replot path). Defaults to eval.out_dir.
#   PACKED_ROOT  eval.packed_root (only used when FRESH=1).
#   CACHE        eval.materialized_validation_cache (only used when FRESH=1).
#   FRESH        0 (default) = --replot from existing *_data.json (cheap); 1 = full recompute.
#   LABEL        subdirectory name under outputs/figures/presentation/ for the collected
#                copies. Defaults to "latest" (fixed, not a timestamp -- see repo convention).
#
# Example:
#   CONFIG=$TINY/config_ysagiv_xaba100k_minply15_maxply75_history.yaml \
#     LABEL=defense_talk_v1 scripts/presentation_figures.sh

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"

source "$TINY/slurm/helpers/setup_env.sh"

DB="${DB:-$($RENDER --get human_analysis.selected_db_default)}"
OUT_DIR="${OUT_DIR:-$($RENDER --get eval.out_dir)}"
BOARD_FIGURES_DIR="${BOARD_FIGURES_DIR:-$($RENDER --get human_analysis.figures_dir)}"
LABEL="${LABEL:-latest}"

echo "=== 1. board.py --plot (rt_distribution + bivariate_n_captures_avail) ==="
python -m analysis.board --plot --db "$DB"

echo "=== 2. evaluate.py: regret-scatter + separation (FRESH=${FRESH:-0}) ==="
if [[ "${FRESH:-0}" == "1" ]]; then
  PACKED_ROOT="${PACKED_ROOT:-$($RENDER --get eval.packed_root)}"
  CACHE="${CACHE:-$($RENDER --get eval.materialized_validation_cache)}"
  python -m analysis.evaluate --which regret-scatter \
    --packed-root "$PACKED_ROOT" --cache "$CACHE" --out-dir "$OUT_DIR"
  python -m analysis.evaluate --which separation \
    --packed-root "$PACKED_ROOT" --cache "$CACHE" --out-dir "$OUT_DIR"
else
  python -m analysis.evaluate --replot --which regret-scatter --out-dir "$OUT_DIR"
  python -m analysis.evaluate --replot --which separation --out-dir "$OUT_DIR"
fi

echo "=== 3. collecting JSON + PDF/PNG into outputs/figures/presentation/$LABEL ==="
COLLECT_DIR="$TINY/outputs/figures/presentation/$LABEL"
mkdir -p "$COLLECT_DIR/board/json" "$COLLECT_DIR/board/pdf" "$COLLECT_DIR/board/png" "$COLLECT_DIR/eval/pdf" "$COLLECT_DIR/eval/png"

for name in rt_distribution bivariate_n_captures_avail; do
  for ext in json; do
    src="$BOARD_FIGURES_DIR/board/json/${name}_data.${ext}"
    [[ -f "$src" ]] && cp -v "$src" "$COLLECT_DIR/board/json/" || echo "missing: $src"
  done
  for kind in pdf png; do
    src="$BOARD_FIGURES_DIR/board/$kind/${name}.${kind}"
    [[ -f "$src" ]] && cp -v "$src" "$COLLECT_DIR/board/$kind/" || echo "missing: $src"
  done
done

for name in regret_scatter delta_regret; do
  src="$OUT_DIR/${name}_data.json"
  [[ -f "$src" ]] && cp -v "$src" "$COLLECT_DIR/eval/" || echo "missing: $src"
done
for name in regret_scatter delta_mean_regret_lambda delta_mean_regret_maintenance; do
  for kind in pdf png; do
    src="$OUT_DIR/$kind/${name}.${kind}"
    [[ -f "$src" ]] && cp -v "$src" "$COLLECT_DIR/eval/$kind/" || echo "missing: $src"
  done
done

echo "Collected presentation figures under $COLLECT_DIR"
