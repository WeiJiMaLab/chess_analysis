#!/bin/bash
# Pipeline step — eval regime-fit (CPU), the ARRAY half of the delta-regret separation sweep.
# Each array task fits stop controllers (stats/zt/ag readouts, 3x 200-epoch PG trainings) for ONE
# independent cost regime out of analysis.evaluate's combined lambda+maintenance grid
# (_regime_grid: DEFAULT_LAMBDA_GRID then DEFAULT_MAINT_GRID, indices 0..N-1) and saves its
# delta-vs-z_t result to <out_dir>/regime_results/<NN>.json. eval.slurm submits this array
# (--wait) at the point that used to run all these regimes sequentially in-process, then runs
# `--which regime-collect` (same job, after this array returns) to reassemble the figures.
#   sbatch --array=0-9 --wait slurm/pipeline/eval_regime_array.slurm
#SBATCH --job-name=eval-regime
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# qos=test (<=1h, della-qos-time-routing): a single regime is ~1/10 of eval.slurm's old 10-regime
# sequential loop, and that full loop (plus the frontier fit + decodability probe) fit inside
# 1:30:00 -- 30min per regime here is generous headroom, first-run untuned. cpus-per-task=2 (not
# eval.slurm's 8): the PG training these regimes do is Python-loop-bound over tiny MLPs
# (in_dim<=33, hidden=64), not BLAS-bound, so extra threads per task were mostly wasted before --
# better spent as more concurrent array tasks. mem=48G kept at eval.slurm's proven value rather
# than guessed down: each task independently reloads the full validation z_t cache (no sharing
# across separate SLURM allocations), so the dominant cost -- the initial data load -- doesn't
# shrink just because one task now only fits one regime instead of ten. Revisit --cpus-per-task
# and --mem after the first real run's sacct numbers (labnotebook 2026-07-11).

set -euo pipefail
TID=${SLURM_ARRAY_TASK_ID:?must be submitted as an array (--array=0-N)}
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
MAINT_LAMBDA=$(field maint_lambda)
MAINTENANCE_EXPONENT=$(field maintenance_exponent)
MAX_EPISODES=$(field max_episodes)

echo "eval_regime_array task=$TID host=$(hostname) config=$CONFIG"; date
python -m analysis.evaluate --which regime-fit --regime-index "$TID" \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" \
  --maint-lambda "$MAINT_LAMBDA" --maintenance-exponent "$MAINTENANCE_EXPONENT" \
  --max-episodes "$MAX_EPISODES"
echo "EVAL REGIME $TID DONE"; date
