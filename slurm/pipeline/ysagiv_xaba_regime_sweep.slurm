#!/bin/bash
# Agent 4 (ysagiv xaba-branch, time_lambda sweep) -- disambiguates whether z_t's win at
# time_lambda=0.0005 (ysagiv_trees.md xaba-branch table) is a real per-episode-adaptivity
# advantage or an artifact of SingleHalt* sitting at k*=95/96 (near-ceiling, right-censored)
# at that ONE regime. Reuses the SAME already-packed/materialized ysagiv xaba population Agent
# 1 built (config_ysagiv_xaba.yaml's eval.packed_root / eval.materialized_validation_cache) --
# no repacking, no new tree generation, no GPU. Two phases in one job:
#   (1) analysis.regime_select's existing sweep() at maintenance_scale=0.0 fixed, over its own
#       already-tuned LAMBDA_GRID (22 log-spaced points, 5e-5..0.3) -- maps k* vs time_lambda on
#       THIS population (own data-derived ceiling, not assumed 96) using the exact same
#       "1 < k* < ceiling-1" auto-criterion REGIME used on our own corpus (regime_select.md).
#   (2) auto-selects a representative spread of meaningful points (continuity point 0.0005,
#       mid-band points, small-k* point if one exists, plus the known 0.01 boundary and a
#       fresh re-run of the nonzero-maintenance 0.001/0.001 point) and runs the full 3-way
#       paired-bootstrap significance test (analysis.ysagiv_sig) at each.
#
#   CONFIG=config_ysagiv_xaba.yaml sbatch slurm/pipeline/ysagiv_xaba_regime_sweep.slurm
#SBATCH --job-name=ysagiv-xaba-regime-sweep
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# Bumped qos=test/00:55:00 -> qos=short/02:00:00 (2026-07-08 evening, 20K->100K raw sample):
# the original 00:55:00 budget was measured at 9m19s actual for the 352-point grid + 6
# significance calls on the 13,140-tree xaba population (job 10846482) -- comfortable then,
# but a straight 5x data-volume scale-up projects to ~45-50min, too close to that cap for an
# unattended overnight run. Della auto-routes QOS by --time, not the --qos flag itself (see
# della-qos-time-routing), so pushing over 1h here also gets short's better priority for free.
# Original reasoning retained below: same smoke-scale-relative population (n_eval=628/n_fit=1465,
# ~2093 validation episodes total at 20K) and same class of job (regime_select's grid sweep is
# "pure arithmetic after one data load, runs in seconds" per its own docstring; ysagiv_sig's 3-regime
# job finished in a couple minutes total per Agent 1's session) as ysagiv_eval_multi.slurm,
# which this script directly copies the qos/time reasoning from -- just ~7-8 ysagiv_sig calls
# instead of 3, still comfortably under the 1h test cap. No GPU anywhere in this chain.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

field() { $RENDER --get "eval.$1"; }
PACKED=$(field packed_root)
MATVAL=$(field materialized_validation_cache)
OUT=$(field out_dir)
MAX_EPISODES=$(field max_episodes)
SWEEPDIR="$OUT/regime_sweep"
mkdir -p "$SWEEPDIR"

echo "ysagiv_xaba_regime_sweep host=$(hostname) config=$CONFIG"; date
echo "PACKED=$PACKED MATVAL=$MATVAL OUT=$OUT MAX_EPISODES=$MAX_EPISODES"

echo "=== phase 1: k*-vs-time_lambda sweep (maintenance_scale=0.0 slice of REGIME's own grid) ==="
python -m analysis.regime_select \
  --packed-root "$PACKED" --out-dir "$SWEEPDIR" \
  --results-json "$SWEEPDIR/regime_select_results.json" \
  --max-episodes "$MAX_EPISODES"

echo "=== phase 2: auto-select representative meaningful points + write chosen_points.txt ==="
python3 - "$SWEEPDIR/regime_select_results.json" "$SWEEPDIR/chosen_points.txt" <<'PY'
import json, sys
import numpy as np

results_path, chosen_path = sys.argv[1], sys.argv[2]
rows = json.load(open(results_path))
m0 = sorted([r for r in rows if r["maintenance_scale"] == 0.0], key=lambda r: r["time_lambda"])
ceiling = m0[0]["ceiling"]
assert all(r["ceiling"] == ceiling for r in m0), "ceiling should be data-derived & regime-independent"

print(f"[phase2] maintenance_scale=0.0 slice, ceiling={ceiling} (data-derived, {len(m0)} lambda points)")
for r in m0:
    flag = "MEANINGFUL" if r["meaningful"] else "degenerate"
    print(f"  lambda={r['time_lambda']:<8g} k*={r['k_star']:>3d}  k*/ceiling={r['k_frac']:.3f}  {flag}")

meaningful = [r for r in m0 if r["meaningful"]]
if not meaningful:
    raise SystemExit("[phase2] no meaningful m=0 points found -- aborting point selection")

chosen: list[dict] = []
def add(r):
    if r is not None and r["time_lambda"] not in [c["time_lambda"] for c in chosen]:
        chosen.append(r)

# (a) continuity point: the already-known/tested near-ceiling regime, 0.0005
add(next((r for r in m0 if abs(r["time_lambda"] - 0.0005) < 1e-12), None))

# (b) known boundary point already in ysagiv_trees.md's table: 0.01, m=0 (k*=3 there)
add(next((r for r in m0 if abs(r["time_lambda"] - 0.01) < 1e-12), None))

# (c) 3 mid-band points spread across the meaningful k*/ceiling range, away from both extremes
kfracs = np.array([r["k_frac"] for r in meaningful])
for target in (0.5, 0.3, 0.15):
    idx = int(np.argmin(np.abs(kfracs - target)))
    add(meaningful[idx])

# (d) symmetric small-k* (near-1, cheap-to-stop / expensive-to-continue) point, if distinct
add(min(meaningful, key=lambda r: r["k_star"]))

chosen = sorted(chosen, key=lambda r: r["time_lambda"])
print(f"[phase2] chosen {len(chosen)} points (incl. continuity/boundary):")
with open(chosen_path, "w") as f:
    for r in chosen:
        tag = "xswp_lambda" + f"{r['time_lambda']:g}".replace(".", "p").replace("-", "m") + "_m0"
        print(f"  {tag}: lambda={r['time_lambda']:g} k*={r['k_star']} k_frac={r['k_frac']:.3f}")
        f.write(f"{r['time_lambda']:g}:0.0:{tag}\n")
PY

echo "=== phase 3: full 3-way paired significance test at each chosen point ==="
while IFS=':' read -r LAM MSCALE TAG; do
  [ -z "${LAM:-}" ] && continue
  echo "--- ysagiv_sig @ time_lambda=$LAM maintenance_scale=$MSCALE tag=$TAG ---"
  python -m analysis.ysagiv_sig \
    --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" --tag "$TAG" \
    --time-mode linear --time-lambda "$LAM" \
    --maintenance-scale "$MSCALE" --maintenance-exponent 1.0 \
    --max-episodes "$MAX_EPISODES"
done < "$SWEEPDIR/chosen_points.txt"

echo "=== phase 3b: fresh re-run of the known nonzero-maintenance point (lambda=0.001, m=0.001) for consistency ==="
python -m analysis.ysagiv_sig \
  --packed-root "$PACKED" --cache "$MATVAL" --out-dir "$OUT" --tag "xswp_lambda0p001_m0p001" \
  --time-mode linear --time-lambda 0.001 \
  --maintenance-scale 0.001 --maintenance-exponent 1.0 \
  --max-episodes "$MAX_EPISODES"

echo "YSAGIV XABA REGIME SWEEP DONE -> $OUT (sweep table: $SWEEPDIR/regime_select_results.json, chosen points: $SWEEPDIR/chosen_points.txt, sig results: $OUT/ysagiv_sig_xswp_*.json)"; date
