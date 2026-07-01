#!/bin/bash
# Analysis pipeline step 1b — generate the SF-1 value trees engine.py reads.
# Drives the `treegen` stage of config.yaml over a contiguous FEN-index range per
# array task. Default knobs (MODE=none, N=1, max_depth=36) reproduce the canonical
# n1md36 set: Stockfish, n=1 leaf eval, depth-36, no PUCT-loop pruning, over the
# 250k FEN sample at treegen.fens (lmcos/fens_250k.txt). Output dir is
# sf_trees/n1md36 — exactly human_analysis.trees_default, so step 2 (engine.py)
# reads it with no extra wiring.
#   sbatch --array=0-99 slurm/pipeline/gen_trees.slurm           # full 250k, n1md36
# Optional knobs (kept from the pruning study; not needed for the baseline):
#   MODE=relative|absolute|rank PARAM=<eps>   -> writes a pruned sibling dir
#   N=100                                     -> 100-node leaf eval (n100md36)
#SBATCH --job-name=gen-trees
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
MODE="${MODE:-none}"       # none (baseline n1md36) | relative (eps) | absolute (win-prob floor) | rank (top-k)
PARAM="${PARAM:-}"         # the prune knob (ignored for MODE=none)
N="${N:-1}"                # leaf-eval node budget (1 or 100)
LANE_END="${LANE_END:-250000}"
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

# Output dir is wired from config: the baseline (MODE=none, N=1) writes EXACTLY to
# human_analysis.trees_default, the same key engine.py (step 2) reads — so gen-write
# and analysis-read can never drift. Pruning siblings / other N hang off the same parent.
TREES_DEFAULT=$($RENDER --get human_analysis.trees_default)   # e.g. .../sf_trees/n1md36
TREES_PARENT=$(dirname "$TREES_DEFAULT")                      # e.g. .../sf_trees
PRUNE_OVERRIDE=()
if [ "$MODE" = "none" ] && [ "$N" = "1" ]; then
  OUTDIR="$TREES_DEFAULT"
elif [ "$MODE" = "none" ]; then
  OUTDIR="$TREES_PARENT/n${N}md36"
else
  case "$MODE" in relative) TAG=rel;; absolute) TAG=abs;; rank) TAG=rank;; *) echo "bad MODE"; exit 1;; esac
  OUTDIR="$TREES_PARENT/n${N}md36_${TAG}${PARAM}"
  PRUNE_OVERRIDE=(--override prune_mode=${MODE} --override prune_epsilon=${PARAM})
fi

SHARD_SIZE=2500; BASE_START=0
start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE )); (( end > LANE_END )) && end=${LANE_END}
if (( start >= LANE_END )); then echo "no-op: start=${start} >= LANE_END"; exit 0; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "GEN-TREES N=${N} MODE=${MODE} PARAM=${PARAM} md=36 task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} out=${OUTDIR} host=$(hostname)"; date
python -m cts.data.build_tree --config "$CONFIG" --stage treegen \
    --override sf_search_limit_nodes=${N} --override max_depth=36 "${PRUNE_OVERRIDE[@]}" \
    --override output_dir="${OUTDIR}" \
    --override start_index=${start} --override end_index=${end}
echo "GEN-TREES MODE=${MODE} PARAM=${PARAM} DONE start=${start}"; date
