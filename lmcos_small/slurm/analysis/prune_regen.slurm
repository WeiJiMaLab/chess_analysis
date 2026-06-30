#!/bin/bash
# R-PRUNING step 2: regenerate trees at max_depth=36 (the correct horizon — depth-4 capped
# the freed budget), WITH value-pruning in the PUCT loop (depth>=1, relative-to-best by
# leaf-eval value). Same SF-2000 engine, n=1 leaf eval, same 250k roots. EPS=none = the
# unpruned md36 baseline (regenerated because md36 != md4 for some positions). Per-eps dir.
#   for E in none 0.05 0.1 0.3; do
#     sbatch --array=0-19 --export=ALL,MODE=absolute,PARAM=0.5,LANE_END=50000 lmcos_small/slurm/analysis/prune_regen.slurm
#   done
#SBATCH --job-name=prune-regen
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
export ELO=2000
MODE="${MODE:-relative}"   # none | relative (eps) | absolute (win-prob floor) | rank (top-k)
PARAM="${PARAM:-}"         # the prune knob (ignored for MODE=none)
N="${N:-1}"                # leaf-eval node budget (1 or 100)
LANE_END="${LANE_END:-250000}"
TINY=/home/hl4291/chess_analysis/lmcos_small
source "$TINY/slurm/helpers/setup_env.sh"

PRUNE_OVERRIDE=()
if [ "$MODE" = "none" ]; then
  OUTDIR="/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/n${N}md36"
else
  case "$MODE" in relative) TAG=rel;; absolute) TAG=abs;; rank) TAG=rank;; *) echo "bad MODE"; exit 1;; esac
  OUTDIR="/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/n${N}md36_${TAG}${PARAM}"
  PRUNE_OVERRIDE=(--override prune_mode=${MODE} --override prune_epsilon=${PARAM})
fi

SHARD_SIZE=2500; BASE_START=0
start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE )); (( end > LANE_END )) && end=${LANE_END}
if (( start >= LANE_END )); then echo "no-op: start=${start} >= LANE_END"; exit 0; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "PRUNE-REGEN N=${N} MODE=${MODE} PARAM=${PARAM} md=36 task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} out=${OUTDIR} host=$(hostname)"; date
python -m cts.data.build_tree --config "$CONFIG" --stage treegen --set globals.sf_elo=2000 \
    --override sf_search_limit_nodes=${N} --override max_depth=36 "${PRUNE_OVERRIDE[@]}" \
    --override output_dir="${OUTDIR}" \
    --override start_index=${start} --override end_index=${end}
echo "PRUNE-REGEN MODE=${MODE} PARAM=${PARAM} DONE start=${start}"; date
