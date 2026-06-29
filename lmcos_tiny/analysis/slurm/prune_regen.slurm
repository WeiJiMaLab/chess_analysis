#!/bin/bash
# R-PRUNING step 2: regenerate trees at max_depth=36 (the correct horizon — depth-4 capped
# the freed budget), WITH value-pruning in the PUCT loop (depth>=1, relative-to-best by
# leaf-eval value). Same SF-2000 engine, n=1 leaf eval, same 250k roots. EPS=none = the
# unpruned md36 baseline (regenerated because md36 != md4 for some positions). Per-eps dir.
#   for E in none 0.05 0.1 0.3; do
#     sbatch --array=0-99 --export=ALL,EPS=$E lmcos_tiny/analysis/slurm/prune_regen.slurm   # full 250k
#   done
#SBATCH --job-name=prune-regen
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
export ELO=2000
EPS="${EPS:?set EPS=none|0.05|0.1|0.3 via --export=ALL,EPS=...}"
N="${N:-1}"          # leaf-eval node budget (1 or 100)
LANE_END="${LANE_END:-250000}"
TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh"

PRUNE_OVERRIDE=()
if [ "$EPS" = "none" ]; then
  OUTDIR="/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/n${N}md36"
else
  OUTDIR="/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/n${N}md36_eps${EPS}"
  PRUNE_OVERRIDE=(--override prune_epsilon=${EPS})
fi

SHARD_SIZE=2500; BASE_START=0
start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE )); (( end > LANE_END )) && end=${LANE_END}
if (( start >= LANE_END )); then echo "no-op: start=${start} >= LANE_END"; exit 0; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "PRUNE-REGEN N=${N} eps=${EPS} md=36 task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} out=${OUTDIR} host=$(hostname)"; date
python -m cts.data.build_tree --config "$CONFIG" --stage treegen --set globals.sf_elo=2000 \
    --override sf_search_limit_nodes=${N} --override max_depth=36 "${PRUNE_OVERRIDE[@]}" \
    --override output_dir="${OUTDIR}" \
    --override start_index=${start} --override end_index=${end}
echo "PRUNE-REGEN eps=${EPS} DONE start=${start}"; date
