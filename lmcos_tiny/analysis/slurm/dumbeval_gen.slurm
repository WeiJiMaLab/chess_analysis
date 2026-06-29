#!/bin/bash
# OVERNIGHT: dumber-eval tree generation — sf_search_limit_nodes 100 -> 1 (near-static,
# noisy/myopic per-node values), same SF-2000 engine + same 250k roots, into a SEPARATE
# dir (elo2000_n1) so it does NOT clobber elo2000. Tests H1/H3 (generative-search thesis):
# does a dumber search's tree-stats recover the +0.31 legal-moves/RT effect?
#   sbatch --array=0-99 lmcos_tiny/analysis/slurm/dumbeval_gen.slurm
#SBATCH --job-name=dumbeval-gen
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
export ELO=2000
TINY=/home/hl4291/chess_analysis/lmcos_tiny
source "$TINY/pipeline/helpers/setup_env.sh"

SHARD_SIZE=2500; BASE_START=0; LANE_END=250000
start=$(( BASE_START + SLURM_ARRAY_TASK_ID * SHARD_SIZE ))
end=$(( start + SHARD_SIZE )); (( end > LANE_END )) && end=${LANE_END}
if (( start >= LANE_END )); then echo "no-op: start=${start} >= LANE_END"; exit 0; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "DUMBEVAL(n1) task=${SLURM_ARRAY_TASK_ID} start=${start} end=${end} host=$(hostname)"; date
python -m cts.data.build_tree --config "$CONFIG" --stage treegen --set globals.sf_elo=2000 \
    --override sf_search_limit_nodes=1 \
    --override output_dir=/scratch/gpfs/GRIFFITHS/hl4291/sf_trees/elo2000_n1 \
    --override start_index=${start} --override end_index=${end}
echo "DUMBEVAL(n1) DONE start=${start}"; date
