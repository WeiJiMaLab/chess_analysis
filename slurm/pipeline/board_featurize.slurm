#!/bin/bash
# Board precondition — featurize the legal-move covariates (captures/checks) over
# filtered_moves' distinct FENs. Array: each task featurizes one hash-partition of
# the FENs → a shard (board_merge joins them). Depends on filter_and_sample.
#   sbatch --array=0-15 slurm/pipeline/board_featurize.slurm
#SBATCH --job-name=board-featurize
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "$TINY"
source "$TINY/slurm/helpers/setup_env.sh"
echo "board featurize task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} on $(hostname) config=$CONFIG"; date
python -m analysis.board --featurize
echo "Finished at $(date)"
