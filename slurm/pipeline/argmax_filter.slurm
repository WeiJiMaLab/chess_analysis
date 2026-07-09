#!/bin/bash
# Pipeline step — argmax_filter (CPU). "Thinking helps" filter, run after
# gen_trees and before pack_trees: keeps only trees where the TRUE oracle
# argmax stop step exceeds 2 (see cts.data.preprocess_mc.filter_argmax and
# labnotebook 2026-07-07 for why: SingleHalt* collapses to a near-zero stop
# on the unfiltered population, but this filter surfaces a genuinely
# non-trivial k*=7 on the 900-FEN pilot). Non-destructive: writes an
# include_list for split.py to read (config split.include_list); gen_trees's
# own output directory is untouched, so engine_analysis_pwin/cp still see
# the full, unfiltered population.
#   sbatch --dependency=afterok:<gen_trees_arrayjob> slurm/pipeline/argmax_filter.slurm
#SBATCH --job-name=argmax-filter
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
# ~0.016s/tree single-threaded (measured); 400K trees ~= 107 min -- generous
# margin above that for filesystem-listing overhead on a large tree directory.
# Single job, no array: at this scale it isn't worth the complexity.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

TREES_DIR=$($RENDER --get human_analysis.trees_default)
OUT=$($RENDER --get split.include_list)

echo "argmax_filter trees_dir=${TREES_DIR} -> ${OUT} host=$(hostname)"; date
python -m cts.data.preprocess_mc.filter_argmax --trees-dir "$TREES_DIR" --output "$OUT"
echo "ARGMAX FILTER DONE"; date
