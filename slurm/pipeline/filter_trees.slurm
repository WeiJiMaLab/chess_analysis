#!/bin/bash
# Pipeline step — filter_trees. Select the in-window root FENs to generate trees
# from: the distinct FENs whose move_ply is in [min_ply, max_ply] (from the run
# config), uniformly subsampled to N_FENS, written to treegen.fens. DB-sourced
# (ply lives only in the preprocessed DB — the raw fens_pool is truncated).
# Depends on preprocess (the move DB). Everything downstream applies the SAME
# window on read, so the population stays consistent.
#   N_FENS=250000 sbatch slurm/pipeline/filter_trees.slurm
#SBATCH --job-name=filter-trees
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"
source "${TINY}/slurm/helpers/setup_env.sh"

N_FENS="${N_FENS:-250000}"
DB=$($RENDER --get human_analysis.selected_db_default)
OUT=$($RENDER --get treegen.fens)
MINPLY=$($RENDER --get globals.min_ply)
MAXPLY=$($RENDER --get globals.max_ply)

echo "filter_trees: db=$DB out=$OUT ply=[$MINPLY,$MAXPLY] n=$N_FENS on $(hostname) at $(date)"
python -m analysis.filter_trees \
    --db "$DB" --out "$OUT" \
    --min-ply "$MINPLY" --max-ply "$MAXPLY" --n "$N_FENS"
echo "Finished at $(date)"
