#!/bin/bash
# Pipeline step — filter_trees. Select the root FENs to generate trees from.
# In this version that is simply a uniform random subsample of the ground-truth
# FEN pool (globals.fens_pool) down to N_FENS roots, written to treegen.fens
# (the file gen_trees reads). In future this is where richer position filtering
# would live. Depends on preprocess (the FEN pool comes from the move data).
#   N_FENS=250000 sbatch slurm/pipeline/filter_trees.slurm
#SBATCH --job-name=filter-trees
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
cd "${TINY}"
source "${TINY}/slurm/helpers/setup_env.sh"

N_FENS="${N_FENS:-250000}"
SRC=$($RENDER --get globals.fens_pool)
DST=$($RENDER --get treegen.fens)

echo "filter_trees: subsample ${N_FENS} of $(wc -l < "$SRC") FENs from $SRC -> $DST on $(hostname) at $(date)"
shuf -n "${N_FENS}" "$SRC" > "$DST"
echo "wrote $(wc -l < "$DST") FENs to $DST at $(date)"
