#!/bin/bash
# Pipeline step — pack_trees_history (CPU). "History" analog of pack_trees.slurm for the
# frozen-value-fix branch (see history.md, repo root). Does NOT modify pack_trees.slurm or
# anything it drives -- that script still runs config_ysagiv_xaba20k.yaml's OLD gnn_pack/mc_pack
# stages unmodified, exactly as before. This script drives config_ysagiv_xaba20k_history.yaml's
# (or any other *_history.yaml's) three packing sub-stages instead:
#   split              -> cts.data.preprocess_gnn.split      (SHARED with the old config --
#                          same seed, same trees, same train/validation assignment; harmless
#                          to re-run, but typically already exists from the old config's run)
#   mc_pack            -> cts.data.preprocess_mc.pack        (packhistory_trees -- Task 1's
#                          in-place rewrite of the SAME module the old config also uses,
#                          `--stage mc_pack` unchanged; the new per-step-replay logic there is
#                          reached the exact same way it always was, not a new module)
#   gnn_pack_history   -> cts.data.preprocess_gnn.pack_history (packhistory_GNNpretrain --
#                          Task 4's WHOLLY NEW module, not a modification of the old
#                          preprocess_gnn.pack, so it needs its own module path and its own
#                          --stage name, gnn_pack_history, matching config_ysagiv_xaba20k_history
#                          .yaml's `gnn_pack_history:` section, not the old `gnn_pack:`.)
#   sbatch --dependency=afterok:<gen_trees_jobid> slurm/pipeline/pack_trees_history.slurm
#SBATCH --job-name=pack-trees-history
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
# Same time budget as pack_trees.slurm (~20min est. at 20K-tree scale) -- bump if driven
# against a much larger corpus.

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
NW=${SLURM_CPUS_PER_TASK:-16}

echo "pack_trees_history host=$(hostname) cpus=$NW"; date

echo "=== split (train/val partition; all trees -- shared with the old config's run) ==="
python -m cts.data.preprocess_gnn.split --config "$CONFIG" --stage split

echo "=== mc_pack (packhistory_trees: per-step backprop replay + sparse update log) ==="
python -m cts.data.preprocess_mc.pack --config "$CONFIG" --stage mc_pack --set num_workers="$NW"

echo "=== gnn_pack_history (packhistory_GNNpretrain: T_n -> T_n+k snapshot pairs) ==="
python -m cts.data.preprocess_gnn.pack_history --config "$CONFIG" --stage gnn_pack_history

echo "PACK_HISTORY DONE"; date
