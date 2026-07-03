#!/bin/bash
# AUX branch (NOT in the default DAG) — monotonicity / no-switchback training filter.
# Computes a per-tree switchback count from the BeFS greedy best-move trace
# (how often the running best root move changes as the tree grows), writes a
# keep-list of monotone/low-switchback FENs, and re-packs so the readout can be
# retrained on monotone-only positions. Run AFTER gen_trees, as an after-the-fact
# variant; default pipeline does not submit this.
#
# STATUS: STUB. Wiring exists (split.filter_trees_by_trace consumes an
# include_list). TODO before use:
#   1. compute switchbacks per tree from the stored trace (analysis.utils.tree_loader
#      already replays best_trace per tree — expose a --emit-switchbacks mode);
#   2. threshold -> keep-list parquet;
#   3. pack_trees --include-list <keeplist> -> retrain readout on the subset.
#SBATCH --job-name=aux-trace-filter
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "aux_trace_filter is a STUB — see header. Not implemented; exiting 0." >&2
exit 0
