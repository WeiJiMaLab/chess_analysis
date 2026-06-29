#!/bin/bash
#SBATCH --job-name=voc_signals
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_tiny/analysis/logs/voc_signals.%j.out

cd /home/hl4291/chess_analysis
set -euo pipefail

# Merge the filter shards (1b writes shards/clean_trees_shard_0..9.txt; the 1c merge
# step is not run here) into the canonical clean_trees.txt before computing — so this
# job always reads the FULL filtered set, not a stale clean_trees.txt.
ELO="${VOC_ELO:-2000}"
FDIR="/scratch/gpfs/GRIFFITHS/hl4291/sf_filtered/elo${ELO}"
if ls "${FDIR}"/shards/clean_trees_shard_0.txt >/dev/null 2>&1; then
  cat "${FDIR}"/shards/clean_trees_shard_{0,1,2,3,4,5,6,7,8,9}.txt > "${FDIR}/clean_trees.txt"
  echo "merged shards 0-9 -> ${FDIR}/clean_trees.txt ($(wc -l < ${FDIR}/clean_trees.txt) trees)"
fi

/home/hl4291/venv/bin/python lmcos_tiny/analysis/compute_voc_signals.py
