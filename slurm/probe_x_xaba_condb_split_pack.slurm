#!/bin/bash
# Probe X, condition B (exclude_xaba ALONE, no argmax pre-filter). Runs split (random 30K
# tractability subsample, NOT a quality filter -- see config_probe_x_xaba_condb.yaml header) then
# mc_pack (exclude_xaba=true, the only real filter). No gnn_pack/encoder step -- reuses the
# already-trained frozen encoder (exclude_xaba is mc_pack-only, confirmed in pack.py).
#   sbatch slurm/probe_x_xaba_condb_split_pack.slurm
#SBATCH --job-name=probe-x-xaba-condb-pack
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:55:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
export CONFIG="$TINY/config_probe_x_xaba_condb.yaml"
source "$TINY/slurm/helpers/setup_env.sh"
NW=${SLURM_CPUS_PER_TASK:-16}

echo "probe_x_xaba_condb_split_pack host=$(hostname) cpus=$NW"; date

echo "=== split (random 30K subsample, no argmax filter) ==="
python -m cts.data.preprocess_gnn.split --config "$CONFIG" --stage split

echo "=== mc_pack (exclude_xaba=true, ONLY filter) ==="
python -m cts.data.preprocess_mc.pack --config "$CONFIG" --stage mc_pack --set num_workers="$NW"

echo "PROBE X XABA CONDB SPLIT+PACK DONE"; date
