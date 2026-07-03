#!/bin/bash
# z_t tree-stat decodability probe: can [height, width, n_nodes] be linearly / MLP-
# decoded from the encoder ROOT EMBEDDING z_t (d_embed=32)? The MCHalt controller reads a
# PURE z_t head (controller_inputs=['z_t','T_t']); the StatsReadout baseline reads the raw
# stats. If z_t can't recover them, the pure-z_t head is missing info the baseline has.
# Reads paths from the config `eval:`/`train:` sections; self-checks cache<->dataset
# alignment (packed n_nodes == cache N_t) before probing. Writes zt_decodability.{csv,png}
# to the run's normative figures dir. Probes the VALIDATION split only (shuffle=False cache).
#   sbatch slurm/pipeline/zt_probe.slurm
#SBATCH --job-name=zt-probe
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "zt_probe host=$(hostname) config=$CONFIG"; date
python -m cts.analysis.zt_probe --config "$CONFIG"
echo "ZT_PROBE DONE"; date
