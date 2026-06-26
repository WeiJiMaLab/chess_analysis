#!/bin/bash
#SBATCH --job-name=minimal_mc
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/logs/minimal_mc_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/logs/minimal_mc_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# NOTE: no `set -e` — inherited active venv makes a harmless 127 in module/activate
# abort before the first line (matches the working pack/materialize scripts).
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3 >/dev/null 2>&1 || true
source /home/hl4291/venv/bin/activate
cd /home/hl4291/chess_analysis
export PYTHONPATH=/home/hl4291/chess_analysis/lmcos/src
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date)"
python minimal_mc.py --config minimal_mc.yaml
echo "done rc=$? $(date)"
