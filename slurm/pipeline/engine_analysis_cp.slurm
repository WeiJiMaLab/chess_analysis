#!/bin/bash
# Analysis pipeline step — engine-level RT analysis in NATIVE CENTIPAWNS.
# Identical to engine_analysis_pwin.slurm except UNIT=cp (signals derived from
# the per-node cp_order feature; figures to figures_dir/engine_cp/).
#   sbatch --dependency=afterok:<gen_trees_arrayjob> slurm/pipeline/engine_analysis_cp.slurm
#SBATCH --job-name=engine-cp
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00

export UNIT=cp
exec bash /home/hl4291/chess_analysis/slurm/pipeline/engine_analysis_pwin.slurm
