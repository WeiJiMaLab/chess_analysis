#!/bin/bash
#SBATCH --job-name=gap_rec
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/gap_reconcile_%j.out
cd /home/hl4291/chess_analysis
/home/hl4291/venv/bin/python lmcos_small/analysis/gap_reconcile.py
