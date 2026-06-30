#!/bin/bash
#SBATCH --job-name=ratio_test
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/hl4291/chess_analysis/lmcos_small/slurm/logs/ratio_test_%j.out
cd /home/hl4291/chess_analysis
/home/hl4291/venv/bin/python lmcos_small/analysis/ratio_test.py
