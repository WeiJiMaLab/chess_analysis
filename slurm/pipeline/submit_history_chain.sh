#!/bin/bash
# Submit the full 5-stage frozen-value-fix ("_history") pipeline for one CONFIG,
# dependency-chained end to end via --dependency=afterok, in one call. See
# history.md (repo root) for the plan this pipeline implements.
#
# Replaces manually typing out 5-6 separate sbatch calls each time (as was done
# for config_ysagiv_xaba20k_history.yaml and
# config_ysagiv_xaba100k_minply15_maxply75_history.yaml) -- eval.slurm is now
# included by default, not a separate follow-up call.
#
# train_readout_pg.slurm is DELIBERATELY NOT part of this chain (removed
# 2026-07-11, was job5 of a 6-stage chain): its only output is
# materialize_history/mchalt_controller.pt, and nothing downstream of it --
# including eval.slurm -- reads that checkpoint (grep-confirmed against
# src/analysis/evaluate.py: zero references to `checkpoint`/`mchalt_controller`
# anywhere). eval.slurm's own `analysis.evaluate --which all` already PG-fits a
# z_t-based readout via the exact same routine (`fit_readout_pg`, imported
# directly from `cts.train.pg_controller_train` -- the same module
# train_readout_pg.slurm drives), then compares it against action-gap/stats/
# steps baselines -- a strictly more informative check than train_readout_pg's
# own stated role in history.md ("verification only... confirm no shape errors,
# no NaNs, sane advantage/loss values"). If a persisted, deployable
# `mchalt_controller.pt` is ever needed for something eval doesn't produce
# (e.g. actual downstream inference use, not just the research comparison),
# run `sbatch --dependency=afterok:<pack_root_merge_jobid> --export=ALL,
# CONFIG=<...> slurm/pipeline/train_readout_pg.slurm` standalone -- the script
# still exists, just no longer auto-chained here.
#
# Usage:
#   bash slurm/pipeline/submit_history_chain.sh <CONFIG.yaml> [--no-eval]
#
# Example:
#   bash slurm/pipeline/submit_history_chain.sh config_ysagiv_xaba100k_minply15_maxply75_history.yaml
#
# Time budgets below match what was actually used (and worked) for both
# xaba20k_history and xaba100k_minply15_maxply75_history runs -- bumped above
# each script's own #SBATCH default because real episode-count scaling per
# dataset is not knowable in advance from raw tree counts alone (see
# history.md's Wave-3 Progress Log for the reasoning). Adjust here if a much
# larger corpus is ever run through this same chain.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <CONFIG.yaml> [--no-eval]" >&2
  exit 1
fi
CONFIG="$1"
RUN_EVAL=1
[[ "${2:-}" == "--no-eval" ]] && RUN_EVAL=0

TINY=/home/hl4291/chess_analysis
cd "$TINY"
if [[ ! -f "$CONFIG" ]]; then
  # allow a bare filename relative to repo root, same convention as every other script here
  CONFIG="$TINY/$CONFIG"
fi
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

J1=$(sbatch --parsable --time=02:30:00 --export=ALL,CONFIG="$CONFIG" slurm/pipeline/pack_trees_history.slurm)
echo "job1 (pack_trees_history): $J1"

J2=$(sbatch --parsable --time=02:00:00 --dependency=afterok:$J1 --export=ALL,CONFIG="$CONFIG" slurm/pipeline/train_encoder.slurm)
echo "job2 (train_encoder): $J2"

J3=$(sbatch --parsable --array=0-39 --time=02:30:00 --dependency=afterok:$J2 --export=ALL,CONFIG="$CONFIG" slurm/pipeline/pack_root.slurm)
echo "job3 (pack_root array): $J3"

J4=$(sbatch --parsable --dependency=afterok:$J3 --export=ALL,CONFIG="$CONFIG" slurm/pipeline/pack_root_merge.slurm)
echo "job4 (pack_root_merge): $J4"

if [[ "$RUN_EVAL" == "1" ]]; then
  J5=$(sbatch --parsable --dependency=afterok:$J4 --export=ALL,CONFIG="$CONFIG" slurm/pipeline/eval.slurm)
  echo "job5 (eval): $J5"
else
  echo "eval.slurm skipped (--no-eval)"
fi

echo "$CONFIG" > /tmp/last_history_chain_config.txt
