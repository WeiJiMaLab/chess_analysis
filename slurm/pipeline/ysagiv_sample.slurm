#!/bin/bash
# Regenerates the seeded raw sample of ysagiv's `human_trees` corpus that every other ysagiv
# pipeline stage (ysagiv_xaba_filter.slurm -> split -> gnn_pack/mc_pack -> ...) reads from.
# This step previously existed only as an uncommitted one-off command (see
# outputs/reports/ysagiv.md Methods) -- committed here so the full chain is reproducible from
# nothing but the permanent, read-only source corpus. N bumped 20,000->100,000 (2026-07-08
# evening): still well under 10% of the 1,078,585-tree source corpus.
#   sbatch slurm/pipeline/ysagiv_sample.slurm
#SBATCH --job-name=ysagiv-sample
#SBATCH --output=/home/hl4291/chess_analysis/slurm/logs/%x_%j.out
#SBATCH --error=/home/hl4291/chess_analysis/slurm/logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
TINY=/home/hl4291/chess_analysis
source "$TINY/slurm/helpers/setup_env.sh"

SRC=/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees   # read-only, never written
OUT=/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/raw_sample
SEED=42
N=${N:-100000}   # override at submission time (e.g. `N=20000 sbatch ...`) for a cheap dry run;
                 # committed default is 100000 (still <10% of the 1,078,585-tree source corpus)

echo "ysagiv_sample src=$SRC out=$OUT seed=$SEED n=$N host=$(hostname)"; date
rm -rf "$OUT"   # clear-and-redo (matches split/gnn_pack/mc_pack's clear:true convention) --
                 # a stale smaller/larger sample left over from a prior N would otherwise silently
                 # mix into this run's population instead of being replaced by it
mkdir -p "$OUT"

python3 - "$SRC" "$OUT" "$SEED" "$N" <<'PY'
import os, random, sys
src, out, seed, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
names = os.listdir(src)
print(f"[sample] {len(names)} files in source corpus")
random.seed(seed)
chosen = random.sample(names, n)
made = 0
for name in chosen:
    link = os.path.join(out, name)
    if not os.path.lexists(link):
        os.symlink(os.path.join(src, name), link)
        made += 1
print(f"[sample] wrote {made} new symlinks ({len(chosen) - made} already present) -> {out}")
PY

echo "YSAGIV SAMPLE DONE -> $OUT"; date
