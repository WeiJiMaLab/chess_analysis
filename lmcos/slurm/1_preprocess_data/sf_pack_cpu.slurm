#!/bin/bash
# P2 Stockfish ladder — CPU packing stages for one Elo rung.
#   stage 3: gnn split            (cts.data.preprocess_gnn.split)
#   stage 4: gnn pack             (cts.data.preprocess_gnn.pack)   -> encoder train examples
#   stage 6: mc pack (oracle DP)  (cts.data.preprocess_mc.pack, num_workers = cpus)
# Stages 5 (encoder) + 7 (materialize) are GPU and live in sf_pack_gpu.slurm,
# submitted with an afterok dependency on this job.
#
# Submit (per rung, after tree-gen):
#   ELO=1800 sbatch --dependency=afterok:<treegen_jobid> slurm/1_preprocess_data/sf_pack_cpu.slurm
#SBATCH --job-name=sf-pack-cpu
#SBATCH --output=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/GRIFFITHS/hl4291/sf_pack_logs/%x_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
: "${ELO:?ELO env var required (e.g. 1800)}"

ROOT=/scratch/gpfs/GRIFFITHS/hl4291
PROJECT_DIR=/home/hl4291/chess_analysis/lmcos
VENV_DIR=/home/hl4291/venv

TREES=$ROOT/sf_trees/elo${ELO}
SPLIT=$ROOT/sf_split/elo${ELO}
GNN=$ROOT/sf_packed/elo${ELO}
MCP=$ROOT/sf_mc_packed/elo${ELO}
CFG=$ROOT/sf_pack_configs/elo${ELO}
mkdir -p "$ROOT/sf_pack_logs" "$CFG"

set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
source "$VENV_DIR/bin/activate"
set -u

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
export PYTHONUNBUFFERED=1
NW=${SLURM_CPUS_PER_TASK:-32}

echo "elo=$ELO host=$(hostname) cpus=$NW start=$(date)"
echo "trees=$(ls "$TREES" | wc -l) under $TREES"

# ── stage 3: split (80/20, seed 0) ────────────────────────────────────────────
cat > "$CFG/split.yaml" <<YAML
source_root: $TREES
split_root: $SPLIT
validation_fraction: 0.2
seed: 0
clear: true
YAML
echo "=== stage 3 split ==="; python -m cts.data.preprocess_gnn.split --config "$CFG/split.yaml"

# ── stage 4: gnn pack (encoder examples) ──────────────────────────────────────
cat > "$CFG/gnnpack.yaml" <<YAML
split_root: $SPLIT
output_root: $GNN
shard_size: 2000
clear: true
YAML
echo "=== stage 4 gnn pack ==="; python -m cts.data.preprocess_gnn.pack --config "$CFG/gnnpack.yaml"

# ── stage 6: mc pack (budgeted oracle DP; canonical cost model) ────────────────
cat > "$CFG/mcpack.yaml" <<YAML
split_root: $SPLIT
output_root: $MCP
shard_size: 500
num_workers: $NW
reward_scale: 1.0
min_halt_reward_range: 0.05
min_decision_margin: 0.0
search_budget: 96
max_depth: 4
maintenance_scale: 0.0
maintenance_ref_nodes: 30.0
maintenance_exponent: 1.1
time_lambda: 18.537
time_p: 2.8
time_tau: 2.5
time_delta: 1
timeout_value: -1.0
time_mode: power_law
samples_per_bucket: 2
seed: 0
clear: true
YAML
echo "=== stage 6 mc pack ==="; python -m cts.data.preprocess_mc.pack --config "$CFG/mcpack.yaml"

echo "CPU PACK DONE elo=$ELO $(date)"
