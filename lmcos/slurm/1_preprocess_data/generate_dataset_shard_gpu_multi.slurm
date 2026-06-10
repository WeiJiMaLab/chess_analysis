#!/bin/bash
# U1.1 multi-GPU lane: ONE job that grabs N GPUs and runs N build_tree workers,
# each pinned to its own GPU (CUDA_VISIBLE_DEVICES) on a disjoint FEN index range.
# This sidesteps the ~3-job-per-user cap (we're job-capped, not GPU-capped): a
# single gpu:N job uses N GPUs but counts as one job. Separate GPUs => additive,
# no contention. Writes into the shared human_trees_50k/ (index-keyed; disjoint
# ranges never collide; resume:true).
#SBATCH --job-name=gen50k-gpu4
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --gres=gpu:4
#SBATCH --constraint=nomig
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -uo pipefail
: "${CONFIG:?CONFIG (base YAML) required}"
: "${BASE_START:?BASE_START required}"
: "${SHARD_PER_WORKER:?SHARD_PER_WORKER required}"
: "${LANE_END:?LANE_END required}"
NGPU="${NGPU:-4}"
PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
VENV_DIR="${VENV_DIR:-/home/hl4291/venv}"

mkdir -p "${PROJECT_DIR}/slurm/logs"
set +u
module --force purge >/dev/null 2>&1 || true
module load anaconda3/2023.3
module load cudatoolkit/12.8
source "${VENV_DIR}/bin/activate"
set -u
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true
date

pids=()
for i in $(seq 0 $((NGPU-1))); do
  s=$(( BASE_START + i*SHARD_PER_WORKER ))
  e=$(( s + SHARD_PER_WORKER )); [ "$e" -gt "$LANE_END" ] && e=$LANE_END
  if [ "$s" -ge "$LANE_END" ]; then continue; fi
  echo "worker ${i}: CUDA_VISIBLE_DEVICES=${i} indices [${s},${e})"
  CUDA_VISIBLE_DEVICES=${i} python3 -m cts.data.build_tree --config "${CONFIG}" \
      --override start_index=${s} --override end_index=${e} \
      > "slurm/logs/${SLURM_JOB_ID}_w${i}.out" 2>&1 &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
date
echo "all ${#pids[@]} workers done rc=${rc}"
exit ${rc}
