#!/usr/bin/env bash
# Copy Yotam's Slurm logs for May 2026 controller comparison runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${SCRIPT_DIR}/logs"
SOURCE="${SOURCE:-/home/ysagiv/chess/cts/async_soph/audit_outputs}"

mkdir -p "${LOGS_DIR}"

for name in \
  cts-fittedq_8527146.out \
  cts-fittedq_8527147.out \
  cts-fittedq_8533204.out \
  cts-fittedq_8550917.out
do
  src="${SOURCE}/${name}"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${LOGS_DIR}/"
    echo "copied ${name}"
  else
    echo "[skip] not found: ${src}"
  fi
done

echo "logs: ${LOGS_DIR}"
ls -lh "${LOGS_DIR}"/*.out 2>/dev/null || true
