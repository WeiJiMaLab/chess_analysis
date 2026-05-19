# Shared environment for hl4291 della jobs (source from other scripts in this directory).

hl4291_slurm_setup() {
  PROJECT_DIR="${PROJECT_DIR:-/home/hl4291/chess_analysis/lmcos}"
  VENV_ACTIVATE="${VENV_ACTIVATE:-/home/hl4291/venv/bin/activate}"

  mkdir -p "${PROJECT_DIR}/logs"

  if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "venv activate not found: ${VENV_ACTIVATE} — set VENV_ACTIVATE"
    exit 1
  fi

  set +u
  source "${VENV_ACTIVATE}"
  set -u

  cd "${PROJECT_DIR}"
  export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
}

hl4291_slurm_setup_gpu() {
  hl4291_slurm_setup
  module --force purge >/dev/null 2>&1 || true
  module load cudatoolkit/12.8
  export PYTHONUNBUFFERED=1
}
