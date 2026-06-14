"""Best-effort GPU-utilization sampling via ``nvidia-smi``.

WHY here: the throughput smoke test wants to corroborate the "the accelerator
is ~85% idle at batch 1, saturated at large batch" story from report §1.2, but
must not *require* a GPU or nvidia-smi to run (the CPU lane has neither). So
this returns ``None`` cleanly when nvidia-smi is absent or fails.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def sample_gpu_utilization_percent() -> Optional[float]:
    """Return current GPU utilization (%) from nvidia-smi, or None if unavailable.

    Averages across visible GPUs. A single instantaneous sample is enough for a
    smoke summary; callers that want a steady-state number should sample while
    the batched eval is in flight.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    values = [line.strip() for line in output.stdout.splitlines() if line.strip()]
    if not values:
        return None
    try:
        numeric = [float(value) for value in values]
    except ValueError:
        return None
    return sum(numeric) / len(numeric)
