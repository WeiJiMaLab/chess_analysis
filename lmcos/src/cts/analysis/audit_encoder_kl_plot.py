"""Render the encoder KL audit heatmap from a JSON payload.

The audit (``cts.analysis.audit_encoder_kl``) writes its numerical results
to JSON before attempting the heatmap render, so if matplotlib is broken
on the cluster (libstdc++ ABI mismatches are common) the audit still
exits with the data on disk. This companion module reads the JSON and
renders the PDF separately — on a node where matplotlib works, or
locally after an rsync.

The JSON payload includes everything the heatmap needs (per-bucket mean
KL, edge counts, size + depth labels, overall mean KL), so no encoder
or dataset is required at render time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from cts.analysis.audit_encoder_kl import _save_heatmap


class AuditEncoderKLPlotConfig(BaseModel):
    """CLI config for the standalone heatmap renderer."""

    model_config = ConfigDict(extra="forbid")

    input_json: str  # JSON payload produced by cts.analysis.audit_encoder_kl
    output_pdf: str  # destination PDF path; parent dirs created on demand
    title_suffix: Optional[str] = None  # overrides the default "— <manifest>.txt" suffix


def main(config: AuditEncoderKLPlotConfig) -> None:
    """CLI entry point: read JSON, render PDF, write."""
    payload = json.loads(Path(config.input_json).read_text())
    if config.title_suffix is not None:
        title_suffix = config.title_suffix
    else:
        manifest_name = Path(payload.get("manifest_path", "")).name or "unknown"
        title_suffix = f" — {manifest_name}"

    output_pdf = Path(config.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    _save_heatmap(
        str(output_pdf),
        mean_kl_grid=payload["mean_kl_grid"],
        edge_count_grid=payload["edge_count_grid"],
        size_bin_labels=payload["size_bin_labels"],
        depth_bin_labels=payload["depth_bin_labels"],
        overall_mean_kl=payload["overall_mean_kl"],
        overall_edge_count=payload["overall_edge_count"],
        title_suffix=title_suffix,
    )
    print(f"[audit-encoder-kl-plot] wrote {output_pdf}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(AuditEncoderKLPlotConfig, main)
