"""Yotam (ysagiv) pipeline outputs on della — inputs for ``chess_analysis/analysis/``.

Update paths here if scratch layout changes. Local analysis artifacts live under
``analysis/logs/`` and ``analysis/outputs/`` (see ``sync_logs.sh``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Roots -------------------------------------------------------------------

YSAGIV_SCRATCH = Path("/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS")
YSAGIV_TIGRESS = Path("/tigress/ysagiv/chess/cts")
YSAGIV_REPO = Path("/home/ysagiv/chess/cts/async_soph")

LMCOS_ROOT = Path(__file__).resolve().parents[1] / "lmcos"
ANALYSIS_ROOT = Path(__file__).resolve().parent

# --- Controller packed episodes (Stage 2b labels / oracle targets) -----------

CONTROLLER_PACKED = YSAGIV_SCRATCH / "data/controller_packed_combined_nomaint_no_xaba"
PACKED_TRAIN_MANIFEST = CONTROLLER_PACKED / "train_manifest.json"
PACKED_VALIDATION_MANIFEST = CONTROLLER_PACKED / "validation_manifest.json"
# Manifest totals (2026-05): ~576_830 train episodes, ~30_630 validation episodes.

# --- Encoder pretrain (Stage 1) ----------------------------------------------

PRETRAIN_PACKED_RERUN = YSAGIV_SCRATCH / "data/pretrain_packed_oracle96_trace_filtered_rerun"

ENCODER_RERUN = YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_rerun.pt"
ENCODER_RERUN_DECODER = YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_rerun_decoder.pt"

ENCODER_BUCKETED = YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_bucketed.pt"
ENCODER_BUCKETED_KL_JSONL = YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_bucketed_kl.jsonl"

ENCODER_SUBTREE_WEIGHTED = YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted.pt"
ENCODER_SUBTREE_WEIGHTED_DECODER = (
    YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted_decoder.pt"
)
ENCODER_SUBTREE_WEIGHTED_KL_JSONL = (
    YSAGIV_SCRATCH / "checkpoints/tree_encoder_child_wdl_async_k1_subtree_weighted_kl.jsonl"
)

# --- Materialized controller caches (Stage 2a) --------------------------------

@dataclass(frozen=True)
class MaterializedCaches:
    train_index: Path
    validation_index: Path
  # Shard payloads live beside the index (``.pt.d/`` dirs); indexes are small JSON-in-torch.

CACHE_SUBTREE_WEIGHTED_TRAIN = YSAGIV_SCRATCH / "data/train_cache_subtree_weighted.pt"
CACHE_SUBTREE_WEIGHTED_VAL = YSAGIV_SCRATCH / "data/validation_cache_subtree_weighted.pt"
CACHE_SUBTREE_WEIGHTED_TRAIN_SHARDS = YSAGIV_SCRATCH / "data/train_cache_subtree_weighted_shards"
CACHE_SUBTREE_WEIGHTED_VAL_SHARDS = YSAGIV_SCRATCH / "data/validation_cache_subtree_weighted_shards"

CACHE_RERUN_TRAIN = YSAGIV_TIGRESS / "train_cache_rerun.pt"
CACHE_RERUN_VAL = YSAGIV_TIGRESS / "validation_cache_rerun.pt"

CACHES_SUBTREE_WEIGHTED = MaterializedCaches(
    train_index=CACHE_SUBTREE_WEIGHTED_TRAIN,
    validation_index=CACHE_SUBTREE_WEIGHTED_VAL,
)
CACHES_RERUN = MaterializedCaches(
    train_index=CACHE_RERUN_TRAIN,
    validation_index=CACHE_RERUN_VAL,
)

# --- Trained controllers + diagnostics (Stage 2b outputs, May 2026) --------

@dataclass(frozen=True)
class ControllerRun:
    """One fitted-Q controller experiment."""

    label: str
    checkpoint: Path
    diagnostics_jsonl: Path | None
  # Slurm stdout with per-epoch ``greedy_epoch=`` lines (copy into analysis/logs/).
    slurm_log_name: str | None = None

    @property
    def slurm_log_local(self) -> Path | None:
        if self.slurm_log_name is None:
            return None
        return ANALYSIS_ROOT / "logs" / self.slurm_log_name

    @property
    def slurm_log_source(self) -> Path | None:
        if self.slurm_log_name is None:
            return None
        return YSAGIV_REPO / "audit_outputs" / self.slurm_log_name


CONTROLLER_SUBTREE_ROOT_BUDGET = ControllerRun(
    label="Subtree-weighted (root, budget)",
    checkpoint=YSAGIV_SCRATCH / "checkpoints/fittedq_subtree_weighted_zt_tt.pt",
    diagnostics_jsonl=YSAGIV_SCRATCH / "checkpoints/fittedq_subtree_weighted_zt_tt_diagnostics.jsonl",
    slurm_log_name="cts-fittedq_8527146.out",
)

CONTROLLER_SUBTREE_ROOT = ControllerRun(
    label="Subtree-weighted (root)",
    checkpoint=YSAGIV_SCRATCH / "checkpoints/fittedq_subtree_weighted_zt_only.pt",
    diagnostics_jsonl=YSAGIV_SCRATCH / "checkpoints/fittedq_subtree_weighted_zt_only_diagnostics.jsonl",
    slurm_log_name="cts-fittedq_8527147.out",
)

CONTROLLER_RERUN_ROOT_BUDGET = ControllerRun(
    label="No-subtree-weighted (root, budget)",
    checkpoint=YSAGIV_SCRATCH / "checkpoints/fittedq_rerun_encoder_zt_tt_ablation.pt",
    diagnostics_jsonl=YSAGIV_SCRATCH / "checkpoints/fittedq_rerun_encoder_zt_tt_ablation_diagnostics.jsonl",
    slurm_log_name="cts-fittedq_8533204.out",
)

CONTROLLER_TT_ONLY = ControllerRun(
    label="T_t only (new pipeline)",
    checkpoint=YSAGIV_SCRATCH / "checkpoints/fittedq_subtree_weighted_tt_only.pt",
    diagnostics_jsonl=YSAGIV_SCRATCH / "checkpoints/fittedq_subtree_weighted_tt_only_diagnostics.jsonl",
    slurm_log_name="cts-fittedq_8550917.out",
)

CONTROLLER_RUNS: tuple[ControllerRun, ...] = (
    CONTROLLER_SUBTREE_ROOT_BUDGET,
    CONTROLLER_SUBTREE_ROOT,
    CONTROLLER_RERUN_ROOT_BUDGET,
    CONTROLLER_TT_ONLY,
)

# --- Encoder KL audit (optional analysis) -------------------------------------

ENCODER_KL_AUDIT_DIR = YSAGIV_SCRATCH / "data/encoder_kl_audit"

# --- Lmcos training entry (for smoke / re-runs only) -------------------------

CONTROLLER_TRAIN_MODULE = "cts.train.controller_train"
CONTROLLER_TRAIN_CONFIG_BEST = (
    LMCOS_ROOT / "configs/train/controller_subtree_weighted_zt_tt.yaml"
)
