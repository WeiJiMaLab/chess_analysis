"""Per-edge KL audit of the child-WDL encoder by (parent_depth, child_subtree_size).

The pretraining loop reports one scalar ``loss_gap`` ≈ KL(target‖pred)
averaged over every supervised edge. That hides whether the encoder is
uniformly good or whether it nails some structural slice and misses
another. This module re-runs encoder + decoder on a packed pretrain
manifest and buckets the per-edge KL by:

  - ``parent_depth``: plies from root of the parent node (column axis).
  - ``child_subtree_size``: number of descendants below the child node,
    including the child itself (row axis, log2 bins). A leaf child has
    ``subtree_size = 1`` and the encoder has nothing to summarize about
    the child's subtree; a child with many descendants forces the encoder
    to compress an arbitrarily large subtree into the child's slot signal
    visible from the parent.

Outputs a JSON blob with per-bucket edge counts and mean KL plus an
optional heatmap PDF. The overall mean KL is reported alongside as a
sanity check against pretraining's reported ``loss_gap``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from cts.core.kl_buckets import (
    compute_subtree_sizes as _compute_subtree_sizes,
    depth_bin as _depth_bin,
    depth_bin_labels as _depth_bin_labels,
    num_children_bin,
    num_children_bin_labels,
    num_children_per_node,
    size_bin as _size_bin,
    size_bin_labels as _size_bin_labels,
)
from cts.core.schema import NodeFeatureSchema, tree_encoder_feature_schema
from cts.core.tensorizer import edge_wdl_target_tensor, tensorize_forest
from cts.data.preprocess_gnn.teacher_targets import (
    RAW_PRETRAIN_FORMAT,
    PretrainExample,
    RawPretrainExampleRecord,
    load_pretrain_example_dataset,
)
from cts.models.gnn import ChildWdlHead, TreeEncoder
from cts.train.gnn_pretrain import load_encoder_architecture, load_encoder_checkpoint


class AuditEncoderKLConfig(BaseModel):
    """CLI config for ``cts.analysis.audit_encoder_kl``."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: str  # packed pretrain manifest (train or validation)
    encoder_checkpoint: str
    decoder_checkpoint: str
    output_json: str  # where to write the per-bucket numbers
    output_heatmap: Optional[str] = None  # optional PDF of the (depth, size) heatmap
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 0
    max_depth_bin: int = 12  # parent depths > this are folded into the last column
    # log2 bin edges over child subtree size; default groups singletons (leaves) on their own
    # then 2–3, 4–7, 8–15, 16–31, 32–63, 64+ (i.e. bins by floor(log2(size))).
    subtree_size_log_max: int = 7
    # Linear axis: parent's number of children. Chess legal moves cap around ~40,
    # so 40 is a roomy ceiling; anything larger folds into the ``≥40`` cell.
    num_children_max_bin: int = 40
    decoder_hidden_dim: Optional[int] = None  # inferred from the decoder state_dict if omitted
    log_interval: int = 100  # batches between progress prints; first/last batch always logged
    # Resume convention: ``<output_json>.resume.pt`` is written every
    # ``checkpoint_every`` batches with the accumulator state + next batch
    # index. On startup the run picks up from the saved index if a matching
    # checkpoint is present, and deletes the checkpoint after a successful
    # JSON write. Set ``checkpoint_every: 0`` to disable resumption entirely.
    checkpoint_every: int = 100


def _load_decoder(
    checkpoint_path: str,
    d_embed: int,
    device: torch.device,
    explicit_hidden_dim: Optional[int],
) -> ChildWdlHead:
    """Load a ``ChildWdlHead`` from a decoder checkpoint.

    The pretrainer's saved decoder dict has no architecture metadata, so
    we infer ``hidden_dim`` from the shape of ``mlp.0.weight`` (which is
    ``[hidden_dim, 2 * d_embed]``) unless the caller passes it explicitly.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["decoder_state_dict"]
    if explicit_hidden_dim is None:
        first_weight = state_dict["mlp.0.weight"]
        expected_in_features = 2 * d_embed
        if first_weight.shape[1] != expected_in_features:
            raise ValueError(
                f"Decoder mlp.0.weight has in_features={first_weight.shape[1]} "
                f"but encoder d_embed={d_embed} implies 2*d_embed={expected_in_features}."
            )
        hidden_dim = int(first_weight.shape[0])
    else:
        hidden_dim = int(explicit_hidden_dim)
    head = ChildWdlHead(d_embed=d_embed, hidden_dim=hidden_dim, device=device)
    head.load_state_dict(state_dict)
    head.eval()
    return head


def _build_encoder(encoder_checkpoint: str, device: torch.device) -> TreeEncoder:
    """Reconstruct a ``TreeEncoder`` from architecture metadata + weights."""
    architecture = load_encoder_architecture(encoder_checkpoint, map_location=str(device))
    encoder = TreeEncoder(
        k=architecture["k"],
        node_feat=architecture["node_feat"],
        device=device,
        node_embed_hidden=architecture["node_embed_hidden"],
        d_embed=architecture["d_embed"],
        d_message=architecture["d_message"],
        n_heads=architecture["n_heads"],
        d_att=architecture["d_att"],
        sequential=True,
    )
    load_encoder_checkpoint(encoder_checkpoint, encoder, map_location=str(device))
    encoder.eval()
    return encoder


_LEGACY_RAW_FORMAT_TAGS = (
    "cts_raw_pretrain_example_v1",
    "cts_raw_pretrain_example_v2",
    "cts_raw_pretrain_example_v3",
    "cts_raw_pretrain_example_v4",
    "cts_raw_pretrain_example_v5",
)
_RESUME_FORMAT = "cts_audit_encoder_kl_resume_v2"  # v2: adds num_children grid (size grid alone was silently incomplete on resume)


def _resume_path(output_json: str) -> Path:
    """Convention: the resume checkpoint lives next to the JSON with a ``.resume.pt`` suffix.

    Mirrors ``ChildWdlPretrainer``'s ``<output>_resume.pt`` pattern. Keeping
    resume metadata next to the output means deleting/renaming the JSON also
    handles the checkpoint, and resuming requires no extra config plumbing.
    """
    return Path(str(output_json) + ".resume.pt")


def _save_resume_checkpoint(
    path: Path,
    *,
    sum_kl: torch.Tensor,
    count: torch.Tensor,
    sum_kl_nc: torch.Tensor,
    count_nc: torch.Tensor,
    overall_sum_kl: torch.Tensor,
    overall_count: int,
    next_batch_index: int,
    manifest_path: str,
    encoder_checkpoint: str,
    decoder_checkpoint: str,
    num_size_bins: int,
    num_depth_bins: int,
    num_nc_bins: int,
    max_depth_bin: int,
    subtree_size_log_max: int,
    num_children_max_bin: int,
) -> None:
    """Atomically write the in-flight accumulator state to ``path``.

    Saves both bucketing grids (``sum_kl``/``count`` keyed by child subtree
    size, and ``sum_kl_nc``/``count_nc`` keyed by parent's number of children)
    so resume produces a complete state for every grid the audit accumulates.
    The size grid alone was the v1 payload; the num_children grid was added
    later and went uncheckpointed for one commit — v2 fixes that.

    Saves to ``<path>.tmp`` and ``os.replace`` swaps it into place, so a
    process killed mid-save leaves the previous checkpoint intact rather
    than a half-written file.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format": _RESUME_FORMAT,
        "manifest_path": manifest_path,
        "encoder_checkpoint": encoder_checkpoint,
        "decoder_checkpoint": decoder_checkpoint,
        "max_depth_bin": int(max_depth_bin),
        "subtree_size_log_max": int(subtree_size_log_max),
        "num_children_max_bin": int(num_children_max_bin),
        "num_size_bins": int(num_size_bins),
        "num_depth_bins": int(num_depth_bins),
        "num_nc_bins": int(num_nc_bins),
        "sum_kl": sum_kl.detach().to("cpu"),
        "count": count.detach().to("cpu"),
        "sum_kl_nc": sum_kl_nc.detach().to("cpu"),
        "count_nc": count_nc.detach().to("cpu"),
        "overall_sum_kl": overall_sum_kl.detach().to("cpu"),
        "overall_count": int(overall_count),
        "next_batch_index": int(next_batch_index),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _load_resume_checkpoint(
    path: Path,
    *,
    device: torch.device,
    expected_manifest: str,
    expected_encoder: str,
    expected_decoder: str,
    expected_num_size_bins: int,
    expected_num_depth_bins: int,
    expected_num_nc_bins: int,
) -> Optional[Dict[str, Any]]:
    """Return the resume state, or ``None`` if no usable checkpoint exists.

    Validates that the checkpoint's identity metadata (manifest + encoder +
    decoder + bin shape for both grids) matches the current config. Mismatch
    means the operator changed configs since the last run; we refuse to
    silently resume into the wrong run and ask them to delete the checkpoint.
    """
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != _RESUME_FORMAT:
        raise ValueError(
            f"Resume checkpoint at {path} has unexpected format {payload.get('format')!r}; "
            f"expected {_RESUME_FORMAT}. Delete it to start fresh."
        )
    for field, expected, found in (
        ("manifest_path", expected_manifest, payload["manifest_path"]),
        ("encoder_checkpoint", expected_encoder, payload["encoder_checkpoint"]),
        ("decoder_checkpoint", expected_decoder, payload["decoder_checkpoint"]),
        ("num_size_bins", expected_num_size_bins, int(payload["num_size_bins"])),
        ("num_depth_bins", expected_num_depth_bins, int(payload["num_depth_bins"])),
        ("num_nc_bins", expected_num_nc_bins, int(payload["num_nc_bins"])),
    ):
        if expected != found:
            raise ValueError(
                f"Resume checkpoint at {path} disagrees on {field}: "
                f"checkpoint has {found!r}, current config has {expected!r}. "
                "Delete the checkpoint to start a fresh run, or re-run with the original config."
            )
    return {
        "sum_kl": payload["sum_kl"].to(device).to(torch.float64),
        "count": payload["count"].to(device).to(torch.long),
        "sum_kl_nc": payload["sum_kl_nc"].to(device).to(torch.float64),
        "count_nc": payload["count_nc"].to(device).to(torch.long),
        "overall_sum_kl": payload["overall_sum_kl"].to(device).to(torch.float64),
        "overall_count": int(payload["overall_count"]),
        "next_batch_index": int(payload["next_batch_index"]),
    }


def _load_v1_or_v2_pretrain_example(path: str) -> PretrainExample:
    """Load a raw pretrain ``.pt`` file, accepting both v1 and v2 dict payloads.

    The on-disk ``v2`` bump (``cts_raw_pretrain_example_v2``) was motivated by
    canonical UCI-sorted slot ordering of the on-disk ``children_index`` array;
    consumers that read ``children_index`` directly (e.g. the packer) need v2.
    The in-memory tensorize path used by this audit re-derives the canonical
    slot ordering on every call via ``_sorted_child_ids_with_slots``, and
    ``edge_wdl_targets`` is a ``(parent_id, child_id)``-keyed dict rather than
    an array aligned to ``children_index`` — so v1 records are functionally
    equivalent to v2 once they're rehydrated into a ``PretrainExample`` and
    tensorized.

    This wrapper retags ``v1`` payloads to ``v2`` before calling
    ``RawPretrainExampleRecord.from_payload`` so the strict-v2 check there
    (``teacher_targets.py:563``) doesn't reject legacy splits. The retag is
    purely a parser concession; nothing on disk is mutated.
    """
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a raw pretrain dict payload at {path}, got {type(payload).__name__}. "
            "Pickled-PretrainExample-style legacy files are not supported by the audit; "
            "use rewrite_compact after that bug is fixed, or skip them."
        )
    format_tag = payload.get("format")
    if format_tag not in _LEGACY_RAW_FORMAT_TAGS:
        raise ValueError(
            f"Unrecognized raw pretrain format {format_tag!r} at {path}; expected one of {_LEGACY_RAW_FORMAT_TAGS}."
        )
    if format_tag != RAW_PRETRAIN_FORMAT:
        # Local-copy retag so we don't mutate the on-disk file's dict via aliasing.
        payload = dict(payload)
        payload["format"] = RAW_PRETRAIN_FORMAT
    record = RawPretrainExampleRecord.from_payload(payload)
    return record.to_pretrain_example()


class _V1OrV2PretrainExamplePathDataset:
    """``PretrainExamplePathDataset`` variant whose ``__getitem__`` accepts v1 dicts.

    Used in place of the universal-loader's per-example dataset when the
    manifest points at v1 raw records (the rerun pretrain split). The
    canonical universal loader's ``__getitem__`` goes through
    ``load_pretrain_example → RawPretrainExampleRecord.load → from_payload``,
    which rejects v1; this dataset routes through ``_load_v1_or_v2_pretrain_example``
    instead. Otherwise has the same interface (length + per-index load) that the
    audit's DataLoader needs.
    """

    def __init__(self, paths: List[str]) -> None:
        if not paths:
            raise ValueError("V1/V2 pretrain example dataset requires at least one path.")
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> PretrainExample:
        return _load_v1_or_v2_pretrain_example(self.paths[index])


def _build_audit_dataset(manifest_path: str):
    """Return a dataset of PretrainExamples (or packed tensorized batches) for ``manifest_path``.

    Mirrors ``load_pretrain_example_dataset``'s dispatch: packed JSON manifests
    keep using the canonical packed dataset, while text manifests and example
    directories route through the v1-or-v2 loader so legacy raw .pt files
    don't trip ``from_payload``'s strict-v2 check.
    """
    if os.path.isdir(manifest_path):
        paths = sorted(
            os.path.join(manifest_path, filename)
            for filename in os.listdir(manifest_path)
            if filename.endswith(".pt")
        )
        return _V1OrV2PretrainExamplePathDataset(paths)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest path does not exist: {manifest_path}")

    if manifest_path.endswith(".json"):
        # Packed shards already store the canonical v1 manifest format and the
        # tensorized v1 shard format, both of which are independent of the raw
        # ``cts_raw_pretrain_example_v*`` format bump.
        return load_pretrain_example_dataset(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        paths = [line.strip() for line in handle if line.strip()]
    return _V1OrV2PretrainExamplePathDataset(paths)


def _tree_batch_and_edge_targets(
    batch_data: Any,
    schema: NodeFeatureSchema,
    device: torch.device,
) -> Tuple[Any, torch.Tensor]:
    """Normalize a DataLoader batch to ``(tree_batch, edge_targets)``.

    Mirrors ``ChildWdlPretrainer._tree_batch_and_edge_targets`` so the audit
    accepts the same set of dataset formats the pretrainer does:

    - packed tensorized shards (JSON manifest, ``cts_tensorized_pretrain_manifest_v1``)
      emit ``(tree_batch, node_targets)`` with ``tree_batch.edge_wdl_targets`` already
      populated;
    - per-example datasets (directory of ``.pt`` files or text manifest pointing at
      them) emit a list of ``PretrainExample`` we still need to tensorize.

    The audit doesn't need ``node_targets``, so we drop them.
    """
    if isinstance(batch_data, tuple) and len(batch_data) == 2:
        tree_batch, _ = batch_data
        if tree_batch.edge_wdl_targets is None:
            raise ValueError(
                "Packed tensorized batch is missing edge_wdl_targets; "
                "the audit requires the per-edge WDL supervision targets."
            )
        return tree_batch, tree_batch.edge_wdl_targets.to(device)

    batch_examples: List[PretrainExample] = list(batch_data)
    tree_batch = tensorize_forest(
        [example.tree for example in batch_examples],
        schema=schema,
        device=device,
    )
    target_parts = []
    for example in batch_examples:
        if not example.edge_wdl_targets:
            raise ValueError(
                "The audit requires every example to carry edge_wdl_targets; "
                "this PretrainExample has none."
            )
        target_parts.append(
            edge_wdl_target_tensor(
                example.tree,
                example.edge_wdl_targets,
                schema=schema,
                device=device,
            )
        )
    edge_targets = torch.cat(target_parts, dim=0).to(device) if target_parts else torch.empty(
        (0, 3), dtype=torch.float32, device=device
    )
    return tree_batch, edge_targets


def _evaluate(
    encoder: TreeEncoder,
    decoder: ChildWdlHead,
    manifest_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_depth_bin: int,
    subtree_size_log_max: int,
    log_interval: int,
    *,
    encoder_checkpoint: str,
    decoder_checkpoint: str,
    resume_path: Path,
    checkpoint_every: int,
    num_children_max_bin: int,
) -> Dict[str, Any]:
    """Stream the manifest through encoder + decoder and accumulate per-bin KL."""
    schema = tree_encoder_feature_schema()
    dataset = _build_audit_dataset(manifest_path)
    # Packed shard datasets expose a custom ``collate_fn`` that returns a
    # tensorized ``(tree_batch, node_targets)`` pair; per-example datasets
    # default to ``list`` so the loop tensorizes them on the fly.
    collate_fn = getattr(dataset, "collate_fn", list)
    dataloader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        # Workers tensorize trees in the background; without persistence and
        # prefetch the GPU stalls waiting for the next batch on every step.
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **dataloader_kwargs)
    total_batches = len(loader)
    print(
        f"[audit-encoder-kl] dataset_size={len(dataset)} batch_size={batch_size} "
        f"num_workers={num_workers} total_batches={total_batches}",
        flush=True,
    )

    # Two accumulator grids: one keyed by (child_subtree_size, parent_depth),
    # the other by (parent_num_children, parent_depth). Both run off the same
    # per-edge KL stream so we get two views of the breakdown in one pass.
    num_size_bins = subtree_size_log_max + 1
    num_depth_bins = max_depth_bin + 1
    num_nc_bins = num_children_max_bin + 1
    sum_kl = torch.zeros(
        (num_size_bins, num_depth_bins), dtype=torch.float64, device=device
    )
    count = torch.zeros(
        (num_size_bins, num_depth_bins), dtype=torch.long, device=device
    )
    sum_kl_nc = torch.zeros(
        (num_nc_bins, num_depth_bins), dtype=torch.float64, device=device
    )
    count_nc = torch.zeros(
        (num_nc_bins, num_depth_bins), dtype=torch.long, device=device
    )
    overall_sum_kl = torch.zeros((), dtype=torch.float64, device=device)
    overall_count = 0
    skip_batches = 0

    if checkpoint_every > 0:
        resume_state = _load_resume_checkpoint(
            resume_path,
            device=device,
            expected_manifest=manifest_path,
            expected_encoder=encoder_checkpoint,
            expected_decoder=decoder_checkpoint,
            expected_num_size_bins=num_size_bins,
            expected_num_depth_bins=num_depth_bins,
            expected_num_nc_bins=num_nc_bins,
        )
        if resume_state is not None:
            sum_kl = resume_state["sum_kl"]
            count = resume_state["count"]
            sum_kl_nc = resume_state["sum_kl_nc"]
            count_nc = resume_state["count_nc"]
            overall_sum_kl = resume_state["overall_sum_kl"]
            overall_count = resume_state["overall_count"]
            skip_batches = resume_state["next_batch_index"]
            print(
                f"[audit-encoder-kl] resuming from batch {skip_batches}/{total_batches} "
                f"(edges_total={overall_count})",
                flush=True,
            )

    import itertools
    import time

    start_time = time.monotonic()
    with torch.no_grad():
        # ``itertools.islice`` advances the DataLoader's internal iterator past
        # already-processed batches without yielding them. Worker tensorizations
        # of the skipped batches still happen (DataLoader prefetches in
        # background), but at <1ms each for 96-node trees that's negligible.
        batch_iter = (
            iter(loader) if skip_batches == 0
            else itertools.islice(iter(loader), skip_batches, None)
        )
        for batch_offset, batch_data in enumerate(batch_iter):
            batch_index = skip_batches + batch_offset + 1
            tree_batch, edge_targets = _tree_batch_and_edge_targets(batch_data, schema, device)
            if edge_targets.numel() == 0:
                continue

            encoded = encoder(tree_batch)
            edge_parent = tree_batch.edge_parent.to(device)
            edge_child = tree_batch.edge_child.to(device)
            edge_slot = tree_batch.edge_slot.to(device)
            depth_all = tree_batch.depth.to(device)
            parent_index_all = tree_batch.parent_index.to(device)

            slot_states = encoder.slot_embeddings(edge_slot)
            edge_logits = decoder(encoded.node_states[edge_parent], slot_states)
            log_probs = F.log_softmax(edge_logits, dim=-1)
            target_log_probs = edge_targets.clamp_min(1e-12).log()
            # KL(target || pred). The pretrain "loss_gap" metric uses the
            # batch mean of this same quantity.
            per_edge_kl = (edge_targets * (target_log_probs - log_probs)).sum(dim=-1)

            subtree_sizes = _compute_subtree_sizes(parent_index_all, depth_all)
            child_size_bin = _size_bin(subtree_sizes[edge_child], subtree_size_log_max)
            parent_depth_bin = _depth_bin(depth_all[edge_parent], max_depth_bin)
            flat_bin = child_size_bin * num_depth_bins + parent_depth_bin

            flat_sum = torch.zeros(num_size_bins * num_depth_bins, dtype=torch.float64, device=device)
            flat_count = torch.zeros(num_size_bins * num_depth_bins, dtype=torch.long, device=device)
            flat_sum.scatter_add_(0, flat_bin, per_edge_kl.to(torch.float64))
            flat_count.scatter_add_(
                0,
                flat_bin,
                torch.ones_like(flat_bin, dtype=torch.long),
            )
            sum_kl += flat_sum.view(num_size_bins, num_depth_bins)
            count += flat_count.view(num_size_bins, num_depth_bins)

            # Second grid: parent's number of children × parent depth.
            # ``edge_parent`` is the per-edge parent id; counting how often
            # each parent id appears gives each node's child count, then we
            # look that up per-edge for the parent.
            num_children_per_parent = num_children_per_node(edge_parent, num_nodes=int(parent_index_all.shape[0]))
            parent_nc_bin = num_children_bin(num_children_per_parent[edge_parent], num_children_max_bin)
            flat_bin_nc = parent_nc_bin * num_depth_bins + parent_depth_bin

            flat_sum_nc = torch.zeros(num_nc_bins * num_depth_bins, dtype=torch.float64, device=device)
            flat_count_nc = torch.zeros(num_nc_bins * num_depth_bins, dtype=torch.long, device=device)
            flat_sum_nc.scatter_add_(0, flat_bin_nc, per_edge_kl.to(torch.float64))
            flat_count_nc.scatter_add_(
                0,
                flat_bin_nc,
                torch.ones_like(flat_bin_nc, dtype=torch.long),
            )
            sum_kl_nc += flat_sum_nc.view(num_nc_bins, num_depth_bins)
            count_nc += flat_count_nc.view(num_nc_bins, num_depth_bins)

            overall_sum_kl += per_edge_kl.to(torch.float64).sum()
            overall_count += int(per_edge_kl.shape[0])

            # Throttled progress line: first batch, last batch, or every
            # ``log_interval`` batches. The running mean KL is a host sync
            # (``.item()``) but only fires on log batches, so it doesn't
            # dominate the inner loop's wall time.
            should_log = (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % log_interval == 0
            )
            if should_log:
                now = time.monotonic()
                elapsed = now - start_time
                running_mean = float((overall_sum_kl / max(overall_count, 1)).item())
                # Rate accounts for skipped batches: only batches we actually
                # processed contribute to elapsed, so dividing the processed
                # count gives the true batches/sec.
                processed = batch_index - skip_batches
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (total_batches - batch_index) / rate if rate > 0 else float("inf")
                print(
                    f"[audit-encoder-kl] batch={batch_index}/{total_batches} "
                    f"edges_total={overall_count} running_mean_kl={running_mean:.6f} "
                    f"elapsed={elapsed:.0f}s rate={rate:.2f}batch/s eta={eta:.0f}s",
                    flush=True,
                )

            # Periodic resume checkpoint. Always also save on the final batch
            # so a clean run leaves a consistent ``next_batch_index == total``
            # checkpoint right before the JSON write deletes it.
            should_checkpoint = (
                checkpoint_every > 0
                and (batch_index % checkpoint_every == 0 or batch_index == total_batches)
            )
            if should_checkpoint:
                _save_resume_checkpoint(
                    resume_path,
                    sum_kl=sum_kl,
                    count=count,
                    sum_kl_nc=sum_kl_nc,
                    count_nc=count_nc,
                    overall_sum_kl=overall_sum_kl,
                    overall_count=overall_count,
                    next_batch_index=batch_index,
                    manifest_path=manifest_path,
                    encoder_checkpoint=encoder_checkpoint,
                    decoder_checkpoint=decoder_checkpoint,
                    num_size_bins=num_size_bins,
                    num_depth_bins=num_depth_bins,
                    num_nc_bins=num_nc_bins,
                    max_depth_bin=max_depth_bin,
                    subtree_size_log_max=subtree_size_log_max,
                    num_children_max_bin=num_children_max_bin,
                )

    mean_kl = torch.where(count > 0, sum_kl / count.clamp_min(1).to(torch.float64), torch.zeros_like(sum_kl))
    mean_kl_nc = torch.where(count_nc > 0, sum_kl_nc / count_nc.clamp_min(1).to(torch.float64), torch.zeros_like(sum_kl_nc))
    overall_mean = float((overall_sum_kl / max(overall_count, 1)).item())
    return {
        "overall_mean_kl": overall_mean,
        "overall_edge_count": overall_count,
        "max_depth_bin": max_depth_bin,
        "subtree_size_log_max": subtree_size_log_max,
        "num_children_max_bin": num_children_max_bin,
        # (child_subtree_size × parent_depth) grid + marginals.
        "mean_kl_grid": mean_kl.cpu().tolist(),
        "edge_count_grid": count.cpu().tolist(),
        "marginal_mean_kl_by_depth": _marginal(sum_kl, count, axis=0),
        "marginal_mean_kl_by_size_bin": _marginal(sum_kl, count, axis=1),
        "marginal_edge_count_by_depth": count.sum(dim=0).cpu().tolist(),
        "marginal_edge_count_by_size_bin": count.sum(dim=1).cpu().tolist(),
        # (parent_num_children × parent_depth) grid + marginals.
        "mean_kl_grid_num_children": mean_kl_nc.cpu().tolist(),
        "edge_count_grid_num_children": count_nc.cpu().tolist(),
        "marginal_mean_kl_by_num_children": _marginal(sum_kl_nc, count_nc, axis=1),
        "marginal_edge_count_by_num_children": count_nc.sum(dim=1).cpu().tolist(),
    }


def _marginal(sum_kl: torch.Tensor, count: torch.Tensor, axis: int) -> List[float]:
    sums = sum_kl.sum(dim=axis)
    counts = count.sum(dim=axis).clamp_min(1).to(torch.float64)
    return (sums / counts).cpu().tolist()


def _save_heatmap(
    output_path: str,
    mean_kl_grid: List[List[float]],
    edge_count_grid: List[List[int]],
    size_bin_labels: List[str],
    depth_bin_labels: List[str],
    overall_mean_kl: float,
    overall_edge_count: int,
    title_suffix: str,
) -> None:
    """Render the per-bucket mean-KL heatmap with edge counts annotated."""
    # Imported here so the module stays importable in environments without matplotlib.
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.array(mean_kl_grid, dtype=np.float64)
    counts = np.array(edge_count_grid, dtype=np.int64)
    masked = np.ma.array(grid, mask=(counts == 0))

    fig, ax = plt.subplots(figsize=(1.1 * len(depth_bin_labels) + 2, 0.6 * len(size_bin_labels) + 2),
                            constrained_layout=True)
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(depth_bin_labels)))
    ax.set_xticklabels(depth_bin_labels)
    ax.set_yticks(range(len(size_bin_labels)))
    ax.set_yticklabels(size_bin_labels)
    ax.set_xlabel("parent depth (plies from root)")
    ax.set_ylabel("child subtree size (log2 bin)")
    ax.set_title(
        f"per-edge KL(target‖pred){title_suffix}\n"
        f"overall mean KL = {overall_mean_kl:.4f} ({overall_edge_count} edges)"
    )
    for size_idx in range(grid.shape[0]):
        for depth_idx in range(grid.shape[1]):
            if counts[size_idx, depth_idx] == 0:
                continue
            ax.text(
                depth_idx,
                size_idx,
                f"{grid[size_idx, depth_idx]:.3f}\n(n={counts[size_idx, depth_idx]})",
                ha="center",
                va="center",
                color="white" if grid[size_idx, depth_idx] > grid.max() / 2 else "black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, label="mean KL")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(config: AuditEncoderKLConfig) -> None:
    """CLI entry point: encode, decode, bucket per-edge KL, emit JSON (+ optional heatmap)."""
    device = torch.device(config.device)
    encoder = _build_encoder(config.encoder_checkpoint, device)
    decoder = _load_decoder(
        config.decoder_checkpoint,
        d_embed=encoder.d_embed,
        device=device,
        explicit_hidden_dim=config.decoder_hidden_dim,
    )

    print(f"[audit-encoder-kl] manifest={config.manifest_path}", flush=True)
    print(f"[audit-encoder-kl] encoder={config.encoder_checkpoint}", flush=True)
    print(f"[audit-encoder-kl] decoder={config.decoder_checkpoint}", flush=True)

    resume_path = _resume_path(config.output_json)
    metrics = _evaluate(
        encoder=encoder,
        decoder=decoder,
        manifest_path=config.manifest_path,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_depth_bin=config.max_depth_bin,
        subtree_size_log_max=config.subtree_size_log_max,
        log_interval=config.log_interval,
        encoder_checkpoint=config.encoder_checkpoint,
        decoder_checkpoint=config.decoder_checkpoint,
        resume_path=resume_path,
        checkpoint_every=config.checkpoint_every,
        num_children_max_bin=config.num_children_max_bin,
    )

    # Labels round out the JSON so downstream consumers don't need to
    # re-derive what each bin means.
    size_labels = _size_bin_labels(config.subtree_size_log_max)
    depth_labels = _depth_bin_labels(config.max_depth_bin)
    num_children_labels = num_children_bin_labels(config.num_children_max_bin)
    payload = {
        **metrics,
        "manifest_path": config.manifest_path,
        "encoder_checkpoint": config.encoder_checkpoint,
        "decoder_checkpoint": config.decoder_checkpoint,
        "size_bin_labels": size_labels,
        "depth_bin_labels": depth_labels,
        "num_children_bin_labels": num_children_labels,
    }
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(
        f"[audit-encoder-kl] wrote {output_json} "
        f"(overall_mean_kl={metrics['overall_mean_kl']:.6f}, edges={metrics['overall_edge_count']})",
        flush=True,
    )
    # JSON is the authoritative output — once it's on disk the resume
    # checkpoint is redundant and should be cleaned up so the next run
    # against this output path starts fresh rather than picking up
    # somebody else's accumulator state.
    if resume_path.exists():
        try:
            resume_path.unlink()
        except OSError as exc:
            print(
                f"[audit-encoder-kl] warning: could not delete resume checkpoint at {resume_path}: {exc}",
                flush=True,
            )

    if config.output_heatmap:
        output_pdf = Path(config.output_heatmap)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        title_suffix = f" — {Path(config.manifest_path).name}"
        # matplotlib may be unavailable on some cluster nodes (libstdc++
        # ABI mismatch with the conda env). The JSON is already on disk,
        # so a failed heatmap is non-fatal — log it and let downstream
        # plot from the JSON.
        try:
            _save_heatmap(
                str(output_pdf),
                mean_kl_grid=metrics["mean_kl_grid"],
                edge_count_grid=metrics["edge_count_grid"],
                size_bin_labels=size_labels,
                depth_bin_labels=depth_labels,
                overall_mean_kl=metrics["overall_mean_kl"],
                overall_edge_count=metrics["overall_edge_count"],
                title_suffix=title_suffix,
            )
            print(f"[audit-encoder-kl] wrote {output_pdf}", flush=True)
        except ImportError as exc:
            print(
                f"[audit-encoder-kl] matplotlib unavailable, skipping heatmap "
                f"({type(exc).__name__}: {exc}); render later with "
                f"``python -m cts.analysis.audit_encoder_kl_plot --input {output_json} --output {output_pdf}``.",
                flush=True,
            )


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(AuditEncoderKLConfig, main)
