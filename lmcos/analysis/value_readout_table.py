"""Step 1 of the value-readout characterization: build the per-snapshot table.

The question (settled with the user): the controller's scalar advantage IS its
value-of-computation estimate (it was trained on the budgeted oracle's A*). The
child-WDL decoder shows the root embedding ``z_root`` carries the value
distribution over the root's candidate moves -- the primitives any VOC rule runs
on. The open question is the *readout*: what function of those decoded move-values
does the controller compute to produce the advantage (a top-two-margin threshold?
something using the whole distribution? something else)?

This module produces the raw material for that analysis: one pass over the packed
validation episodes that, per snapshot, records

  - the controller's actual advantage output a_t (the real network, not a surrogate),
  - the decoded value of every root child move (ragged: a flat array keyed by
    snapshot, so any summary -- margin, spread, entropy, full sorted shape -- can
    be computed downstream without re-deciding it here),
  - N_t, T_t, and the regression target A*_t.

Nothing is fit here and no functional form is assumed; later steps read the form
off this table and confirm it causally on the same frozen controller.

Decode correctness + perspective check (the analogue of the baseline-regret check
in wdl_subspace_ablation): the child WDL is predicted for the child position
(opponent to move), so the root-player value of a move may be the *negated* child
value. Rather than guess, we compute the best-child value under both sign
conventions and report which correlates with the known committed-move reward
(``halt_reward`` = oracle final-Q of the move you'd play now). The convention with
the strong positive correlation is the right one, and a strong correlation at all
confirms the decode is wired correctly.

Everything runs on the controller + paired decoder we already have, re-encoding the
packed trees (the materialized cache stores only z_root, not the per-child slot
states the decoder needs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from cts.analysis.wdl_subspace_ablation import _load_controller, _load_decoder
from cts.train.controller_train import _build_packed_loader


class ValueReadoutTableConfig(BaseModel):
    """CLI config for ``cts.analysis.value_readout_table``."""

    model_config = ConfigDict(extra="forbid")

    controller_checkpoint: str  # trained fitted-Q controller ({model_state_dict, metadata})
    decoder_checkpoint: str  # paired child-WDL decoder ({decoder_state_dict, ...})
    manifest_path: str  # packed validation manifest (v4); we re-encode its trees
    output_npz: str  # per-snapshot + ragged-child arrays
    output_json: str  # metadata, decode-validation correlations, summary counts
    device: str = "cuda"
    batch_size: int = 32  # episodes per loader batch
    num_workers: int = 0
    decoder_hidden_dim: Optional[int] = None  # inferred from decoder state_dict if omitted


def _grouped_max(values: np.ndarray, snapshot_ids: np.ndarray, num_snapshots: int) -> np.ndarray:
    """Per-snapshot max of ``values`` keyed by ``snapshot_ids``; NaN for snapshots with no children."""
    out = np.full(num_snapshots, np.nan, dtype=np.float64)
    if values.size == 0:
        return out
    order = np.argsort(snapshot_ids, kind="stable")
    sid = snapshot_ids[order]
    val = values[order]
    uniq, start = np.unique(sid, return_index=True)
    out[uniq] = np.maximum.reduceat(val, start)
    return out


def main(config: ValueReadoutTableConfig) -> None:
    """Re-encode the packed episodes, decode root-child values, emit the per-snapshot table."""
    device = torch.device(config.device)
    model, controller_metadata = _load_controller(config.controller_checkpoint, device)
    d_embed = int(model.encoder.d_embed)
    decoder = _load_decoder(config.decoder_checkpoint, d_embed, device, config.decoder_hidden_dim)
    encoder_device = model.encoder.device

    loader = _build_packed_loader(
        config.manifest_path,
        batch_size=config.batch_size,
        shuffle=False,  # deterministic snapshot order
        seed=0,
        num_workers=config.num_workers,
    )

    print(f"[value-readout] controller={config.controller_checkpoint}", flush=True)
    print(f"[value-readout] decoder={config.decoder_checkpoint}", flush=True)
    print(f"[value-readout] controller_inputs={list(model.controller_inputs)} d_embed={d_embed}", flush=True)

    # Per-snapshot scalars (one row per snapshot, in deterministic loader order).
    adv_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    nt_chunks: List[np.ndarray] = []
    tt_chunks: List[np.ndarray] = []
    halt_chunks: List[np.ndarray] = []
    # Ragged root-child rows (flat, keyed to a global snapshot id).
    child_value_chunks: List[np.ndarray] = []  # W - L of the child position
    child_slot_chunks: List[np.ndarray] = []  # UCI-lex slot index of the child
    child_snap_chunks: List[np.ndarray] = []  # global snapshot id

    offset = 0  # global snapshot id of the first snapshot in the current batch
    for batch_index, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        tree_batch = batch.tree_batch
        with torch.inference_mode():
            encoded = model.encoder(tree_batch)
            scalars = torch.stack(
                [
                    batch.tree_sizes.to(encoder_device, dtype=torch.float32),
                    batch.time_budgets.to(encoder_device, dtype=torch.float32),
                ],
                dim=-1,
            )
            features = torch.cat([encoded.root_states, scalars], dim=-1)
            advantage, _ = model.predict_from_features(features)  # [B]

            edge_parent = tree_batch.edge_parent
            root_child = torch.isin(edge_parent, tree_batch.root_index).nonzero(as_tuple=True)[0]
            rc_parent = edge_parent[root_child]
            slot_states = model.encoder.slot_embeddings(tree_batch.edge_slot[root_child])
            wdl = torch.softmax(decoder(encoded.node_states[rc_parent], slot_states), dim=-1)  # [E_rc, 3]
            child_value = (wdl[:, 0] - wdl[:, 2])  # value of the child position (W - L), child to move
            child_snap_local = tree_batch.tree_index[rc_parent]  # [E_rc] in 0..B-1

        batch_snapshots = int(advantage.shape[0])
        adv_chunks.append(advantage.detach().float().cpu().numpy())
        target_chunks.append(batch.target_advantages.detach().float().cpu().numpy())
        nt_chunks.append(batch.tree_sizes.detach().float().cpu().numpy())
        tt_chunks.append(batch.time_budgets.detach().float().cpu().numpy())
        # Per-episode halt-reward sequences flatten to per-snapshot in tree order.
        halt_chunks.append(np.concatenate([np.asarray(seq, dtype=np.float64) for seq in batch.halt_rewards]))

        child_value_chunks.append(child_value.detach().float().cpu().numpy())
        child_slot_chunks.append(tree_batch.edge_slot[root_child].detach().cpu().numpy())
        child_snap_chunks.append(offset + child_snap_local.detach().cpu().numpy())
        offset += batch_snapshots

        if batch_index % 50 == 0:
            print(f"[value-readout] batch={batch_index} snapshots={offset}", flush=True)

    advantage = np.concatenate(adv_chunks)
    target_advantage = np.concatenate(target_chunks)
    n_t = np.concatenate(nt_chunks)
    t_t = np.concatenate(tt_chunks)
    halt_reward = np.concatenate(halt_chunks)
    child_value = np.concatenate(child_value_chunks)
    child_slot = np.concatenate(child_slot_chunks)
    child_snapshot_id = np.concatenate(child_snap_chunks)
    num_snapshots = int(advantage.shape[0])

    if not (num_snapshots == target_advantage.shape[0] == halt_reward.shape[0]):
        raise ValueError(
            f"Per-snapshot arrays disagree: advantage={num_snapshots} "
            f"target={target_advantage.shape[0]} halt={halt_reward.shape[0]}."
        )

    # Decode-correctness + perspective resolution: which sign convention for the
    # root-player move value lines up with the known committed-move reward.
    best_as_child = _grouped_max(child_value, child_snapshot_id, num_snapshots)  # value to child (opponent)
    best_as_root = _grouped_max(-child_value, child_snapshot_id, num_snapshots)  # value to root player
    has_children = ~np.isnan(best_as_child)

    def _corr(a: np.ndarray) -> float:
        mask = has_children & np.isfinite(a) & np.isfinite(halt_reward)
        if mask.sum() < 2:
            return float("nan")
        return float(np.corrcoef(a[mask], halt_reward[mask])[0, 1])

    corr_child_perspective = _corr(best_as_child)
    corr_root_perspective = _corr(best_as_root)
    resolved = "root_player(-child_value)" if corr_root_perspective >= corr_child_perspective else "child(child_value)"

    print(
        f"[value-readout] decode check: corr(best_child_value, halt_reward)="
        f"{corr_child_perspective:.3f}  corr(-best, halt_reward)={corr_root_perspective:.3f}  "
        f"-> root-player value = {resolved}",
        flush=True,
    )

    output_npz = Path(config.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        advantage=advantage.astype(np.float32),
        target_advantage=target_advantage.astype(np.float32),
        n_t=n_t.astype(np.float32),
        t_t=t_t.astype(np.float32),
        halt_reward=halt_reward.astype(np.float32),
        child_value=child_value.astype(np.float32),
        child_slot=child_slot.astype(np.int32),
        child_snapshot_id=child_snapshot_id.astype(np.int64),
    )
    print(f"[value-readout] wrote {output_npz} ({num_snapshots} snapshots, {child_value.shape[0]} child rows)", flush=True)

    payload = {
        "controller_checkpoint": config.controller_checkpoint,
        "decoder_checkpoint": config.decoder_checkpoint,
        "manifest_path": config.manifest_path,
        "controller_inputs": list(model.controller_inputs),
        "d_embed": d_embed,
        "num_snapshots": num_snapshots,
        "num_child_rows": int(child_value.shape[0]),
        "snapshots_with_children": int(has_children.sum()),
        "decode_check": {
            "corr_child_value_vs_halt_reward": corr_child_perspective,
            "corr_neg_child_value_vs_halt_reward": corr_root_perspective,
            "resolved_root_player_value": resolved,
            "note": "child WDL is W-L of the child (opponent-to-move) position; the "
            "convention whose best-move value correlates with halt_reward is the "
            "root-player value. A strong correlation confirms the decode is wired right.",
        },
        "value_field": "child_value = P(win) - P(loss) of the child position (child to move); "
        "apply the resolved sign for root-player value.",
    }
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"[value-readout] wrote {output_json}", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(ValueReadoutTableConfig, main)
