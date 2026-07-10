"""End-to-end (encoder-unfrozen) MetaController training loop (plan.md Z2 / endtoend.md).

Today's production controller path (``cts.train.pg_controller_train``) always trains
against a FROZEN encoder: ``z_t`` is precomputed once by ``materialize.py`` and cached,
and only the small advantage head ever gets a gradient. The encoder never sees the
downstream regret objective. This script removes that: it builds a ``MetaController``
WITHOUT calling ``freeze_encoder()``, initializes the encoder from the pretrained
child-WDL checkpoint (not from scratch), and trains encoder + head JOINTLY against the
real closed-form expected-regret loss (``pg_controller_train.expected_regret_batched``),
via one joint optimizer.

Reuses, verbatim, the three pieces plan.md/endtoend.md identified as already-existing
and sufficient:
  - ``ControllerEpisodeDataset`` / ``collate_controller_episodes`` (``controller_train.py``)
    — reconstruct + batch REAL per-step raw tree data from the packed manifest.
  - ``MetaController`` (``cts.models.mc``) with ``freeze_encoder()`` skipped.
  - ``expected_regret_batched`` (``pg_controller_train.py``) — the closed-form
    expected-regret loss, no REINFORCE rollout variance.
  - ``_build_model_and_optimizer`` / ``_seed_and_resolve_paths`` (``controller_train.py``)
    — same encoder-loading + optimizer-construction path ``pg_controller_train.py`` uses;
    passing ``unfreeze_encoder: true`` is the one flag flip that makes the encoder trainable
    (that flag already existed in ``ControllerTrainConfig`` but nothing previously called it
    with a live/unfrozen encoder).

Config: reuses the existing ``train`` section of the pipeline config (same
``ControllerTrainConfig`` pydantic model pg_controller_train.py uses) — no new config class.
Point ``--set output_checkpoint=...`` at a SCRATCH path; never at the real
``${packed_dir}/mchalt_controller.pt`` production checkpoint.

    python -m cts.train.e2e_controller_train --config config_minply15_maxply75.yaml \\
        --stage train --set unfreeze_encoder=true \\
        --set output_checkpoint=/scratch/.../e2e_controller.pt --set epochs=20

Data-loading note (important for wall-clock, NOT covered by Z0's timing script — that
script timed compute only, on one pre-loaded batch): ``ControllerEpisodeDataset`` caches
just the ONE most-recently-loaded shard (``_load_shard``'s single-slot memo), because its
designed access pattern is shard-local, not fully-random. A naive ``torch.randperm`` over
the WHOLE dataset (~22K episodes / ~45 shards for the train split) would evict and reload
a ~500-episode shard from disk on nearly every 8-episode batch — pure I/O thrashing Z0
never measured. ``_epoch_batches`` below shuffles the ORDER OF SHARDS and the order of
episodes WITHIN each shard, but keeps every batch's episodes drawn from a single shard, so
each shard is ``torch.load``-ed at most once per epoch.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from cts._config import run_with_config_cli
from cts.core.schema import tree_encoder_feature_schema
from cts.data.preprocess_mc.oracle import return_for_stop_step
from cts.train.controller_train import (
    ControllerTrainConfig,
    ControllerEpisodeDataset,
    collate_controller_episodes,
    _build_model_and_optimizer,
    _seed_and_resolve_paths,
)
from cts.train.gnn_pretrain import save_encoder_checkpoint
from cts.train.pg_controller_train import expected_regret_batched, _scatter_to_padded


def _epoch_batches(
    dataset: ControllerEpisodeDataset,
    episode_batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator,
    max_episodes: int | None = None,
) -> List[List[int]]:
    """Shard-local batch index groups (see module docstring for the I/O rationale).

    Shuffles shard order and within-shard episode order each call (when
    ``shuffle``), but never crosses a shard boundary within one batch, so
    ``ControllerEpisodeDataset``'s single-shard memo cache stays hot across an
    entire shard's worth of batches instead of reloading per batch.
    """
    shard_ranges = []
    start = 0
    for size in dataset.shard_num_episodes:
        shard_ranges.append((start, start + size))
        start += size

    shard_order = list(range(len(shard_ranges)))
    if shuffle:
        perm = torch.randperm(len(shard_order), generator=generator).tolist()
        shard_order = [shard_order[i] for i in perm]

    batches: List[List[int]] = []
    for shard_idx in shard_order:
        s, e = shard_ranges[shard_idx]
        idxs = list(range(s, e))
        if shuffle:
            local_perm = torch.randperm(len(idxs), generator=generator).tolist()
            idxs = [idxs[i] for i in local_perm]
        for i in range(0, len(idxs), episode_batch_size):
            batches.append(idxs[i : i + episode_batch_size])

    if max_episodes is not None:
        # Cap TOTAL episodes processed this epoch (smoke / quick-turnaround runs).
        # Batches stay shard-local (each already drawn from one shard), so this
        # just stops early rather than breaking the I/O-locality guarantee above.
        kept: List[List[int]] = []
        seen = 0
        for b in batches:
            if seen >= max_episodes:
                break
            kept.append(b)
            seen += len(b)
        batches = kept
    return batches


def build_regret_targets(episodes, oracle_config, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-batch ``(g_pad, mask, lengths, oracle_values)`` from real per-episode halt-reward
    traces -- the same closed-form oracle machinery ``pg_controller_train.py`` uses.
    ``oracle_value`` is ``max(g)`` recomputed fresh under ``oracle_config``, NOT the packed
    shard's frozen ``oracle_value`` (see ``pg_controller_train._episode_returns`` for why:
    this keeps cost-model changes independent of repacking)."""
    lengths = torch.tensor(
        [len(ep.step_node_features) for ep in episodes], dtype=torch.long, device=device
    )
    g_parts = []
    oracle_values: List[float] = []
    for ep in episodes:
        tree_sizes = ep.tree_sizes.tolist()
        time_budgets = ep.time_budgets.tolist()
        g = [
            return_for_stop_step(ep.halt_rewards, tree_sizes, time_budgets, s, oracle_config)
            for s in range(len(ep.step_node_features))
        ]
        g_parts.append(torch.tensor(g, dtype=torch.float32, device=device))
        oracle_values.append(float(max(g)))
    g_flat = torch.cat(g_parts)
    g_pad = _scatter_to_padded(g_flat, lengths)
    mask = _scatter_to_padded(torch.ones_like(g_flat), lengths)
    oracle_values_t = torch.tensor(oracle_values, dtype=torch.float32, device=device)
    return g_pad, mask, lengths, oracle_values_t


def forward_batch_loss(model, episodes, oracle_config, device) -> torch.Tensor:
    """Collate ``episodes`` -> one MetaController forward -> expected-regret loss.

    Single shared helper for train/val/tests so the exact same tensor plumbing
    (collate -> forward -> scatter -> expected_regret_batched) is exercised
    everywhere, instead of the training loop and tests each re-deriving it.
    """
    batch = collate_controller_episodes(episodes)
    assert batch is not None
    g_pad, mask, lengths, oracle_values = build_regret_targets(episodes, oracle_config, device)
    advantage, _sign_logit = model.forward(batch.tree_batch, batch.tree_sizes, batch.time_budgets)
    adv_pad = _scatter_to_padded(advantage, lengths)
    return expected_regret_batched(adv_pad, g_pad, mask, lengths, oracle_values)


def run_epoch(
    model,
    dataset: ControllerEpisodeDataset,
    oracle_config,
    device,
    *,
    episode_batch_size: int,
    optimizer,
    training: bool,
    max_grad_norm: float,
    generator: torch.Generator,
    epoch_index: int,
    log_interval: int = 200,
    max_episodes: int | None = None,
) -> tuple[float, list[float]]:
    """One pass (train or eval) over ``dataset``. Returns ``(episode-weighted mean loss,
    per-batch loss trace)`` -- the trace lets callers plot a within-epoch loss curve, which
    matters when a run is capped at very few epochs (see ``_write_history``/``_plot_within_epoch``)."""
    model.train(training)
    batches = _epoch_batches(dataset, episode_batch_size, shuffle=training, generator=generator, max_episodes=max_episodes)
    num_batches = len(batches)
    phase = "train" if training else "validation"

    total_loss = 0.0
    total_episodes = 0
    batch_loss_trace: list[float] = []  # per-batch loss, in batch order -- the promised 2nd
                                        # return value, previously computed (line below) but
                                        # never collected/returned (plan.md Agent 2 fix, flagged
                                        # by a direct user request: within-epoch curves matter
                                        # when a run is capped at very few epochs).
    started = time.time()
    for batch_index, idxs in enumerate(batches, start=1):
        episodes = [dataset[i] for i in idxs]
        with torch.set_grad_enabled(training):
            loss = forward_batch_loss(model, episodes, oracle_config, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_grad_norm
                    )
                optimizer.step()
        batch_loss = float(loss.detach())
        batch_loss_trace.append(batch_loss)
        total_loss += batch_loss * len(idxs)
        total_episodes += len(idxs)
        if batch_index == 1 or batch_index == num_batches or batch_index % log_interval == 0:
            elapsed = time.time() - started
            print(
                f"[e2e-controller] epoch={epoch_index} phase={phase} "
                f"batch={batch_index}/{num_batches} "
                f"running_mean_loss={total_loss / max(total_episodes, 1):.6f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )
    return total_loss / max(total_episodes, 1), batch_loss_trace


def _save_checkpoint(path: Path, model, epoch: int, train_loss: float, val_loss: float, config: ControllerTrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "stage": "e2e_joint_encoder_head",
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "controller_inputs": list(config.controller_inputs),
                "encoder_checkpoint_init": str(config.encoder_checkpoint),
                "objective": "pg_expected_regret_joint",
                "unfreeze_encoder": True,
            },
        },
        path,
    )
    # ALSO write an encoder-only checkpoint in the exact format
    # `gnn_pretrain.save_encoder_checkpoint`/`load_encoder_checkpoint`/`load_encoder_architecture`
    # expect ({"encoder_state_dict", "metadata": {"encoder_architecture": ...}}) -- this is the
    # format `cts.data.preprocess_mc.materialize.materialize_worker` consumes directly (it only
    # ever reads the encoder half of a checkpoint, discarding whatever head is paired with it —
    # see materialize_worker's own comment). Writing this here means Z3 can re-materialize z_t
    # from the jointly-trained encoder with the EXISTING materialize.py pipeline unchanged,
    # instead of a bespoke loader for this one checkpoint format.
    encoder_path = path.with_name(f"{path.stem}_encoder{path.suffix}")
    save_encoder_checkpoint(
        str(encoder_path),
        model.encoder,
        metadata={
            "stage": "e2e_joint_encoder_head",
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "encoder_checkpoint_init": str(config.encoder_checkpoint),
        },
    )


def _write_history(history: list[dict], out_path: Path) -> None:
    """CSV + loss-vs-epoch PNG/PDF, same visual style as the E1 encoder-loss plot
    (outputs/figures/.../diagnosis/{pdf,png}/e1_encoder_loss_vs_epoch.*)."""
    csv_path = out_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[e2e-controller] history csv -> {csv_path}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, [h["train_loss"] for h in history], "-o", color="#b3382c", label="train E[regret] (joint encoder+head)")
    ax.plot(epochs, [h["val_loss"] for h in history], "-s", color="#2c6fb3", label="val E[regret] (joint encoder+head)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("expected regret (soft-stop objective)")
    ax.set_title("Z2: end-to-end (encoder-unfrozen) MetaController loss vs epoch")
    ax.legend(loc="upper right")
    ax.grid(axis="both", linestyle=":", alpha=0.5)
    fig.tight_layout()

    png_dir = out_path.parent / "png"
    pdf_dir = out_path.parent / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{out_path.name}.png"
    pdf_path = pdf_dir / f"{out_path.name}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[e2e-controller] loss curve -> {png_path} , {pdf_path}", flush=True)


def _write_within_epoch_trace(trace: list[dict], out_path: Path) -> str:
    """Write the per-BATCH (not just per-epoch) loss trace this job collected -- CSV
    (``<out_path>_within_epoch_trace.csv``) + a plot via ``_plot_within_epoch``. Matters when a
    run is capped at very few epochs (this investigation's chained 1-epoch-per-job continuation,
    plan.md Agent 2): the per-EPOCH curve has only 1 point per job and can't show whether loss is
    still moving within that single epoch, which this trace can. No-ops (prints a warning, writes
    nothing) if ``trace`` is empty rather than crashing on an empty CSV/plot -- callers running 0
    train batches (e.g. a degenerate max_episodes=0 smoke config) still get a clean exit.
    """
    if not trace:
        print("[e2e-controller] within_epoch_trace is empty, skipping trace CSV/plot", flush=True)
        return ""
    csv_path = out_path.with_name(f"{out_path.name}_within_epoch_trace.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"[e2e-controller] within-epoch trace csv -> {csv_path}", flush=True)
    _plot_within_epoch(trace, out_path)
    return str(csv_path)


def _plot_within_epoch(trace: list[dict], out_path: Path) -> None:
    """Plot per-batch train/validation loss against a CONTINUOUS cumulative batch index (spanning
    every epoch this job ran, not reset per epoch) -- the within-epoch curve
    ``run_epoch``'s docstring promises callers, previously never implemented. Same visual house
    style as ``_write_history``'s per-epoch curve (same colors/labels), figures saved next to it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_rows = [r for r in trace if r["phase"] == "train"]
    val_rows = [r for r in trace if r["phase"] == "validation"]

    fig, ax = plt.subplots(figsize=(9, 6))
    if train_rows:
        ax.plot(range(1, len(train_rows) + 1), [r["loss"] for r in train_rows], "-", lw=0.9,
               color="#b3382c", alpha=0.85, label="train batch loss (joint encoder+head)")
    if val_rows:
        # Validation batches are typically far fewer than train batches (no gradient steps) and
        # would be invisible plotted 1:1 against train's x-axis -- spread evenly across the SAME
        # x-range instead, so both series are readable on one panel.
        n = max(len(train_rows), 1)
        xs = np.linspace(1, n, num=len(val_rows)) if val_rows else []
        ax.plot(xs, [r["loss"] for r in val_rows], "o", ms=3, color="#2c6fb3", alpha=0.85,
               label="val batch loss (joint encoder+head)")
    ax.set_xlabel("cumulative batch index (this job's epoch(s))")
    ax.set_ylabel("expected regret (soft-stop objective), per batch")
    ax.set_title("Z2/Z-EXT: within-epoch batch loss trace")
    ax.legend(loc="upper right")
    ax.grid(axis="both", linestyle=":", alpha=0.5)
    fig.tight_layout()

    png_dir = out_path.parent / "png"
    pdf_dir = out_path.parent / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    base = f"{out_path.name}_within_epoch_trace"
    png_path = png_dir / f"{base}.png"
    pdf_path = pdf_dir / f"{base}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[e2e-controller] within-epoch trace plot -> {png_path} , {pdf_path}", flush=True)


def main(config: ControllerTrainConfig) -> None:
    if not config.unfreeze_encoder:
        raise ValueError(
            "e2e_controller_train requires unfreeze_encoder=true (pass --set unfreeze_encoder=true); "
            "with it False this script degenerates to the same frozen-encoder training "
            "pg_controller_train.py already does, defeating the whole point of this script."
        )
    if not config.output_checkpoint:
        raise ValueError("output_checkpoint is required (point it at a SCRATCH path, not the production checkpoint).")

    # require_packed_oracle_match=False: this script's loss (forward_batch_loss ->
    # build_regret_targets, above) recomputes BOTH the return curve g(s) and oracle_value
    # fresh from raw halt_rewards/tree_sizes/time_budgets under the CURRENT oracle_config on
    # every batch -- it never reads the packed manifest's baked target_advantages/oracle_values.
    # So a packed-vs-requested maintenance_scale/time_lambda/etc. mismatch (e.g. training at
    # maintenance_scale=0.2 against a manifest packed at the default maintenance_scale=0.0) is
    # harmless here and shouldn't block the run -- see _seed_and_resolve_paths' docstring and
    # 2026-07-09 e2e-probe notebook entry for the full trace (this is NOT the same bug class as
    # the already-fixed pg_controller_train.py oracle_value staleness: this script never had a
    # staleness bug, the packed-manifest check was just stricter than this loss requires).
    device, _train_cache_path, _val_cache_path, oracle_config, schema = _seed_and_resolve_paths(
        config, require_packed_oracle_match=False
    )
    print(
        f"[e2e-controller] device={device} k={config.k} node_embed_hidden={config.node_embed_hidden} "
        f"d_embed={config.d_embed} d_message={config.d_message} hidden_dim={config.hidden_dim} "
        f"hidden_layers={config.hidden_layers} episode_batch_size={config.episode_batch_size} "
        f"epochs={config.epochs} learning_rate={config.learning_rate}",
        flush=True,
    )
    model, optimizer, scheduler = _build_model_and_optimizer(config, schema)
    assert all(p.requires_grad for p in model.encoder.parameters()), (
        "encoder parameters are not trainable -- freeze_encoder() must have been called "
        "despite unfreeze_encoder=true; refusing to run a training that silently isn't end-to-end."
    )

    train_dataset = ControllerEpisodeDataset(config.packed_train_data)
    val_dataset = ControllerEpisodeDataset(config.packed_validation_data)
    print(
        f"[e2e-controller] train_episodes={len(train_dataset)} val_episodes={len(val_dataset)}",
        flush=True,
    )

    out_path = Path(config.output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config.seed)

    # Reuses the SAME config field pg_controller_train.py already uses to cap train/val
    # episodes for smoke/quick-turnaround runs (ControllerTrainConfig.pg_max_episodes,
    # default None = no cap). Under the whole effort's 1h-wall-clock hard constraint this
    # is the "cap further via a max_episodes-style override" headroom knob -- no new config
    # field, just wiring the existing one through (previously dead in this script).
    episode_cap = config.pg_max_episodes
    if episode_cap is not None:
        print(f"[e2e-controller] pg_max_episodes cap active: {episode_cap} episodes/epoch (train+val each)", flush=True)

    history: list[dict] = []
    within_epoch_trace: list[dict] = []  # per-batch {epoch, phase, batch_index, loss} rows across
                                         # ALL epochs of THIS job -- the within-epoch curve a
                                         # capped-epoch run needs (see run_epoch's docstring;
                                         # previously promised but never collected -- fixed here).
    best_val_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.time()
        train_loss, train_batch_trace = run_epoch(
            model, train_dataset, oracle_config, device,
            episode_batch_size=config.episode_batch_size, optimizer=optimizer,
            training=True, max_grad_norm=config.max_grad_norm, generator=generator, epoch_index=epoch,
            max_episodes=episode_cap,
        )
        model.eval()
        with torch.no_grad():
            val_loss, val_batch_trace = run_epoch(
                model, val_dataset, oracle_config, device,
                episode_batch_size=config.episode_batch_size, optimizer=optimizer,
                training=False, max_grad_norm=config.max_grad_norm, generator=generator, epoch_index=epoch,
                max_episodes=episode_cap,
            )
        within_epoch_trace.extend(
            {"epoch": epoch, "phase": "train", "batch_index": i, "loss": v}
            for i, v in enumerate(train_batch_trace, start=1)
        )
        within_epoch_trace.extend(
            {"epoch": epoch, "phase": "validation", "batch_index": i, "loss": v}
            for i, v in enumerate(val_batch_trace, start=1)
        )
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - epoch_started

        # Same epoch-summary log style/shape as gnn_pretrain.py / build_tree.py's
        # `_log_epoch` (see cts.data.build_tree.pretrain_child_wdl_encoder_command).
        print(
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": elapsed})

        # Per-epoch checkpoint (current weights) so a partial/interrupted run is still usable,
        # plus a separate best-val snapshot -- mirrors gnn_pretrain.py's checkpoint_every_epochs
        # + best-validation-restore convention.
        epoch_ckpt = out_path.with_name(f"{out_path.stem}_epoch{epoch:03d}{out_path.suffix}")
        _save_checkpoint(epoch_ckpt, model, epoch, train_loss, val_loss, config)
        latest_ckpt = out_path.with_name(f"{out_path.stem}_latest{out_path.suffix}")
        _save_checkpoint(latest_ckpt, model, epoch, train_loss, val_loss, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(out_path, model, epoch, train_loss, val_loss, config)
            print(f"[e2e-controller] saved best val_loss={best_val_loss:.6f} -> {out_path}", flush=True)

        _write_history(history, out_path.with_suffix(""))

    _write_within_epoch_trace(within_epoch_trace, out_path.with_suffix(""))
    print(f"[e2e-controller] DONE best_val_loss={best_val_loss:.6f} checkpoint={out_path}", flush=True)


if __name__ == "__main__":
    run_with_config_cli(ControllerTrainConfig, main)
