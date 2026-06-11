"""WDL-subspace ablation: does the controller's stopping decision causally use the child-WDL subspace of ``z_t``?

Section-3 instrument, rebuilt as a ``cts.analysis`` module after the deleted
``analyze_advantage_head.py``. The question it answers causally:

The child-WDL decoder defines a set of directions in root-embedding (``z_t``)
space that carry the decoded child WDLs. Does the trained advantage head's
*stopping decision* depend on those directions, or does it read ``z_t`` from
directions orthogonal to them? On the rerun encoder the answer was the latter
(ablating the WDL subspace did not hurt -- it slightly helped -- the
halt/continue decision; the head loaded ~78% orthogonal to the WDL subspace).
The open question is whether the subtree-weighted encoder flips this.

Method (in regret units -- the prior pass only had sign accuracy):

  - SVD the decoder's root-readout submatrix ``mlp.0.weight[:, :d_embed]`` to
    get an orthonormal basis ``V`` of ``z_t`` directions ordered by how
    strongly the decoder reads them. ``V[:, :k]`` spans the top-k child-WDL
    subspace.
  - KEEP-ONLY top-k: replace ``z_t`` with its projection onto ``span(V[:, :k])``.
  - ABLATE top-k: replace ``z_t`` with the orthogonal complement.
  - Re-run the frozen advantage head on the modified cache features; report
    greedy stopping regret + snapshot sign accuracy at each k.

If ablating the WDL subspace wrecks regret (and keep-only preserves it), the
controller's decision causally depends on child-WDL competition -- the VOC
reading. If ablation leaves regret flat (or improves it), the decision is
carried by non-WDL directions, as on the rerun encoder.

Also reports weight-space alignment: the fraction of the head's ``z_t`` input
weight energy in the top-k WDL subspace, vs. chance (``k / d_embed``).

Everything runs from the materialized cache (``z_t`` already computed) plus the
decoder readout matrix and the controller head; the encoder is reconstructed
only so the controller ``state_dict`` loads cleanly -- it is never run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict

from cts.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    budgeted_oracle_config_from_metadata,
)
from cts.models.gnn import ChildWdlHead
from cts.models.mc import MetaController, validate_controller_inputs

# Reuse the trainer's exact greedy rule + episode-metadata extraction + cache
# loader. This is deliberate: the baseline (no ablation) must reproduce the
# controller's recorded greedy regret, which only holds if we score with the
# same code path the trainer used.
from cts.train.controller_train import (
    ControllerEpisodeDataset,
    MaterializedCache,
    _aggregate_greedy_rollout_metrics,
    _extract_all_episode_metadata,
    _load_materialized_cache,
)
from cts.train.gnn_pretrain import load_encoder_architecture


class WdlSubspaceAblationConfig(BaseModel):
    """CLI config for ``cts.analysis.wdl_subspace_ablation``."""

    model_config = ConfigDict(extra="forbid")

    controller_checkpoint: str  # trained fitted-Q controller ({model_state_dict, metadata})
    decoder_checkpoint: str  # paired child-WDL decoder ({decoder_state_dict, ...})
    cache_path: str  # materialized validation cache ([z_t, N_t, T_t] features)
    manifest_path: str  # packed validation manifest (v4): episode metadata + oracle config
    output_json: str
    output_pdf: Optional[str] = None
    device: str = "cuda"
    predict_batch_size: int = 65536
    # k values to sweep over; if omitted, a log-spaced set up to d_embed is used.
    k_values: Optional[List[int]] = None
    decoder_hidden_dim: Optional[int] = None  # inferred from decoder state_dict if omitted
    greedy_log_interval: int = 0  # 0 = quiet; we run many greedy evals per run


def _infer_head_dims(state_dict: Dict[str, torch.Tensor], separate_sign_head: bool) -> Tuple[int, int]:
    """Infer ``(hidden_dim, hidden_layers)`` of the advantage MLP from saved weights.

    Single head (``advantage_head``): Linear/ReLU pairs then a scalar Linear, so
    Linear layers sit at even ``nn.Sequential`` indices; ``hidden_dim`` is the
    first Linear's ``out_features`` and ``hidden_layers = #Linear - 1`` (the last
    Linear is the scalar output). Separate-sign head (``advantage_backbone``)
    holds exactly ``hidden_layers`` Linears, each ``out_features = hidden_dim``.
    """
    prefix = "advantage_backbone." if separate_sign_head else "advantage_head."
    linear_indices = sorted(
        int(key[len(prefix):].split(".")[0])
        for key in state_dict
        if key.startswith(prefix) and key.endswith(".weight")
    )
    if not linear_indices:
        raise ValueError(f"No advantage-head Linear weights found under prefix {prefix!r}.")
    hidden_dim = int(state_dict[f"{prefix}{linear_indices[0]}.weight"].shape[0])
    hidden_layers = len(linear_indices) if separate_sign_head else len(linear_indices) - 1
    return hidden_dim, hidden_layers


def _load_controller(checkpoint_path: str, device: torch.device) -> Tuple[MetaController, Dict[str, Any]]:
    """Reconstruct + load a ``MetaController`` from a controller checkpoint.

    The checkpoint metadata stores ``controller_inputs``, ``separate_sign_head``,
    and the ``encoder_checkpoint`` but not the network dims, so we read the
    encoder architecture from the encoder checkpoint and infer the head dims
    from the saved weights. The encoder is rebuilt only so the ``state_dict``
    loads strictly -- it is never run; ``z_t`` comes from the materialized cache.
    """
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"]
    metadata = payload.get("metadata", {})
    encoder_checkpoint = metadata["encoder_checkpoint"]
    controller_inputs = validate_controller_inputs(metadata["controller_inputs"])
    if "z_t" not in controller_inputs:
        raise ValueError(f"Controller has no z_t input ({controller_inputs}); nothing to ablate.")
    separate_sign_head = bool(metadata.get("separate_sign_head", False))
    arch = load_encoder_architecture(encoder_checkpoint, map_location=str(device))
    hidden_dim, hidden_layers = _infer_head_dims(state_dict, separate_sign_head)
    model = MetaController(
        k=arch["k"],
        node_feat=arch["node_feat"],
        device=str(device),
        node_embed_hidden=arch["node_embed_hidden"],
        d_embed=arch["d_embed"],
        d_message=arch["d_message"],
        n_heads=arch["n_heads"],
        d_att=arch["d_att"],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        separate_sign_head=separate_sign_head,
        controller_inputs=list(controller_inputs),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


def _load_decoder(
    checkpoint_path: str,
    d_embed: int,
    device: torch.device,
    explicit_hidden_dim: Optional[int],
) -> ChildWdlHead:
    """Load a ``ChildWdlHead`` from a decoder checkpoint (``hidden_dim`` inferred if omitted)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["decoder_state_dict"]
    if explicit_hidden_dim is None:
        first_weight = state_dict["mlp.0.weight"]
        expected_in_features = 2 * d_embed
        if first_weight.shape[1] != expected_in_features:
            raise ValueError(
                f"Decoder mlp.0.weight in_features={first_weight.shape[1]} "
                f"but encoder d_embed={d_embed} implies 2*d_embed={expected_in_features}."
            )
        hidden_dim = int(first_weight.shape[0])
    else:
        hidden_dim = int(explicit_hidden_dim)
    head = ChildWdlHead(d_embed=d_embed, hidden_dim=hidden_dim, device=device)
    head.load_state_dict(state_dict)
    head.eval()
    return head


def _wdl_readout_basis(decoder: ChildWdlHead, d_embed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """SVD the decoder's parent-state readout; return ``(V [d_embed, d_embed], singular_values)``.

    ``mlp.0.weight`` is ``[hidden_dim, 2*d_embed]``; the first ``d_embed`` columns
    multiply the parent (root) state. SVD that submatrix ``W = U S V^T``; the
    columns of ``V`` are orthonormal ``z_t`` directions ordered by how strongly
    the decoder reads them for child WDL. ``V[:, :k]`` spans the top-k subspace.
    """
    w_parent = decoder.mlp[0].weight.detach()[:, :d_embed].float().cpu()  # [hidden_dim, d_embed]
    _, singular_values, vh = torch.linalg.svd(w_parent, full_matrices=True)
    basis = vh.transpose(0, 1).contiguous()  # [d_embed, d_embed]; columns ordered by descending singular value
    return basis, singular_values


def _subspace_projection_matrix(basis: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetric idempotent projector onto ``span(basis[:, :k])`` (``V_k V_k^T``, shape ``[d_embed, d_embed]``)."""
    v_k = basis[:, :k]
    return v_k @ v_k.transpose(0, 1)


def _apply_projection(z: torch.Tensor, projection: torch.Tensor, mode: str) -> torch.Tensor:
    """Project row vectors ``z`` (``[N, d_embed]``) with ``projection``; ``keep`` keeps the subspace, ``ablate`` removes it."""
    z_proj = z @ projection
    if mode == "keep":
        return z_proj
    if mode == "ablate":
        return z - z_proj
    raise ValueError(f"Unknown projection mode: {mode!r}")


def _load_all_cache_features(cache: MaterializedCache) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate every shard's ``features`` + ``target_advantages`` into flat CPU tensors (snapshot order)."""
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for shard_path in cache.shard_paths:
        payload = torch.load(shard_path, weights_only=False)
        features.append(payload["features"])
        targets.append(payload["target_advantages"])
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _predict_advantages(
    model: MetaController,
    features: torch.Tensor,
    d_embed: int,
    projection: Optional[torch.Tensor],
    mode: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run the frozen head over (optionally ``z_t``-projected) cache features; return flat advantages ``[N]`` (CPU).

    ``projection`` is ``None`` for the untouched baseline; otherwise it is the
    ``[d_embed, d_embed]`` projector and ``mode`` selects keep/ablate. Only the
    ``z_t`` columns (the first ``d_embed``) are transformed; the ``N_t``/``T_t``
    scalar columns pass through unchanged.
    """
    out: List[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start:start + batch_size].to(device, non_blocking=True)
            if projection is not None:
                z_new = _apply_projection(batch[:, :d_embed], projection, mode)
                batch = torch.cat([z_new, batch[:, d_embed:]], dim=-1)
            predicted, _ = model.predict_from_features(batch)
            out.append(predicted.detach().float().cpu())
    return torch.cat(out, dim=0)


def _sign_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of snapshots where ``sign(predicted)`` matches ``sign(target)`` (the prior pass's metric)."""
    return float(((predicted > 0) == (target > 0)).float().mean().item())


def _greedy_regret(
    episode_metadata: List[Any],
    advantages: torch.Tensor,
    oracle_config: BudgetedOracleConfig,
    log_interval: int,
):
    """Score a flat advantage vector with the trainer's exact greedy stop-when-advantage<=0 rule."""
    return _aggregate_greedy_rollout_metrics(
        episode_metadata,
        advantages,
        oracle_config,
        log_interval=log_interval,
        started=time.time(),
        diagnostics_out=None,
    )


def _weight_alignment(
    model: MetaController,
    basis: torch.Tensor,
    d_embed: int,
    k_values: List[int],
    separate_sign_head: bool,
) -> Dict[str, Dict[str, float]]:
    """Fraction of the head's ``z_t`` input-weight energy lying in the top-k WDL subspace, vs. chance.

    ``_select_features`` concatenates the head input in canonical order, so when
    ``z_t`` is present it occupies the first ``d_embed`` columns of the first
    Linear's weight. The fraction is ``||W_z V_k||_F^2 / ||W_z||_F^2``; chance is
    ``k / d_embed``.
    """
    first_linear = model.advantage_backbone[0] if separate_sign_head else model.advantage_head[0]
    w_z = first_linear.weight.detach().float().cpu()[:, :d_embed]  # [hidden_dim, d_embed]
    total = float(w_z.pow(2).sum().item())
    alignment: Dict[str, Dict[str, float]] = {}
    for k in k_values:
        v_k = basis[:, :k]
        fraction = float((w_z @ v_k).pow(2).sum().item() / total) if total > 0 else 0.0
        alignment[str(k)] = {"fraction_in_subspace": fraction, "chance": k / d_embed}
    return alignment


def _default_k_values(d_embed: int) -> List[int]:
    """Log-spaced k sweep up to ``d_embed`` (always includes 1 and the full dim)."""
    candidates = [1, 2, 3, 5, 8, 16, 32, 64, d_embed // 2, d_embed]
    return sorted({k for k in candidates if 1 <= k <= d_embed})


def _save_plot(output_pdf: str, payload: Dict[str, Any]) -> None:
    """Two-panel PDF: greedy regret and snapshot sign accuracy vs. k for keep-only and ablate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_values = payload["k_values"]
    sweep = payload["sweep"]
    base = payload["baseline"]
    keep_regret = [row["keep_only"]["regret"] for row in sweep]
    ablate_regret = [row["ablate"]["regret"] for row in sweep]
    keep_sign = [row["keep_only"]["sign_accuracy"] for row in sweep]
    ablate_sign = [row["ablate"]["sign_accuracy"] for row in sweep]

    fig, (ax_regret, ax_sign) = plt.subplots(1, 2, figsize=(11, 4))
    ax_regret.axhline(base["regret"], ls="--", c="k", label=f"baseline {base['regret']:.3f}")
    ax_regret.plot(k_values, keep_regret, "o-", label="keep-only top-k WDL")
    ax_regret.plot(k_values, ablate_regret, "s-", label="ablate top-k WDL")
    ax_regret.set_xscale("log", base=2)
    ax_regret.set_xlabel("k (WDL subspace dim)")
    ax_regret.set_ylabel("greedy regret")
    ax_regret.set_title("Regret vs WDL-subspace ablation")
    ax_regret.legend()

    ax_sign.axhline(base["sign_accuracy"], ls="--", c="k", label=f"baseline {base['sign_accuracy']:.3f}")
    ax_sign.plot(k_values, keep_sign, "o-", label="keep-only")
    ax_sign.plot(k_values, ablate_sign, "s-", label="ablate")
    ax_sign.set_xscale("log", base=2)
    ax_sign.set_xlabel("k (WDL subspace dim)")
    ax_sign.set_ylabel("snapshot sign accuracy")
    ax_sign.set_title("Sign accuracy vs WDL-subspace ablation")
    ax_sign.legend()

    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def main(config: WdlSubspaceAblationConfig) -> None:
    """Load controller + decoder + cache, sweep the WDL-subspace ablation, emit JSON (+ optional PDF)."""
    device = torch.device(config.device)
    model, controller_metadata = _load_controller(config.controller_checkpoint, device)
    d_embed = int(model.encoder.d_embed)
    decoder = _load_decoder(config.decoder_checkpoint, d_embed, device, config.decoder_hidden_dim)
    basis, singular_values = _wdl_readout_basis(decoder, d_embed)
    basis_device = basis.to(device)

    # Oracle config straight from the packed manifest, so greedy regret is
    # scored against the exact reward the data was packed with.
    with open(config.manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    oracle_config = budgeted_oracle_config_from_metadata(manifest)
    if oracle_config is None:
        raise ValueError(f"Manifest missing budgeted oracle metadata: {config.manifest_path}")

    cache = _load_materialized_cache(
        config.cache_path,
        manifest_path=config.manifest_path,
        encoder_checkpoint=controller_metadata["encoder_checkpoint"],
    )
    dataset = ControllerEpisodeDataset(config.manifest_path)
    episode_metadata = _extract_all_episode_metadata(dataset)
    features, targets = _load_all_cache_features(cache)
    total_steps = sum(m.num_steps for m in episode_metadata)
    if features.shape[0] != total_steps:
        raise ValueError(
            f"Cache has {features.shape[0]} snapshots but manifest has {total_steps} steps."
        )

    print(f"[wdl-ablation] controller={config.controller_checkpoint}", flush=True)
    print(f"[wdl-ablation] decoder={config.decoder_checkpoint}", flush=True)
    print(f"[wdl-ablation] controller_inputs={list(model.controller_inputs)} d_embed={d_embed}", flush=True)
    print(f"[wdl-ablation] snapshots={features.shape[0]} episodes={len(episode_metadata)}", flush=True)

    # Baseline (no transform) must reproduce the controller's recorded greedy regret.
    baseline_pred = _predict_advantages(
        model, features, d_embed, None, "baseline", config.predict_batch_size, device
    )
    baseline_metrics = _greedy_regret(episode_metadata, baseline_pred, oracle_config, config.greedy_log_interval)
    baseline_sign = _sign_accuracy(baseline_pred, targets)
    print(
        f"[wdl-ablation] baseline regret={baseline_metrics.average_regret:.4f} "
        f"sign_acc={baseline_sign:.4f} (should match the controller's recorded regret)",
        flush=True,
    )

    k_values = config.k_values or _default_k_values(d_embed)
    sweep: List[Dict[str, Any]] = []
    for k in k_values:
        projection = _subspace_projection_matrix(basis_device, k)
        keep_pred = _predict_advantages(model, features, d_embed, projection, "keep", config.predict_batch_size, device)
        ablate_pred = _predict_advantages(model, features, d_embed, projection, "ablate", config.predict_batch_size, device)
        keep_metrics = _greedy_regret(episode_metadata, keep_pred, oracle_config, config.greedy_log_interval)
        ablate_metrics = _greedy_regret(episode_metadata, ablate_pred, oracle_config, config.greedy_log_interval)
        row = {
            "k": k,
            "keep_only": {
                "regret": keep_metrics.average_regret,
                "sign_accuracy": _sign_accuracy(keep_pred, targets),
                "average_expansions": keep_metrics.average_expansions,
                "exact_stop_step_accuracy": keep_metrics.exact_stop_step_accuracy,
            },
            "ablate": {
                "regret": ablate_metrics.average_regret,
                "sign_accuracy": _sign_accuracy(ablate_pred, targets),
                "average_expansions": ablate_metrics.average_expansions,
                "exact_stop_step_accuracy": ablate_metrics.exact_stop_step_accuracy,
            },
        }
        sweep.append(row)
        print(
            f"[wdl-ablation] k={k:>4d}  "
            f"keep regret={keep_metrics.average_regret:.4f} sign={row['keep_only']['sign_accuracy']:.3f}  "
            f"ablate regret={ablate_metrics.average_regret:.4f} sign={row['ablate']['sign_accuracy']:.3f}",
            flush=True,
        )

    alignment = _weight_alignment(
        model, basis, d_embed, k_values, bool(controller_metadata.get("separate_sign_head", False))
    )

    payload = {
        "controller_checkpoint": config.controller_checkpoint,
        "decoder_checkpoint": config.decoder_checkpoint,
        "cache_path": config.cache_path,
        "manifest_path": config.manifest_path,
        "controller_inputs": list(model.controller_inputs),
        "d_embed": d_embed,
        "num_snapshots": int(features.shape[0]),
        "num_episodes": len(episode_metadata),
        "baseline": {
            "regret": baseline_metrics.average_regret,
            "sign_accuracy": baseline_sign,
            "average_expansions": baseline_metrics.average_expansions,
            "exact_stop_step_accuracy": baseline_metrics.exact_stop_step_accuracy,
            "average_oracle_value": baseline_metrics.average_oracle_value,
        },
        "k_values": list(k_values),
        "sweep": sweep,
        "weight_alignment": alignment,
        "singular_values": singular_values.tolist(),
    }
    output_json = Path(config.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"[wdl-ablation] wrote {output_json}", flush=True)

    if config.output_pdf:
        output_pdf = Path(config.output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        try:
            _save_plot(str(output_pdf), payload)
            print(f"[wdl-ablation] wrote {output_pdf}", flush=True)
        except ImportError as exc:
            print(f"[wdl-ablation] matplotlib unavailable, skipping plot ({type(exc).__name__}: {exc})", flush=True)


if __name__ == "__main__":
    from cts._config import run_with_config_cli

    run_with_config_cli(WdlSubspaceAblationConfig, main)
