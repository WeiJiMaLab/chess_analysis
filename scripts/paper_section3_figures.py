"""Paper Section 3 figures: what does the controller compute?

Generates Figures 3a-3f testing whether the controller implements VOC
using children's WDL distributions decoded from the root embedding.

All five models share one frozen encoder and one retrained decoder.
They differ only in advantage_head weights. Features are materialized
once from packed controller episodes; VOC is computed in the same pass.

Usage:
    python paper_section3_figures.py \
        --model-configs configs.json \
        --encoder-checkpoint encoder.pt \
        --decoder-checkpoint decoder.pt \
        --packed-controller-episodes validation_manifest.json \
        --output-dir figures/section3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from GNN import ChildWdlHead, TreeNN
from paper_figure_utils import MODEL_COLORS, MODEL_MARKERS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    label: str
    controller_checkpoint: str
    diagnostics_path: str


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _infer_encoder_params(state_dict: dict) -> dict:
    d_embed = state_dict["node_embed.2.weight"].shape[0]
    model_width = state_dict["upward_msg.W_q.weight"].shape[0]
    n_heads = 4
    return dict(
        d_embed=d_embed,
        node_feat=state_dict["node_embed.0.weight"].shape[1],
        node_embed_hidden=state_dict["node_embed.0.weight"].shape[0],
        d_message=state_dict["node_gru.weight_ih"].shape[1],
        n_heads=n_heads,
        d_att=model_width // n_heads,
    )


def load_encoder(checkpoint_path: str, device: torch.device, k: int = 1) -> TreeNN:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["encoder_state_dict"]
    p = _infer_encoder_params(state)
    encoder = TreeNN(
        k=k, node_feat=p["node_feat"], device=str(device),
        node_embed_hidden=p["node_embed_hidden"],
        d_embed=p["d_embed"], d_message=p["d_message"],
        n_heads=p["n_heads"], d_att=p["d_att"], sequential=True,
    )
    encoder.load_state_dict(state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def load_decoder(checkpoint_path: str, d_embed: int, device: torch.device) -> ChildWdlHead:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    hidden = int(meta.get("decoder_hidden", 128))
    head = ChildWdlHead(d_embed=d_embed, hidden_dim=hidden, device=str(device))
    head.load_state_dict(ckpt["decoder_state_dict"])
    head.eval()
    return head


def load_advantage_head(controller_ckpt: str, d_embed: int, device: torch.device) -> nn.Sequential:
    """Extract just the advantage_head from a full controller checkpoint."""
    ckpt = torch.load(controller_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    prefix = "advantage_head."
    head_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not head_state:
        raise KeyError(f"No advantage_head keys in checkpoint {controller_ckpt}")

    # nn.Sequential: Linear at even indices (0, 2, 4, ...), ReLU at odd
    layers: list[nn.Module] = []
    idx = 0
    while f"{idx}.weight" in head_state:
        w = head_state[f"{idx}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0], device=device))
        if f"{idx + 2}.weight" in head_state:
            layers.append(nn.ReLU())
        idx += 2

    head = nn.Sequential(*layers)
    head.load_state_dict(head_state)
    head.eval()
    return head


def load_diagnostics(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Materialization + VOC (one pass over packed controller episodes)
# ---------------------------------------------------------------------------


@dataclass
class MaterializedData:
    features: torch.Tensor       # [N, d_embed+2]
    target_advantages: torch.Tensor  # [N]
    voc: torch.Tensor            # [N]


def _vpi_from_wdl(w: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Closed-form VPI (Dearden) from decoded WDL distributions.

    Given C children with win probabilities w[C] and loss probabilities l[C],
    VPI = E[max_c V_c] - max_c E[V_c] where V_c ~ Cat(+1,0,-1; w_c,d_c,l_c).

    E[max_c V_c] = 1 - prod_c(1 - w_c) - prod_c(l_c)
    max_c E[V_c] = max_c(w_c - l_c)

    Returns a scalar tensor.
    """
    e_max = 1.0 - (1.0 - w).prod() - l.prod()
    return e_max - (w - l).max()


def materialize_features_and_voc(
    encoder: TreeNN,
    decoder: ChildWdlHead,
    packed_manifest: str,
    device: torch.device,
    episode_batch_size: int = 8,
    num_workers: int = 0,
) -> MaterializedData:
    """One pass over packed controller episodes: materialize encoder features
    and compute VPI at each snapshot. Returns aligned tensors."""
    from train_fitted_q_controller import PackedControllerEpisodeDataset, PackedControllerCollator

    ds = PackedControllerEpisodeDataset(packed_manifest)
    collator = PackedControllerCollator()
    loader = DataLoader(ds, batch_size=episode_batch_size, shuffle=False,
                        collate_fn=collator, num_workers=num_workers)

    encoder.eval()
    decoder.eval()

    all_features: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_voc: list[torch.Tensor] = []
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue

            # --- Materialize features [z_root, N_t, T_t] ---
            encoded = encoder(batch.tree_batch)
            root_states = encoded.root_states  # [B, d_embed]
            ts = batch.tree_sizes.to(device, dtype=torch.float32)
            tb = batch.time_budgets.to(device, dtype=torch.float32)
            features = torch.cat([root_states, ts.unsqueeze(-1), tb.unsqueeze(-1)], dim=-1)
            all_features.append(features.cpu())
            all_targets.append(batch.target_advantages.cpu())

            # --- VPI per snapshot (closed-form) ---
            edge_parent = batch.tree_batch.edge_parent.to(device)
            edge_slot = batch.tree_batch.edge_slot.to(device)
            root_index = batch.tree_batch.root_index.to(device)

            parent_states = encoded.node_states[edge_parent]
            slot_embeds = encoder.slot_embeddings(edge_slot)
            wdl_probs = F.softmax(decoder(parent_states, slot_embeds), dim=-1)

            B = features.shape[0]
            batch_voc = torch.zeros(B, device=device)

            for i in range(B):
                child_mask = edge_parent == root_index[i]
                if not child_mask.any():
                    continue
                child_wdl = wdl_probs[child_mask]  # [C, 3]
                batch_voc[i] = _vpi_from_wdl(child_wdl[:, 0], child_wdl[:, 2])

            all_voc.append(batch_voc.cpu())
            total += B

            if (batch_idx + 1) % 100 == 0:
                logger.info("  materialized %d batches, %d snapshots", batch_idx + 1, total)

    logger.info("Materialized %d total snapshots", total)
    return MaterializedData(
        features=torch.cat(all_features),
        target_advantages=torch.cat(all_targets),
        voc=torch.cat(all_voc),
    )


# ---------------------------------------------------------------------------
# Fig 3a: Controller output vs decoded VOC
# ---------------------------------------------------------------------------


def fig_3a_voc_correlation(
    model_specs: List[ModelSpec],
    data: MaterializedData,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    n = len(model_specs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    v = data.voc.numpy()

    for i, spec in enumerate(model_specs):
        ax = axes[0][i]
        head = load_advantage_head(spec.controller_checkpoint, d_embed, device)
        with torch.no_grad():
            pred = head(data.features.to(device)).squeeze(-1).cpu().numpy()

        # Subsample for readability
        rng = np.random.default_rng(0)
        if len(pred) > 5000:
            idx = rng.choice(len(pred), 5000, replace=False)
            ps, vs = pred[idx], v[idx]
        else:
            ps, vs = pred, v

        color = MODEL_COLORS.get(spec.label, "gray")
        ax.scatter(vs, ps, s=1, alpha=0.3, color=color)

        r_pearson = float(np.corrcoef(pred, v)[0, 1])
        r_spearman = float(stats.spearmanr(pred, v).statistic)
        ax.set_title(f"{spec.label}\nr={r_pearson:.3f} ρ={r_spearman:.3f}", fontsize=9)
        ax.set_xlabel("Decoded VOC")
        ax.set_ylabel("Predicted advantage")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_3a_voc_correlation.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_3a_voc_correlation.pdf")


# ---------------------------------------------------------------------------
# Fig 3b: WDL subspace ablation
# ---------------------------------------------------------------------------


def fig_3b_wdl_ablation(
    model_specs: List[ModelSpec],
    data: MaterializedData,
    decoder: ChildWdlHead,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
    max_k: int = 64,
) -> None:
    W = decoder.mlp[0].weight.detach().cpu()  # [hidden, 2*d_embed]
    W_root = W[:, :d_embed]
    _, _, Vt = torch.linalg.svd(W_root, full_matrices=False)
    max_k = min(max_k, Vt.shape[0], d_embed)
    ks = list(range(1, max_k + 1))

    z_root = data.features[:, :d_embed]
    scalars = data.features[:, d_embed:]
    target_sign = np.sign(data.target_advantages.numpy())

    fig, (ax_keep, ax_abl) = plt.subplots(1, 2, figsize=(12, 5))

    for spec in model_specs:
        head = load_advantage_head(spec.controller_checkpoint, d_embed, device)
        keep_acc, ablate_acc = [], []
        zr = z_root.to(device)
        sc = scalars.to(device)

        for k in ks:
            Vk = Vt[:k].to(device)

            with torch.no_grad():
                proj = zr @ Vk.T @ Vk
                pred_keep = head(torch.cat([proj, sc], dim=-1)).squeeze(-1).cpu().numpy()
                ablated = zr - proj
                pred_abl = head(torch.cat([ablated, sc], dim=-1)).squeeze(-1).cpu().numpy()

            keep_acc.append(float((np.sign(pred_keep) == target_sign).mean()))
            ablate_acc.append(float((np.sign(pred_abl) == target_sign).mean()))

        color = MODEL_COLORS.get(spec.label, "gray")
        ax_keep.plot(ks, keep_acc, color=color, label=spec.label)
        ax_abl.plot(ks, ablate_acc, color=color, label=spec.label)

    ax_keep.set_title("Keep top-k WDL directions")
    ax_abl.set_title("Ablate top-k WDL directions")
    for ax in (ax_keep, ax_abl):
        ax.set_xlabel("k (WDL principal directions)")
        ax.set_ylabel("Sign accuracy")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_3b_wdl_ablation.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_3b_wdl_ablation.pdf")


# ---------------------------------------------------------------------------
# Fig 3c: Layer-by-layer VOC correlation
# ---------------------------------------------------------------------------


def fig_3c_layer_correlation(
    model_specs: List[ModelSpec],
    data: MaterializedData,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    v = data.voc.numpy()
    fig, (ax_voc, ax_out) = plt.subplots(1, 2, figsize=(12, 5))

    for spec in model_specs:
        head = load_advantage_head(spec.controller_checkpoint, d_embed, device)
        with torch.no_grad():
            final_pred = head(data.features.to(device)).squeeze(-1).cpu().numpy()

        # Capture activations at each ReLU
        layer_activations: list[np.ndarray] = []
        hooks = []
        for module in head.modules():
            if isinstance(module, nn.ReLU):
                store: list[np.ndarray] = []
                layer_activations.append(store)  # type: ignore[arg-type]
                hooks.append(module.register_forward_hook(
                    lambda m, inp, out, s=store: s.append(out.detach().cpu().numpy())
                ))

        with torch.no_grad():
            head(data.features.to(device))
        for h in hooks:
            h.remove()

        voc_corrs, out_corrs = [], []
        for store in layer_activations:
            act = store[0]
            cv = np.array([abs(float(np.corrcoef(act[:, j], v)[0, 1]))
                           for j in range(act.shape[1])])
            co = np.array([abs(float(np.corrcoef(act[:, j], final_pred)[0, 1]))
                           for j in range(act.shape[1])])
            voc_corrs.append(float(np.nanmean(cv)))
            out_corrs.append(float(np.nanmean(co)))

        color = MODEL_COLORS.get(spec.label, "gray")
        layers = list(range(len(voc_corrs)))
        ax_voc.plot(layers, voc_corrs, "o-", color=color, label=spec.label)
        ax_out.plot(layers, out_corrs, "o-", color=color, label=spec.label)

    ax_voc.set_title("Mean |corr| with VOC")
    ax_out.set_title("Mean |corr| with output")
    for ax in (ax_voc, ax_out):
        ax.set_xlabel("Layer (ReLU index)")
        ax.set_ylabel("Mean |r|")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_3c_layer_correlation.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_3c_layer_correlation.pdf")


# ---------------------------------------------------------------------------
# Fig 3d: Residual analysis beyond VOC
# ---------------------------------------------------------------------------


def fig_3d_residuals(
    model_specs: List[ModelSpec],
    data: MaterializedData,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    v = data.voc.numpy()
    t_t = data.features[:, d_embed + 1].numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    predictors = ["T_t"]
    x = np.arange(len(predictors))
    w = 0.8 / len(model_specs)

    for mi, spec in enumerate(model_specs):
        head = load_advantage_head(spec.controller_checkpoint, d_embed, device)
        with torch.no_grad():
            pred = head(data.features.to(device)).squeeze(-1).cpu().numpy()

        slope, intercept = np.polyfit(v, pred, 1)
        residuals = pred - (slope * v + intercept)
        corr_tt = float(np.corrcoef(residuals, t_t)[0, 1])

        offset = (mi - len(model_specs) / 2 + 0.5) * w
        ax.bar(x + offset, [corr_tt], w,
               color=MODEL_COLORS.get(spec.label, "gray"), label=spec.label)

    ax.set_xticks(x)
    ax.set_xticklabels(predictors)
    ax.set_ylabel("Correlation with residual")
    ax.set_title("Residuals after VOC regression")
    ax.legend(fontsize=7)
    ax.axhline(0, color="k", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_3d_residuals.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_3d_residuals.pdf")


# ---------------------------------------------------------------------------
# Fig 3e: T_t gating
# ---------------------------------------------------------------------------


def fig_3e_tt_gating(
    model_specs: List[ModelSpec],
    data: MaterializedData,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
    top_k: int = 10,
) -> None:
    v = data.voc.numpy()
    t_t = data.features[:, d_embed + 1].numpy()
    n = len(model_specs)
    fig, axes = plt.subplots(2, max(n, 1), figsize=(4 * n, 7), squeeze=False)

    for mi, spec in enumerate(model_specs):
        head = load_advantage_head(spec.controller_checkpoint, d_embed, device)

        # First hidden layer activations
        first_linear = None
        for m in head.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            continue

        with torch.no_grad():
            act = F.relu(first_linear(data.features.to(device))).cpu().numpy()

        # Partial correlation of each neuron with T_t, controlling for VOC
        n_neurons = act.shape[1]
        partial_corrs = np.zeros(n_neurons)
        for j in range(n_neurons):
            sl_a, int_a = np.polyfit(v, act[:, j], 1)
            sl_t, int_t = np.polyfit(v, t_t, 1)
            res_a = act[:, j] - (sl_a * v + int_a)
            res_t = t_t - (sl_t * v + int_t)
            if res_a.std() > 0 and res_t.std() > 0:
                partial_corrs[j] = float(np.corrcoef(res_a, res_t)[0, 1])

        top_idx = np.argsort(np.abs(partial_corrs))[-top_k:][::-1]

        ax_bar = axes[0][mi]
        ax_bar.barh(range(top_k), partial_corrs[top_idx],
                    color=MODEL_COLORS.get(spec.label, "gray"))
        ax_bar.set_yticks(range(top_k))
        ax_bar.set_yticklabels([f"n{i}" for i in top_idx], fontsize=7)
        ax_bar.set_xlabel("Partial corr(neuron, T_t | VOC)")
        ax_bar.set_title(spec.label, fontsize=9)

        ax_sc = axes[1][mi]
        best = top_idx[0]
        n_plot = min(2000, len(t_t))
        ax_sc.scatter(t_t[:n_plot], act[:n_plot, best], s=1, alpha=0.3,
                      color=MODEL_COLORS.get(spec.label, "gray"))
        ax_sc.set_xlabel("T_t")
        ax_sc.set_ylabel(f"Neuron {best} activation")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_3e_tt_gating.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_3e_tt_gating.pdf")


# ---------------------------------------------------------------------------
# Fig 3f: Example episodes
# ---------------------------------------------------------------------------


def fig_3f_example_episodes(
    model_specs: List[ModelSpec],
    output_dir: Path,
) -> None:
    ref_diag = load_diagnostics(model_specs[0].diagnostics_path)

    easy = [e for e in ref_diag if e["oracle_stop_step"] == 0 and e["episode_length"] > 1]
    medium = [e for e in ref_diag if 3 <= e["oracle_stop_step"] <= 5]
    hard = [e for e in ref_diag if e["oracle_stop_step"] >= 8]
    by_regret = sorted(ref_diag, key=lambda e: -e["regret"])

    selected = []
    for pool in [easy, medium, hard]:
        if pool:
            selected.append(pool[0])
    for e in by_regret:
        if e not in selected:
            selected.append(e)
            break

    if not selected:
        logger.warning("No episodes for Fig 3f; skipping")
        return

    all_diag: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in model_specs:
        diag = load_diagnostics(spec.diagnostics_path)
        all_diag[spec.label] = {e["path"]: e for e in diag}

    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 3.5 * len(selected)), squeeze=False)

    for row, ep in enumerate(selected):
        ax = axes[row][0]
        L = int(ep["episode_length"])
        steps = list(range(L))

        ax.plot(steps, ep["target_advantages"][:L], "k-", linewidth=2,
                label="Target adv.", alpha=0.7)

        for spec in model_specs:
            model_ep = all_diag[spec.label].get(ep["path"])
            if model_ep is not None:
                pred = model_ep["predicted_advantages"][:L]
                ax.plot(steps[:len(pred)], pred,
                        color=MODEL_COLORS.get(spec.label, "gray"),
                        linewidth=1, label=spec.label, alpha=0.8)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(int(ep["oracle_stop_step"]), color="red", linewidth=0.8,
                   linestyle=":", label="Oracle stop")
        ax.set_xlabel("Step")
        ax.set_ylabel("Advantage")
        ax.set_title(f"Oracle stop={ep['oracle_stop_step']}, budget={ep['starting_budget']}, "
                      f"bucket={ep['budget_bucket_name']}, regret={ep['regret']:.4f}",
                      fontsize=9)
        if row == 0:
            ax.legend(fontsize=6, ncol=3)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_3f_example_episodes.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved fig_3f_example_episodes.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 3 paper figures")
    parser.add_argument("--model-configs", required=True,
                        help="JSON: [{label, controller_checkpoint, diagnostics_path}, ...]")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--packed-controller-episodes", required=True,
                        help="Validation manifest for packed controller episodes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--ablation-max-k", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache", help="Path to save/load materialized features+VOC")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)

    with open(args.model_configs) as f:
        raw_configs = json.load(f)
    model_specs = [ModelSpec(**cfg) for cfg in raw_configs]
    logger.info("Loaded %d model configs", len(model_specs))

    logger.info("Loading encoder: %s", args.encoder_checkpoint)
    encoder = load_encoder(args.encoder_checkpoint, device)
    d_embed = encoder.d_embed

    logger.info("Loading decoder: %s", args.decoder_checkpoint)
    decoder = load_decoder(args.decoder_checkpoint, d_embed, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Materialize features + VOC (cached to disk for reruns)
    if args.cache and Path(args.cache).exists():
        logger.info("Loading cached data from %s", args.cache)
        cached = torch.load(args.cache, weights_only=False)
        data = MaterializedData(
            features=cached["features"],
            target_advantages=cached["target_advantages"],
            voc=cached["voc"],
        )
    else:
        logger.info("Materializing features + VOC from %s", args.packed_controller_episodes)
        data = materialize_features_and_voc(
            encoder, decoder, args.packed_controller_episodes, device,
            episode_batch_size=args.episode_batch_size,
            num_workers=args.num_workers,
        )
        if args.cache:
            torch.save({
                "features": data.features,
                "target_advantages": data.target_advantages,
                "voc": data.voc,
            }, args.cache)
            logger.info("Cached to %s", args.cache)

    logger.info("Data: %d snapshots, feature dim %d", data.features.shape[0], data.features.shape[1])

    fig_3a_voc_correlation(model_specs, data, d_embed, device, output_dir)
    fig_3b_wdl_ablation(model_specs, data, decoder, d_embed, device, output_dir,
                        max_k=args.ablation_max_k)
    fig_3c_layer_correlation(model_specs, data, d_embed, device, output_dir)
    fig_3d_residuals(model_specs, data, d_embed, device, output_dir)
    fig_3e_tt_gating(model_specs, data, d_embed, device, output_dir)
    fig_3f_example_episodes(model_specs, output_dir)

    logger.info("All Section 3 figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
