"""Mechanistic analysis of the advantage head MLP.

For each model, decompose the advantage head layer by layer:
- What z_root directions does each hidden neuron read from?
- Do those directions align with WDL decoder readout directions (VPI-relevant)?
- How do T_t and N_t scalar weights compare to z_root weights?
- Which neurons does the output layer weight most, and are those VPI-sensitive or T_t-sensitive?

Outputs a JSON report + summary figures.

Usage:
    python analyze_advantage_head.py \
        --model-configs configs.json \
        --decoder-checkpoint decoder.pt \
        --cache materialized_voc_cache.pt \
        --output-dir analysis/advantage_head
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from GNN import ChildWdlHead

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    label: str
    controller_checkpoint: str
    diagnostics_path: str


def load_advantage_head(controller_ckpt: str, device: torch.device) -> nn.Sequential:
    ckpt = torch.load(controller_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    prefix = "advantage_head."
    head_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
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


def load_decoder(checkpoint_path: str, d_embed: int, device: torch.device) -> ChildWdlHead:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    hidden = int(meta.get("decoder_hidden", 128))
    head = ChildWdlHead(d_embed=d_embed, hidden_dim=hidden, device=str(device))
    head.load_state_dict(ckpt["decoder_state_dict"])
    head.eval()
    return head


# ---------------------------------------------------------------------------
# WDL readout directions from decoder
# ---------------------------------------------------------------------------

def get_wdl_readout_basis(decoder: ChildWdlHead, d_embed: int) -> torch.Tensor:
    """SVD of decoder's first-layer root-readout submatrix.

    Returns Vt [K, d_embed] — rows are principal readout directions
    in z_root space, ordered by singular value.
    """
    W = decoder.mlp[0].weight.detach().cpu()  # [hidden, 2*d_embed]
    W_root = W[:, :d_embed]  # [hidden, d_embed]
    _, S, Vt = torch.linalg.svd(W_root, full_matrices=False)
    return Vt, S


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def analyze_model(
    label: str,
    head: nn.Sequential,
    wdl_Vt: torch.Tensor,
    wdl_S: torch.Tensor,
    features: torch.Tensor,
    voc: np.ndarray,
    target: np.ndarray,
    d_embed: int,
) -> dict:
    """Full mechanistic decomposition of one model's advantage head."""
    device = next(head.parameters()).device

    # Extract weight matrices from the Sequential
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    report = {"label": label, "num_layers": len(linears), "layers": []}

    # --- First layer: decompose into z_root vs scalar contributions ---
    W0 = linears[0].weight.detach().cpu().numpy()  # [H, 130]
    b0 = linears[0].bias.detach().cpu().numpy()     # [H]
    H = W0.shape[0]

    W0_z = W0[:, :d_embed]           # [H, 128] — z_root weights
    W0_nt = W0[:, d_embed]           # [H] — N_t weights
    W0_tt = W0[:, d_embed + 1]       # [H] — T_t weights

    # For each neuron: norm of z_root weight vs scalar weights
    z_norms = np.linalg.norm(W0_z, axis=1)       # [H]
    nt_abs = np.abs(W0_nt)                         # [H]
    tt_abs = np.abs(W0_tt)                         # [H]
    total_norms = np.linalg.norm(W0, axis=1)       # [H]

    # Alignment of each neuron's z_root weight with WDL readout directions
    Vt_np = wdl_Vt.numpy()  # [K, d_embed]
    # Project each neuron's z_root weight onto the WDL basis
    # cos similarity with top-k WDL directions
    W0_z_normed = W0_z / (z_norms[:, None] + 1e-10)
    wdl_alignment = W0_z_normed @ Vt_np.T  # [H, K] — cosine with each WDL direction

    # Fraction of z_root weight variance in WDL subspace (top-k directions)
    for k in [5, 10, 20]:
        proj = W0_z @ Vt_np[:k].T @ Vt_np[:k]  # [H, d_embed]
        frac_in_wdl = (np.linalg.norm(proj, axis=1) ** 2) / (z_norms ** 2 + 1e-10)
        report[f"layer0_mean_frac_in_wdl_top{k}"] = float(frac_in_wdl.mean())

    layer0_report = {
        "shape": list(W0.shape),
        "z_root_norm_mean": float(z_norms.mean()),
        "z_root_norm_std": float(z_norms.std()),
        "nt_weight_mean_abs": float(nt_abs.mean()),
        "tt_weight_mean_abs": float(tt_abs.mean()),
        "frac_z_norm": float((z_norms / (total_norms + 1e-10)).mean()),
        "frac_tt_of_total": float((tt_abs / (total_norms + 1e-10)).mean()),
        "max_wdl_alignment_per_neuron_mean": float(np.abs(wdl_alignment).max(axis=1).mean()),
    }
    report["layers"].append(layer0_report)

    # --- Run features through to get activations ---
    feat_dev = features.to(device)
    activations = [feat_dev]
    x = feat_dev
    with torch.no_grad():
        for module in head:
            x = module(x)
            if isinstance(module, nn.ReLU):
                activations.append(x)
        activations.append(x)  # final output

    # --- Per hidden layer: neuron activity statistics ---
    for li in range(1, len(linears)):
        W = linears[li].weight.detach().cpu().numpy()
        layer_report = {"shape": list(W.shape)}
        report["layers"].append(layer_report)

    # --- Output layer: which neurons matter most? ---
    W_out = linears[-1].weight.detach().cpu().numpy().flatten()  # [H_last]
    b_out = linears[-1].bias.detach().cpu().numpy().item()

    # Last hidden activations
    last_hidden = activations[-2].cpu().numpy()  # [N, H_last]

    # Effective contribution of each neuron to output: |w_out_j| * std(h_j)
    neuron_std = last_hidden.std(axis=0)
    neuron_contribution = np.abs(W_out) * neuron_std
    top_neurons = np.argsort(neuron_contribution)[::-1][:20]

    # For each top neuron: correlate its activation with VPI and T_t
    t_t = features[:, d_embed + 1].numpy()
    neuron_details = []
    for j in top_neurons:
        act_j = last_hidden[:, j]
        r_voc = float(np.corrcoef(act_j, voc)[0, 1]) if act_j.std() > 0 else 0.0
        r_tt = float(np.corrcoef(act_j, t_t)[0, 1]) if act_j.std() > 0 else 0.0
        r_tgt = float(np.corrcoef(act_j, target)[0, 1]) if act_j.std() > 0 else 0.0
        neuron_details.append({
            "neuron": int(j),
            "output_weight": float(W_out[j]),
            "activation_std": float(neuron_std[j]),
            "contribution": float(neuron_contribution[j]),
            "r_vpi": r_voc,
            "r_tt": r_tt,
            "r_target": r_tgt,
            "frac_active": float((act_j > 0).mean()),
        })

    report["output_layer"] = {
        "bias": float(b_out),
        "top_neurons": neuron_details,
        "total_contribution": float(neuron_contribution.sum()),
    }

    # --- First-layer neurons: T_t-gated vs VPI-sensitive classification ---
    first_hidden = activations[1].cpu().numpy()  # [N, H]
    neuron_types = {"tt_dominated": 0, "vpi_sensitive": 0, "mixed": 0, "dead": 0}
    first_layer_neurons = []
    for j in range(first_hidden.shape[1]):
        act_j = first_hidden[:, j]
        if act_j.std() < 1e-8:
            neuron_types["dead"] += 1
            continue
        r_voc = abs(float(np.corrcoef(act_j, voc)[0, 1]))
        r_tt = abs(float(np.corrcoef(act_j, t_t)[0, 1]))
        if r_tt > 0.3 and r_voc < 0.1:
            neuron_types["tt_dominated"] += 1
        elif r_voc > 0.1 and r_tt < 0.1:
            neuron_types["vpi_sensitive"] += 1
        elif r_tt > 0.1 and r_voc > 0.1:
            neuron_types["mixed"] += 1
        first_layer_neurons.append({
            "neuron": int(j),
            "r_vpi": float(r_voc),
            "r_tt": float(r_tt),
            "tt_weight": float(W0_tt[j]),
            "nt_weight": float(W0_nt[j]),
            "z_norm": float(z_norms[j]),
            "frac_active": float((act_j > 0).mean()),
        })

    report["first_layer_neuron_types"] = neuron_types
    report["first_layer_neurons"] = sorted(
        first_layer_neurons, key=lambda d: d["r_vpi"], reverse=True,
    )[:20]

    return report


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_neuron_type_summary(reports: list[dict], output_dir: Path) -> None:
    labels = [r["label"] for r in reports]
    types = ["tt_dominated", "vpi_sensitive", "mixed", "dead"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.8 / len(types)
    for ti, t in enumerate(types):
        vals = [r["first_layer_neuron_types"][t] for r in reports]
        ax.bar(x + ti * w, vals, w, label=t)
    ax.set_xticks(x + w * len(types) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of first-layer neurons")
    ax.legend(fontsize=8)
    ax.set_title("First-layer neuron classification (|r| thresholds: T_t>0.3, VPI>0.1)")
    fig.tight_layout()
    fig.savefig(output_dir / "neuron_type_summary.pdf", dpi=200)
    plt.close(fig)


def plot_output_neuron_profiles(reports: list[dict], output_dir: Path) -> None:
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
    for i, r in enumerate(reports):
        ax = axes[0][i]
        neurons = r["output_layer"]["top_neurons"][:15]
        y = np.arange(len(neurons))
        r_vpi = [n["r_vpi"] for n in neurons]
        r_tt = [n["r_tt"] for n in neurons]
        ax.barh(y - 0.15, r_vpi, 0.3, label="r(VPI)", color="#2ca02c")
        ax.barh(y + 0.15, r_tt, 0.3, label="r(T_t)", color="#d62728")
        ax.set_yticks(y)
        ax.set_yticklabels([f"n{n['neuron']} ({n['contribution']:.2f})" for n in neurons],
                           fontsize=7)
        ax.set_xlabel("|r|")
        ax.set_title(r["label"], fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Top output neurons: VPI vs T_t correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "output_neuron_profiles.pdf", dpi=200)
    plt.close(fig)


def plot_wdl_subspace_fraction(reports: list[dict], output_dir: Path) -> None:
    labels = [r["label"] for r in reports]
    ks = [5, 10, 20]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    w = 0.8 / len(ks)
    for ki, k in enumerate(ks):
        vals = [r[f"layer0_mean_frac_in_wdl_top{k}"] for r in reports]
        ax.bar(x + ki * w, vals, w, label=f"top-{k}")
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of z_root weight in WDL subspace")
    ax.legend(fontsize=8)
    ax.set_title("First-layer z_root weights: projection onto WDL readout basis")
    fig.tight_layout()
    fig.savefig(output_dir / "wdl_subspace_fraction.pdf", dpi=200)
    plt.close(fig)


def plot_first_layer_weight_decomposition(reports: list[dict], output_dir: Path) -> None:
    labels = [r["label"] for r in reports]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    z_fracs = [r["layers"][0]["frac_z_norm"] for r in reports]
    tt_fracs = [r["layers"][0]["frac_tt_of_total"] for r in reports]
    ax.bar(x - 0.15, z_fracs, 0.3, label="z_root fraction", color="#1f77b4")
    ax.bar(x + 0.15, tt_fracs, 0.3, label="T_t fraction", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of total weight norm")
    ax.legend(fontsize=8)
    ax.set_title("First-layer input weight decomposition")
    fig.tight_layout()
    fig.savefig(output_dir / "first_layer_weight_decomposition.pdf", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hook helper
# ---------------------------------------------------------------------------


def _run_with_hooks(head: nn.Sequential, feat: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run advantage head, return (output, [post-ReLU activations per hidden layer]).

    Captures the output of each ReLU in the Sequential, giving the post-activation
    hidden state at every hidden layer.  Does NOT include the final Linear output.
    """
    layer_acts: list[torch.Tensor] = []
    hooks = []

    def _make_hook():
        def fn(_mod, _inp, out):
            layer_acts.append(out.detach().cpu())
        return fn

    for module in head:
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(_make_hook()))

    with torch.no_grad():
        output = head(feat)

    for h in hooks:
        h.remove()

    return output.detach().cpu(), layer_acts


def _top_neuron_indices(head: nn.Sequential, layer_acts: list[torch.Tensor],
                        n_top: int = 15) -> np.ndarray:
    """Return indices of top-n penultimate-layer neurons by output contribution."""
    W_out = list(head.modules())
    linears = [m for m in W_out if isinstance(m, nn.Linear)]
    w_last = linears[-1].weight.detach().cpu().numpy().flatten()
    last_hidden = layer_acts[-1].numpy()
    contribution = np.abs(w_last) * last_hidden.std(axis=0)
    return np.argsort(contribution)[::-1][:n_top]


# ---------------------------------------------------------------------------
# Intervention 1: T_t sweep (hold z_root fixed)
# ---------------------------------------------------------------------------


def intervention_tt_sweep(
    model_specs: list[ModelSpec],
    features: torch.Tensor,
    target: np.ndarray,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
    n_examples: int = 50,
    n_tt_points: int = 100,
) -> None:
    """For a sample of z_roots, sweep T_t from 0→120 and plot the response.

    Produces two figures:
      intervention_tt_sweep.pdf         — final scalar output vs T_t
      intervention_tt_sweep_activations.pdf — per-neuron activations vs T_t
    """
    # Stratify: pick examples spanning the target advantage range
    sorted_idx = np.argsort(target)
    step = max(1, len(sorted_idx) // n_examples)
    sample_idx = sorted_idx[::step][:n_examples]

    z_roots = features[sample_idx, :d_embed]  # [n_examples, 128]
    n_ts = features[sample_idx, d_embed]       # [n_examples] — keep original N_t

    tt_grid = torch.linspace(0, 120, n_tt_points)  # [n_tt_points]

    n_models = len(model_specs)
    fig_out, axes_out = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)
    # Activation figure: 2 rows (first hidden layer, penultimate layer) × n_models cols
    n_hidden_layers = None  # determined on first model

    per_model_acts = []  # collect for the activation figure

    for mi, spec in enumerate(model_specs):
        ax = axes_out[0][mi]
        head = load_advantage_head(spec.controller_checkpoint, device)

        # Build [n_examples * n_tt_points, 130] feature matrix
        z_rep = z_roots.unsqueeze(1).expand(-1, n_tt_points, -1)  # [E, T, 128]
        nt_rep = n_ts.unsqueeze(1).expand(-1, n_tt_points)        # [E, T]
        tt_rep = tt_grid.unsqueeze(0).expand(n_examples, -1)      # [E, T]
        feat_grid = torch.cat([
            z_rep.reshape(-1, d_embed),
            nt_rep.reshape(-1, 1),
            tt_rep.reshape(-1, 1),
        ], dim=-1)  # [E*T, 130]

        output, layer_acts = _run_with_hooks(head, feat_grid.to(device))
        pred = output.squeeze(-1).numpy().reshape(n_examples, n_tt_points)

        if n_hidden_layers is None:
            n_hidden_layers = len(layer_acts)

        # --- scalar output plot (same as before) ---
        tt_np = tt_grid.numpy()
        for i in range(n_examples):
            ax.plot(tt_np, pred[i], alpha=0.15, linewidth=0.5, color="steelblue")
        ax.plot(tt_np, pred.mean(axis=0), color="black", linewidth=2, label="mean")
        ax.set_xlabel("T_t (time budget)")
        ax.set_ylabel("Predicted advantage")
        ax.set_title(spec.label, fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        if mi == 0:
            ax.legend(fontsize=7)

        # --- collect activation data ---
        top_idx = _top_neuron_indices(head, layer_acts, n_top=15)
        # Reshape each layer's activations to [E, T, H]
        reshaped = [a.numpy().reshape(n_examples, n_tt_points, -1) for a in layer_acts]
        per_model_acts.append({
            "label": spec.label, "top_idx": top_idx, "layer_acts": reshaped,
            "head": head,
        })

    fig_out.suptitle("T_t sweep: hold z_root fixed, vary time budget")
    fig_out.tight_layout()
    fig_out.savefig(output_dir / "intervention_tt_sweep.pdf", dpi=200)
    plt.close(fig_out)
    logger.info("Saved intervention_tt_sweep.pdf")

    # --- Activation figure ---
    # Rows: first hidden layer, last hidden layer.  Cols: one per model.
    # Each panel: mean activation (across z_root samples) vs T_t for top neurons.
    layer_indices = [0, n_hidden_layers - 1] if n_hidden_layers > 1 else [0]
    layer_names = ["First hidden layer", "Penultimate layer"] if n_hidden_layers > 1 else ["Hidden layer"]

    fig_act, axes_act = plt.subplots(
        len(layer_indices), n_models,
        figsize=(4 * n_models, 4 * len(layer_indices)), squeeze=False,
    )

    cmap = plt.cm.tab20
    for mi, info in enumerate(per_model_acts):
        top_idx = info["top_idx"]
        head = info["head"]
        linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
        w_last = linears[-1].weight.detach().cpu().numpy().flatten()

        for ri, li in enumerate(layer_indices):
            ax = axes_act[ri][mi]
            acts = info["layer_acts"][li]  # [E, T, H]
            mean_acts = acts.mean(axis=0)  # [T, H]

            # For penultimate layer, use top_idx; for first layer, pick by variance
            if li == layer_indices[-1]:
                neurons = top_idx
            else:
                var_per_neuron = mean_acts.var(axis=0)
                neurons = np.argsort(var_per_neuron)[::-1][:15]

            for rank, j in enumerate(neurons):
                color = cmap(rank / 15)
                sign_str = "+" if (li == layer_indices[-1] and w_last[j] > 0) else "-"
                lbl = f"n{j} ({sign_str})" if li == layer_indices[-1] else f"n{j}"
                ax.plot(tt_np, mean_acts[:, j], color=color, linewidth=1,
                        alpha=0.8, label=lbl)

            ax.set_xlabel("T_t")
            ax.set_ylabel("Mean activation")
            if ri == 0:
                ax.set_title(info["label"], fontsize=9)
            if mi == 0:
                ax.set_ylabel(f"{layer_names[ri]}\nMean activation")
            ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
            ax.legend(fontsize=4, ncol=3, loc="upper left")

    fig_act.suptitle("T_t sweep: per-neuron activations (mean over z_root samples)")
    fig_act.tight_layout()
    fig_act.savefig(output_dir / "intervention_tt_sweep_activations.pdf", dpi=200)
    plt.close(fig_act)
    logger.info("Saved intervention_tt_sweep_activations.pdf")


# ---------------------------------------------------------------------------
# Intervention 2: z_root progression along episodes
# ---------------------------------------------------------------------------


def intervention_zroot_progression(
    model_specs: list[ModelSpec],
    encoder: object,  # TreeNN
    packed_manifest: str,
    d_embed: int,
    device: torch.device,
    output_dir: Path,
    n_episodes: int = 12,
    fixed_tt_values: list[float] | None = None,
) -> None:
    """For selected episodes, encode each step's tree, hold T_t fixed,
    and plot the advantage head's output as z_root evolves."""
    from train_fitted_q_controller import (
        PackedControllerEpisodeDataset, PackedControllerCollator,
    )
    from paper_figure_utils import MODEL_COLORS

    if fixed_tt_values is None:
        fixed_tt_values = [10.0, 40.0, 80.0]

    ds = PackedControllerEpisodeDataset(packed_manifest)
    collator = PackedControllerCollator()
    n_total = len(ds)

    # Select episodes with diverse lengths
    rng = np.random.default_rng(42)
    candidates = rng.choice(n_total, size=min(200, n_total), replace=False)
    selected = []
    for idx in candidates:
        ep = ds[int(idx)]
        if len(ep.halt_rewards) > 10:
            selected.append(int(idx))
        if len(selected) >= n_episodes:
            break

    if not selected:
        logger.warning("No suitable episodes for z_root progression; skipping")
        return

    heads = {}
    for spec in model_specs:
        heads[spec.label] = load_advantage_head(spec.controller_checkpoint, device)

    nrows = len(selected)
    ncols = len(fixed_tt_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    all_ep_act_data = []

    encoder.eval()
    with torch.no_grad():
        for row, ep_idx in enumerate(selected):
            ep = ds[ep_idx]
            num_steps = len(ep.halt_rewards)

            # Batch all steps of this episode via the collator → one TreeBatch
            batch = collator([ep])
            encoded = encoder(batch.tree_batch)
            # root_states: one per step in the episode
            z_seq = encoded.root_states.cpu()  # [num_steps, d_embed]
            n_t_seq = ep.tree_sizes.float()     # [num_steps]

            # Collect per-model activations for this episode at each T_t
            ep_act_data = {}

            for col, fixed_tt in enumerate(fixed_tt_values):
                ax = axes[row][col]

                feat = torch.cat([
                    z_seq,
                    n_t_seq.unsqueeze(-1),
                    torch.full((num_steps, 1), fixed_tt),
                ], dim=-1)  # [num_steps, 130]

                steps = np.arange(num_steps)
                for spec in model_specs:
                    output, layer_acts = _run_with_hooks(
                        heads[spec.label], feat.to(device))
                    pred = output.squeeze(-1).numpy()
                    ax.plot(steps, pred, color=MODEL_COLORS.get(spec.label, "gray"),
                            linewidth=1, label=spec.label if row == 0 and col == 0 else None,
                            alpha=0.8)

                    # Store penultimate-layer activations for the activation figure
                    if layer_acts:
                        ep_act_data.setdefault(col, {})[spec.label] = \
                            layer_acts[-1].numpy()  # [num_steps, H]

                target_adv = ep.target_advantages.numpy()
                ax.plot(steps, target_adv, "k-", linewidth=1.5, alpha=0.5,
                        label="target (orig T_t)" if row == 0 and col == 0 else None)

                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.axvline(ep.oracle_stop_step, color="red", linewidth=0.5, linestyle=":")

                if row == 0:
                    ax.set_title(f"T_t={fixed_tt:.0f}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"ep {ep_idx}\n(L={num_steps})", fontsize=7)
                if row == nrows - 1:
                    ax.set_xlabel("Step")

            all_ep_act_data.append((ep_idx, num_steps, ep.oracle_stop_step, ep_act_data))

    if nrows > 0 and ncols > 0:
        axes[0][0].legend(fontsize=5, ncol=2)

    fig.suptitle("z_root progression: hold T_t fixed, vary tree state along episode")
    fig.tight_layout()
    fig.savefig(output_dir / "intervention_zroot_progression.pdf", dpi=200)
    plt.close(fig)
    logger.info("Saved intervention_zroot_progression.pdf")

    # --- Activation figure: per-neuron activations along episodes ---
    # Pick middle T_t column and first model to determine top neurons by variance
    if not all_ep_act_data:
        return
    mid_col = len(fixed_tt_values) // 2
    first_label = model_specs[0].label
    all_acts = np.concatenate(
        [d[3][mid_col][first_label] for d in all_ep_act_data
         if mid_col in d[3] and first_label in d[3][mid_col]], axis=0)
    var_per = all_acts.var(axis=0)
    top_idx = np.argsort(var_per)[::-1][:10]

    n_eps_show = min(4, len(all_ep_act_data))
    fig_act, axes_act = plt.subplots(
        n_eps_show, ncols, figsize=(5 * ncols, 3 * n_eps_show), squeeze=False)
    cmap = plt.cm.tab10

    for ei in range(n_eps_show):
        ep_idx, num_steps, oracle_stop, ep_act_data = all_ep_act_data[ei]
        steps = np.arange(num_steps)
        for col in range(ncols):
            ax = axes_act[ei][col]
            if col not in ep_act_data or first_label not in ep_act_data[col]:
                continue
            acts = ep_act_data[col][first_label]  # [num_steps, H]
            for rank, j in enumerate(top_idx):
                ax.plot(steps, acts[:, j], color=cmap(rank / 10),
                        linewidth=1, alpha=0.8, label=f"n{j}" if ei == 0 and col == 0 else None)
            ax.axvline(oracle_stop, color="red", linewidth=0.5, linestyle=":")
            ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
            if ei == 0:
                ax.set_title(f"T_t={fixed_tt_values[col]:.0f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"ep {ep_idx}\nActivation", fontsize=7)
            if ei == n_eps_show - 1:
                ax.set_xlabel("Step")

    if n_eps_show > 0 and ncols > 0:
        axes_act[0][0].legend(fontsize=5, ncol=2)
    fig_act.suptitle(f"z_root progression: penultimate-layer activations ({first_label})")
    fig_act.tight_layout()
    fig_act.savefig(output_dir / "intervention_zroot_activations.pdf", dpi=200)
    plt.close(fig_act)
    logger.info("Saved intervention_zroot_activations.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanistic analysis of advantage head")
    parser.add_argument("--model-configs", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--cache", required=True, help="Materialized VPI cache from section3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--packed-validation-data",
                        help="Validation manifest for z_root progression (requires GPU)")
    parser.add_argument("--encoder-checkpoint",
                        help="Encoder checkpoint for z_root progression")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)

    with open(args.model_configs) as f:
        raw_configs = json.load(f)
    model_specs = [ModelSpec(**cfg) for cfg in raw_configs]

    # Load cached data
    logger.info("Loading cache: %s", args.cache)
    cached = torch.load(args.cache, weights_only=False)
    features = cached["features"]
    voc = cached["voc"].numpy()
    target = cached["target_advantages"].numpy()
    d_embed = features.shape[1] - 2  # 130 - 2 = 128

    # Load decoder and get WDL readout basis
    logger.info("Loading decoder: %s", args.decoder_checkpoint)
    decoder = load_decoder(args.decoder_checkpoint, d_embed, device)
    wdl_Vt, wdl_S = get_wdl_readout_basis(decoder, d_embed)
    logger.info("WDL readout basis: top-5 singular values: %s",
                ", ".join(f"{s:.3f}" for s in wdl_S[:5].tolist()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each model
    reports = []
    for spec in model_specs:
        logger.info("Analyzing: %s (%s)", spec.label, spec.controller_checkpoint)
        head = load_advantage_head(spec.controller_checkpoint, device)
        report = analyze_model(
            spec.label, head, wdl_Vt, wdl_S, features, voc, target, d_embed,
        )
        reports.append(report)

    # Save JSON report
    with open(output_dir / "advantage_head_analysis.json", "w") as f:
        json.dump(reports, f, indent=2)
    logger.info("Saved analysis to %s", output_dir / "advantage_head_analysis.json")

    # Figures
    plot_neuron_type_summary(reports, output_dir)
    plot_output_neuron_profiles(reports, output_dir)
    plot_wdl_subspace_fraction(reports, output_dir)
    plot_first_layer_weight_decomposition(reports, output_dir)

    # Intervention 1: T_t sweep (runs on cached data, no encoder needed)
    intervention_tt_sweep(model_specs, features, target, d_embed, device, output_dir)

    # Intervention 2: z_root progression (requires encoder + packed data)
    if args.packed_validation_data and args.encoder_checkpoint:
        from paper_section3_figures import load_encoder
        logger.info("Loading encoder for z_root progression: %s", args.encoder_checkpoint)
        encoder = load_encoder(args.encoder_checkpoint, device)
        intervention_zroot_progression(
            model_specs, encoder, args.packed_validation_data,
            d_embed, device, output_dir,
        )
    else:
        logger.info("Skipping z_root progression (no --packed-validation-data / --encoder-checkpoint)")

    logger.info("All analysis saved to %s", output_dir)


if __name__ == "__main__":
    main()
