"""Retrain ChildWdlHead decoder on frozen encoder outputs at ALL edges.

Steps:
1. Load frozen encoder from checkpoint.
2. One-pass materialization: encode every tree, collect per-edge
   (parent_state || slot_embed, wdl_target) tuples.
3. Train a fresh ChildWdlHead on the materialized flat dataset with
   cross-entropy loss.
4. Save decoder checkpoint.

Usage:
    python retrain_child_wdl_decoder.py \
        --encoder-checkpoint path/to/encoder.pt \
        --train-data path/to/train_manifest.json \
        --val-data path/to/val_manifest.json \
        --output-checkpoint path/to/decoder.pt
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

from GNN import ChildWdlHead, TreeNN
from cts_pretrain import load_pretrain_example_dataset
from schema import tree_encoder_feature_schema
from tensorizer import TreeTensorizer, TensorizedTreeExample, collate_tensorized_examples

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder construction helpers
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


def _build_encoder(state_dict: dict, device: torch.device, k: int = 1) -> TreeNN:
    p = _infer_encoder_params(state_dict)
    encoder = TreeNN(
        k=k, node_feat=p["node_feat"], device=str(device),
        node_embed_hidden=p["node_embed_hidden"],
        d_embed=p["d_embed"], d_message=p["d_message"],
        n_heads=p["n_heads"], d_att=p["d_att"], sequential=True,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _collate_raw(batch, tensorizer: TreeTensorizer):
    trees = [ex.tree for ex in batch]
    tree_batch = tensorizer.tensorize_forest(trees)
    targets = []
    for ex in batch:
        targets.append(tensorizer.edge_wdl_target_tensor(ex.tree, ex.edge_wdl_targets))
    return tree_batch, torch.cat(targets) if targets else torch.empty(0, 3)


def materialize_edges(
    encoder: TreeNN, dataset, device: torch.device,
    batch_size: int = 4, num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all trees and collect per-edge (features, targets)."""
    sample = dataset[0]
    is_packed = isinstance(sample, TensorizedTreeExample)

    if is_packed:
        collate_fn = getattr(dataset, "collate_fn", collate_tensorized_examples)
    else:
        tensorizer = TreeTensorizer(tree_encoder_feature_schema(), device="cpu")
        collate_fn = lambda batch: _collate_raw(batch, tensorizer)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    all_features: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_edges = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if is_packed:
                tree_batch, _ = batch_data
                edge_targets = tree_batch.edge_wdl_targets
                if edge_targets is None:
                    raise ValueError("Packed data must include edge_wdl_targets")
            else:
                tree_batch, edge_targets = batch_data

            encoded = encoder(tree_batch)
            ep = tree_batch.edge_parent.to(device)
            es = tree_batch.edge_slot.to(device)

            parent_states = encoded.node_states[ep]
            slot_embeds = encoder.slot_embeddings(es)
            features = torch.cat([parent_states, slot_embeds], dim=-1)

            all_features.append(features.cpu())
            all_targets.append(edge_targets.cpu())
            total_edges += features.shape[0]

            if (batch_idx + 1) % 200 == 0:
                logger.info("  %d batches, %d edges", batch_idx + 1, total_edges)

    logger.info("Materialized %d edges", total_edges)
    return torch.cat(all_features), torch.cat(all_targets)


# ---------------------------------------------------------------------------
# Decoder training
# ---------------------------------------------------------------------------


def _evaluate(decoder: ChildWdlHead, features: torch.Tensor, targets: torch.Tensor,
              d_embed: int, device: torch.device, batch_size: int = 8192) -> float:
    decoder.eval()
    total = 0.0
    n = 0
    ds = TensorDataset(features, targets)
    for fb, tb in DataLoader(ds, batch_size=batch_size):
        fb, tb = fb.to(device), tb.to(device)
        logits = decoder(fb[:, :d_embed], fb[:, d_embed:])
        loss = -(tb * F.log_softmax(logits, dim=-1)).sum(dim=-1).sum()
        total += loss.item()
        n += fb.shape[0]
    return total / n


def train_decoder(
    train_feat: torch.Tensor, train_tgt: torch.Tensor,
    val_feat: torch.Tensor | None, val_tgt: torch.Tensor | None,
    d_embed: int, hidden_dim: int,
    epochs: int, lr: float, batch_size: int, device: torch.device,
) -> ChildWdlHead:
    decoder = ChildWdlHead(d_embed=d_embed, hidden_dim=hidden_dim, device=str(device))
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    ds = TensorDataset(train_feat, train_tgt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0.0
        n = 0
        for fb, tb in loader:
            fb, tb = fb.to(device), tb.to(device)
            logits = decoder(fb[:, :d_embed], fb[:, d_embed:])
            loss = -(tb * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * fb.shape[0]
            n += fb.shape[0]

        train_loss = total_loss / n
        if val_feat is not None and val_tgt is not None:
            val_loss = _evaluate(decoder, val_feat, val_tgt, d_embed, device)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = copy.deepcopy(decoder.state_dict())
            logger.info("Epoch %d/%d  train=%.4f  val=%.4f%s",
                        epoch, epochs, train_loss, val_loss, " *" if improved else "")
        else:
            logger.info("Epoch %d/%d  train=%.4f", epoch, epochs, train_loss)

    if best_state is not None:
        decoder.load_state_dict(best_state)
        logger.info("Restored best decoder (val_loss=%.4f)", best_val_loss)

    return decoder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain ChildWdlHead on frozen encoder")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--train-data", required=True, help="Train manifest/dir")
    parser.add_argument("--val-data", help="Validation manifest/dir")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048, help="Decoder training batch")
    parser.add_argument("--materialize-batch-size", type=int, default=4, help="Encoder fwd batch")
    parser.add_argument("--decoder-hidden", type=int, default=128)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)

    # Load encoder
    ckpt = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["encoder_state_dict"]
    p = _infer_encoder_params(state)
    logger.info("Encoder params: d_embed=%d node_feat=%d d_message=%d d_att=%d",
                p["d_embed"], p["node_feat"], p["d_message"], p["d_att"])

    encoder = _build_encoder(state, device, k=args.k)

    # Load data
    logger.info("Loading training data: %s", args.train_data)
    train_ds = load_pretrain_example_dataset(args.train_data)
    logger.info("  %d examples", len(train_ds))

    val_ds = None
    if args.val_data:
        logger.info("Loading validation data: %s", args.val_data)
        val_ds = load_pretrain_example_dataset(args.val_data)
        logger.info("  %d examples", len(val_ds))

    # Materialize
    logger.info("Materializing training edges...")
    train_feat, train_tgt = materialize_edges(
        encoder, train_ds, device, args.materialize_batch_size, args.num_workers,
    )

    val_feat, val_tgt = None, None
    if val_ds is not None:
        logger.info("Materializing validation edges...")
        val_feat, val_tgt = materialize_edges(
            encoder, val_ds, device, args.materialize_batch_size, args.num_workers,
        )

    # Train
    logger.info("Training decoder: epochs=%d lr=%g batch=%d hidden=%d",
                args.epochs, args.lr, args.batch_size, args.decoder_hidden)
    decoder = train_decoder(
        train_feat, train_tgt, val_feat, val_tgt,
        d_embed=p["d_embed"], hidden_dim=args.decoder_hidden,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device,
    )

    # Final validation
    final_val_loss = None
    if val_feat is not None and val_tgt is not None:
        final_val_loss = _evaluate(decoder, val_feat, val_tgt, p["d_embed"], device)
        logger.info("Final validation loss: %.4f", final_val_loss)

    # Save
    torch.save({
        "decoder_state_dict": decoder.state_dict(),
        "metadata": {
            "d_embed": p["d_embed"],
            "decoder_hidden": args.decoder_hidden,
            "encoder_checkpoint": args.encoder_checkpoint,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_edges": int(train_feat.shape[0]),
            "val_edges": int(val_feat.shape[0]) if val_feat is not None else 0,
            "final_val_loss": final_val_loss,
        },
    }, args.output_checkpoint)
    logger.info("Saved decoder to %s", args.output_checkpoint)


if __name__ == "__main__":
    main()
