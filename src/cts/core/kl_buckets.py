"""Shared by ``cts.analysis.audit_encoder_kl`` and ``cts.train.gnn_pretrain`` so
their bucketed-KL numbers are directly comparable."""

from __future__ import annotations

from typing import List

import torch


def compute_subtree_sizes(
    parent_index: torch.Tensor,
    depth: torch.Tensor,
    *,
    max_depth_hint: int = 30,
) -> torch.Tensor:
    """Vectorized descendant count per node, including the node itself.

    Reverse-BFS accumulation, processing depth ``d`` from ``max_depth_hint``
    down to ``1`` and adding each depth-``d`` node's current subtree size to
    its parent via ``scatter_add_``. The inner loop is **sync-free**: each
    iteration uses only fixed-shape mask-multiply + scatter ops, so no
    ``.nonzero()`` / ``.item()`` / ``.any()`` calls force GPU→CPU stalls.
    The Python iteration count is a constant ``max_depth_hint`` (extra
    no-op iterations on shallower trees are cheap — each is a handful of
    elementwise ops on the node count).

    Roots have ``parent_index == -1``; we clamp to 0 and mask their
    contribution to 0 via ``is_not_root`` so the scatter into slot 0 is
    a no-op for root entries.

    Args:
        parent_index: ``[N]`` long tensor; ``-1`` for roots in a batched forest.
        depth: ``[N]`` long tensor; plies from each node's root.
        max_depth_hint: upper bound on tree depth in the input; iterations
            past the actual max are no-op-equivalent. Default 30 matches
            the project's ``MAX_DEPTH`` setting for chess trees.
    """
    num_nodes = parent_index.shape[0]
    if num_nodes == 0:
        return torch.zeros(0, dtype=torch.long, device=parent_index.device)

    subtree_size = torch.ones(num_nodes, dtype=torch.long, device=parent_index.device)
    # Clamp roots' parent_index from -1 to 0 so it's a valid scatter target;
    # is_not_root zeroes their contribution so it doesn't matter where they
    # would have scattered.
    parent_index_clamped = parent_index.clamp_min(0)
    is_not_root = (parent_index >= 0).to(torch.long)

    for current_depth in range(max_depth_hint, 0, -1):
        at_this_depth = (depth == current_depth).to(torch.long)
        update = subtree_size * at_this_depth * is_not_root
        subtree_size.scatter_add_(0, parent_index_clamped, update)
    return subtree_size


def size_bin(subtree_size: torch.Tensor, max_bin: int) -> torch.Tensor:
    """Map subtree size to ``floor(log2(size))`` clamped to ``[0, max_bin]``.

    Edges: bin 0 ↔ size==1 (leaves), bin 1 ↔ size∈[2,3], bin 2 ↔ [4,7],
    bin 3 ↔ [8,15], … bin ``max_bin`` ↔ ≥2**max_bin.
    """
    size_clamped = subtree_size.clamp_min(1).to(torch.float64)
    bins = torch.floor(torch.log2(size_clamped)).to(torch.long)
    return bins.clamp_max(max_bin)


def depth_bin(depth: torch.Tensor, max_bin: int) -> torch.Tensor:
    """Clamp parent depth to ``[0, max_bin]``; deeper parents fall in the last column."""
    return depth.clamp_max(max_bin)


def size_bin_labels(max_bin: int) -> List[str]:
    """Pretty labels matching ``size_bin``'s ``floor(log2(size))`` semantics."""
    labels: List[str] = []
    for bin_index in range(max_bin + 1):
        if bin_index == max_bin:
            labels.append(f"≥{2 ** bin_index}")
        elif bin_index == 0:
            labels.append("1")
        else:
            labels.append(f"{2 ** bin_index}–{2 ** (bin_index + 1) - 1}")
    return labels


def depth_bin_labels(max_bin: int) -> List[str]:
    """Pretty labels: ``"0", "1", …, "≥max_bin"``."""
    labels = [str(i) for i in range(max_bin)]
    labels.append(f"≥{max_bin}")
    return labels


