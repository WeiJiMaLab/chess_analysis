"""Shared data-loading helpers for lmcos analysis scripts.

All analysis scripts load trees from the same filtered_shard directory tree.
This module centralises the discovery, shuffle, and load loop so each script
only needs to supply a per-tree processing function.
"""
from __future__ import annotations

import os
from typing import Callable, TypeVar

import numpy as np
import torch
from tqdm import tqdm

T = TypeVar("T")


def list_shard_files(trees_root: str, seed: int = 42) -> list[str]:
    """Return a deterministically shuffled list of all .pt files under filtered_shard dirs."""
    dirs = sorted(d for d in os.listdir(trees_root) if d.startswith("filtered_shard"))
    files: list[str] = []
    for d in dirs:
        p = os.path.join(trees_root, d)
        files.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pt"))
    np.random.default_rng(seed).shuffle(files)
    return files


def load_shard_trees(
    trees_root: str,
    n_max: int,
    process_fn: Callable[[dict], T | None],
    seed: int = 42,
    desc: str = "Loading trees",
) -> list[T]:
    """Load up to n_max trees from filtered_shard dirs, apply process_fn, collect non-None results.

    Silently skips files that fail to load or whose process_fn raises / returns None.
    """
    files = list_shard_files(trees_root, seed)
    results: list[T] = []
    for path in tqdm(files[:n_max], desc=desc):
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            row = process_fn(t)
            if row is not None:
                results.append(row)
        except Exception:
            pass
    return results
