"""Generate one search tree from a CSV FEN row, pack tensors, profile."""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import time
from pathlib import Path

import pandas as pd
import torch

from metacontrol.core.providers import LC0ExpansionProvider, StockfishExpansionProvider
from metacontrol.core.schemas import GeneratorConfig
from metacontrol.data.generator import TreeSearch
from metacontrol.data.tree_pack import pack_single_tree_like_legacy_shard


DEFAULT_LC0 = "/scratch/gpfs/GRIFFITHS/ysagiv/tools/lc0/build/release/lc0"
DEFAULT_WEIGHTS = "/scratch/gpfs/GRIFFITHS/ysagiv/chess/weights/t1-256x10-distilled-swa-2432500.pb.gz"
DEFAULT_CSV = "/scratch/gpfs/GRIFFITHS/hl4291/data/metacontrol_example.csv"
DEFAULT_OUT = "/scratch/gpfs/GRIFFITHS/hl4291/data/trees/example_tree_00001.pt"
DEFAULT_PROFILE = "/scratch/gpfs/GRIFFITHS/hl4291/data/trees/generate_and_profile.pstats"


def _make_provider(args: argparse.Namespace):
    if args.engine == "lc0":
        if not os.path.isfile(args.lc0_bin):
            raise FileNotFoundError(f"lc0 not found: {args.lc0_bin}")
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"weights not found: {args.weights}")
        return LC0ExpansionProvider(args.lc0_bin, args.weights, nodes=args.nodes)
    if not os.path.isfile(args.stockfish_bin):
        raise FileNotFoundError(f"stockfish not found: {args.stockfish_bin}")
    return StockfishExpansionProvider(args.stockfish_bin, nodes=args.nodes)


def generate_pack_profile(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty")
    fen = str(df.iloc[0]["full_fen"])

    provider = _make_provider(args)
    config = GeneratorConfig(max_nodes=args.max_nodes, max_depth=args.max_depth, c_puct=args.c_puct)
    search = TreeSearch(provider, config)

    t_wall0 = time.perf_counter()
    result = search.generate(fen)
    t_grow = time.perf_counter() - t_wall0

    t_pack0 = time.perf_counter()
    packed = pack_single_tree_like_legacy_shard(result, continue_cost=args.continue_cost)
    torch.save(packed, out_path)
    t_pack = time.perf_counter() - t_pack0

    total = time.perf_counter() - t_wall0
    print(
        f"FEN row 0: nodes={packed['num_nodes']} edges={packed['num_edges']} "
        f"snapshots={packed['num_snapshots']} expansions={result.num_expansions}"
    )
    print(f"Wall: grow={t_grow:.3f}s pack+save={t_pack:.3f}s total={total:.3f}s → {out_path}")
    if total > args.max_wall_seconds:
        print(f"WARNING: total wall time {total:.1f}s exceeds budget {args.max_wall_seconds}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--engine", choices=("lc0", "stockfish"), default="lc0")
    parser.add_argument("--lc0-bin", default=os.environ.get("LC0_BIN", DEFAULT_LC0))
    parser.add_argument(
        "--weights",
        default=os.environ.get("LC0_WEIGHTS", DEFAULT_WEIGHTS),
    )
    parser.add_argument("--stockfish-bin", default=os.environ.get("STOCKFISH_BIN", "stockfish"))
    parser.add_argument("--nodes", type=int, default=64, help="Per-expansion engine node budget")
    parser.add_argument("--max-nodes", type=int, default=64, help="PUCT expansion budget")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--continue-cost", type=float, default=1e-3)
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    parser.add_argument("--profile-out", default=DEFAULT_PROFILE)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()

    if args.no_profile:
        generate_pack_profile(args)
        return

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        generate_pack_profile(args)
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
        print("\n--- Top 25 by cumulative time ---")
        stats.print_stats(25)
        pout = Path(args.profile_out)
        pout.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(pout))
        print(f"\nWrote pstats: {pout}")


if __name__ == "__main__":
    main()
