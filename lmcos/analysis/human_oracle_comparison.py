"""
Analysis 1: human vs oracle stop step comparison on human FEN positions.

Computes oracle_stop_step on the generated human trees, joins with the export
manifest for exact move RT, optionally enriches with VOC from pos_with_engine_eval,
and creates comparison plots.

Usage:
    python analysis/human_oracle_comparison.py \\
        --trees-dir /scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_1k \\
        --manifest /scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k_manifest.parquet
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import chess
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "human_analytics"))

from src.data.preprocess_mc.oracle import (
    BudgetedOracleConfig,
    compute_budgeted_oracle,
)
from src.core.tensorizer import TensorizedTreeExample
from src.models.mc import MetaController

_DB_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_CONFIG = BudgetedOracleConfig()
_INDEX_RE = re.compile(r"^(\d+)_root_\d+\.pt$")

_RE_WHITE = re.compile(r"[RNBQK]")
_RE_BLACK = re.compile(r"[rnbqk]")


def compute_oracle_stop_step(
    halt_rewards: list[float],
    tree_sizes: list[int],
    budget: int,
    config: BudgetedOracleConfig,
) -> int:
    """Run the DP oracle and return the optimal halt step from snapshot 0."""
    policy = compute_budgeted_oracle(halt_rewards, tree_sizes, budget, config)
    return policy.optimal_stop_step


def extract_board_features(fen: str) -> dict:
    """Position features from a root FEN string."""
    parts = fen.split()
    side = parts[1] if len(parts) > 1 else "w"
    placement = parts[0]
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    move_ply = (fullmove - 1) * 2 + (0 if side == "w" else 1)

    board = chess.Board(fen)
    n_possible = board.legal_moves.count()
    flat = placement.replace("/", "")
    n_self = len((_RE_WHITE if side == "w" else _RE_BLACK).findall(flat))

    return {"n_possible_moves": n_possible, "n_self_pieces_exc_pawns": n_self, "move_ply": move_ply}


def extract_tree_features(t: dict) -> dict:
    """toptwo_equiv and gain_depth_equiv from oracle_root_q_trace."""
    q = t["oracle_root_q_trace"]
    final_q = q[-1]
    nonzero = final_q[final_q != 0]

    if len(nonzero) >= 2:
        v = nonzero.topk(2).values
        toptwo = float((v[0] - v[1]).abs().item())
    else:
        toptwo = float("nan")

    first_nz = (q.sum(dim=1) != 0).nonzero(as_tuple=True)[0]
    best_q_final = final_q.max().item()
    if len(first_nz) > 0:
        gain = best_q_final - q[first_nz[0].item()].max().item()
    else:
        gain = float("nan")

    return {"toptwo_equiv": toptwo, "gain_depth_equiv": gain}


def process_tree(t: dict, budget: int) -> dict | None:
    """Extract features and oracle stop step from tree dictionary."""
    try:
        fen = t["root_position_spec"]
        board_feats = extract_board_features(fen)
        tree_feats = extract_tree_features(t)

        q = t["oracle_root_q_trace"]
        best_idx = t["oracle_best_move_index"]
        T = len(q)

        halt_rewards = [float(q[s, best_idx[s].item()].item()) for s in range(T)]
        tree_sizes = [1] * T

        row = {**board_feats, **tree_feats}
        row["fen_6field"] = fen
        row["fen_4field"] = " ".join(fen.split()[:4])

        b = min(budget, T)
        row["oracle_stop_step"] = compute_oracle_stop_step(
            halt_rewards[:b], tree_sizes[:b], b, _CONFIG
        )
        return row
    except Exception as e:
        print(f"Error processing tree: {e}")
        return None


def _tree_index(path: str) -> int | None:
    match = _INDEX_RE.match(os.path.basename(path))
    return int(match.group(1)) if match else None


def _load_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_parquet(manifest_path)
    required = {"index", "gid", "move_ply", "move_time", "full_fen"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df


def _enrich_with_voc(df: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """Left-join Stockfish VOC/MQ from pos_with_engine_eval when available."""
    conn = duckdb.connect(db_path, read_only=True)
    tables = {row[0] for row in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'pos_with_engine_eval'"
    ).fetchall()}
    if "pos_with_engine_eval" not in tables:
        conn.close()
        print("pos_with_engine_eval not found; skipping VOC join.")
        return df

    conn.register("manifest_join", df[["gid", "move_ply"]])
    voc_df = conn.execute("""
        SELECT m.gid, m.move_ply, p.voc, p.toptwo, p.mq
        FROM manifest_join m
        LEFT JOIN pos_with_engine_eval p USING (gid, move_ply)
    """).df()
    conn.close()
    return df.merge(voc_df, on=["gid", "move_ply"], how="left")


def _analysis_style() -> None:
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _print_correlations(df: pd.DataFrame) -> None:
    print("\n--- Pearson correlations ---")
    features = ["n_possible_moves", "n_self_pieces_exc_pawns", "gain_depth_equiv", "toptwo_equiv"]
    for feat in features:
        r_rt = df["log_rt"].corr(df[feat])
        r_oracle = df["oracle_stop_step"].corr(df[feat])
        print(f"{feat:25s} | log RT r = {r_rt:+.3f} | oracle r = {r_oracle:+.3f}")

    r_rt_oracle = df["log_rt"].corr(df["oracle_stop_step"])
    print(f"\nr(log RT, oracle_stop_step) = {r_rt_oracle:+.3f}")

    if "voc" in df.columns and df["voc"].notna().any():
        n_voc = int(df["voc"].notna().sum())
        r_voc_rt = df["log_rt"].corr(df["voc"])
        r_voc_oracle = df["oracle_stop_step"].corr(df["voc"])
        print(f"r(log RT, VOC)              = {r_voc_rt:+.3f}  (n={n_voc})")
        print(f"r(oracle_stop_step, VOC)    = {r_voc_oracle:+.3f}  (n={n_voc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_1k")
    parser.add_argument(
        "--manifest",
        default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k_manifest.parquet",
    )
    parser.add_argument("--db-path", default=_DB_PATH)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--output-tag", default="", help="Suffix for figure filenames")
    parser.add_argument("--model-checkpoint", default=None, help="Optional GNN+MC checkpoint for Plot C")
    args = parser.parse_args()

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""

    manifest = _load_manifest(args.manifest)

    print(f"Scanning trees in {args.trees_dir}...")
    files = sorted(
        os.path.join(args.trees_dir, f)
        for f in os.listdir(args.trees_dir)
        if f.endswith(".pt")
    )
    print(f"Found {len(files)} tree files.")

    rows = []
    for path in tqdm(files, desc="Processing trees"):
        idx = _tree_index(path)
        if idx is None:
            print(f"Skipping unrecognized filename: {path}")
            continue
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
            row = process_tree(t, args.budget)
            if row is not None:
                row["index"] = idx
                rows.append(row)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    df_trees = pd.DataFrame(rows)
    if df_trees.empty:
        print("No trees successfully processed. Exiting.")
        return

    df = df_trees.merge(manifest, on="index", how="inner", suffixes=("", "_manifest"))
    if df.empty:
        print("No rows matched manifest by index. Check export vs tree filenames.")
        return

    fen_mismatch = (df["fen_6field"] != df["full_fen"]).sum()
    if fen_mismatch:
        print(f"Warning: {fen_mismatch} rows have tree FEN != manifest FEN.")

    df["log_rt"] = np.log(df["move_time"].astype(float))
    df = _enrich_with_voc(df, args.db_path)

    print(f"Matched {len(df)} / {len(manifest)} manifest rows with trees.")
    _print_correlations(df)

    r_rt_oracle = df["log_rt"].corr(df["oracle_stop_step"])

    _analysis_style()
    plt.figure(figsize=(7, 5))
    plt.scatter(df["oracle_stop_step"], df["log_rt"], color="#2563EB", alpha=0.4, edgecolors="none")
    bins = np.unique(np.percentile(df["oracle_stop_step"], [0, 25, 50, 75, 100]))
    if len(bins) > 1:
        df["bin"] = pd.cut(df["oracle_stop_step"], bins, include_lowest=True)
        binned = df.groupby("bin", observed=False)["log_rt"].mean()
        bin_centers = df.groupby("bin", observed=False)["oracle_stop_step"].mean()
        plt.plot(bin_centers, binned, color="#DC2626", linewidth=2.5, marker="o", label="Binned trend")

    plt.xlabel("oracle_stop_step")
    plt.ylabel("human log(RT)")
    plt.title(f"A1: oracle_stop_step vs. human log(RT) (r = {r_rt_oracle:+.3f}, n={len(df)})")
    plt.tight_layout()
    rt_path = _FIGURES_DIR / f"human_oracle_rt_comparison{tag}.png"
    plt.savefig(rt_path, dpi=150)
    plt.close()
    print(f"Saved Plot A to {rt_path}")

    features = ["n_possible_moves", "n_self_pieces_exc_pawns", "gain_depth_equiv", "toptwo_equiv"]
    plt.figure(figsize=(8, 5))
    x = np.arange(len(features))
    width = 0.35
    oracle_rs = [df["oracle_stop_step"].corr(df[f]) for f in features]
    human_rs = [df["log_rt"].corr(df[f]) for f in features]
    plt.bar(x - width / 2, oracle_rs, width, label="oracle_stop_step", color="#2563EB")
    plt.bar(x + width / 2, human_rs, width, label="human log(RT)", color="#10B981")
    plt.xticks(x, [f.replace("_equiv", "") for f in features])
    plt.ylabel("Pearson correlation (r)")
    plt.title(f"A1: Correlation with Board/Tree Features (n={len(df)})")
    plt.legend()
    plt.tight_layout()
    feat_path = _FIGURES_DIR / f"human_oracle_features_comparison{tag}.png"
    plt.savefig(feat_path, dpi=150)
    plt.close()
    print(f"Saved Plot B to {feat_path}")

    if "voc" in df.columns and df["voc"].notna().sum() >= 10:
        sub = df.dropna(subset=["voc"])
        r_voc_oracle = sub["oracle_stop_step"].corr(sub["voc"])
        plt.figure(figsize=(7, 5))
        plt.scatter(sub["voc"], sub["oracle_stop_step"], color="#7C3AED", alpha=0.4, edgecolors="none")
        plt.xlabel("VOC (Stockfish)")
        plt.ylabel("oracle_stop_step (Lc0)")
        plt.title(f"oracle_stop_step vs VOC (r = {r_voc_oracle:+.3f}, n={len(sub)})")
        plt.tight_layout()
        voc_path = _FIGURES_DIR / f"human_oracle_voc_comparison{tag}.png"
        plt.savefig(voc_path, dpi=150)
        plt.close()
        print(f"Saved Plot C to {voc_path}")


if __name__ == "__main__":
    main()
