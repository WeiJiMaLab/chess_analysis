"""
Analysis 1: human vs oracle stop step comparison on human FEN positions.

Computes oracle_stop_step on the generated human trees, joins with the export
manifest for exact move RT, optionally enriches with VOC from pos_with_engine_eval,
and creates comparison plots.

Oracle labels use the controller pack path (``pack.py``):
    trajectory = build_compact_trajectory_from_payload(tree)
    oracle_stop_step = budgeted_oracle_from_trajectory(trajectory, budget, config).optimal_stop_step

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

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.board_tree_features import extract_board_features, extract_tree_features
from src.data.preprocess_mc.oracle import BudgetedOracleConfig
from src.data.preprocess_mc.pack import (
    budgeted_oracle_from_trajectory,
    build_compact_trajectory_from_payload,
)
from analysis._plots import analysis_style, save_fig

_CONFIG = BudgetedOracleConfig()
_DB_PATH = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"
_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
_INDEX_RE = re.compile(r"^(\d+)_root_\d+\.pt$")


def process_tree(t: dict, budget: int) -> dict | None:
    try:
        trajectory = build_compact_trajectory_from_payload(t)
        if trajectory is None:
            return None
        stop_step = budgeted_oracle_from_trajectory(trajectory, budget, _CONFIG).optimal_stop_step
        return {
            **extract_board_features(t["root_position_spec"]),
            **extract_tree_features(t),
            "fen": t["root_position_spec"],
            "oracle_stop_step": stop_step,
        }
    except Exception as exc:
        print(f"Error processing tree: {exc}")
        return None


def _tree_index(path: str) -> int | None:
    match = _INDEX_RE.match(os.path.basename(path))
    return int(match.group(1)) if match else None


def _load_manifest(manifest_path: str) -> pd.DataFrame:
    path = Path(manifest_path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"index", "gid", "move_ply", "move_time", "full_fen"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return df


def enrich_with_voc(df: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """Left-join Stockfish VOC/MQ from pos_with_engine_eval when available."""
    conn = duckdb.connect(db_path, read_only=True)
    tables = {row[0] for row in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'pos_with_engine_eval'"
    ).fetchall()}
    if "pos_with_engine_eval" not in tables:
        conn.close()
        print("pos_with_engine_eval not found; skipping VOC join.")
        return df
    conn.register("_join_frame", df[["gid", "move_ply"]])
    voc_df = conn.execute("""
        SELECT m.gid, m.move_ply, p.voc, p.toptwo AS toptwo_sf, p.mq
        FROM _join_frame m
        LEFT JOIN pos_with_engine_eval p USING (gid, move_ply)
    """).df()
    conn.close()
    return df.merge(voc_df, on=["gid", "move_ply"], how="left")


def _print_correlations(df: pd.DataFrame) -> None:
    print("\n--- Pearson correlations ---")
    features = ["n_possible_moves", "n_self_pieces_exc_pawns", "gain_depth_equiv", "toptwo_equiv"]
    for feat in features:
        r_rt = df["log_rt"].corr(df[feat])
        r_oracle = df["oracle_stop_step"].corr(df[feat])
        print(f"{feat:25s} | log RT r = {r_rt:+.3f} | oracle r = {r_oracle:+.3f}")
    r_rt_oracle = df["log_rt"].corr(df["oracle_stop_step"])
    print(f"\nr(log RT, oracle_stop_step) = {r_rt_oracle:+.3f}  (n={len(df)})")
    if "voc" in df.columns and df["voc"].notna().any():
        n_voc = int(df["voc"].notna().sum())
        print(f"r(log RT, VOC_SF)           = {df['log_rt'].corr(df['voc']):+.3f}  (n={n_voc})")
        print(f"r(oracle_stop_step, VOC_SF) = {df['oracle_stop_step'].corr(df['voc']):+.3f}  (n={n_voc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees-dir", default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_trees_1k")
    parser.add_argument("--manifest", default="/scratch/gpfs/GRIFFITHS/hl4291/tmp/human_fens_1k_manifest.parquet")
    parser.add_argument("--db-path", default=_DB_PATH)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""

    manifest = _load_manifest(args.manifest)
    manifest_fen_by_idx = dict(zip(manifest["index"], manifest["full_fen"]))

    print(f"Scanning trees in {args.trees_dir}...")
    files = sorted(
        os.path.join(args.trees_dir, f)
        for f in os.listdir(args.trees_dir)
        if f.endswith(".pt")
    )
    print(f"Found {len(files)} tree files.")

    rows = []
    n_fen_mismatch = 0
    for path in tqdm(files, desc="Processing trees"):
        idx = _tree_index(path)
        if idx is None:
            continue
        try:
            t = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        expected_fen = manifest_fen_by_idx.get(idx)
        if expected_fen is None:
            continue
        if t.get("root_position_spec") != expected_fen:
            n_fen_mismatch += 1
            continue

        row = process_tree(t, args.budget)
        if row is not None:
            row["index"] = idx
            rows.append(row)

    if n_fen_mismatch:
        print(f"Skipped {n_fen_mismatch} trees whose FEN did not match the manifest "
              f"(stale files from a prior run).")

    df_trees = pd.DataFrame(rows)
    if df_trees.empty:
        print("No trees successfully processed. Exiting.")
        return

    df = df_trees.merge(manifest, on="index", how="inner", suffixes=("", "_manifest"))
    if df.empty:
        print("No rows matched manifest. Check export vs tree filenames.")
        return

    print(f"Matched {len(df)} / {len(manifest)} manifest rows with trees.")
    df["log_rt"] = np.log(df["move_time"].astype(float))
    df = enrich_with_voc(df, args.db_path)
    _print_correlations(df)

    r_rt_oracle = float(df["log_rt"].corr(df["oracle_stop_step"]))
    analysis_style()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["oracle_stop_step"], df["log_rt"], color="#2563EB", alpha=0.4, edgecolors="none")
    bins = np.unique(np.percentile(df["oracle_stop_step"], [0, 25, 50, 75, 100]))
    if len(bins) > 1:
        df["bin"] = pd.cut(df["oracle_stop_step"], bins, include_lowest=True)
        bin_centers = df.groupby("bin", observed=False)["oracle_stop_step"].mean()
        binned = df.groupby("bin", observed=False)["log_rt"].mean()
        plt.plot(bin_centers, binned, color="#DC2626", linewidth=2.5, marker="o", label="Binned trend")
    plt.xlabel("oracle_stop_step")
    plt.ylabel("human log(RT)")
    plt.title(f"A1: oracle_stop_step vs. human log(RT)  r={r_rt_oracle:+.3f}  n={len(df)}")
    plt.tight_layout()
    save_fig(str(_FIGURES_DIR / f"human_oracle_rt_comparison{tag}.png"))

    features = ["n_possible_moves", "n_self_pieces_exc_pawns", "gain_depth_equiv", "toptwo_equiv"]
    labels = ["Branching", "Material", "gain_depth", "toptwo"]
    oracle_rs = [float(df["oracle_stop_step"].corr(df[f])) for f in features]
    human_rs = [float(df["log_rt"].corr(df[f])) for f in features]
    x = np.arange(len(features))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, oracle_rs, width, label="oracle_stop_step (Lc0, 96-node)", color="#2563EB")
    plt.bar(x + width / 2, human_rs, width, label="human log(RT)", color="#10B981")
    plt.axhline(0, color="black", lw=1.2, linestyle="--")
    plt.xticks(x, labels)
    plt.ylabel("Pearson r")
    plt.title(f"A1: Feature correlations  n={len(df)}")
    plt.legend()
    plt.tight_layout()
    save_fig(str(_FIGURES_DIR / f"human_oracle_features_comparison{tag}.png"))


if __name__ == "__main__":
    main()
