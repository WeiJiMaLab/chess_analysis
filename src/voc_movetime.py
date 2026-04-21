import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from utils import (
    compute_metrics_by_qbin, plot_metrics, 
    apply_poster_style, MAIN_COLOR, FONT_SIZE_LABEL
)

# Poster Style Constants
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(base_dir, "src", "figures", "voc_movetime")

def plot_quad_view(df: pd.DataFrame, n_games: int):
    """Generate 2x2 Quad-View for VOC analysis."""
    apply_poster_style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Density [0, 0]
    axes[0, 0].hexbin(df["voc_sqrt"], df["ln_move_time"], gridsize=20, cmap="Purples", mincnt=1)
    sns.regplot(x="voc_sqrt", y="ln_move_time", data=df, scatter=False, color="indigo", ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\sqrt{VOC}$", fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel(r"$\log(T)$", fontsize=FONT_SIZE_LABEL)
    
    # 2. Binned Trend [0, 1]
    df_sorted = df.sort_values("voc_sqrt")
    df_sorted["qbins"], qbin_edges = pd.qcut(df_sorted["voc_sqrt"], q=40, labels=False, retbins=True, duplicates="drop")
    df_tmp = df_sorted.copy()
    df_tmp["move_time"] = df_tmp["ln_move_time"]
    metrics = compute_metrics_by_qbin(df_tmp, qbin_edges)
    plot_metrics(metrics, color="indigo", ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\sqrt{VOC}$", fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel(r"Mean $\log(T)$", fontsize=FONT_SIZE_LABEL)
    
    # 3. Ply Stability [1, 0]
    results = []
    # Filter to plys with enough data
    common_plys = df["ply"].value_counts()[df["ply"].value_counts() > 40].index
    for ply in sorted(common_plys):
        df_ply = df[df["ply"] == ply]
        m = smf.ols("ln_move_time ~ voc_sqrt", data=df_ply).fit()
        results.append({"ply": ply, "coeff": m.params["voc_sqrt"], "bse": m.bse["voc_sqrt"]})
    
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        axes[1, 0].errorbar(df_res["ply"], df_res["coeff"], yerr=1.96 * df_res["bse"], fmt='o', color="indigo", ecolor='lightgray', elinewidth=3, capsize=0)
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_ylabel(r"Slope ($\beta_{\sqrt{VOC}}$)", fontsize=FONT_SIZE_LABEL)
    
    # 4. Quantile Rank [1, 1]
    metrics_rank = {k: list(v) for k, v in metrics.items()}
    metrics_rank["x"] = np.linspace(0, 1, len(metrics_rank["x"]))
    plot_metrics(metrics_rank, color="indigo", ax=axes[1, 1])
    axes[1, 1].set_xlabel("Quantile Rank (VOC)", fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel(r"Mean $\log(T)$", fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    save_path = os.path.join(FIGURE_DIR, f"voc_quad_view_{n_games}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Poster saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze Value of Computation (VOC) results.")
    parser.add_argument("--n_games", type=int, default=500, help="Number of games to analyze")
    args = parser.parse_args()

    # Ground paths relative to the project root (parent of 'src')
    results_file = os.path.join(base_dir, "data", "processed", f"voc_{args.n_games}.csv")
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run slurm/script_compute_voc.py first.")
        return

    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)

    # Preprocess
    df["ln_move_time"] = np.log(pd.to_numeric(df["move_time"], errors="coerce").clip(lower=0.1))
    df["voc_sqrt"] = np.sqrt(pd.to_numeric(df["voc"], errors="coerce"))
    
    # Clean NaNs
    df = df.dropna(subset=["ln_move_time", "voc_sqrt", "ply"])
    
    # Analysis summary
    model = smf.ols("ln_move_time ~ voc_sqrt + ply", data=df).fit()
    print(f"\n--- VOC Analysis Summary ({args.n_games} games) ---")
    print(model.summary().tables[1])
    
    plot_quad_view(df, args.n_games)
    print("\n✅ VOC analysis complete.")

if __name__ == "__main__":
    main()
