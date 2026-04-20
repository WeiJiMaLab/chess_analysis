"""
Analysis of the impact of move ply on response times.
This script replicates the peak-finding logic used in timing models,
testing for the significance of the middle-game response time arc.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from utils import (
    MAIN_COLOR, compute_metrics_by_bin, compute_metrics_by_qbin, plot_metrics,
    apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS
)

# Constants for this analysis (Poster Style)
MAX_PLY_FOR_PLOT = 120

def analyze_ply_effect(df: pd.DataFrame):
    """
    Perform statistical analysis of move_ply on move_time, including
    quadratic regressions on natural log scale.
    """
    df = df.copy()
    df["move_time"] = pd.to_numeric(df["move_time"], errors="coerce")
    df["move_ply"] = pd.to_numeric(df["move_ply"], errors="coerce")
    df = df.dropna(subset=["move_time", "move_ply"])
    
    # Quadratic model on natural log scale
    df["move_ply_sq"] = df["move_ply"] ** 2
    df["ln_move_time"] = np.log(df["move_time"].astype(float).clip(lower=0.1))
    
    m_log = smf.ols("ln_move_time ~ move_ply + move_ply_sq", data=df).fit()
    print("\n--- Log-Quadratic Model: ln(T) ~ move_ply + move_ply^2 ---")
    print(m_log.summary().tables[1])
    
    b_log, a_log = m_log.params["move_ply"], m_log.params["move_ply_sq"]
    print(f"Predicted Peak Ply (ln scale): {-b_log / (2 * a_log):.2f}")
    
    return df

def plot_ply_impact(df: pd.DataFrame, figures_dir: str):
    """
    Generate unified plots showing the impact of move ply on response times
    using Poster Style.
    """
    apply_poster_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Drop raw move_time to avoid collision after renaming ln_move_time -> move_time
    df_ln = df.drop(columns=["move_time"])
    col_map = {"ln_move_time": "move_time"}

    # --- Plot A: ln(T) by ply (Raw) ---
    df_early = df_ln[df_ln["move_ply"] <= MAX_PLY_FOR_PLOT].copy()
    df_early["bin"] = df_early["move_ply"]
    
    df_early_utils = df_early.rename(columns=col_map)
    metrics_raw = compute_metrics_by_bin(df_early_utils)
    
    plt.sca(axes[0])
    plot_metrics(metrics_raw, color=MAIN_COLOR)
    sns.regplot(x="move_ply", y="move_time", data=df_early_utils, scatter=False, 
                order=2, color="red", label=r"$\ln(T)$ Quadratic Fit")
    
    axes[0].set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel(r"Mean $\ln(T)$", fontsize=FONT_SIZE_LABEL)
    axes[0].legend(fontsize=FONT_SIZE_TICKS)

    # --- Plot B: Q-binned ln(T) impact ---
    df_qbin = df_ln.copy()
    df_qbin["qbins"], qbin_edges = pd.qcut(df_qbin["move_ply"], q=10, 
                                           duplicates="drop", retbins=True, labels=False)
    
    df_qbin_utils = df_qbin.rename(columns=col_map)
    metrics_qbin = compute_metrics_by_qbin(df_qbin_utils, qbin_edges)
    
    plt.sca(axes[1])
    plot_metrics(metrics_qbin, color=MAIN_COLOR)
    axes[1].set_xlabel("Move Ply (Quantile-binned)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel(r"Mean $\ln(T)$", fontsize=FONT_SIZE_LABEL)

    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, "ply_impact_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {save_path}")

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    data_path = os.path.join(base_dir, "data", "moves_500.parquet")
    figures_dir = os.path.join(base_dir, "src/figures/ply_analysis") # Dedicated subdir
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df_cleaned = analyze_ply_effect(df)
    plot_ply_impact(df_cleaned, figures_dir)
    print("\n✅ Ply analysis complete. Poster-Style figures generated.")

if __name__ == "__main__":
    main()
