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
    MAIN_COLOR, FONT_SIZE_LABEL, FONT_SIZE_TICKS, FONT_SIZE_TITLE,
    compute_metrics_by_bin, compute_metrics_by_qbin, plot_metrics
)

# Constants for this analysis
MAX_PLY_FOR_PLOT = 120

def analyze_ply_effect(df: pd.DataFrame):
    """
    Perform statistical analysis of move_ply on move_time, including
    quadratic regressions on both linear and log scales.
    """
    df = df.copy()
    df["move_time"] = pd.to_numeric(df["move_time"], errors="coerce")
    df["move_ply"] = pd.to_numeric(df["move_ply"], errors="coerce")
    df = df.dropna(subset=["move_time", "move_ply"])
    
    # Linear scale quadratic model
    df["move_ply_sq"] = df["move_ply"] ** 2
    m2 = smf.ols("move_time ~ move_ply + move_ply_sq", data=df).fit()
    print("\n--- Quadratic Model: move_time ~ move_ply + move_ply^2 ---")
    print(m2.summary().tables[1])
    
    # Peak calculation: -b / 2a
    b, a = m2.params["move_ply"], m2.params["move_ply_sq"]
    print(f"\nPredicted Peak Ply (Linear scale): {-b / (2 * a):.2f}")

    # Log scale quadratic model (more robust for response times)
    df["log_move_time"] = np.log(df["move_time"] + 1e-3)
    m3 = smf.ols("log_move_time ~ move_ply + move_ply_sq", data=df).fit()
    print("\n--- Log-Quadratic Model: log(move_time) ~ move_ply + move_ply^2 ---")
    print(m3.summary().tables[1])
    
    b_log, a_log = m3.params["move_ply"], m3.params["move_ply_sq"]
    print(f"Predicted Peak Ply (Log scale): {-b_log / (2 * a_log):.2f}")
    
    return df

def plot_ply_impact(df: pd.DataFrame, figures_dir: str):
    """
    Generate unified plots showing the impact of move ply on response times
    using Raw Mean and Quantile-Binned approaches.
    """
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Column rename mapping to use shared metrics helpers from utils.py
    col_map = {"move_time": "move_time_raw", "log_move_time": "move_time"}

    # --- Plot A: Log Mean move time by ply (Raw) ---
    df_early = df[df["move_ply"] <= MAX_PLY_FOR_PLOT].copy()
    df_early["bin"] = df_early["move_ply"]
    
    df_early_utils = df_early.rename(columns=col_map)
    metrics_raw = compute_metrics_by_bin(df_early_utils)
    
    plt.sca(axes[0])
    plot_metrics(metrics_raw, color=MAIN_COLOR)
    sns.regplot(x="move_ply", y="move_time", data=df_early_utils, scatter=False, 
                order=2, color="red", label="Log-Quadratic Fit")
    
    axes[0].set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel("Log Move Time", fontsize=FONT_SIZE_LABEL)
    axes[0].set_title("A. Log-Transformed Mean Time", fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[0].tick_params(labelsize=FONT_SIZE_TICKS)
    axes[0].legend(fontsize=FONT_SIZE_TICKS)

    # --- Plot B: Q-binned Log ply impact ---
    df_qbin = df.copy()
    df_qbin["qbins"], qbin_edges = pd.qcut(df_qbin["move_ply"], q=10, 
                                           duplicates="drop", retbins=True, labels=False)
    
    df_qbin_utils = df_qbin.rename(columns=col_map)
    metrics_qbin = compute_metrics_by_qbin(df_qbin_utils, qbin_edges)
    
    plt.sca(axes[1])
    plot_metrics(metrics_qbin, color=MAIN_COLOR)
    axes[1].set_xlabel("Move Ply (Quantile-binned)", fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel("Log Move Time", fontsize=FONT_SIZE_LABEL)
    axes[1].set_title("B. Binned Log-Impact", fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[1].tick_params(labelsize=FONT_SIZE_TICKS)

    plt.suptitle("Significance of the Middle-Game Response Time Peak", 
                 fontsize=24, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, "ply_impact_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {save_path}")

def main():
    # Setup paths relative to script location
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(src_dir)
    data_path = os.path.join(base_dir, "data", "moves_200.parquet")
    figures_dir = os.path.join(base_dir, "figures")
    
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
    print("\nAnalysis complete. Figures generated in 'figures/'")

if __name__ == "__main__":
    main()
