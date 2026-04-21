import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from utils import (
    load_games, preprocess_data, 
    plot_standard_analysis_quad,
    MAIN_COLOR
)

# Constants
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(base_dir, "src", "figures", "voc_movetime2")
DATA_PATH = os.path.join(base_dir, "data", "processed", "processed_moves_500.parquet")

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return None
    df = pd.read_parquet(DATA_PATH)
    df = df[df["move_time"] > 0].copy()
    df["ln_move_time"] = np.log(df["move_time"])
    # Standard residual
    df["ply_median"] = df.groupby("move_ply")["ln_move_time"].transform("median")
    df["ln_move_resid_ply"] = df["ln_move_time"] - df["ply_median"]
    return df

def main():
    df = load_data()
    if df is None: return
    
    analysis_configs = [
        ("voc_single_sample",    "#2E86C1", "VOC (Single Sample)"),
        ("voc_consideration_set", "#A330C9", "VOC (Consideration Set)"),
        ("best_two_diff",        "#E67E22", "Best-Two Difference")
    ]
    
    for var, color, label in analysis_configs:
        if var in df.columns:
            plot_standard_analysis_quad(
                df, 
                x_var=var, 
                y_var="ln_move_time",
                x_label=label, 
                y_label=r"$\log(T)$",
                color=color,
                save_path=os.path.join(FIGURE_DIR, f"standard_{var}.png")
            )
            print(f"✅ Generated standard analysis for {var}")

if __name__ == "__main__":
    main()
