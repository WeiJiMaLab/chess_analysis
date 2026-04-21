import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from utils import (
    plot_standard_analysis_quad,
    apply_poster_style, MAIN_COLOR, FONT_SIZE_LABEL
)

# Poster Style Constants
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(base_dir, "src", "figures", "voc_movetime")

def main():
    parser = argparse.ArgumentParser(description="Analyze Value of Computation (VOC) results.")
    parser.add_argument("--n_games", type=int, default=500, help="Number of games to analyze")
    args = parser.parse_args()

    results_file = os.path.join(base_dir, "data", "processed", f"voc_{args.n_games}.csv")
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)

    # Preprocess
    df["ln_move_time"] = np.log(pd.to_numeric(df["move_time"], errors="coerce").clip(lower=0.1))
    df["voc_sqrt"] = np.sqrt(pd.to_numeric(df["voc"], errors="coerce"))
    
    # Clean NaNs
    df = df.dropna(subset=["ln_move_time", "voc_sqrt", "ply"])
    
    # Standard Analysis Quad
    plot_standard_analysis_quad(
        df, 
        x_var="voc_sqrt", 
        y_var="ln_move_time",
        x_label=r"$\sqrt{VOC}$",
        y_label=r"$\log(T)$",
        color="indigo",
        save_path=os.path.join(FIGURE_DIR, f"voc_standard_quad_{args.n_games}.png"),
        ply_var="ply" # voc_csv uses 'ply' instead of 'move_ply'
    )
    
    print("\n✅ VOC analysis complete using standardized plots.")

if __name__ == "__main__":
    main()
