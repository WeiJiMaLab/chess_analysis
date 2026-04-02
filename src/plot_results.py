import pandas as pd
import numpy as np
from voc_analysis import analyze_data, plot_data

if __name__ == "__main__":
    try:
        df = pd.read_csv("voc_results.csv")
        df["voc_sqrt"] = np.sqrt(df["voc"])
        print(f"Loaded {len(df)} positions from voc_results.csv")
        
        # Run analysis and save plots
        df = analyze_data(df)
        plot_data(df)
        
        print("\nPlot saved: voc_sqrt_vs_move_time.png")
    except FileNotFoundError:
        print("Error: voc_results.csv not found. Please run the full analysis script first.")
    except Exception as e:
        print(f"Error: {e}")
