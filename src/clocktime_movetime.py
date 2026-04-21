import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils import (
    load_games, preprocess_data, 
    plot_standard_analysis_quad, plot_distribution_side_by_side,
    MAIN_COLOR
)

# Poster Style Constants
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(base_dir, "src", "figures", "clocktime_movetime")
POSITIVE_COLOR = "#2ecc71"

def load_and_preprocess():
    """Load data and compute natural-log features."""
    data_path = os.path.join(base_dir, "data", "moves_500.parquet")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return None
    
    df = pd.read_parquet(data_path)
    df = df.sort_values(["gid", "move_ply"]).copy()
    df["move_time"] = df["move_time"].fillna(0).clip(lower=0)
    
    grouped = df.groupby(["gid", "player_white"])
    df["spent_prior"] = grouped["move_time"].cumsum() - df["move_time"]
    df["n_prior"] = grouped.cumcount()
    df["player_time_left"] = df["initial_clock"] - df["spent_prior"] + df["n_prior"] * df["clock_increment"]
    
    df = df[df["move_time"] > 0].copy()
    df = df[df["player_time_left"] > 0].copy()
    
    df["log_move_time"] = np.log(df["move_time"])
    df["log_clock"] = np.log(df["player_time_left"])
    df["player_id"] = np.where(df["player_white"], df["white_id"], df["black_id"])
    
    # 3. Controls
    df["ply_median"] = df.groupby("move_ply")["log_move_time"].transform("median")
    df["log_move_resid_ply"] = df["log_move_time"] - df["ply_median"]
    
    # 4. Double Control (Player + Ply)
    # Identifiers needed for player control
    df["side"] = np.where(df["player_white"], "white", "black")
    df["game_player"] = df["gid"].astype(str) + "_" + df["side"]
    player_medians = df.groupby("game_player")["log_move_time"].transform("median")
    df["log_move_resid_double"] = df["log_move_time"] - player_medians - df["ply_median"]
    
    return df

def run_mixed_effects(df: pd.DataFrame):
    """Formal Mixed Linear Effects model validation."""
    print("\nEstimating Mixed Effects Model...")
    try:
        model = smf.mixedlm("log_move_time ~ log_clock + move_ply", df, groups=df["player_id"])
        result = model.fit()
        print(result.summary())
    except Exception as e:
        print(f"Mixed Effects failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR, exist_ok=True)
        
    data = load_and_preprocess()
    if data is not None:
        # 1. Distribution
        plot_distribution_side_by_side(
            data,
            log_col="log_move_time",
            color=POSITIVE_COLOR,
            save_path=os.path.join(FIGURE_DIR, "move_time_distribution.png"),
        )
        
        # 2. Standard Analysis
        print("Generating standard clock analysis...")
        plot_standard_analysis_quad(
            data, 
            x_var="log_clock", 
            y_var="log_move_time",
            x_label=r"$\log(\text{Clock Time})$",
            y_label=r"$\log(T)$",
            color=POSITIVE_COLOR,
            save_path=os.path.join(FIGURE_DIR, "clock_standard_analysis.png")
        )
        
        # 3. Ply-Controlled Analysis
        print("Generating ply-controlled clock analysis...")
        plot_standard_analysis_quad(
            data, 
            x_var="log_clock", 
            y_var="log_move_resid_ply",
            x_label=r"$\log(\text{Clock Time})$",
            y_label=r"$\log T$" + "\n" + r"$-\,\log\,\mathrm{med}_{ply}$",
            color="#e74c3c", # Red for ply-control phase
            save_path=os.path.join(FIGURE_DIR, "clock_ply_controlled.png")
        )
        
        # 4. Double-Controlled Analysis (Ply + Player)
        print("Generating double-controlled clock analysis...")
        plot_standard_analysis_quad(
            data, 
            x_var="log_clock", 
            y_var="log_move_resid_double",
            x_label=r"$\log(\text{Clock Time})$",
            y_label=(
                r"$\log T$"
                + "\n"
                + r"$-\,\log\,\mathrm{med}_{ply}$"
                + "\n"
                + r"$-\,\log\,\mathrm{med}_{player}$"
            ),
            color=MAIN_COLOR,
            save_path=os.path.join(FIGURE_DIR, "clock_double_controlled.png"),
        )
        
        # 5. Mixed Effects
        run_mixed_effects(data)
        
        print("\n✅ All clock-time analysis posters generated successfully.")
