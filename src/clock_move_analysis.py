import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from utils import compute_metrics_by_qbin, plot_metrics, MAIN_COLOR

# Poster Style Constants
FONT_SIZE_LABEL = 22
FONT_SIZE_TICKS = 18
FIGURE_DIR = "src/figures"
NEGATIVE_COLOR = "#e74c3c"
POSITIVE_COLOR = "#2ecc71"

# Apply global aesthetics
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICKS
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.usetex'] = False

def load_and_preprocess():
    """Load data and compute natural-log features."""
    if not os.path.exists("data/moves_200.parquet"):
        print("Error: data/moves_200.parquet not found. Run load_games.py first.")
        return None
    
    df = pd.read_parquet("data/moves_200.parquet")
    
    # 1. Calculate player_time_left
    df = df.sort_values(["gid", "move_ply"]).copy()
    df["move_time"] = df["move_time"].fillna(0).clip(lower=0)
    
    grouped = df.groupby(["gid", "player_white"])
    df["spent_prior"] = grouped["move_time"].cumsum() - df["move_time"]
    df["n_prior"] = grouped.cumcount()
    
    df["player_time_left"] = df["initial_clock"] - df["spent_prior"] + df["n_prior"] * df["clock_increment"]
    
    # 2. Filter and log-transform (Natural Log)
    df = df[df["move_time"] > 0].copy()
    df = df[df["player_time_left"] > 0].copy()
    
    df["ln_move_time"] = np.log(df["move_time"])
    df["ln_clock"] = np.log(df["player_time_left"])
    
    # 3. Identifiers
    df["side"] = np.where(df["player_white"], "white", "black")
    df["game_player"] = df["gid"].astype(str) + "_" + df["side"]
    df["player_id"] = np.where(df["player_white"], df["white_id"], df["black_id"])
    
    return df

def analyze_distribution(df: pd.DataFrame):
    """Visualize the distribution of move times side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.histplot(df["move_time"], bins=20, kde=True, color=MAIN_COLOR, alpha=0.6, ax=axes[0])
    axes[0].set_xlabel("Move Time (s)", fontsize=FONT_SIZE_LABEL)
    
    sns.histplot(df["ln_move_time"], bins=20, kde=True, color=MAIN_COLOR, alpha=0.6, ax=axes[1])
    axes[1].set_xlabel(r"$\ln(T)$", fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "move_time_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

def analyze_naive_trend(df: pd.DataFrame):
    """Attempt 1: Naive analysis."""
    print("Running Naive Analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # [0, 0] Density
    axes[0, 0].hexbin(df["ln_clock"], df["ln_move_time"], gridsize=40, cmap="Blues", mincnt=1)
    sns.regplot(x="ln_clock", y="ln_move_time", data=df, scatter=False, color=MAIN_COLOR, ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\ln(\text{Clock Time})$", fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel(r"$\ln(T)$", fontsize=FONT_SIZE_LABEL)
    
    # [0, 1] Trend by Value
    df_sorted = df.sort_values("ln_clock")
    df_sorted["qbins"], qbin_edges = pd.qcut(df_sorted["ln_clock"], q=25, labels=False, retbins=True, duplicates="drop")
    df_tmp = df_sorted.copy()
    df_tmp["move_time"] = df_tmp["ln_move_time"]
    metrics = compute_metrics_by_qbin(df_tmp, qbin_edges)
    plot_metrics(metrics, color=MAIN_COLOR, ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\ln(\text{Clock Time})$", fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel(r"Mean $\ln(T)$", fontsize=FONT_SIZE_LABEL)

    # [1, 1] Trend by Quantile Rank
    metrics_rank = {k: list(v) for k, v in metrics.items()}
    metrics_rank["x"] = np.linspace(0, 1, len(metrics_rank["x"]))
    plot_metrics(metrics_rank, color=MAIN_COLOR, ax=axes[1, 1])
    axes[1, 1].set_xlabel("Quantile Rank (Clock)", fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel(r"Mean $\ln(T)$", fontsize=FONT_SIZE_LABEL)

    # [1, 0] Information Pane
    axes[1, 0].text(0.5, 0.5, r"$\mathbf{Attempt\ 1\ (Naive)}$" + "\nNo Controls\n(Confounded)", 
                    ha='center', va='center', fontsize=26, alpha=0.8)
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "attempt1_naive_trend.png"), dpi=300, bbox_inches="tight")
    plt.close()

def analyze_ply_wise_regression(df: pd.DataFrame):
    """Attempt 2 (Ply-Controlled)."""
    print("Running Attempt 2 (Ply-Controlled)...")
    df["ply_median"] = df.groupby("move_ply")["ln_move_time"].transform("median")
    df["ln_move_resid_ply"] = df["ln_move_time"] - df["ply_median"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # [0, 0] Density
    axes[0, 0].hexbin(df["ln_clock"], df["ln_move_resid_ply"], gridsize=40, cmap="Reds", mincnt=1)
    sns.regplot(x="ln_clock", y="ln_move_resid_ply", data=df, scatter=False, color="firebrick", ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\ln(\text{Clock Time})$", fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel(r"$\ln T - \ln \text{med}_{ply}$", fontsize=FONT_SIZE_LABEL)

    # [0, 1] Binned by Value
    df_sorted = df.sort_values("ln_clock")
    df_sorted["qbins"], qbin_edges = pd.qcut(df_sorted["ln_clock"], q=25, labels=False, retbins=True, duplicates="drop")
    df_tmp = df_sorted.copy()
    df_tmp["move_time"] = df_tmp["ln_move_resid_ply"]
    metrics = compute_metrics_by_qbin(df_tmp, qbin_edges)
    plot_metrics(metrics, color="firebrick", ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\ln(\text{Clock Time})$", fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel(r"Mean $[\ln T - \ln \text{med}_{ply}]$", fontsize=FONT_SIZE_LABEL)

    # [1, 0] Ply Control (Stability)
    results = []
    for ply in range(10, 81, 2):
        df_ply = df[df["move_ply"] == ply]
        if len(df_ply) > 30:
            model = smf.ols("ln_move_time ~ ln_clock", data=df_ply).fit()
            results.append({"ply": ply, "coeff": model.params["ln_clock"], "bse": model.bse["ln_clock"]})
    df_res = pd.DataFrame(results)
    axes[1, 0].errorbar(df_res["ply"], df_res["coeff"], yerr=1.96 * df_res["bse"], fmt='o', color="firebrick", ecolor='lightgray', elinewidth=3, capsize=0)
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_ylabel(r"Slope ($\beta_{ply}$)", fontsize=FONT_SIZE_LABEL)

    # [1, 1] Binned by Quantile Rank
    metrics_rank = {k: list(v) for k, v in metrics.items()}
    metrics_rank["x"] = np.linspace(0, 1, len(metrics_rank["x"]))
    plot_metrics(metrics_rank, color="firebrick", ax=axes[1, 1])
    axes[1, 1].set_xlabel("Quantile Rank (Clock)", fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel(r"Mean $[\ln T - \ln \text{med}_{ply}]$", fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "attempt2_ply_wise_trend.png"), dpi=300, bbox_inches="tight")
    plt.close()

def analyze_controlled_trend(df: pd.DataFrame):
    """Attempt 3 (Double-Controlled)."""
    print("Running Attempt 3 (Double-Controlled)...")
    player_medians = df.groupby("game_player")["ln_move_time"].transform("median")
    ply_medians = df.groupby("move_ply")["ln_move_time"].transform("median")
    
    df["ln_move_resid"] = df["ln_move_time"] - player_medians - ply_medians
    df["ln_move_norm_player"] = df["ln_move_time"] - player_medians

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # [0, 0] Density
    axes[0, 0].hexbin(df["ln_clock"], df["ln_move_resid"], gridsize=40, cmap="Greens", mincnt=1)
    sns.regplot(x="ln_clock", y="ln_move_resid", data=df, scatter=False, color=POSITIVE_COLOR, ax=axes[0, 0])
    axes[0, 0].set_xlabel(r"$\ln(\text{Clock Time})$", fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel(r"$\ln T - \ln med_{player} - \ln med_{ply}$", fontsize=FONT_SIZE_LABEL)

    # [0, 1] Binned by Value
    df_sorted = df.sort_values("ln_clock")
    df_sorted["qbins"], qbin_edges = pd.qcut(df_sorted["ln_clock"], q=25, labels=False, retbins=True, duplicates="drop")
    df_tmp = df_sorted.copy()
    df_tmp["move_time"] = df_tmp["ln_move_resid"]
    metrics = compute_metrics_by_qbin(df_tmp, qbin_edges)
    plot_metrics(metrics, color=POSITIVE_COLOR, ax=axes[0, 1])
    axes[0, 1].set_xlabel(r"$\ln(\text{Clock Time})$", fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel(r"Mean $[\ln T - \ln med_{player} - \ln med_{ply}]$", fontsize=FONT_SIZE_LABEL)

    # [1, 0] Ply Control (Stability)
    results = []
    for ply in range(10, 81, 2):
        df_ply = df[df["move_ply"] == ply]
        if len(df_ply) > 30:
            model = smf.ols("ln_move_norm_player ~ ln_clock", data=df_ply).fit()
            results.append({"ply": ply, "coeff": model.params["ln_clock"], "bse": model.bse["ln_clock"]})
    df_res = pd.DataFrame(results)
    axes[1, 0].errorbar(df_res["ply"], df_res["coeff"], yerr=1.96 * df_res["bse"], fmt='o', color=POSITIVE_COLOR, ecolor='lightgray', elinewidth=3, capsize=0)
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel("Move Ply", fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_ylabel(r"Slope ($\beta$)", fontsize=FONT_SIZE_LABEL)

    # [1, 1] Binned by Quantile Rank
    metrics_rank = {k: list(v) for k, v in metrics.items()}
    metrics_rank["x"] = np.linspace(0, 1, len(metrics_rank["x"]))
    plot_metrics(metrics_rank, color=POSITIVE_COLOR, ax=axes[1, 1])
    axes[1, 1].set_xlabel("Quantile Rank (Clock)", fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel(r"Mean $[\ln T - \ln med_{player} - \ln med_{ply}]$", fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "attempt3_controlled_trend.png"), dpi=300, bbox_inches="tight")
    plt.close()

def run_mixed_effects(df: pd.DataFrame):
    """Formal Mixed Linear Effects model validation."""
    print("\nEstimating Mixed Effects Model...")
    try:
        model = smf.mixedlm("ln_move_time ~ ln_clock + move_ply", df, groups=df["player_id"])
        result = model.fit()
        print(result.summary())
    except Exception as e:
        print(f"Mixed Effects failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    data = load_and_preprocess()
    if data is not None:
        analyze_distribution(data)
        analyze_naive_trend(data)
        analyze_ply_wise_regression(data)
        analyze_controlled_trend(data)
        run_mixed_effects(data)
        print("\n✅ Analysis complete. Posters generated with natural log and XY grids.")
