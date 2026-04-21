import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from utils.plots import plot_standard_analysis_quad
from utils.helpers import apply_poster_style, MAIN_COLOR

# Poster Style Constants
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(base_dir, "src", "figures", "fe_oppclock_movetime")
OPP_COLOR = "#9b59b6" # Amethyst purple for distinction

def load_and_preprocess():
    """Load and compute log features for opponent clock."""
    data_path = os.path.join(base_dir, "data", "moves_500.parquet")
    if not os.path.exists(data_path):
        return None
    
    df = pd.read_parquet(data_path)
    df = df.sort_values(["gid", "move_ply"]).copy()
    
    # Standard time calculations
    df["move_time"] = df["move_time"].fillna(0).clip(lower=0)
    grouped = df.groupby(["gid", "player_white"])
    df["spent_prior"] = grouped["move_time"].cumsum() - df["move_time"]
    df["n_prior"] = grouped.cumcount()
    df["player_time_left"] = df["initial_clock"] - df["spent_prior"] + df["n_prior"] * df["clock_increment"]
    
    # Opponent clock logic (shift by -1 within each game)
    df["opponent_time_left"] = df.groupby("gid")["player_time_left"].shift(-1)
    
    df = df[(df["move_time"] > 0) & (df["opponent_time_left"] > 0)].copy()
    df["log_T"] = np.log(df["move_time"])
    df["log_opp_clock"] = np.log(df["opponent_time_left"])
    
    # Identifier for pgi (player-game instance)
    df["side"] = np.where(df["player_white"], "white", "black")
    df["pgi"] = df["gid"].astype(str) + "_" + df["side"]
    
    return df

def run_fe_analyses(df):
    """Run Fixed Effects analyses for opponent clock."""
    results = []
    
    # 1. Baseline OLS
    print("Running Model 1: baseline OLS (Opponent)...")
    m1 = sm.OLS(df['log_T'].astype(float), sm.add_constant(df['log_opp_clock']).astype(float)).fit()
    results.append({
        "specification": "OLS\n(Baseline)",
        "coef": m1.params["log_opp_clock"],
        "ci_lower": m1.conf_int().loc["log_opp_clock", 0],
        "ci_upper": m1.conf_int().loc["log_opp_clock", 1]
    })
    plot_standard_analysis_quad(
        df, "log_opp_clock", "log_T", 
        x_label=r"$\log(\mathrm{Opponent~Clock})$", 
        y_label=r"$\log(T)$",
        color=OPP_COLOR, save_path=os.path.join(FIGURE_DIR, "quad_0_baseline.png")
    )
    
    # 2. FE (ply)
    print("Running Model 2: FE (ply) (Opponent)...")
    df['log_T_ply'] = df['log_T'] - df.groupby('move_ply')['log_T'].transform('mean')
    df['log_opp_clock_ply'] = df['log_opp_clock'] - df.groupby('move_ply')['log_opp_clock'].transform('mean')
    m2 = sm.OLS(df['log_T_ply'].astype(float), df['log_opp_clock_ply'].astype(float)).fit()
    results.append({
        "specification": "Fixed Effects\n(Move Number / Ply)",
        "coef": m2.params.iloc[0],
        "ci_lower": m2.conf_int().iloc[0, 0],
        "ci_upper": m2.conf_int().iloc[0, 1]
    })
    plot_standard_analysis_quad(
        df, "log_opp_clock_ply", "log_T_ply", 
        x_label=r"$\log(\mathrm{Opponent~Clock})$" + "\n" + r"$- \mathbb{E}[\log(\mathrm{Opp.~Cl.}) | \mathrm{Ply}]$", 
        y_label=r"$\log(T)$" + "\n" + r"$- \mathbb{E}[\log(T) | \mathrm{Ply}]$",
        color="#e67e22", save_path=os.path.join(FIGURE_DIR, "quad_1_ply_fe.png")
    )
    
    # 3. FE (pgi)
    print("Running Model 3: FE (pgi) (Opponent)...")
    df['log_T_pgi'] = df['log_T'] - df.groupby('pgi')['log_T'].transform('mean')
    df['log_opp_clock_pgi'] = df['log_opp_clock'] - df.groupby('pgi')['log_opp_clock'].transform('mean')
    m3 = sm.OLS(df['log_T_pgi'].astype(float), df['log_opp_clock_pgi'].astype(float)).fit()
    results.append({
        "specification": "Fixed Effects\n(Player-Game Instance)",
        "coef": m3.params.iloc[0],
        "ci_lower": m3.conf_int().iloc[0, 0],
        "ci_upper": m3.conf_int().iloc[0, 1]
    })
    plot_standard_analysis_quad(
        df, "log_opp_clock_pgi", "log_T_pgi", 
        x_label=r"$\log(\mathrm{Opponent~Clock})$" + "\n" + r"$- \mathbb{E}[\log(\mathrm{Opp.~Cl.}) | \mathrm{Game}]$", 
        y_label=r"$\log(T)$" + "\n" + r"$- \mathbb{E}[\log(T) | \mathrm{Game}]$",
        color=MAIN_COLOR, save_path=os.path.join(FIGURE_DIR, "quad_2_pgi_fe.png")
    )
    
    # 4. Two-Way FE (ply + pgi)
    print("Running Model 4: Two-Way FE (ply + pgi) (Opponent)...")
    df['log_T_2way'] = df['log_T_pgi'] - df['log_T_pgi'].groupby(df['move_ply']).transform('mean')
    df['log_opp_clock_2way'] = df['log_opp_clock_pgi'] - df['log_opp_clock_pgi'].groupby(df['move_ply']).transform('mean')
    m4 = sm.OLS(df['log_T_2way'].astype(float), df['log_opp_clock_2way'].astype(float)).fit()
    results.append({
        "specification": "Two-Way Fixed Effects\n(Game + Ply Controls)",
        "coef": m4.params.iloc[0],
        "ci_lower": m4.conf_int().iloc[0, 0],
        "ci_upper": m4.conf_int().iloc[0, 1]
    })
    plot_standard_analysis_quad(
        df, "log_opp_clock_2way", "log_T_2way", 
        x_label=r"Residual $\log(\mathrm{Opponent~Clock})$" + "\n" + r"$\mid \mathrm{Game, Ply}$", 
        y_label=r"Residual $\log(T)$" + "\n" + r"$\mid \mathrm{Game, Ply}$",
        color=OPP_COLOR, save_path=os.path.join(FIGURE_DIR, "quad_3_2way_fe.png")
    )
    
    return pd.DataFrame(results)

def plot_coefficients(rdf):
    """Create a premium horizontal barplot for Opponent FE results."""
    plt.figure(figsize=(11, 7))
    apply_poster_style()
    
    # Aesthetic distinction for opponent
    PRIMARY_COLOR = "#8e44ad"  # Dark purple
    
    rdf_plot = rdf.copy()
    
    # Plotting
    ax = sns.barplot(
        y="specification", x="coef", data=rdf_plot, 
        color=PRIMARY_COLOR, alpha=0.85, edgecolor="none"
    )
    
    y_coords = np.arange(len(rdf_plot))
    x_errs = [rdf_plot["coef"] - rdf_plot["ci_lower"], rdf_plot["ci_upper"] - rdf_plot["coef"]]
    plt.errorbar(
        x=rdf_plot["coef"], y=y_coords, xerr=x_errs, 
        fmt='none', c='black', capsize=6, elinewidth=1.5, capthick=1.5
    )
    
    plt.axvline(0, color='black', lw=1, ls='-')
    plt.title("Effect of Opponent Clock on Player Move Time", fontsize=18, fontweight='bold', pad=25, loc='left')
    plt.xlabel(r"Elasticity: $\Delta \log(\mathrm{Player~T}) / \Delta \log(\mathrm{Opponent~Clock})$", fontsize=13, labelpad=15)
    plt.ylabel("", fontsize=1)
    
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=12, fontweight='bold')
    
    for i, coef in enumerate(rdf_plot["coef"]):
        plt.text(coef + 0.005 if coef > 0 else coef - 0.04, i, f"{coef:.3f}", 
                va='center', fontweight='bold', fontsize=12, color=PRIMARY_COLOR)

    plt.figtext(0.1, 0.02, 
                "Interpretation: How the opponent's remaining time affects the current player's thinking duration.\n" + 
                "Two-Way FE isolates the response after controlling for both game instance and stage (ply).", 
                fontsize=10, style='italic', alpha=0.7)

    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR, exist_ok=True)
    save_path = os.path.join(FIGURE_DIR, "fe_oppclock_coefficients_premium.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Premium Opponent FE Plot saved to: {save_path}")

if __name__ == "__main__":
    df = load_and_preprocess()
    if df is not None:
        rdf = run_fe_analyses(df)
        print("\nOpponent Fixed Effects Analysis Results:")
        print(rdf)
        plot_coefficients(rdf)
