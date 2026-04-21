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
FIGURE_DIR = os.path.join(base_dir, "src", "figures", "fe_clocktime_movetime")
MAIN_COLOR = "#2ecc71"

def load_and_preprocess():
    """Load and compute log features consistently."""
    data_path = os.path.join(base_dir, "data", "moves_500.parquet")
    if not os.path.exists(data_path):
        return None
    
    df = pd.read_parquet(data_path)
    df = df.sort_values(["gid", "move_ply"]).copy()
    df["move_time"] = df["move_time"].fillna(0).clip(lower=0)
    
    grouped = df.groupby(["gid", "player_white"])
    df["spent_prior"] = grouped["move_time"].cumsum() - df["move_time"]
    df["n_prior"] = grouped.cumcount()
    df["player_time_left"] = df["initial_clock"] - df["spent_prior"] + df["n_prior"] * df["clock_increment"]
    
    df = df[(df["move_time"] > 0) & (df["player_time_left"] > 0)].copy()
    df["log_T"] = np.log(df["move_time"])
    df["log_clock"] = np.log(df["player_time_left"])
    
    # Identifier for pgi (player-game instance)
    df["side"] = np.where(df["player_white"], "white", "black")
    df["pgi"] = df["gid"].astype(str) + "_" + df["side"]
    
    return df

def run_fe_analyses(df):
    """Run Fixed Effects analyses via de-meaning."""
    results = []
    
    # 1. Baseline OLS
    print("Running Model 1: baseline OLS...")
    m1 = sm.OLS(df['log_T'].astype(float), sm.add_constant(df['log_clock']).astype(float)).fit()
    results.append({
        "specification": "OLS\n(Baseline)",
        "coef": m1.params["log_clock"],
        "ci_lower": m1.conf_int().loc["log_clock", 0],
        "ci_upper": m1.conf_int().loc["log_clock", 1]
    })
    plot_standard_analysis_quad(
        df, "log_clock", "log_T", 
        x_label=r"$\log(\mathrm{Clock})$", 
        y_label=r"$\log(\mathrm{Move~Time})$",
        color=MAIN_COLOR, save_path=os.path.join(FIGURE_DIR, "quad_0_baseline.png")
    )
    
    # 2. FE (ply)
    print("Running Model 2: FE (ply)...")
    df['log_T_ply'] = df['log_T'] - df.groupby('move_ply')['log_T'].transform('mean')
    df['log_clock_ply'] = df['log_clock'] - df.groupby('move_ply')['log_clock'].transform('mean')
    m2 = sm.OLS(df['log_T_ply'].astype(float), df['log_clock_ply'].astype(float)).fit()
    results.append({
        "specification": "Fixed Effects\n(Move Number / Ply)",
        "coef": m2.params.iloc[0],
        "ci_lower": m2.conf_int().iloc[0, 0],
        "ci_upper": m2.conf_int().iloc[0, 1]
    })
    plot_standard_analysis_quad(
        df, "log_clock_ply", "log_T_ply", 
        x_label=r"$\log(\mathrm{Clock}) - \mathbb{E}[\log(\mathrm{Clock}) | \mathrm{Ply}]$", 
        y_label=r"$\log(T) - \mathbb{E}[\log(T) | \mathrm{Ply}]$",
        color="#e74c3c", save_path=os.path.join(FIGURE_DIR, "quad_1_ply_fe.png")
    )
    
    # 3. FE (pgi)
    print("Running Model 3: FE (pgi)...")
    df['log_T_pgi'] = df['log_T'] - df.groupby('pgi')['log_T'].transform('mean')
    df['log_clock_pgi'] = df['log_clock'] - df.groupby('pgi')['log_clock'].transform('mean')
    m3 = sm.OLS(df['log_T_pgi'].astype(float), df['log_clock_pgi'].astype(float)).fit()
    results.append({
        "specification": "Fixed Effects\n(Player-Game Instance)",
        "coef": m3.params.iloc[0],
        "ci_lower": m3.conf_int().iloc[0, 0],
        "ci_upper": m3.conf_int().iloc[0, 1]
    })
    plot_standard_analysis_quad(
        df, "log_clock_pgi", "log_T_pgi", 
        x_label=r"$\log(\mathrm{Clock}) - \mathbb{E}[\log(\mathrm{Clock}) | \mathrm{Game}]$", 
        y_label=r"$\log(T) - \mathbb{E}[\log(T) | \mathrm{Game}]$",
        color="#3498db", save_path=os.path.join(FIGURE_DIR, "quad_2_pgi_fe.png")
    )
    
    # 4. Two-Way FE (ply + pgi)
    print("Running Model 4: Two-Way FE (ply + pgi)...")
    df['log_T_2way'] = df['log_T_pgi'] - df['log_T_pgi'].groupby(df['move_ply']).transform('mean')
    df['log_clock_2way'] = df['log_clock_pgi'] - df['log_clock_pgi'].groupby(df['move_ply']).transform('mean')
    m4 = sm.OLS(df['log_T_2way'].astype(float), df['log_clock_2way'].astype(float)).fit()
    results.append({
        "specification": "Two-Way Fixed Effects\n(Game + Ply Controls)",
        "coef": m4.params.iloc[0],
        "ci_lower": m4.conf_int().iloc[0, 0],
        "ci_upper": m4.conf_int().iloc[0, 1]
    })
    plot_standard_analysis_quad(
        df, "log_clock_2way", "log_T_2way", 
        x_label=r"Residual $\log(\mathrm{Clock}) \mid \mathrm{Game, Ply}$", 
        y_label=r"Residual $\log(T) \mid \mathrm{Game, Ply}$",
        color="#9b59b6", save_path=os.path.join(FIGURE_DIR, "quad_3_2way_fe.png")
    )
    
    return pd.DataFrame(results)

def plot_coefficients(rdf):
    """Create a premium horizontal barplot for FE results."""
    plt.figure(figsize=(11, 7))
    sns.set_style("white")
    
    # Modern professional colors
    PRIMARY_COLOR = "#2c3e50"  # Deep slate
    
    # Sort or prepare data for horizontal plotting
    rdf_plot = rdf.copy()
    
    # Plotting
    ax = sns.barplot(
        y="specification", x="coef", data=rdf_plot, 
        color=PRIMARY_COLOR, alpha=0.85, edgecolor="none"
    )
    
    # Manual error bars for horizontal layout
    y_coords = np.arange(len(rdf_plot))
    x_errs = [rdf_plot["coef"] - rdf_plot["ci_lower"], rdf_plot["ci_upper"] - rdf_plot["coef"]]
    plt.errorbar(
        x=rdf_plot["coef"], y=y_coords, xerr=x_errs, 
        fmt='none', c='black', capsize=6, elinewidth=1.5, capthick=1.5
    )
    
    plt.axvline(0, color='black', lw=1, ls='-')
    plt.title("The 'Within-Game' Effect of Clock on Move Time", fontsize=18, fontweight='bold', pad=25, loc='left')
    plt.xlabel(r"Elasticity: $\Delta \log(\mathrm{Move Time}) / \Delta \log(\mathrm{Clock Time})$", fontsize=13, labelpad=15)
    plt.ylabel("", fontsize=1) # Remove y-label to save space, names are on bars
    
    # Aesthetics polish
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Annotate values
    for i, coef in enumerate(rdf_plot["coef"]):
        plt.text(coef + 0.015, i, f"{coef:.3f}", 
                va='center', fontweight='bold', fontsize=12, color=PRIMARY_COLOR)

    # Descriptive footnote
    plt.figtext(0.1, 0.02, 
                "Interpretation: A coefficient of 0.40 means a 10% reduction in clock leads to a ~4% faster move.\n" + 
                "Two-Way FE controls for both the specific game instance and the move number (ply).", 
                fontsize=10, style='italic', alpha=0.7)

    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR, exist_ok=True)
    save_path = os.path.join(FIGURE_DIR, "fe_clock_coefficients_premium.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Premium FE Plot saved to: {save_path}")

if __name__ == "__main__":
    df = load_and_preprocess()
    if df is not None:
        rdf = run_fe_analyses(df)
        print("\nFixed Effects Analysis Results:")
        print(rdf)
        plot_coefficients(rdf)
