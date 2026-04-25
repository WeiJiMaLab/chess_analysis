"""
Analysis of the relationship between OPPONENT clock time and move time, segmented by game phase.
Phases (Early, Mid, End) are pre-defined in the database based on move ply quantiles.
This script re-calculates opponent clock quantiles within each phase and plots the results.
"""

import os
import sys
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory (src) to sys.path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    apply_poster_style, FONT_SIZE_LABEL, FONT_SIZE_TICKS, MAIN_COLOR, PHASE_COLORS
)

import argparse

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    parser = argparse.ArgumentParser(description="Phase-segmented opponent clock analysis.")
    parser.add_argument("--overlaid", action="store_true", help="Overlay phases on a single plot instead of stacking.")
    args = parser.parse_args()

    print(f"Connecting to {PERSONAL_DB}...")
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    base_table = "_selected_moves_nonzero_T"
    
    # 1. Report Gross Statistics (Ranges)
    print("\n--- Phase Statistics (Move Ply Ranges) ---")
    phase_stats = conn.execute(f"""
        SELECT 
            game_phase, 
            min(move_ply) as min_ply, 
            max(move_ply) as max_ply, 
            count(*) as n_moves,
            count(distinct gid) as n_games
        FROM {base_table} 
        GROUP BY game_phase 
        ORDER BY game_phase
    """).df()
    
    phase_names = {1: "Opening", 2: "Midgame", 3: "Endgame"}
    phase_stats['phase_name'] = phase_stats['game_phase'].map(phase_names)
    
    for _, row in phase_stats.iterrows():
        ply_desc = f"Ply {int(row['min_ply'])} - {int(row['max_ply'])}"
        if row['game_phase'] == 3:
            ply_desc = f"Ply > {int(row['min_ply'])}"
        print(f"{row['phase_name']}: {ply_desc} | Moves: {int(row['n_moves']):,}")

    # 2. Segmented Quantile Calculation for Opponent Clock
    print("\nCalculating per-phase opponent clock quantiles...")
    query = f"""
    WITH phase_data AS (
        SELECT 
            game_phase,
            opponent_clock_time,
            ln_move_time,
            ntile(20) OVER (PARTITION BY game_phase ORDER BY opponent_clock_time) as qbin
        FROM {base_table}
    )
    SELECT 
        game_phase,
        qbin,
        avg(opponent_clock_time) as mean_clock,
        avg(ln_move_time) as mean_y,
        stddev(ln_move_time) / sqrt(count(*)) as sem_y,
        count(*) as n
    FROM phase_data
    GROUP BY game_phase, qbin
    ORDER BY game_phase, qbin
    """
    df_q = conn.execute(query).df()
    conn.close()

    # 3. Plotting
    apply_poster_style()
    
    if args.overlaid:
        fig, ax = plt.subplots(figsize=(24, 12))
        axes = [ax] * 3
    else:
        fig, axes = plt.subplots(3, 1, figsize=(18, 30), sharex=False)
    
    for i, phase in enumerate([1, 2, 3]):
        ax = axes[i]
        subset = df_q[df_q['game_phase'] == phase]
        name = phase_names[phase]
        min_p = int(phase_stats.loc[phase_stats['game_phase'] == phase, 'min_ply'].values[0])
        max_p = int(phase_stats.loc[phase_stats['game_phase'] == phase, 'max_ply'].values[0])
        ply_range = f"Ply {min_p}-{max_p}"
        if phase == 3:
            ply_range = f"Ply > {min_p}"
        
        # Plot mean trends with error bars (SEM)
        ax.errorbar(
            subset['mean_clock'], subset['mean_y'], 
            yerr=subset['sem_y']*2, # 95% CI roughly
            fmt='o-', color=PHASE_COLORS[phase], lw=5, markersize=12, capsize=8,
            label=f"{name} ({ply_range})"
        )
        
        if args.overlaid:
            ax.fill_between(
                subset['mean_clock'], 
                subset['mean_y'] - subset['sem_y']*2,
                subset['mean_y'] + subset['sem_y']*2,
                color=PHASE_COLORS[phase], alpha=0.1
            )
        else:
            ax.set_title(f"{name} ({ply_range})", fontsize=FONT_SIZE_LABEL, pad=20)
            ax.set_ylabel("Mean log T (s)", fontsize=FONT_SIZE_LABEL)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=FONT_SIZE_TICKS)
            ax.set_xscale('log')
            ax.set_xlabel("Mean Opponent Remaining Clock (s) [log scale]", fontsize=FONT_SIZE_LABEL)

    if args.overlaid:
        ax.set_xlabel("Opponent Time Left (s)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Mean log T (s)", fontsize=FONT_SIZE_LABEL)
        ax.legend(fontsize=FONT_SIZE_TICKS, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
        ax.grid(True, alpha=0.3)
        suffix = "overlaid"
    else:
        plt.subplots_adjust(hspace=0.5)
        suffix = "trend"

    n_games = int(phase_stats['n_games'].sum())
    total_moves = int(phase_stats['n_moves'].sum())
    fig.suptitle(f"{n_games:,} games | {total_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10, y=1.02 if not args.overlaid else 0.95)
    
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "exploratory")
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, f"phase_segmented_opp_clock_{suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot to: {output_path}")

if __name__ == "__main__":
    main()
