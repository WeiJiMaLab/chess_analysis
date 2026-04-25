"""
Analysis of the relationship between player or opponent clock time and move time.
"""

import os
import duckdb
import argparse
from utils import Variable, Analyzer

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--opp', action='store_true', help='Analyze opponent clock time instead of player clock time')
    args = parser.parse_args()
    
    player = "opponent" if args.opp else "player"
    clock_col = f"{player}_clock_time"
    
    # Configure Variables
    x_var = Variable(column=clock_col, is_log=False, name=f"{player.capitalize()} Clock")
    y_var = Variable(column="move_time", is_log=True, name="T")
    
    # Run Analysis
    analyzer = Analyzer(
        db_conn=conn, 
        table_name="_selected_moves_nonzero_T", 
        x_var=x_var, 
        y_var=y_var, 
        filter_query=f"{clock_col} < 600",
        title=f"{player.capitalize()} Clock Pressure"
    )
    
    # Save Plots
    figures_dir = os.path.join(src_dir, "figures", "clock_movetime")
    filename = f"combined{'_opp' if args.opp else ''}.png"
    analyzer.save_dashboard(os.path.join(figures_dir, filename))
    
    conn.close()

if __name__ == "__main__":
    main()
