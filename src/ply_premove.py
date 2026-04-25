"""
Analysis of the relationship between game stage (ply) and the probability of instant moves.
Note: This analysis includes moves with 0 move time.
"""

import os
import duckdb
from utils import Variable, Analyzer

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    conn = duckdb.connect(database=PERSONAL_DB, read_only=True)
    
    # Configure Variables
    # Y is the indicator function for an "instant move" (T=0)
    x_var = Variable(column="move_ply", is_log=False, name="Move Ply")
    y_var = Variable(
        column="(CASE WHEN move_time = 0 THEN 1 ELSE 0 END)", 
        is_log=False, 
        name="Instant Move Probability"
    )
    
    # Run Analysis
    # We use _selected_moves to include T=0 moves
    analyzer = Analyzer(
        db_conn=conn, 
        table_name="_selected_moves", 
        x_var=x_var, 
        y_var=y_var, 
        filter_query="move_ply <= 150",
        title="Instant Move Arc: Probability vs Game Stage"
    )
    
    # Save Plots
    figures_dir = os.path.join(src_dir, "figures", "ply_instantmove")
    analyzer.save_dashboard(os.path.join(figures_dir, "combined.png"), layout='1x2')
    
    conn.close()

if __name__ == "__main__":
    main()
