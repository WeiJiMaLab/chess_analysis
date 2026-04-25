"""
Analysis of the relationship between number of legal moves and move time.
"""

import os
import duckdb
from utils import Variable, Analyzer

# Constants
PERSONAL_DB = "/scratch/gpfs/GRIFFITHS/hl4291/personal.db"

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    conn = duckdb.connect(database=PERSONAL_DB, read_only=False)
    
    # Configure Variables
    x_var = Variable(column="n_possible_moves", is_log=False, name="Number of Legal Moves")
    y_var = Variable(column="move_time", is_log=False, name="Move Time (s)")
    
    # Run Analysis
    analyzer = Analyzer(
        db_conn=conn, 
        table_name="_selected_moves_nonzero_T", 
        x_var=x_var, 
        y_var=y_var, 
        filter_query="n_possible_moves < 50",
        title="Branching Factor Influence"
    )
    
    # Save Plots
    figures_dir = os.path.join(src_dir, "figures", "npossiblemoves_movetime")
    analyzer.save_dashboard(os.path.join(figures_dir, "combined.png"))
    
    conn.close()

if __name__ == "__main__":
    main()
