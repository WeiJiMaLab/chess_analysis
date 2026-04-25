"""
Analysis of the relationship between game stage (ply) and move time.
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
    x_var = Variable(column="move_ply", is_log=False, name="Move Ply")
    y_var = Variable(column="move_time", is_log=True, name="T")
    
    # Run Analysis
    analyzer = Analyzer(
        db_conn=conn, 
        table_name="_selected_moves_nonzero_T", 
        x_var=x_var, 
        y_var=y_var, 
        filter_query="move_ply <= 150",
        title="Thinking Arc: Time vs Game Stage"
    )
    
    # Save Plots
    figures_dir = os.path.join(src_dir, "figures", "ply_movetime")
    analyzer.save_dashboard(os.path.join(figures_dir, "combined.png"), layout='1x2')
    
    conn.close()

if __name__ == "__main__":
    main()
