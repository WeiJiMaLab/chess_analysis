"""
Unified analysis framework for chess thinking dynamics.
Standardizes SQL-native statistics calculation and publication-ready plotting
using a clean, object-oriented approach.
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .helpers import (
    EPSILON, apply_poster_style, MAIN_COLOR, FONT_SIZE_LABEL, FONT_SIZE_TICKS
)
from .plots import plot_raw_trend, plot_qbin_stats, plot_subset_scatterplot

@dataclass
class Variable:
    column: str
    is_log: bool = False
    name: str = None

    @property
    def label(self):
        """Returns the human-readable name of the variable."""
        return self.name or self.column

    @property
    def sql_expression(self):
        """Returns the SQL expression for the variable, including log-transform if needed."""
        if self.is_log:
            return f"ln({self.column} + {EPSILON})"
        return self.column

class Analyzer:
    """
    Standard analyzer for the relationship between two variables (X and Y).
    Runs SQL-native aggregations in DuckDB.
    """
    def __init__(self, db_conn, table_name, x_var: Variable, y_var: Variable, n_bins=20, filter_query=None, title=None):
        self.conn = db_conn
        self.table = table_name
        self.x = x_var
        self.y = y_var
        self.n_bins = n_bins
        self.filter_query = filter_query
        self.title = title or f"{self.x.label} vs {self.y.label}"
        
        # Results to be populated by _run_sql_pipeline()
        self.n_games = 0
        self.n_moves = 0
        self.raw_trend_df = None
        self.quantile_df = None
        self.sample_df = None
        
        self._run_sql_pipeline()

    def _run_sql_pipeline(self):
        """Executes the core SQL aggregation logic."""
        x_expr = self.x.sql_expression
        y_expr = self.y.sql_expression
        
        # 1. Prepare temporary analysis view
        where_clause = f"WHERE {self.filter_query}" if self.filter_query else ""
        self.conn.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW _analyzer_view AS
            SELECT 
                *,
                {x_expr} as _x_transformed,
                {y_expr} as _y_transformed
            FROM {self.table}
            {where_clause}
        """)

        # 2. Extract dataset metadata
        self.n_games = self.conn.execute("SELECT count(distinct gid) FROM _analyzer_view").fetchone()[0]
        self.n_moves = self.conn.execute("SELECT count(*) FROM _analyzer_view").fetchone()[0]
        print(f"📊 Analyzing {self.n_moves:,} moves from {self.n_games:,} games...")

        # 3. Compute Raw Trend Data
        self.raw_trend_df = self.conn.execute(f"""
            SELECT 
                _x_transformed as x_val,
                avg(_y_transformed) as mean_y,
                stddev(_y_transformed) as std_y,
                count(*) as n
            FROM _analyzer_view
            GROUP BY 1
        """).df()

        # 4. Compute Quantile-Binned Data
        self.quantile_df = self.conn.execute(f"""
            SELECT 
                qbin,
                avg({self.x.column}) as mean_x,
                avg(_y_transformed) as mean_y,
                stddev(_y_transformed) as std_y,
                count(*) as n
            FROM (
                SELECT *, ntile({self.n_bins}) over (order by {self.x.column}) as qbin
                FROM _analyzer_view
            )
            GROUP BY qbin
        """).df()
        
        # 5. Pull sample for scatter visualization (sampled in SQL for memory efficiency)
        print("Sampling moves for scatterplot...")
        self.sample_df = self.conn.execute(f"""
            SELECT _x_transformed as x, _y_transformed as y 
            FROM _analyzer_view 
            USING SAMPLE 100000 ROWS
        """).df()

    def plot_raw_trend(self, ax):
        """Plots the raw trend of mean Y vs X."""
        x_label = self.x.label + (" (log)" if self.x.is_log else "")
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label
        
        plot_raw_trend(
            ax, self.raw_trend_df,
            x_col="x_val",
            y_col="mean_y",
            std_col="std_y",
            n_col="n",
            x_label=x_label,
            y_label=y_label,
            min_n=5,
            show_legend=False
        )

    def plot_quantile_bins(self, ax):
        """Plots the trend across equal-sized quantile bins."""
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label
        
        plot_qbin_stats(
            ax, self.quantile_df,
            x_col="mean_x",
            x_label=f"Qbin {self.x.label}",
            y_label=y_label,
            normalized=False,
            show_legend=False
        )

    def plot_scatter(self, ax, n_points=10000):
        """Plots a density scatterplot."""
        x_label = self.x.label + (" (log)" if self.x.is_log else "")
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label
        
        plot_subset_scatterplot(
            ax, self.sample_df,
            x_col="x",
            y_col="y",
            x_label=x_label,
            y_label=y_label,
            n=n_points
        )

    def save_dashboard(self, output_path, layout='1x3'):
        """Generates and saves a publication-ready dashboard."""
        apply_poster_style()
        
        if layout == '1x3':
            fig, axes = plt.subplots(1, 3, figsize=(36, 11))
            self.plot_raw_trend(axes[0])
            self.plot_quantile_bins(axes[1])
            self.plot_scatter(axes[2])
        elif layout == '1x2':
            fig, axes = plt.subplots(1, 2, figsize=(24, 11))
            self.plot_raw_trend(axes[0])
            self.plot_quantile_bins(axes[1])
        elif layout == '2x2':
            # If 2x2 is requested, we use 3 slots and leave one empty or reorganize
            fig, axes = plt.subplots(2, 2, figsize=(24, 18))
            axes_flat = axes.flatten()
            self.plot_raw_trend(axes_flat[0])
            self.plot_quantile_bins(axes_flat[1])
            self.plot_scatter(axes_flat[2])
            axes_flat[3].axis('off')
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        subtitle = f"{self.n_games:,} games | {self.n_moves:,} moves"
        fig.suptitle(f"{self.title}\n{subtitle}", fontsize=FONT_SIZE_LABEL + 10, y=1.05)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Dashboard saved to {output_path}")
