"""
Unified analysis framework for chess thinking dynamics.
Standardizes SQL-native statistics calculation and publication-ready plotting
using a clean, object-oriented approach.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt

from .helpers import (
    EPSILON,
    apply_poster_style,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    MAIN_COLOR,
    PHASE_COLORS,
)
from .plots import (
    get_isoluminant_cmap,
    plot_heatmap_with_alpha,
    plot_qbin_stats,
    plot_raw_trend,
    plot_subset_scatterplot,
)

_PLY_TERTILE_LEGEND_FALLBACK = {1: "Tertile 1", 2: "Tertile 2", 3: "Tertile 3"}

_SQL_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_sql_identifier(name: str) -> str:
    if not _SQL_IDENTIFIER.match(name):
        raise ValueError(
            f"quantile_heatmap_row must be a simple SQL identifier (letters, digits, underscore); got {name!r}"
        )
    return name


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

    Segmentation by **ply tertiles** uses ``ply_tertiles`` from preprocess:
    ``ntile(3) OVER (ORDER BY move_ply)`` globally on the move table (values 1–3).
    ``raw_trend_tertile_df`` and ``quantile_tertile_df`` hold per-tertile aggregates;
    quantile bins use ``ntile`` **partitioned by** ``ply_tertiles`` so ranks are recomputed
    within each tertile. Legend labels use observed ``move_ply`` ranges per tertile (min/max
    in the filtered data); tertiles split rows by global ``ntile``, so ply ranges may overlap.
    ``ply_tertile_bounds_df`` stores those min/max (and move counts) per tertile.

    Optional ``quantile_heatmap_row``: second column (e.g. ``move_ply``) for a
    quantile×quantile heatmap of mean transformed Y, saved as its own figure when
    ``save_dashboard(..., include_quantile_heatmap=True)`` (not embedded in the 2×2 grid).
    """

    def __init__(
        self,
        db_conn,
        table_name,
        x_var: Variable,
        y_var: Variable,
        n_bins=20,
        filter_query=None,
        title=None,
        quantile_heatmap_row: str | None = None,
        quantile_heatmap_row_label: str | None = None,
    ):
        self.conn = db_conn
        self.table = table_name
        self.x = x_var
        self.y = y_var
        self.n_bins = n_bins
        self.filter_query = filter_query
        self.title = title or f"{self.x.label} vs {self.y.label}"

        self.quantile_heatmap_row = None
        self._quantile_heatmap_row_label = ""
        self.quantile_heatmap_mean_df = None
        self.quantile_heatmap_count_df = None

        if quantile_heatmap_row is not None:
            qhr = _validate_sql_identifier(quantile_heatmap_row)
            if _SQL_IDENTIFIER.match(self.x.column.strip()) and qhr == self.x.column.strip():
                raise ValueError(
                    "quantile_heatmap_row must differ from x_var.column (otherwise the quantile grid is degenerate)."
                )
            self.quantile_heatmap_row = qhr
            self._quantile_heatmap_row_label = quantile_heatmap_row_label or qhr.replace("_", " ")

        # Results to be populated by _run_sql_pipeline()
        self.n_games = 0
        self.n_moves = 0
        self.raw_trend_df = None
        self.quantile_df = None
        self.sample_df = None
        self.raw_trend_tertile_df = None
        self.quantile_tertile_df = None
        self.ply_tertile_bounds_df = None

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

        self.ply_tertile_bounds_df = self.conn.execute("""
            SELECT
                ply_tertiles AS tertile_id,
                min(move_ply) AS min_ply,
                max(move_ply) AS max_ply,
                count(*) AS n_moves
            FROM _analyzer_view
            GROUP BY ply_tertiles
            ORDER BY tertile_id
        """).df()

        # 3. Compute Raw Trend Data
        self.raw_trend_df = self.conn.execute("""
            SELECT
                _x_transformed as x_val,
                avg(_y_transformed) as mean_y,
                stddev(_y_transformed) as std_y,
                count(*) as n
            FROM _analyzer_view
            GROUP BY _x_transformed
        """).df()

        # 4. Compute Quantile-Binned Data (global ranks)
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

        self.raw_trend_tertile_df = self.conn.execute("""
            SELECT
                ply_tertiles as tertile_id,
                _x_transformed as x_val,
                avg(_y_transformed) as mean_y,
                stddev(_y_transformed) as std_y,
                count(*) as n
            FROM _analyzer_view
            GROUP BY ply_tertiles, _x_transformed
        """).df()

        self.quantile_tertile_df = self.conn.execute(f"""
            SELECT
                tertile_id,
                qbin,
                avg({self.x.column}) as mean_x,
                avg(_y_transformed) as mean_y,
                stddev(_y_transformed) as std_y,
                count(*) as n
            FROM (
                SELECT
                    ply_tertiles as tertile_id,
                    {self.x.column},
                    _y_transformed,
                    ntile({self.n_bins}) OVER (
                        PARTITION BY ply_tertiles ORDER BY {self.x.column}
                    ) as qbin
                FROM _analyzer_view
            )
            GROUP BY tertile_id, qbin
            ORDER BY tertile_id, qbin
        """).df()

        # 5. Pull sample for scatter visualization (sampled in SQL for memory efficiency)
        print("Sampling moves for scatterplot...")
        self.sample_df = self.conn.execute("""
            SELECT _x_transformed as x, _y_transformed as y
            FROM _analyzer_view
            USING SAMPLE 100000 ROWS
        """).df()

        if self.quantile_heatmap_row:
            hr = self.quantile_heatmap_row
            nb = self.n_bins
            self.quantile_heatmap_mean_df = self.conn.execute(f"""
                PIVOT (
                    SELECT
                        ntile({nb}) OVER (ORDER BY {self.x.column}) AS x_qbin,
                        ntile({nb}) OVER (ORDER BY {hr}) AS row_qbin,
                        _y_transformed
                    FROM _analyzer_view
                )
                ON x_qbin USING avg(_y_transformed) GROUP BY row_qbin
            """).df().set_index("row_qbin")
            self.quantile_heatmap_count_df = self.conn.execute(f"""
                PIVOT (
                    SELECT
                        ntile({nb}) OVER (ORDER BY {self.x.column}) AS x_qbin,
                        ntile({nb}) OVER (ORDER BY {hr}) AS row_qbin
                    FROM _analyzer_view
                )
                ON x_qbin USING count(*) GROUP BY row_qbin
            """).df().set_index("row_qbin")

    def _tertile_sorted_ids(self) -> list:
        df = self.raw_trend_tertile_df
        if df.empty:
            return []
        return sorted(df["tertile_id"].unique().tolist())

    def _ply_tertile_legend_label(self, tertile_id: int) -> str:
        """Human-readable legend segment from observed move_ply bounds for this tertile."""
        df = self.ply_tertile_bounds_df
        if df is None or df.empty:
            return _PLY_TERTILE_LEGEND_FALLBACK.get(tertile_id, f"Tertile {tertile_id}")
        match = df[df["tertile_id"] == tertile_id]
        if match.empty:
            return _PLY_TERTILE_LEGEND_FALLBACK.get(tertile_id, f"Tertile {tertile_id}")
        lo_raw = match.iloc[0]["min_ply"]
        hi_raw = match.iloc[0]["max_ply"]
        lo, hi = int(lo_raw), int(hi_raw)
        tid = int(tertile_id)
        if lo == hi:
            return f"Tertile {tid} (ply = {lo})"
        return f"Tertile {tid} ({lo} ≤ ply ≤ {hi})"

    def plot_raw_trend_tertile_segmented(self, ax, *, min_n: int = 5):
        """
        Mean Y vs X (distinct transformed X), one curve + CI per ply tertile, all on ``ax``.
        Legend text includes observed ``move_ply`` bounds per tertile.
        """
        if self.raw_trend_tertile_df.empty:
            raise ValueError(
                "Ply-tertile-segmented raw trend needs non-empty data with column ply_tertiles "
                "(run preprocess on move tables)."
            )
        tertiles = self._tertile_sorted_ids()
        x_label = self.x.label + (" (log)" if self.x.is_log else "")
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label

        for t in tertiles:
            subset = self.raw_trend_tertile_df[self.raw_trend_tertile_df["tertile_id"] == t]
            color = PHASE_COLORS.get(int(t), MAIN_COLOR)
            lbl = self._ply_tertile_legend_label(int(t))
            plot_raw_trend(
                ax,
                subset,
                x_col="x_val",
                y_col="mean_y",
                std_col="std_y",
                n_col="n",
                x_label=x_label,
                y_label=y_label,
                color=color,
                label=lbl,
                min_n=min_n,
                show_legend=False,
                ci_legend_label=None,
            )
        ax.legend(fontsize=FONT_SIZE_TICKS)

    def plot_quantile_bins_tertile_segmented(self, ax, *, min_n: int = 1):
        """
        Quantile bins with **ntile recomputed within each ply tertile**; all tertiles on ``ax``.
        Legend text includes observed ``move_ply`` bounds per tertile.
        """
        if self.quantile_tertile_df.empty:
            raise ValueError(
                "Ply-tertile-segmented quantile bins need non-empty data with column ply_tertiles "
                "(run preprocess on move tables)."
            )
        tertiles = sorted(self.quantile_tertile_df["tertile_id"].unique().tolist())
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label
        x_label = f"Qbin {self.x.label}"

        for t in tertiles:
            subset = self.quantile_tertile_df[self.quantile_tertile_df["tertile_id"] == t]
            if min_n > 1:
                subset = subset[subset["n"] >= min_n]
            color = PHASE_COLORS.get(int(t), MAIN_COLOR)
            lbl = self._ply_tertile_legend_label(int(t))
            plot_qbin_stats(
                ax,
                subset,
                x_col="mean_x",
                x_label=x_label,
                y_label=y_label,
                color=color,
                label=lbl,
                normalized=False,
                show_legend=False,
                ci_legend_label=None,
            )
        ax.legend(fontsize=FONT_SIZE_TICKS)

    def plot_quantile_heatmap(self, ax, *, alpha_mode: str = "log"):
        """
        Joint quantile bins of ``x`` and ``quantile_heatmap_row``: cell color = mean transformed Y;
        alpha = cell frequency (see ``plot_heatmap_with_alpha``).
        """
        if not self.quantile_heatmap_row:
            raise ValueError(
                "Quantile heatmap requires Analyzer(..., quantile_heatmap_row='<column>', ...), "
                "e.g. quantile_heatmap_row='move_ply' for clock vs ply."
            )
        mean_df = self.quantile_heatmap_mean_df
        count_df = self.quantile_heatmap_count_df
        if mean_df is None or mean_df.empty:
            return

        y_disp = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label
        plot_heatmap_with_alpha(
            ax,
            mean_df,
            count_df,
            get_isoluminant_cmap(),
            alpha_mode=alpha_mode,
            value_label=f"Mean {y_disp}",
            imshow_aspect="equal",
        )
        ax.set_xlabel(f"{self.x.label} quantile bin", fontsize=FONT_SIZE_LABEL, labelpad=36)
        ax.set_ylabel(f"{self._quantile_heatmap_row_label} quantile bin", fontsize=FONT_SIZE_LABEL, labelpad=48)
        ax.set_title("Quantile × quantile heatmap", fontsize=FONT_SIZE_LABEL, pad=12)

    def plot_raw_trend(self, ax):
        """Plots the raw trend of mean Y vs X."""
        x_label = self.x.label + (" (log)" if self.x.is_log else "")
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label

        plot_raw_trend(
            ax,
            self.raw_trend_df,
            x_col="x_val",
            y_col="mean_y",
            std_col="std_y",
            n_col="n",
            x_label=x_label,
            y_label=y_label,
            min_n=5,
            show_legend=False,
        )

    def plot_quantile_bins(self, ax):
        """Plots the trend across equal-sized quantile bins."""
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label

        plot_qbin_stats(
            ax,
            self.quantile_df,
            x_col="mean_x",
            x_label=f"Qbin {self.x.label}",
            y_label=y_label,
            normalized=False,
            show_legend=False,
        )

    def plot_scatter(self, ax, n_points=10000):
        """Plots a density scatterplot."""
        x_label = self.x.label + (" (log)" if self.x.is_log else "")
        y_label = (r"$\log(" + self.y.label + ")$") if self.y.is_log else self.y.label

        plot_subset_scatterplot(
            ax,
            self.sample_df,
            x_col="x",
            y_col="y",
            x_label=x_label,
            y_label=y_label,
            n=n_points,
        )

    def save_quantile_heatmap_figure(
        self,
        output_path,
        *,
        heatmap_alpha_mode: str = "log",
    ):
        """Save a single-panel quantile×quantile heatmap (not embedded in the 2×2 dashboard)."""
        if not self.quantile_heatmap_row:
            raise ValueError(
                "save_quantile_heatmap_figure requires Analyzer(..., quantile_heatmap_row='<column>')."
            )
        apply_poster_style()
        fig, ax = plt.subplots(figsize=(22, 18))
        self.plot_quantile_heatmap(ax, alpha_mode=heatmap_alpha_mode)
        subtitle = f"{self.n_games:,} games | {self.n_moves:,} moves"
        fig.suptitle(f"{self.title}\n{subtitle}", fontsize=FONT_SIZE_LABEL + 10, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Quantile heatmap saved to {output_path}")

    def save_dashboard(
        self,
        output_path,
        layout="2x2",
        *,
        include_quantile_heatmap: bool = False,
        heatmap_alpha_mode: str = "log",
        heatmap_output_path: str | None = None,
    ):
        """Generates and saves a publication-ready dashboard.

        ``2x2`` (default): global raw trend | global quantile bins; then the same pair by
        ``ply_tertiles`` (overlaid tertiles). ``1x3`` keeps raw | quantile | scatter. ``1x2`` is
        raw | quantile only.

        Set ``include_quantile_heatmap=True`` (with ``layout="2x2"`` only, and
        ``quantile_heatmap_row='...'`` on construction) to also write the quantile×quantile
        heatmap as a **separate** PNG next to the dashboard (by default:
        ``<stem>_quantile_heatmap<ext>`` beside ``output_path``). Pass ``heatmap_output_path``
        to override the heatmap destination.
        """
        apply_poster_style()

        if include_quantile_heatmap and layout != "2x2":
            raise ValueError("include_quantile_heatmap is only supported with layout='2x2'.")
        if include_quantile_heatmap and not self.quantile_heatmap_row:
            raise ValueError(
                "include_quantile_heatmap requires Analyzer(..., quantile_heatmap_row='<column>')."
            )

        if layout == "2x2":
            fig, axes = plt.subplots(2, 2, figsize=(36, 26))
            self.plot_raw_trend(axes[0, 0])
            self.plot_quantile_bins(axes[0, 1])
            self.plot_raw_trend_tertile_segmented(axes[1, 0])
            self.plot_quantile_bins_tertile_segmented(axes[1, 1])
            axes[0, 0].set_title("Raw trend", fontsize=FONT_SIZE_LABEL, pad=12)
            axes[0, 1].set_title("Quantile bins", fontsize=FONT_SIZE_LABEL, pad=12)
            axes[1, 0].set_title("Raw trend (by ply tertile)", fontsize=FONT_SIZE_LABEL, pad=12)
            axes[1, 1].set_title("Quantile bins (by ply tertile)", fontsize=FONT_SIZE_LABEL, pad=12)
        elif layout == "1x3":
            fig, axes = plt.subplots(1, 3, figsize=(36, 12))
            self.plot_raw_trend(axes[0])
            self.plot_quantile_bins(axes[1])
            self.plot_scatter(axes[2])
        elif layout == "1x2":
            fig, axes = plt.subplots(1, 2, figsize=(24, 12))
            self.plot_raw_trend(axes[0])
            self.plot_quantile_bins(axes[1])
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        subtitle = f"{self.n_games:,} games | {self.n_moves:,} moves"
        fig.suptitle(f"{self.title}\n{subtitle}", fontsize=FONT_SIZE_LABEL + 10, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Dashboard saved to {output_path}")

        if include_quantile_heatmap:
            if heatmap_output_path is None:
                root, ext = os.path.splitext(output_path)
                heatmap_output_path = f"{root}_quantile_heatmap{ext}"
            self.save_quantile_heatmap_figure(
                heatmap_output_path,
                heatmap_alpha_mode=heatmap_alpha_mode,
            )
