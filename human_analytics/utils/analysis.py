"""
Unified analysis framework for chess thinking dynamics.
Standardizes SQL-native statistics calculation and publication-ready plotting
using a clean, object-oriented approach.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def _seconds_from_log(axis) -> None:
    """Relabel a log-valued axis to show ``round(exp(v), 1)`` — i.e. seconds."""
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{np.round(np.exp(v), 1)}"))

from .helpers import (
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
)

# Ply tertiles are settled **a priori from the whole move dataset**, not from the
# (filtered) analysis subset: we infer the two ``move_ply`` tertile cutpoints once
# from this source table and apply the *same* boundaries to every plot, so the
# segmentation is identical and comparable across analyses regardless of filtering.
_PLY_TERTILE_SOURCE_DEFAULT = "processed_moves_nonzero"
_PLY_CUTS_CACHE: dict[str, tuple[int, int]] = {}


def _infer_ply_cuts(conn, source: str) -> tuple[int, int]:
    """Return the (1/3, 2/3) ``move_ply`` tertile cutpoints over the whole ``source`` table."""
    if source not in _PLY_CUTS_CACHE:
        c1, c2 = conn.execute(
            f"SELECT quantile_disc(move_ply, 1.0/3), quantile_disc(move_ply, 2.0/3) FROM {source}"
        ).fetchone()
        _PLY_CUTS_CACHE[source] = (int(c1), int(c2))
    return _PLY_CUTS_CACHE[source]


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
            return f"ln({self.column})"
        return self.column


class Analyzer:
    """
    Standard analyzer for the relationship between two variables (X and Y).
    Runs SQL-native aggregations in DuckDB.

    Segmentation by **ply tertiles** is settled a priori: the two ``move_ply``
    cutpoints are inferred once from the whole ``ply_tertile_source`` dataset
    (``quantile_disc`` at 1/3, 2/3), then applied as fixed boundaries to this
    (possibly filtered) analysis — so the segmentation is identical and comparable
    across plots. ``quantile_tertile_df`` holds per-tertile aggregates with the
    qbin ``ntile`` recomputed **within** each fixed tertile; legend labels are the
    fixed ply ranges (``self.ply_cuts``).

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
        zero_inflated: bool = False,
        zero_threshold: float = 0.0,
        ply_tertile_source: str = _PLY_TERTILE_SOURCE_DEFAULT,
    ):
        self.conn = db_conn
        self.table = table_name
        self.x = x_var
        self.y = y_var
        self.n_bins = n_bins
        self.filter_query = filter_query
        self.title = title or f"{self.x.label} vs {self.y.label}"
        # Source table whose whole move_ply distribution settles the tertile
        # cutpoints (applied identically to this — possibly filtered — analysis).
        self.ply_tertile_source = _validate_sql_identifier(ply_tertile_source)
        self.ply_cuts: tuple[int, int] | None = None
        # When the x distribution has a large mass near 0 (e.g. VOC: ~⅔ of moves
        # are 0), plain ``ntile`` wastes most bins on that mass. ``zero_inflated``
        # instead lumps the near-zero rows (``abs(x) <= zero_threshold``) into a
        # single leftmost point and quantile-bins only the remaining rows.
        # ``zero_threshold=0`` lumps exactly-zero rows.
        self.zero_inflated = zero_inflated
        self.zero_threshold = float(zero_threshold)

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
        self.quantile_df = None
        self.quantile_tertile_df = None

        self._run_sql_pipeline()

    def _run_sql_pipeline(self):
        """Executes the core SQL aggregation logic."""
        x_expr = self.x.sql_expression
        y_expr = self.y.sql_expression

        # 0. Settle ply tertile cutpoints from the WHOLE source dataset (a priori),
        # then assign each row's tertile by those fixed boundaries (no ntile over
        # the filtered subset). Tertile = 1 + #cutpoints exceeded (avoids CASE).
        c1, c2 = _infer_ply_cuts(self.conn, self.ply_tertile_source)
        self.ply_cuts = (c1, c2)
        tertile_expr = f"(1 + (move_ply > {c1})::INT + (move_ply > {c2})::INT)"

        # 1. Prepare temporary analysis view
        where_clause = f"WHERE {self.filter_query}" if self.filter_query else ""
        self.conn.execute(f"""
            CREATE OR REPLACE TEMPORARY VIEW _analyzer_view AS
            SELECT
                *,
                {x_expr} as _x_transformed,
                {y_expr} as _y_transformed,
                {tertile_expr} as _ply_tertile
            FROM {self.table}
            {where_clause}
        """)

        # 2. Extract dataset metadata
        self.n_games = self.conn.execute("SELECT count(distinct gid) FROM _analyzer_view").fetchone()[0]
        self.n_moves = self.conn.execute("SELECT count(*) FROM _analyzer_view").fetchone()[0]
        print(f"📊 Analyzing {self.n_moves:,} moves from {self.n_games:,} games...")

        # 3. Compute Quantile-Binned Data (global ranks). With ``zero_inflated``,
        # the x==0 point mass collapses to a single leftmost bin and only the
        # nonzero rows are ntiled (avoids wasting bins on the zero mass).
        col = self.x.column
        thr = self.zero_threshold
        if self.zero_inflated:
            global_inner = f"""
                SELECT {col}, _y_transformed, 0 AS qbin
                FROM _analyzer_view WHERE abs({col}) <= {thr}
                UNION ALL
                SELECT {col}, _y_transformed,
                       ntile({self.n_bins}) OVER (ORDER BY {col}) AS qbin
                FROM _analyzer_view WHERE abs({col}) > {thr}
            """
        else:
            global_inner = f"SELECT *, ntile({self.n_bins}) OVER (ORDER BY {col}) AS qbin FROM _analyzer_view"
        self.quantile_df = self.conn.execute(f"""
            SELECT qbin,
                   avg({col}) as mean_x,
                   avg(_y_transformed) as mean_y,
                   stddev(_y_transformed) as std_y,
                   count(*) as n
            FROM ({global_inner})
            GROUP BY qbin
        """).df()

        if self.zero_inflated:
            tertile_inner = f"""
                SELECT _ply_tertile as tertile_id, {col}, _y_transformed, 0 AS qbin
                FROM _analyzer_view WHERE abs({col}) <= {thr}
                UNION ALL
                SELECT _ply_tertile as tertile_id, {col}, _y_transformed,
                       ntile({self.n_bins}) OVER (PARTITION BY _ply_tertile ORDER BY {col}) AS qbin
                FROM _analyzer_view WHERE abs({col}) > {thr}
            """
        else:
            tertile_inner = f"""
                SELECT _ply_tertile as tertile_id, {col}, _y_transformed,
                       ntile({self.n_bins}) OVER (PARTITION BY _ply_tertile ORDER BY {col}) AS qbin
                FROM _analyzer_view
            """
        self.quantile_tertile_df = self.conn.execute(f"""
            SELECT tertile_id, qbin,
                   avg({col}) as mean_x,
                   avg(_y_transformed) as mean_y,
                   stddev(_y_transformed) as std_y,
                   count(*) as n
            FROM ({tertile_inner})
            GROUP BY tertile_id, qbin
            ORDER BY tertile_id, qbin
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

    def _ply_tertile_legend_label(self, tertile_id: int) -> str:
        """Fixed legend label from the a-priori (whole-dataset) tertile cutpoints."""
        c1, c2 = self.ply_cuts
        return {1: f"ply < {c1 + 1}", 2: f"ply {c1 + 1}–{c2}", 3: f"ply > {c2}"}.get(
            tertile_id, f"tertile {tertile_id}"
        )

    def plot_quantile_bins_tertile_segmented(self, ax, *, min_n: int = 1):
        """
        Quantile bins with **ntile recomputed within each fixed ply tertile**; all tertiles on ``ax``.
        Legend labels are the a-priori ply ranges (``self.ply_cuts``).
        """
        if self.quantile_tertile_df.empty:
            raise ValueError(
                "Ply-tertile-segmented quantile bins need non-empty data (check move_ply / filter)."
            )
        tertiles = sorted(self.quantile_tertile_df["tertile_id"].unique().tolist())
        y_label = "Move time (s)" if self.y.is_log else self.y.label
        x_label = f"{self.x.label} (qbin)"

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
        if self.y.is_log:
            _seconds_from_log(ax.yaxis)  # log-spaced positions, second-valued tick labels
        ax.legend(
            fontsize=FONT_SIZE_TICKS,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=1,
            frameon=False,
        )

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

    def plot_quantile_bins(self, ax):
        """Plots the trend across equal-sized quantile bins."""
        y_label = "Move time (s)" if self.y.is_log else self.y.label

        plot_qbin_stats(
            ax,
            self.quantile_df,
            x_col="mean_x",
            x_label=f"{self.x.label} (qbin)",
            y_label=y_label,
            normalized=False,
            show_legend=False,
        )
        if self.y.is_log:
            _seconds_from_log(ax.yaxis)  # log-spaced positions, second-valued tick labels

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
        fig.suptitle(f"{self.title}\nn = {self.n_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10, y=0.98)
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
        *,
        include_quantile_heatmap: bool = False,
        heatmap_alpha_mode: str = "log",
        heatmap_output_path: str | None = None,
    ):
        """Generates and saves a publication-ready dashboard.

        Fixed 1×2 layout: **Quantile bins** (global ranks, left) and **Quantile bins
        (by ply tertile)** (ntile recomputed within each fixed, a-priori ply tertile, right).
        Raw-trend and scatter panels were removed — quantile binning is the canonical view.

        Set ``include_quantile_heatmap=True`` (with ``quantile_heatmap_row='...'`` on
        construction) to also write the quantile×quantile heatmap as a **separate** PNG
        next to the dashboard (default ``<stem>_quantile_heatmap<ext>``). Pass
        ``heatmap_output_path`` to override the heatmap destination.
        """
        apply_poster_style()

        if include_quantile_heatmap and not self.quantile_heatmap_row:
            raise ValueError(
                "include_quantile_heatmap requires Analyzer(..., quantile_heatmap_row='<column>')."
            )

        fig, axes = plt.subplots(1, 2, figsize=(24, 19.6))
        self.plot_quantile_bins(axes[0])
        self.plot_quantile_bins_tertile_segmented(axes[1])
        # Panel titles omitted — left = global, right = by ply tertile (implied by
        # the legend + the "(qbin)" x-axis).

        suptitle = fig.suptitle(f"{self.title}\nn = {self.n_moves:,} moves", fontsize=FONT_SIZE_LABEL + 10)

        # Crop tightly on save and explicitly include the below-axes legend +
        # suptitle as extra artists so they aren't clipped (bbox_inches='tight'
        # alone misses the legend placed outside the axes via bbox_to_anchor).
        extra_artists = [suptitle]
        legend = axes[1].get_legend()
        if legend is not None:
            extra_artists.append(legend)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=extra_artists,
            pad_inches=0.3,
        )
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
