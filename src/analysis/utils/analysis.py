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

from analysis.utils.helpers import (
    apply_poster_style,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICKS,
    MAIN_COLOR,
    PHASE_COLORS,
)
from analysis.utils.plots import (
    get_isoluminant_cmap,
    plot_heatmap_with_alpha,
    plot_qbin_stats,
)

# Ply tertiles are settled **a priori from the whole move dataset**, not from the
# (filtered) analysis subset: we infer the two ``move_ply`` tertile cutpoints once
# from this source table and apply the *same* boundaries to every plot, so the
# segmentation is identical and comparable across analyses regardless of filtering.
_PLY_TERTILE_SOURCE_DEFAULT = "processed_moves_nonzero"
_TERTILE_CUTS_CACHE: dict[tuple[str, str], tuple[int, int]] = {}

# Highlight color for an isolated value point-mass (e.g. Gain==0) on a LOWESS panel:
# black ``x`` reads as a discrete "special point" against the steel-blue smoother.
_MASS_COLOR = "black"


def _infer_tertile_cuts(conn, source: str, column: str = "move_ply") -> tuple[int, int]:
    """Return the (1/3, 2/3) tertile cutpoints of ``column`` over the whole ``source`` table.
    Default ``column='move_ply'`` gives the a-priori ply tertiles; pass another column (e.g.
    ``'gss'``) to segment the same plot by a different variable instead."""
    key = (source, column)
    if key not in _TERTILE_CUTS_CACHE:
        c1, c2 = conn.execute(
            f"SELECT quantile_disc({column}, 1.0/3), quantile_disc({column}, 2.0/3) FROM {source}"
        ).fetchone()
        if c1 is None or c2 is None:
            c1, c2 = 0, 0
        _TERTILE_CUTS_CACHE[key] = (int(c1), int(c2))
    return _TERTILE_CUTS_CACHE[key]


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
            return f"CASE WHEN {self.column} > 0 THEN ln({self.column}) ELSE NULL END"
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
        segment_column: str = "move_ply",
        segment_source: str | None = None,
        segment_label: str = "ply",
        bin_mode: str = "ntile",
        integer_tail_cut: float | None = None,
        integer_bin_width: int = 1,
        edge_mass: tuple[str, float] | list[tuple[str, float]] | None = None,
        tie_safe: bool = False,
        min_bin_count: int = 0,
    ):
        """Binning of x is controlled by three opt-in mechanisms; all default to the
        legacy equal-count ``ntile`` behavior so existing callers are unchanged.

        ``bin_mode``:
          * ``"ntile"`` (default): equal-count quantile bins via SQL ``ntile``.
          * ``"integer"``: group x into fixed-width integer bins of ``integer_bin_width``
            (default 1 = one point per distinct integer; e.g. 5 = groups 0–4, 5–9, …).
            Use for discrete/integer x (e.g. OSS) where ``ntile`` would split a heavy
            point-mass across several identical-mean bins. Each point is plotted at the
            group's mean x. The long thin tail above ``integer_tail_cut`` (if given) is
            merged into a single point, so sparse high values don't read as noise.
            Applies to BOTH the global and the by-ply-tertile panels.

        ``edge_mass`` (tie-safe boundary masses): one ``(op, value)`` pair or a list
        of them, where ``op`` is ``">="``/``"<="``/``">"``/``"<"``/``"=="``. Rows
        satisfying a clause are pulled out into their *own* dedicated point at their
        mean x, instead of being straddled across ``ntile`` edges. Use for discrete
        boundary masses that ``ntile`` otherwise splits (VOC: ``(">=", 1.0)`` for the
        spike at 1.0 + the >1 tail; MQ: ``("<=", -1.0)``). Only meaningful together
        with ``zero_inflated`` / ``tie_safe`` (i.e. the non-ntile interior path).

        ``tie_safe``: when binning the nonzero interior (``zero_inflated`` and/or
        ``edge_mass``), assign rows to bins by *value range* (``width_bucket`` over
        quantile cut-points) rather than equal-count rank, so all rows sharing one
        x-value land in the same bin (no within-value RT noise from a split mass).
        ``zero_inflated`` alone keeps the legacy rank-based interior unless
        ``tie_safe=True``.
        """
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
        # The by-segment panel splits on ``segment_column`` (default ``move_ply`` → ply
        # tertiles); ``segment_source`` is the table its a-priori cutpoints come from
        # (default = ``ply_tertile_source``), and ``segment_label`` is the legend prefix.
        # Pass e.g. (segment_column='gss', segment_source='mq_rt', segment_label='GSS') to
        # get the *same* MQ-vs-RT plot segmented by GSS stratum instead of ply.
        self.segment_column = _validate_sql_identifier(segment_column)
        self.segment_source = _validate_sql_identifier(segment_source or ply_tertile_source)
        self.segment_label = segment_label
        self.ply_cuts: tuple[int, int] | None = None
        # When the x distribution has a large mass near 0 (e.g. VOC: ~⅔ of moves
        # are 0), plain ``ntile`` wastes most bins on that mass. ``zero_inflated``
        # instead lumps the near-zero rows (``abs(x) <= zero_threshold``) into a
        # single leftmost point and quantile-bins only the remaining rows.
        # ``zero_threshold=0`` lumps exactly-zero rows.
        self.zero_inflated = zero_inflated
        self.zero_threshold = float(zero_threshold)

        if bin_mode not in ("ntile", "integer"):
            raise ValueError(f"bin_mode must be 'ntile' or 'integer'; got {bin_mode!r}")
        self.bin_mode = bin_mode
        self.integer_tail_cut = None if integer_tail_cut is None else float(integer_tail_cut)
        self.integer_bin_width = int(integer_bin_width)
        if self.integer_bin_width < 1:
            raise ValueError(f"integer_bin_width must be >= 1; got {integer_bin_width!r}")
        self.tie_safe = bool(tie_safe)
        # Drop plotted bins with fewer than this many rows (0 = keep all). Lets the
        # noisy sparse tails (e.g. high-VOC, high-OSS) be cut from both panels.
        self.min_bin_count = int(min_bin_count)
        # Normalize edge_mass to a list of (op, value) clauses (validated below).
        if edge_mass is None:
            self.edge_mass = []
        elif isinstance(edge_mass, tuple):
            self.edge_mass = [edge_mass]
        else:
            self.edge_mass = list(edge_mass)
        _allowed_ops = {">=", "<=", ">", "<", "=="}
        for op, val in self.edge_mass:
            if op not in _allowed_ops:
                raise ValueError(f"edge_mass op must be one of {_allowed_ops}; got {op!r}")
            float(val)

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

    def _edge_predicate(self) -> str:
        """SQL boolean: TRUE for rows captured by any ``edge_mass`` clause."""
        if not self.edge_mass:
            return "FALSE"
        return " OR ".join(f"({self.x.column} {op} {val})" for op, val in self.edge_mass)

    def _bin_assignment_sql(self, *, partition_by: str | None) -> str:
        """Emit an inner SELECT assigning each row a ``qbin``, honoring the active
        binning mode. Columns produced: ``[<partition_by> as tertile_id,] {col},
        _y_transformed, qbin``. ``partition_by`` (e.g. ``_ply_tertile``) makes the
        interior binning independent per group; the global panel passes ``None``.

        qbin is only used to GROUP rows into points and is plotted sorted by mean_x,
        so its absolute values need only be distinct per group (not globally aligned).
        Dedicated points use sentinel qbins offset from the interior range.
        """
        col = self.x.column
        sel = f"{partition_by} as tertile_id, " if partition_by else ""
        part_clause = f"PARTITION BY {partition_by}" if partition_by else ""

        # --- integer mode: fixed-width integer groups, optional tail merge -----
        if self.bin_mode == "integer":
            w = self.integer_bin_width
            # Group index = floor(x / w); width 1 ⇒ one point per integer. Points are
            # plotted at the group mean x (avg(col)), so the bin width need not show.
            grp = f"CAST(floor({col} / {w}) AS BIGINT)" if w != 1 else f"CAST({col} AS BIGINT)"
            if self.integer_tail_cut is not None:
                # At/above cut: a single merged point (sentinel qbin sorts rightmost;
                # plot orders by mean_x anyway).
                qbin = f"CASE WHEN {col} >= {self.integer_tail_cut} THEN 1000000000 ELSE {grp} END"
            else:
                qbin = grp
            return f"SELECT {sel}{col}, _y_transformed, {qbin} AS qbin FROM _analyzer_view"

        # --- ntile / zero-inflated / tie-safe / edge-mass modes ----------------
        has_zero = self.zero_inflated
        has_edge = bool(self.edge_mass)

        if not has_zero and not has_edge and not self.tie_safe:
            # Legacy: pure equal-count ntile (unchanged behavior).
            return (
                f"SELECT {sel}{col}, _y_transformed, "
                f"ntile({self.n_bins}) OVER ({part_clause} ORDER BY {col}) AS qbin "
                f"FROM _analyzer_view"
            )

        thr = self.zero_threshold
        edge_pred = self._edge_predicate()
        # Interior = rows that are neither the near-zero lump nor an edge mass.
        zero_pred = f"abs({col}) <= {thr}" if has_zero else "FALSE"
        interior_pred = f"NOT ({zero_pred}) AND NOT ({edge_pred})"

        parts: list[str] = []
        if has_zero:
            # Near-zero lump → dedicated leftmost point (qbin = 0).
            parts.append(
                f"SELECT {sel}{col}, _y_transformed, 0 AS qbin "
                f"FROM _analyzer_view WHERE {zero_pred}"
            )

        # Interior binning.
        if self.tie_safe:
            # Tie-safe: bucket by VALUE over per-group quantile cut-points, so
            # identical x-values never split across adjacent bins. The bin index is
            # the number of cut-points the value exceeds (computed with list_filter,
            # which is portable across DuckDB builds that lack width_bucket).
            edges = ", ".join(
                f"quantile_cont({col}, {i / self.n_bins})" for i in range(1, self.n_bins)
            )
            edges_subq = (
                f"(SELECT [{edges}] FROM _analyzer_view i2 WHERE {interior_pred}"
                + (f" AND i2.{partition_by} = i.{partition_by}" if partition_by else "")
                + ")"
            )
            interior = (
                f"SELECT {sel}{col}, _y_transformed, "
                f"1 + len(list_filter({edges_subq}, e -> {col} > e)) AS qbin "
                f"FROM _analyzer_view i WHERE {interior_pred}"
            )
        else:
            interior = (
                f"SELECT {sel}{col}, _y_transformed, "
                f"ntile({self.n_bins}) OVER ({part_clause} ORDER BY {col}) AS qbin "
                f"FROM _analyzer_view WHERE {interior_pred}"
            )
        parts.append(interior)

        if has_edge:
            # Each edge-mass clause → its own dedicated point, sorted to the right
            # of the interior (sentinel qbins above n_bins).
            for j, (op, val) in enumerate(self.edge_mass):
                parts.append(
                    f"SELECT {sel}{col}, _y_transformed, {self.n_bins + 1 + j} AS qbin "
                    f"FROM _analyzer_view WHERE {col} {op} {val}"
                )

        return "\nUNION ALL\n".join(parts)

    def _run_sql_pipeline(self):
        """Executes the core SQL aggregation logic."""
        x_expr = self.x.sql_expression
        y_expr = self.y.sql_expression

        # 0. Settle ply tertile cutpoints from the WHOLE source dataset (a priori),
        # then assign each row's tertile by those fixed boundaries (no ntile over
        # the filtered subset). Tertile = 1 + #cutpoints exceeded (avoids CASE).
        c1, c2 = _infer_tertile_cuts(self.conn, self.segment_source, self.segment_column)
        self.ply_cuts = (c1, c2)
        tertile_expr = f"(1 + ({self.segment_column} > {c1})::INT + ({self.segment_column} > {c2})::INT)"

        # 1. Prepare temporary analysis view.
        # Filter out non-finite (NaN, Inf, -Inf) values to prevent mathematical out-of-range errors in stddev/aggregates.
        finite_cond = (
            f"({self.x.column} IS NOT NULL AND NOT isnan({self.x.column}) AND NOT isinf({self.x.column})) AND "
            f"({self.y.column} IS NOT NULL AND NOT isnan({self.y.column}) AND NOT isinf({self.y.column}))"
        )
        if self.filter_query:
            where_clause = f"WHERE ({self.filter_query}) AND {finite_cond}"
        else:
            where_clause = f"WHERE {finite_cond}"
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

        # 3. Compute Quantile-Binned Data. Binning of x is one of three modes
        # (see __init__): legacy equal-count ``ntile``; ``zero_inflated`` lumping
        # the near-zero mass into a leftmost point; ``bin_mode="integer"`` grouping
        # by integer value; plus tie-safe interior bucketing and dedicated
        # ``edge_mass`` boundary points. The qbin assignment is emitted by
        # ``_bin_assignment_sql`` so the global and per-tertile panels stay identical.
        col = self.x.column

        global_inner = self._bin_assignment_sql(partition_by=None)
        self.quantile_df = self.conn.execute(f"""
            SELECT qbin,
                   avg({col}) as mean_x,
                   avg(_y_transformed) as mean_y,
                   stddev(_y_transformed) as std_y,
                   count(*) as n
            FROM ({global_inner})
            GROUP BY qbin
        """).df()

        tertile_inner = self._bin_assignment_sql(partition_by="_ply_tertile")
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
        """Fixed legend label from the a-priori (whole-dataset) tertile cutpoints, prefixed
        with ``segment_label`` (e.g. 'ply < 28' for ply, 'GSS 2–31' for a GSS segmentation)."""
        c1, c2 = self.ply_cuts
        lab = self.segment_label
        return {1: f"{lab} < {c1 + 1}", 2: f"{lab} {c1 + 1}–{c2}", 3: f"{lab} > {c2}"}.get(
            tertile_id, f"{lab} tertile {tertile_id}"
        )

    @property
    def _x_axis_label(self) -> str:
        """X label; only suffix "(qbin)" when x is binned by plain equal-count ntile.
        Integer / tie-safe / zero-inflated / edge-mass modes are not qbins."""
        plain_qbin = (
            self.bin_mode == "ntile"
            and not self.zero_inflated
            and not self.tie_safe
            and not self.edge_mass
        )
        return f"{self.x.label} (qbin)" if plain_qbin else self.x.label

    def plot_quantile_bins_tertile_segmented(self, ax, *, min_n: int | None = None):
        """
        Quantile bins with **ntile recomputed within each fixed ply tertile**; all tertiles on ``ax``.
        Legend labels are the a-priori ply ranges (``self.ply_cuts``).
        """
        if self.quantile_tertile_df.empty:
            raise ValueError(
                "Ply-tertile-segmented quantile bins need non-empty data (check move_ply / filter)."
            )
        min_n = self.min_bin_count if min_n is None else min_n
        tertiles = sorted(self.quantile_tertile_df["tertile_id"].unique().tolist())
        y_label = "Move time (s)" if self.y.is_log else self.y.label
        x_label = self._x_axis_label

        any_pos_x = False
        for t in tertiles:
            subset = self.quantile_tertile_df[self.quantile_tertile_df["tertile_id"] == t]
            if min_n:
                subset = subset[subset["n"] >= min_n]
            any_pos_x = any_pos_x or (len(subset) and (subset["mean_x"] > 0).any())
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
        if self.x.is_log and any_pos_x:
            ax.set_xscale("log")  # mean_x is raw units; log-scale the axis (skip empty/no-positive panels)
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

        df = self.quantile_df
        if self.min_bin_count:
            df = df[df["n"] >= self.min_bin_count]
        plot_qbin_stats(
            ax,
            df,
            x_col="mean_x",
            x_label=self._x_axis_label,
            y_label=y_label,
            normalized=False,
            show_legend=False,
        )
        if self.y.is_log:
            _seconds_from_log(ax.yaxis)  # log-spaced positions, second-valued tick labels
        if self.x.is_log and len(df) and (df["mean_x"] > 0).any():
            ax.set_xscale("log")  # mean_x is raw units; log-scale the axis (skip empty/no-positive panels)

    # --- Binning-free estimator (LOWESS + curve-level bootstrap band) ----------
    # For a CONTINUOUS predictor, a fixed-K binned staircase does not converge to the
    # smooth regression function as n grows (its resolution is frozen by K); a bandwidth
    # smoother does. We isolate any value point-mass (e.g. Gain's 55% at exactly 0, which
    # violates local smoothness) as its own labeled point and LOWESS only the continuous
    # remainder. See diagnose_gain.md "Gain binning (K-vs-n)".

    def _fetch_xy(self, tertile: int | None = None):
        """Raw (x, _y_transformed) arrays from the analyzer view, optionally one ply tertile."""
        where = "" if tertile is None else f" WHERE _ply_tertile = {int(tertile)}"
        cols = self.conn.execute(
            f"SELECT {self.x.column} AS x, _y_transformed AS y FROM _analyzer_view{where}"
        ).fetchnumpy()
        x = np.asarray(cols["x"], dtype=float)
        y = np.asarray(cols["y"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        return x[finite], y[finite]

    def plot_lowess(self, ax, *, mass_values=(), frac: float = 0.3, n_boot: int = 120,
                    grid_n: int = 120, color=None, tertile: int | None = None,
                    show_band: bool = True, band_alpha: float = 0.22, show_mass: bool = True,
                    overlay_binned: bool = False, label_prefix: str = "", max_fit_n: int = 500_000):
        """LOWESS fit of (transformed) Y on X + curve-level bootstrap band, with value
        point-masses in ``mass_values`` isolated (always excluded from the fit; drawn as
        their own SEM point when ``show_mass``)."""
        from analysis.utils.jaggedness import lowess_bootstrap
        color = color or MAIN_COLOR
        x, y = self._fetch_xy(tertile=tertile)
        if x.size == 0:
            return
        keep = np.ones(x.size, dtype=bool)
        mass_points = []
        for v in mass_values:
            at = np.isclose(x, v, atol=1e-9)
            if at.sum() >= max(self.min_bin_count, 1):
                yc = y[at]
                mass_points.append((float(v), float(yc.mean()),
                                    float(1.96 * yc.std() / np.sqrt(yc.size)), int(at.sum())))
                keep &= ~at
        # Fit in the DISPLAYED x-coordinate: log(x) when the x-axis is log (e.g. RT), raw
        # otherwise, so the smoother's bandwidth matches what the eye sees.
        xc_fit = np.log(x[keep]) if self.x.is_log else x[keep]
        yc = y[keep]
        good = np.isfinite(xc_fit) & np.isfinite(yc)
        xc_fit, yc = xc_fit[good], yc[good]
        if xc_fit.size > max_fit_n:
            # The LOWESS curve + (already very tight) band are unchanged by capping the fit
            # sample; this just keeps the bootstrap tractable on the multi-million-row panels.
            sel = np.random.default_rng(0).choice(xc_fit.size, max_fit_n, replace=False)
            xc_fit, yc = xc_fit[sel], yc[sel]
        if xc_fit.size >= 50:
            lo_x, hi_x = np.quantile(xc_fit, [0.002, 0.998])  # trim sparse extremes for a stable band
            grid = np.linspace(lo_x, hi_x, grid_n)
            delta = 0.01 * float(xc_fit.max() - xc_fit.min() + 1e-12)  # statsmodels large-n speedup
            res = lowess_bootstrap(xc_fit, yc, grid, frac=frac, n_boot=n_boot, delta=delta)
            grid_plot = np.exp(grid) if self.x.is_log else grid  # back to raw units for a log-scaled axis
            # Method (LOWESS + band) is stated in the figure title, not the legend.
            # Only the per-tertile curve carries a (tertile-name) label; the global curve
            # and the band carry none, to keep the legend minimal.
            curve_label = label_prefix.rstrip(": ") or None
            if show_band:
                ax.fill_between(grid_plot, res["lo"], res["hi"], color=color, alpha=band_alpha,
                                lw=0)
            ax.plot(grid_plot, res["fit"], color=color, lw=3.5, label=curve_label)
        if show_mass:
            for v, c, half, n in mass_points:
                ax.errorbar([v], [c], yerr=[half], marker="x", ms=11, color=_MASS_COLOR,
                            mew=2.5, ecolor=_MASS_COLOR, elinewidth=1.8,
                            capsize=3, zorder=5, label=f"x={v:g} mass (n={n:,})")
        if overlay_binned and tertile is None:
            df = self.quantile_df
            if self.min_bin_count:
                df = df[df["n"] >= self.min_bin_count]
            ax.scatter(df["mean_x"], df["mean_y"], s=16, color="0.5", alpha=0.5, zorder=1,
                       label="binned means (sanity)")
        ax.set_xlabel(self.x.label)  # raw label — LOWESS is binning-free, so no "(qbin)" suffix
        ax.set_ylabel("Move time (s)" if self.y.is_log else self.y.label)
        if self.y.is_log:
            _seconds_from_log(ax.yaxis)
        if self.x.is_log:
            ax.set_xscale("log")

    def plot_lowess_tertile_segmented(self, ax, *, mass_values=(), frac: float = 0.3,
                                      n_boot: int = 60, grid_n: int = 120):
        """One LOWESS curve per fixed ply tertile (thin bands), colors/labels matching the
        binned tertile panel. Masses are isolated from each fit but not drawn (global feature)."""
        for t in (1, 2, 3):
            self.plot_lowess(ax, mass_values=mass_values, frac=frac, n_boot=n_boot,
                             grid_n=grid_n, color=PHASE_COLORS.get(t, MAIN_COLOR), tertile=t,
                             show_band=True, band_alpha=0.12, show_mass=False,
                             label_prefix=self._ply_tertile_legend_label(t) + ": ")
        ax.legend(fontsize=FONT_SIZE_TICKS, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)

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
        estimator: str = "binned",
        mass_values=(),
        lowess_frac: float = 0.3,
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

        fig, axes = plt.subplots(1, 2, figsize=(30, 13.72))
        if estimator == "lowess":
            # CONTINUOUS predictor: binning-free LOWESS + bootstrap band (left = global,
            # right = by ply tertile). Value masses isolated as their own points. Method
            # named in the title; legend minimal (mass point only, when present).
            self.plot_lowess(axes[0], mass_values=mass_values, frac=lowess_frac)
            if axes[0].get_legend_handles_labels()[1]:
                axes[0].legend(fontsize=FONT_SIZE_TICKS, loc="upper center",
                               bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)  # below the panel
            self.plot_lowess_tertile_segmented(axes[1], mass_values=mass_values, frac=lowess_frac)
        else:
            self.plot_quantile_bins(axes[0])
            self.plot_quantile_bins_tertile_segmented(axes[1])
        # Panel titles omitted — left = global, right = by ply tertile (implied by
        # the legend + the "(qbin)" x-axis).

        # Push the panels down so the suptitle clears them with a comfortable gap.
        fig.subplots_adjust(top=0.80)
        # Name the estimator in the title (not the legend): LOWESS for the smoother,
        # "quantile bins" only when it is actually qbin'd; native-integer panels say nothing.
        method = (" — LOWESS + 95% bootstrap band" if estimator == "lowess"
                  else " — quantile bins" if self.bin_mode == "ntile" else "")
        suptitle = fig.suptitle(f"{self.title}{method}\nn = {self.n_moves:,} moves",
                                fontsize=FONT_SIZE_LABEL + 10, y=1.0)

        # Crop tightly on save and explicitly include the below-axes legend +
        # suptitle as extra artists so they aren't clipped (bbox_inches='tight'
        # alone misses the legend placed outside the axes via bbox_to_anchor).
        extra_artists = [suptitle]
        for _ax in axes:  # capture every below-panel legend (both panels can have one now)
            _lg = _ax.get_legend()
            if _lg is not None:
                extra_artists.append(_lg)
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
