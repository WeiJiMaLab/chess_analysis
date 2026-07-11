from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.ticker import FuncFormatter


def _seconds_from_log(axis) -> None:
    """Relabel a log-valued axis to show ``round(exp(v), 1)`` — i.e. seconds."""
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{np.round(np.exp(v), 1)}"))

from analysis.utils.helpers import (
    FONT_SIZE_LABEL,
    LEGEND_FONTSIZE,
    MAIN_COLOR,
    PHASE_COLORS,
)
from analysis.utils.plots import (
    plot_qbin_stats,
    _wrap_long_label,
)


def _infer_tertile_cuts(conn, source: str, column: str = "move_ply") -> tuple[int, int]:
    """The (1/3, 2/3) tertile cutpoints of ``column`` over the whole ``source`` table.
    Default ``column='move_ply'`` gives the ply tertiles; pass another column (e.g.
    ``'gss'``) to segment the same plot by a different variable instead."""
    cut_lo, cut_hi = conn.execute(
        f"SELECT quantile_disc({column}, 1.0/3), quantile_disc({column}, 2.0/3) FROM {source}"
    ).fetchone()
    if cut_lo is None or cut_hi is None:
        return 0, 0
    # Integer columns (move_ply) -> int cutpoints (clean "< n" labels); float
    # columns (e.g. game_fraction) keep their float cutpoints.
    if isinstance(cut_lo, int) or (isinstance(cut_lo, float) and cut_lo.is_integer() and float(cut_hi).is_integer()):
        return int(cut_lo), int(cut_hi)
    return cut_lo, cut_hi


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

    Segmentation by **ply tertiles**: the two ``move_ply`` cutpoints are inferred
    from ``ply_tertile_source`` (``quantile_disc`` at 1/3, 2/3) and applied as fixed
    boundaries, so the segmentation is comparable across plots. ``ply_tertile_source``
    defaults to the analysis ``table_name`` (pass a broader table to share cutpoints
    across several filtered analyses). ``quantile_tertile_df`` holds per-tertile
    aggregates with the qbin ``ntile`` recomputed **within** each fixed tertile;
    legend labels are the fixed ply ranges (``self.ply_cuts``).
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
        zero_inflated: bool = False,
        zero_threshold: float = 0.0,
        ply_tertile_source: str | None = None,
        segment_column: str = "move_ply",
        segment_source: str | None = None,
        segment_label: str = "ply",
        bin_mode: str = "ntile",
        integer_tail_cut: float | None = None,
        integer_floor_cut: float | None = None,
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
            ``integer_floor_cut`` is the mirror-image merge for the LOW end (e.g. a
            signed variable's sparse negative tail) — set both for a symmetric
            two-sided merge. Applies to BOTH the global and the by-ply-tertile panels.

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
        self.ply_tertile_source = ply_tertile_source or table_name
        # The by-segment panel splits on ``segment_column`` (default ``move_ply`` → ply
        # tertiles); ``segment_source`` is the table its cutpoints come from (default =
        # ``ply_tertile_source``), and ``segment_label`` is the legend prefix. Pass e.g.
        # (segment_column='gss', segment_source='mq_rt', segment_label='GSS') to get the
        # *same* MQ-vs-RT plot segmented by GSS stratum instead of ply.
        self.segment_column = segment_column
        self.segment_source = segment_source or self.ply_tertile_source
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
        self.integer_floor_cut = None if integer_floor_cut is None else float(integer_floor_cut)
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
        # quantile_cont/abs reject BOOLEAN (e.g. in_check, prev_move_was_capture);
        # cast to DOUBLE wherever x is used numerically (no-op for numeric columns).
        col_num = f"{col}::DOUBLE"
        sel = f"{partition_by} as tertile_id, " if partition_by else ""
        part_clause = f"PARTITION BY {partition_by}" if partition_by else ""

        # --- integer mode: fixed-width integer groups, optional tail merge -----
        if self.bin_mode == "integer":
            w = self.integer_bin_width
            # Group index = floor(x / w); width 1 ⇒ one point per integer. Points are
            # plotted at the group mean x (avg(col)), so the bin width need not show.
            grp = f"CAST(floor({col} / {w}) AS BIGINT)" if w != 1 else f"CAST({col} AS BIGINT)"
            qbin = grp
            if self.integer_tail_cut is not None:
                # At/above cut: a single merged point (sentinel qbin sorts rightmost;
                # plot orders by mean_x anyway).
                qbin = f"CASE WHEN {col} >= {self.integer_tail_cut} THEN 1000000000 ELSE ({qbin}) END"
            if self.integer_floor_cut is not None:
                # At/below floor: the mirror-image merged point (sentinel sorts leftmost).
                qbin = f"CASE WHEN {col} <= {self.integer_floor_cut} THEN -1000000000 ELSE ({qbin}) END"
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
        zero_pred = f"abs({col_num}) <= {thr}" if has_zero else "FALSE"
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
                f"quantile_cont({col_num}, {i / self.n_bins})" for i in range(1, self.n_bins)
            )
            edges_subq = (
                f"(SELECT [{edges}] FROM _analyzer_view i2 WHERE {interior_pred}"
                + (f" AND i2.{partition_by} = i.{partition_by}" if partition_by else "")
                + ")"
            )
            interior = (
                f"SELECT {sel}{col}, _y_transformed, "
                f"1 + len(list_filter({edges_subq}, e -> {col_num} > e)) AS qbin "
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
        # Cast to DOUBLE before isnan/isinf: they reject BOOLEAN (e.g. in_check,
        # prev_move_was_capture), and DOUBLE is a safe supertype for every column
        # kind this pipeline passes here (bool/int/float).
        finite_cond = (
            f"({self.x.column} IS NOT NULL AND NOT isnan({self.x.column}::DOUBLE) AND NOT isinf({self.x.column}::DOUBLE)) AND "
            f"({self.y.column} IS NOT NULL AND NOT isnan({self.y.column}::DOUBLE) AND NOT isinf({self.y.column}::DOUBLE))"
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

        # avg() rejects BOOLEAN (e.g. in_check, prev_move_was_capture) — cast to
        # DOUBLE, a no-op for the numeric columns that already worked.
        global_inner = self._bin_assignment_sql(partition_by=None)
        self.quantile_df = self.conn.execute(f"""
            SELECT qbin,
                   avg({col}::DOUBLE) as mean_x,
                   avg(_y_transformed) as mean_y,
                   stddev(_y_transformed) as std_y,
                   count(*) as n
            FROM ({global_inner})
            GROUP BY qbin
        """).df()

        tertile_inner = self._bin_assignment_sql(partition_by="_ply_tertile")
        self.quantile_tertile_df = self.conn.execute(f"""
            SELECT tertile_id, qbin,
                   avg({col}::DOUBLE) as mean_x,
                   avg(_y_transformed) as mean_y,
                   stddev(_y_transformed) as std_y,
                   count(*) as n
            FROM ({tertile_inner})
            GROUP BY tertile_id, qbin
            ORDER BY tertile_id, qbin
        """).df()

        # Flag the dedicated zero/edge-mass point(s) (sentinel qbins — see
        # ``_bin_assignment_sql``) so plotting can render them as a ✕ marker
        # instead of folding them into the interior circle-marker trend line.
        self.quantile_df["is_mass"] = self.quantile_df["qbin"].apply(self._is_mass_qbin)
        self.quantile_tertile_df["is_mass"] = self.quantile_tertile_df["qbin"].apply(self._is_mass_qbin)

    def _is_mass_qbin(self, qbin: int) -> bool:
        """True iff ``qbin`` is the sentinel for a dedicated zero/edge point-mass
        (0 for the zero_inflated lump, > n_bins for an edge_mass clause — see
        ``_bin_assignment_sql``). NOT true for the integer-mode merged tail point
        (that's just a sparse bin, not a qualitatively different kind of point)."""
        if self.bin_mode == "integer":
            return False
        if self.zero_inflated and qbin == 0:
            return True
        if self.edge_mass and qbin > self.n_bins:
            return True
        return False

    def _ply_tertile_legend_label(self, tertile_id: int) -> str:
        """Fixed legend label from the a-priori (whole-dataset) tertile cutpoints, prefixed
        with ``segment_label`` (e.g. 'ply < 28' for ply, 'GSS 2–31' for a GSS segmentation)."""
        c1, c2 = self.ply_cuts
        lab = self.segment_label
        if isinstance(c1, int):  # integer segment (e.g. ply): clean "< n" boundaries
            ranges = {1: f"{lab} < {c1 + 1}", 2: f"{lab} {c1 + 1}–{c2}", 3: f"{lab} > {c2}"}
        else:                    # float segment (e.g. game fraction)
            ranges = {1: f"{lab} ≤ {c1:.2f}", 2: f"{lab} {c1:.2f}–{c2:.2f}", 3: f"{lab} > {c2:.2f}"}
        return ranges.get(tertile_id, f"{lab} tertile {tertile_id}")

    @property
    def _x_axis_label(self) -> str:
        """X label; only suffix "(qbin)" when x is binned by plain equal-count ntile.
        Integer / tie-safe modes are not qbins and get the bare label. A
        zero_inflated/edge_mass axis gets an explicit "(bin; ... isolated)" suffix
        instead, so the reader knows one of the plotted points is a dedicated
        hurdle/mass point (rendered with a ✕ marker), not an interior bin."""
        plain_qbin = (
            self.bin_mode == "ntile"
            and not self.zero_inflated
            and not self.tie_safe
            and not self.edge_mass
        )
        if plain_qbin:
            return f"{self.x.label} (qbin)"
        if self.zero_inflated or self.edge_mass:
            parts = []
            if self.zero_inflated:
                thr = self.zero_threshold
                parts.append("0 isolated" if thr == 0.0 else f"|x|≤{thr:g} isolated")
            if self.edge_mass:
                parts.append("edge mass isolated")
            # Newline before the qualifier: at FONT_SIZE_LABEL (52pt) the full
            # single-line string is wider than a 1x3 dashboard panel and bleeds
            # into the neighboring panel's own x-label.
            return f"{self.x.label}\n(bin; {', '.join(parts)})"
        return _wrap_long_label(self.x.label)

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
        y_label = "Response time (s)" if self.y.is_log else self.y.label
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
        # Docked to the RIGHT of the panel (not stacked below): a below-axes legend
        # needed extra vertical push for a two-line x_label (zero_inflated/edge_mass
        # qualifier, see _x_axis_label) to avoid colliding with it — a right-side
        # legend has no such interaction with the x-label at all.
        ax.legend(
            fontsize=LEGEND_FONTSIZE,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
            handlelength=1.2,
        )

    def plot_quantile_bins(self, ax):
        """Plots the trend across equal-sized quantile bins."""
        y_label = "Response time (s)" if self.y.is_log else self.y.label

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
