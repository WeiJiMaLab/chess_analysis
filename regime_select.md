# REGIME — confirming the set of meaningful cost regimes

Record of the `REGIME` node in `plan.md`'s (now-closed) diagnosis-phase DAG. The agent that ran
this grid was lost mid-session (background task no longer trackable) before it could write this
up — the grid itself completed successfully and the raw JSON + plot survived, so this file
reconstructs the write-up directly from that data rather than re-running anything.

## Method

Auto-criterion, not hand-picked values (per direct instruction): a `(time_lambda,
maintenance_scale)` point counts as **meaningful** iff SingleHalt*'s own fitted optimal stop
step `k*` (`fit_singlehalt_stop`, pure `argmin` over `_regret_at` — no model fitting, cheap) lands
strictly between 1 and the 96-expansion ceiling. This excludes both a `k*→0` collapse
("always stop immediately" — SingleHalt* degenerates, nothing left to learn) and a `k*→ceiling`
pileup (right-censoring against the expansion budget). Grid: 22 `time_lambda` values
(`5e-05` … `0.3`, log-spaced) × 16 `maintenance_scale` values (`0.0` … `0.3`) = 352 points, run
on the `puctvalue_md36` validation split. Only the eval-side computation was swept — no new tree
generation (T3's tree-generation grid is separately closed and untouched).

## Result: 102/352 points (29%) are meaningful, forming a clean, monotonically-shrinking band

**By `maintenance_scale`, independent of `time_lambda`:** meaningful `maintenance_scale` values
are exactly `{0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005}` — **nothing at
`maintenance_scale ≥ 0.01` is ever meaningful, at any `time_lambda` tested.** This directly and
fully explains the collapse observed earlier in the investigation at `maintenance_scale∈{0.05,
0.1}` (`Z`'s open item, `SIG-Z`'s independent confirmation): those values are 10-20x above the
highest meaningful ceiling in the entire grid. Not a bug — those settings are just outside the
region where SingleHalt* itself has anything left to solve.

**The meaningful maintenance ceiling shrinks as `time_lambda` grows** (larger time cost already
eats into the "budget" before maintenance cost pushes `k*` the rest of the way to 0) — a clean,
monotonic trapezoid, not a scattered/noisy region:

| `time_lambda` | meaningful `maintenance_scale` range | # meaningful points |
|---|---|---|
| 5e-05 | [0.0005, 0.005] (m=0 itself is NOT meaningful here — `k*=95`, right-censored) | 7 |
| 0.0001 – 0.003 | [0, 0.005] (full ceiling) | 8 each |
| 0.005 – 0.007 | [0, 0.003] | 7 each |
| 0.01 | [0, 0.0025] | 6 |
| 0.011 – 0.013 | [0, 0.0015–0.002] | 4-5 |
| 0.014 – 0.017 | [0, 0.0005–0.001] | 2-3 |
| ≥ 0.02 | **none** — even `m=0` collapses (`k*=0`) | 0 |

**At `maintenance_scale=0` specifically** (the single regime validated everywhere else in this
investigation so far): 16 of 22 `time_lambda` values are meaningful, spanning `[0.0001, 0.017]`.
`time_lambda=0.01` (the existing shipped default) sits comfortably mid-band (`k*=10`, `k_frac
=0.105`) — **the currently-used default is well inside the meaningful region, not a lucky edge
case.** Below `0.0001` and above `0.017`, `m=0` degenerates (`k*→95` or `k*→0` respectively).

## Confirmed regime set for downstream use

Any of the 102 `(time_lambda, maintenance_scale)` pairs above are valid to evaluate at. For
representative coverage without exhaustively re-running everything at all 102: the existing
default (`λ=0.01, m=0`) plus at least one genuinely nonzero-maintenance point per `λ` band, e.g.
`λ=0.01, m=0.001` (mid-band, `k*=6`) and `λ=0.001, m=0.0005` (small-`λ`/small-`m` corner,
`k*` — see table above) — this matches the 3 points `our_trees_continued.md`'s sub-line B eval
already used, which is good independent validation that those picks land inside the confirmed
region.

Figure: `outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/regime_select.png` (heatmap,
cell text = `k*`, hatched = not meaningful — matches this table). Raw grid: `outputs/figures/
minply15_maxply75/diagnosis/regime_select_results.json` (352 rows, `{time_lambda,
maintenance_scale, k_star, ceiling, k_frac, meaningful}`).

## Bottom line

The "beats FixedHalt* under any cost regime" goal has a well-characterized, non-trivial target
region to test against — not a single point, and not a vague "somewhere out there." It's a
specific, small trapezoid: `maintenance_scale` from 0 up to (at most) 0.005, with the exact
ceiling depending on `time_lambda`, and `time_lambda` itself bounded to roughly `[0.0001,
0.017]` at `m=0`. Anything swept outside this region (as the earlier `m∈{0.05,0.1}` sweep did)
will mechanically collapse regardless of which controller is being tested — that's a property of
this population under `SingleHalt*`, not a signal about any learned method's quality.
