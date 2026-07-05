# Human response time — what board features predict it

**Ref:** `R-MOVETIME-BOARD` · [Index](reference.md)

## What board features predict how long humans think?

Working model-free (no engine), we ask which features *of the position itself* predict how long a
human thinks before moving — **89,218,280** non-zero-time moves from Lichess **10+0** games
(Elo ≥ 2000, no berserk, Oct–Dec 2023), windowed to **ply 15–75** (Russek et al.'s window: past the
memorized opening, before late-game noise). Legal captures/checks (`n_captures_avail`,
`n_checks_avail`) come from a python-chess enumeration over the run's **84,034,834** distinct FENs,
joined back to every move.

Every dashboard is the same **1×3 layout**: marginal histogram (left), overall trend (middle),
trend by ply tertile (right: Ply < 32 / 32–49 / > 49).

> **Result:** Response time is driven most by the **width of the decision** (legal moves,
> ρ ≈ +0.20) — more than material, clock, or checks/captures available. Legal moves, material
> imbalance, and checks available each bend in informative, non-monotonic ways once you look past
> the headline trend.

### How is response time distributed?

![RT distribution — histogram, normal P-P plot, RT vs ply](../figures/minply15_maxply75/board/png/rt_distribution.png)

Mean **6.5 s** (key: `board/rt/mean_s`), median **6.4 s** (key: `board/rt/median_s`) in log-space —
close together, the signature of a log-symmetric distribution. The middle panel is a **P-P plot**
(both axes 0–1: theoretical vs. empirical quantile probability under the fitted Normal) and hugs
y = x closely. In raw seconds the arithmetic mean is **11.4 s** vs. an exact median of **6.0 s** —
the long right tail of slow moves pulls the mean up, exactly as expected for a **log-normal**
distribution. RT vs. ply (whole game, unwindowed, right panel) rises from the opening, peaks around
ply 40–45 (**> 7.4 s**), then falls — this report's [15, 75] window sits on the rising-into-peak part
of that arc.

> **Result:** Response time is **log-normal** — players scale thinking multiplicatively with
> difficulty. Every trend panel plots RT on a log axis; correlations below are against **log(RT)**.

### How does remaining clock time affect response time?

![player clock vs response time](../figures/minply15_maxply75/board/png/bivariate_player_clock_time.png)

An inverted U: rises from ~7.1 s at low clock, peaks **8.4 s** near 300 s remaining, falls to ~3.3 s
near the full 600 s (Pearson r = **−0.135**, key: `board/corr/pearson/player_clock_time`). The
by-ply panel shows why: the three ply tertiles sit in different, mostly non-overlapping clock ranges
(early game = high clock, late game = low clock), and each tertile's *own* trend is close to
monotonically falling. The "hump" is mostly two effects glued together — fast opening moves that
happen to co-occur with a full clock, plus real time pressure as any tertile's own clock runs low.

> **Clarification:** The apparent inverted-U mixes a game-stage effect (not a clock effect) with
> genuine time pressure (present within every tertile) — not evidence that clock itself is
> non-monotonic once game stage is held fixed.

### How do legal moves affect response time — and why does the curve dip at 9–10?

![legal moves vs response time](../figures/minply15_maxply75/board/png/bivariate_n_possible_moves.png)

Legal moves is the strongest board-feature correlate of RT (r = **+0.198**, key:
`board/corr/pearson/n_possible_moves` — decision **width**, not depth, is what people spend time
on). But the trend isn't monotonic: RT rises from 1 move (~2.5 s) to a local peak around **7 moves**
(~5.2 s), **dips at 9–10** (~3.7–3.9 s), then resumes a long climb past 60 moves (~11 s) — in every
ply tertile.

**Is this the in-check effect?** In-check positions are drastically restricted (block/capture/
king-move only), so they cluster at low move counts. Excluding them entirely
(`n = 84,210,269`, 94.4% of the table):

![legal moves vs response time, excluding in-check positions](../figures/minply15_maxply75/board/png/supp_n_legal.png)

The dip is simply **gone** — a clean monotonic rise (6→3.33 s straight through 13→4.06 s) in every
tertile. In-check moves are slower than not-in-check moves at the same move count (a careful
decision even with few options, not a reflex), and in-check's population share collapses sharply
right around 8–10 legal moves, dragging the pooled mean down exactly there.

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/legalmoves9_incheck.svg" width="240"><br>in check (RT 9 s, ply 45)</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/legalmoves9_nocheck.svg" width="240"><br>not in check (RT 5 s, ply 55)</td>
</tr></table>

Both land at 9 legal moves for unrelated reasons — one a forced king-safety scramble, one a quiet
pawn endgame — and only the first inflates the pooled mean.

> **Clarification:** The dip at 9–10 is a **composition-shift artifact**: in-check carries a real RT
> premium at every move count, and its population share falls off a cliff right there — not
> evidence that ~9–10 legal moves are an easier decision than 7 or 11.

### How does own material affect response time?

![self material vs response time](../figures/minply15_maxply75/board/png/bivariate_self_material.png)

RT climbs from ~1.7 s near an empty board to a broad ~7–9 s plateau at self-material 25–39, but the
plateau is **jagged** — uneven spikes near 35/37/38/39.

**Why?** The weight function (P/N/B/R/Q = 1/3/3/5/9) collapses different piece losses onto the same
integer — N and B both weigh 3, so "lost a knight" and "lost a bishop" land on the same total. A
full untouched army is 8+2·3+2·3+2·5+9 = **39**, confirmed as one of the densest values (4.6M rows).
Parsing real FENs at each total:

| self-material | rows | mean RT | dominant composition |
|---:|---:|---:|---|
| 39 | 4,596,768 | 8.59 s | 100% full army, no losses |
| 38 | 7,548,663 | 10.01 s | 100% down exactly 1 pawn |
| 37 | 2,505,884 | 12.52 s | 100% down exactly 2 pawns |
| 35 | 8,226,020 | 11.34 s | 53% knight-for-pawn / 46% bishop-for-pawn |

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/selfmat39_full.svg" width="190"><br>39: full army</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/selfmat37_2pawns.svg" width="190"><br>37: 2 pawns down</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/selfmat35_knight_missing.svg" width="190"><br>35: knight for pawn</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/selfmat35_bishop_missing.svg" width="190"><br>35: bishop for pawn</td>
</tr></table>

35 and 38 are the two densest points in the covariate, for different reasons: 38 because "down
exactly one pawn" is simply a common game state; 35 because it pools two genuinely different
histories (knight lost vs. bishop lost) that happen to weigh the same. 37 (down two pawns, no piece
losses) is comparatively rare and sits between them as the local minimum.

> **Clarification:** The jaggedness is a genuine **population-mixture** effect, not sampling noise
> (every bin is multi-million-row) — different real board states collide onto the same weighted
> total at uneven rates, and the zigzag tracks that shifting composition, not a real causal
> non-monotonicity in material itself.

### How does material imbalance affect response time — direction, magnitude, and the staggered dips?

**Is being ahead the same as being behind?** No. `material_imbalance` is signed (self − opponent,
mover POV), and it's sharply asymmetric, not a mirror-image U:

![material imbalance (signed) vs response time](../figures/minply15_maxply75/board/png/bivariate_material_imbalance.png)

| imbalance | −15 | −10 | −5 | −3 | −1 | 0 | +1 | +3 | +5 | +10 | +15 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo. mean RT | 2.78 s | 2.95 s | 4.16 s | 4.45 s | 6.64 s | 6.75 s | **8.02 s** | 7.37 s | 6.70 s | 5.04 s | 3.80 s |

The curve peaks just past parity, at **+1**, not at 0 — being slightly ahead is the single most
deliberated state. The two arms are wildly asymmetric: at the same magnitude, being ahead is always
slower than being behind (+5 → 6.70 s vs. −5 → 4.16 s). And there's a local dip exactly at −3, the
same |imbalance| = 3 dip the absolute-value curve shows below — here it's visibly the "behind" side
driving it.

> **Result:** Material imbalance is fundamentally about **who is ahead**, not just "how big is the
> gap." A mover behind plays fast (urgency, more forcing continuations); a mover ahead by the same
> amount deliberates much longer (converting an advantage carefully) — and the very top of the curve
> is a *small* advantage, not a large one.

**How much does the magnitude alone matter?** `abs_material_imbalance` answers "how decided is this
position" instead of "who's ahead" — and has its own wrinkle: staggered dips, not a smooth decline.

![material imbalance (absolute) vs response time](../figures/minply15_maxply75/board/png/bivariate_abs_material_imbalance.png)

RT falls as the gap grows — more decided, faster — but not smoothly: sharp dips around
|imbalance| ≈ 3 and ≈ 9–10, sharpest in the **Ply < 32** tertile. These aren't small-n noise
(imbalance = 3 has 1.7M rows in that tertile alone) — they're the same ahead/behind asymmetry from
above, now visible bin-by-bin:

| \|imbalance\| | mover behind (n, RT) | mover ahead (n, RT) | ratio |
|---:|---|---|---:|
| 2 (no dip) | 816,282, 9.08 s | 456,791, 13.19 s | 1.8:1 |
| **3** (dip) | 1,504,604, **5.39 s** | 210,058, 12.64 s | **7.2:1** |
| **9** (dip) | 80,505, **3.81 s** | 3,476, 9.92 s | **23.2:1** |

At both dips the population is overwhelmingly the disadvantaged mover, who plays much faster than
the advantaged mover at the same magnitude — the pooled mean is pulled down by an increasingly
lopsided mix, and the skew (1.8:1 → 7.2:1 → 23.2:1) tracks the dip depth. These are also short-lived
states: imbalance = 0 lingers ~12 moves/game on average; imbalance = 3 only ~1.65; imbalance = 9
essentially once — consistent with transient, quickly-resolved swings rather than settled gaps.

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_m3_behind.svg" width="190"><br>|3|, behind</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_p3_ahead.svg" width="190"><br>|3|, ahead</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_m9_behind.svg" width="190"><br>|9|, behind</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_p9_ahead.svg" width="190"><br>|9|, ahead</td>
</tr></table>

> **Clarification:** The staggered dips are confounded by an ahead/behind composition shift, not
> noise — the disadvantaged mover dominates the population at exactly these imbalance levels and
> plays much faster, which is what pulls the pooled mean down. Why "behind" outnumbers "ahead" this
> sharply at these specific levels is consistent with short-lived, quickly-resolved states, but the
> exact game-level story (blunder vs. voluntary sacrifice) isn't traced here.

### How does the number of available checks affect response time — and is the inverted-U real?

![checks available vs response time](../figures/minply15_maxply75/board/png/bivariate_n_checks_avail.png)

RT rises from 0 checks available (**5.9 s**, 54.5% of rows) to a peak at **4** (**8.2 s**), then
falls back through 8–9 (**~7 s**) — in the overall trend and every ply tertile. This shape was
adversarially re-tested on three fronts.

**Is the feature even computed correctly?** `n_checks_avail` uses `chess.Board.legal_moves`, which
should already exclude any move that leaves the mover's own king in check. Verified empirically, not
just trusted: re-enumerating all 995 legal moves from 280 real in-check FENs in this dataset, zero
violations. Separately: can an in-check mover's escape *also* deliver check (discovered check while
capturing/blocking)? Yes, legitimately — it happens in 1.33% of in-check rows, e.g. **Rxd2+**
(capturing the checking bishop while itself giving check).

> **Result:** Not a bug. The feature is counting exactly what it should.

**Is this a disguised endgame/game-phase artifact?** Checks-available correlates modestly with ply
(+0.234) and negatively with total pieces on the board (−0.258). Fine ply-bin stratification shows
the curve's *rise* (0→peak) is ply-invariant from ply 25 onward, but its *fall* (peak→9) genuinely
strengthens with ply — absent at ply 15–25, growing to ~15–19% of peak RT by ply 45+. The sharper
test — restricting to genuinely simplified endgames (≤10 total pieces) vs. piece-rich positions
(≥26 pieces) — gives the opposite of what "trivial endgame" predicts: the fall is already fully
present across 76% of the data (≥20 pieces) and comparably sized in true sparse endgames; it's
*only absent* in the piece-richest, most opening-like slice.

> **Clarification:** Game phase modulates the inverted-U's **falling half only** (not its rise, not
> its peak) — but not in the "trivial endgame" direction; the fall is present across most of the
> dataset and only vanishes in the piece-richest positions.

**Is the fall at the sparse high end just small-n noise?** No — bootstrap 95% CIs at every integer
bin from the peak down through the merged 10+ tail are all non-overlapping (8.25 s at k=4 down to
6.77 s at k≥10, cleanly separated at every step), and an independent kernel-regression estimator
with no hard bins reproduces the same shape.

> **Overall verdict:** The inverted-U is a **genuine residual effect** — not a bug, not noise — that
> survives conditioning on legal moves, self-material, and total pieces (banded or as a subset). Its
> falling half specifically is phase-modulated; the rise and peak location are not. What produces
> the *rise* itself remains an open question.

#### Why does material context flip the direction of the checks-avail effect?

Splitting checks-available by the mover's signed material imbalance reveals the real driver of the
fall. As `n_checks_avail` rises, positions increasingly leave material parity (53.5% even at k=0 →
11.9% at k≥10) and lean toward "ahead" on net — but "behind" doesn't disappear from the high end
either (correlation with |imbalance| is a moderate +0.153: checks-available tracks *decisiveness*,
not direction).

![checks available × material imbalance — interaction heatmap](../figures/minply15_maxply75/board/png/checks_material_interaction_heatmap.png)

![checks available vs RT, by material band](../figures/minply15_maxply75/board/png/checks_material_band_curves.png)

Splitting the curve into five material bands:

| band | k=0 | k=4 | k=9 | k=10+ | shape |
|---|---:|---:|---:|---:|---|
| ≤−5 (heavily behind) | 3.18 s | 5.88 s | 6.64 s | **7.26 s** | rises, never falls |
| 0 (even) | 6.28 s | **9.22 s** | 7.98 s | 8.20 s | cleanest inverted-U |
| ≥+5 (heavily ahead) | 5.68 s | 6.36 s | 5.04 s | **4.76 s** | peaks at k=1, falls steadily |

Bootstrap-confirmed real at both extremes, and replicated within a single fixed ply band (so it
isn't a disguised phase effect). The classic inverted-U lives specifically in the **material-even**
band; ahead and behind positions each pull it in a different direction.

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/checks_mech_behind_k8.svg" width="200"><br>k=8, crushed materially<br>RT 14 s</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/checks_mech_ahead_k9.svg" width="200"><br>k=9, comfortably ahead<br>RT 2 s</td>
</tr></table>

Why would more checks mean *less* time when ahead? A first guess — satisficing, stop searching once
an option looks good enough — doesn't fit: satisficing predicts a plateau once 2–3 acceptable
options exist, not the smooth decline actually seen all the way to k=10+. What *does* vary smoothly
alongside k in the heavily-ahead band is total pieces on the board — falling continuously with no
plateau either (19.8 pieces at k=0 down to 15.0 at k=20). That points to a better-supported reading:
when ahead, k is less a count of options to weigh and more a **continuous proxy for how far a
winning conversion has progressed** — the board keeps opening up (more reach for the attacker,
usually the queen), and RT tracks that continuing simplification.

> **Result:** Material context changes the **shape**, not just the level, of the checks-vs-RT curve.
> Ahead: fast, mechanical continuations of an already-winning position, tracking continued
> board-opening rather than option-counting. Behind: RT keeps rising with no turnover — genuinely
> hard, must-find-a-try decisions (though we didn't verify this against engine-optimal "only
> moves"). The residual inverted-U from the verdict above lives specifically in the material-even
> middle.

### How does the number of available captures affect response time?

![captures available vs response time](../figures/minply15_maxply75/board/png/bivariate_n_captures_avail.png)

RT rises monotonically from 5.1 s at zero captures available to a plateau around 9.2–9.7 s at 9–13
(r = **+0.131**, key: `board/corr/pearson/n_captures_avail`) — no inverted-U, unlike checks. The
reason is structural: captures-available correlates *positively* with total pieces on the board
(+0.248 — more captures when the board is busier/denser), the opposite sign from checks-available
(−0.258 — more checks when the board has simplified/opened up), and it's essentially uncorrelated
with |material imbalance| (≈ 0.00, vs. checks' +0.153). Checks track how far a winning conversion
has progressed; captures just track how tactically busy a position is, which has no special
relationship to who's ahead.

> **Result:** Captures available is a **clean, monotonically-increasing** correlate of RT — more
> capture options always means more to weigh, with none of the material-dependent reversal seen for
> checks.

## Methods

- **Dataset:** Lichess 10+0, Elo ≥ 2000, no berserk, Oct–Dec 2023 (shared `personal.db`), windowed
  to ply 15–75.
- **Pipeline:** `filter_and_sample.py` builds the windowed table (89,218,280 rows) →
  `board.py --featurize`/`--merge` enumerates legal captures/checks over 84,034,834 distinct FENs
  (python-chess, Slurm array) → `board.py --plot` runs the RT-distribution summary, per-covariate
  bivariate dashboards, correlation matrix, and feature histograms.
- **Binning:** every dashboard is a fixed 1×3 (histogram / overall / by-ply-tertile). Continuous
  covariates use quantile bins; discrete/integer covariates (legal moves, material, checks/captures
  available) use one bin per integer value, sparse tail merged, `min_bin_count=300`.
  `in_check`/`prev_move_was_capture` are not tracked as standalone features; the in-check
  decomposition above uses one-off diagnostic queries, not regenerated dashboards.
- **Material imbalance** is tracked both signed (`material_imbalance`, self − opponent, mover POV)
  and as magnitude (`abs_material_imbalance`) — two separate dashboards.
- **Correlations** quoted are Pearson r between `ln(move_time)` and the raw covariate, computed in
  SQL over a 1M-row sample (not the Spearman board matrix, a separate artifact).
- **FEN examples** are rendered with `chess.svg.board` (raw SVG; `cairosvg`/`svglib` aren't
  installed in this environment).
- **Numbers this report cites that board.py computes** (RT mean/median, each Pearson r) are
  registered via `analysis.utils.pgfvals` to `outputs/reports/board_stats.tex` as
  `\pgfkeyssetvalue{key}{value}` — the `key: ...` next to a number traces it to the code that
  produced it. Written only by a real (non-`--smoke`) run.
