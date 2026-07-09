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

### Player Clock

![player clock vs response time](../figures/minply15_maxply75/board/png/bivariate_player_clock_time.png)

Pearson r = **−0.135** (key: `board/corr/pearson/player_clock_time`) against log(RT).

#### Nonmonotonicity

An inverted U: rises from ~7.1 s at low clock, peaks **8.4 s** near 300 s remaining, falls to ~3.3 s
near the full 600 s. The by-ply panel shows why: the three ply tertiles sit in different, mostly
non-overlapping clock ranges (early game = high clock, late game = low clock), and each tertile's
*own* trend is close to monotonically falling. The "hump" is mostly two effects glued together —
fast opening moves that happen to co-occur with a full clock, plus real time pressure as any
tertile's own clock runs low.

> **Clarification:** The apparent inverted-U mixes a game-stage effect (not a clock effect) with
> genuine time pressure (present within every tertile) — not evidence that clock itself is
> non-monotonic once game stage is held fixed.

### Legal Moves

![legal moves vs response time](../figures/minply15_maxply75/board/png/bivariate_n_possible_moves.png)

Legal moves is the strongest board-feature correlate of RT (r = **+0.198**, key:
`board/corr/pearson/n_possible_moves` — decision **width**, not depth, is what people spend time
on).

#### Nonmonotonicity

The trend isn't monotonic: RT rises from 1 move (~2.5 s) to a local peak around **7 moves**
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

### Own Material

![self material vs response time](../figures/minply15_maxply75/board/png/bivariate_self_material.png)

RT climbs from ~1.7 s near an empty board to a broad ~7–9 s plateau at self-material 25–39.

#### Nonmonotonicity

The plateau is **jagged** — uneven spikes near 35/37/38/39.

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
> non-monotonicity in material itself. (The evidence here is the composition table above, not a
> regenerated filtered dashboard like `supp_n_legal` below — the underlying finding is solid, but a
> split "knight-missing vs. bishop-missing" recovery plot would make it visual rather than tabular.)

### Material Imbalance (Signed)

`material_imbalance` is signed (self − opponent, mover POV) — is being ahead the same as being
behind?

![material imbalance (signed) vs response time](../figures/minply15_maxply75/board/png/bivariate_material_imbalance.png)

| imbalance | −15 | −10 | −5 | −3 | −1 | 0 | +1 | +3 | +5 | +10 | +15 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo. mean RT | 2.78 s | 2.95 s | 4.16 s | 4.45 s | 6.64 s | 6.75 s | **8.02 s** | 7.37 s | 6.70 s | 5.04 s | 3.80 s |

No — sharply asymmetric, not a mirror-image U. The curve peaks just past parity, at **+1**, not at
0 — being slightly ahead is the single most deliberated state. The two arms are wildly asymmetric:
at the same magnitude, being ahead is always slower than being behind (+5 → 6.70 s vs. −5 → 4.16 s).

> **Result:** Material imbalance is fundamentally about **who is ahead**, not just "how big is the
> gap." A mover behind plays fast (urgency, more forcing continuations); a mover ahead by the same
> amount deliberates much longer (converting an advantage carefully) — and the very top of the curve
> is a *small* advantage, not a large one.

#### Nonmonotonicity

There are local dips exactly at −3 and −9 (4.45 s and 2.82 s vs. smooth neighbors on both sides) —
but **only on the negative (behind) side**: +3 and +9 decline perfectly smoothly. Same mechanism as
the absolute-value curve below (`prev_move_was_capture`), viewed from one side: a clean single-piece
capture nets −3/−9 for whoever just got captured from and +3/+9 for whoever just captured, but the
dip only shows up where the disadvantaged mover's population share is much larger (7.2:1/23.2:1
behind:ahead at |3|/|9|, see below) — enough to pull the pooled mean down on that side, not enough
on the other. Excluding recaptures (`supp_material_imbalance`, same filter, n = 66,770,171):

![material imbalance (signed, excluding recapture) vs response time](../figures/minply15_maxply75/board/png/supp_material_imbalance.png)

Both dips vanish — a smooth peak at +1 with monotonic decline on either side (e.g. −4→−3→−2 runs
6.85→7.18→7.64 s, no kink).

> **Clarification:** The −3/−9 dips are the same **recent-capture composition-shift artifact** as
> the absolute-value curve, asymmetric here only because the disadvantaged side's population share
> is what's large enough to show it.

### Material Imbalance (Absolute)

`abs_material_imbalance` answers "how decided is this position" instead of "who's ahead":

![material imbalance (absolute) vs response time](../figures/minply15_maxply75/board/png/bivariate_abs_material_imbalance.png)

RT falls as the gap grows — more decided, faster.

#### Nonmonotonicity

Not smoothly: sharp dips around |imbalance| ≈ 3 and ≈ 9–10, sharpest in the **Ply < 32** tertile, and
not small-n noise (imbalance = 3 has 1.7M rows in that tertile alone). The dips look at first like
the ahead/behind asymmetry above — the disadvantaged mover dominates the population at exactly these
levels and plays much faster — but restricting to the disadvantaged side alone doesn't remove the
dip. A second guess — excluding positions reachable via a one-sided "hang" (one side still has every
piece) — doesn't remove it either. Both are correlates, not the cause.

**The actual driver: `prev_move_was_capture`** — whether the position was reached immediately after
an opponent's capture. Its population share spikes locally exactly at 3 and 9 (**49.6%**/**74.1%**
vs. 30–42%/58–66% at their neighbors), and post-capture positions are dramatically faster at the
same magnitude (geo. mean RT 3.63 s vs. 7.50 s at |3|; 2.61 s vs. 5.45 s at |9|, both over millions
of rows). Excluding recaptures entirely — `supp_abs_material_imbalance`
(filter `NOT prev_move_was_capture`, n = 66,770,171, 74.8% of the table), same recipe as
`supp_n_legal` below — makes the dip and trough vanish completely:

![material imbalance (absolute, excluding recapture) vs response time](../figures/minply15_maxply75/board/png/supp_abs_material_imbalance.png)

A clean monotonic decline from |1| onward, no wrinkle at 3 or 9.

**Why exactly 3 and 9?** A material difference is reachable via a single clean piece-for-piece
capture — nothing else traded — only when it equals one piece's raw value: pawn = 1, minor = 3,
rook = 5, queen = 9. The share of positions reachable via exactly one piece-category difference
jumps sharply and specifically at 1/3/5/9 (**61.9%** at |3| vs. 2.2% at |4|; **40.3%** at |9| vs.
3.9% at |10|). The same mechanism operates at |1| and |5| too, just harder to see: |1| is swamped by
the curve's own early rise, and |5|'s dip is real but proportionally smaller.

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_m3_behind.svg" width="190"><br>|3|, behind</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_p3_ahead.svg" width="190"><br>|3|, ahead</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_m9_behind.svg" width="190"><br>|9|, behind</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/imb_p9_ahead.svg" width="190"><br>|9|, ahead</td>
</tr></table>

**Why are recapture positions faster, not just more forced?** Total legal moves is barely different
at |3| (31.3 vs. 31.2) and only modestly narrower at |9| (22.9 vs. 29.2) — not enough to explain a
~2× RT gap — and captures still available actually run slightly *higher* after a capture, not lower.
The better-supported reading: recapturing continues an exchange the mover was already tracking while
calculating their *previous* move, so much of the search already happened a half-move earlier —
consistent with classic chunking/anticipation findings in chess cognition, though not directly
measurable from this dataset.

> **Clarification:** The staggered dips are a **recent-capture composition-shift artifact**, not an
> ahead/behind effect. |3| and |9| are singled out because they're the only magnitudes reachable by
> a single clean piece-for-piece capture; recapture positions are fast most plausibly because the
> decision was substantially pre-computed a half-move earlier, not because they're objectively
> narrower.

### Checks Available

![checks available vs response time](../figures/minply15_maxply75/board/png/bivariate_n_checks_avail.png)

RT rises from 0 checks available (**5.9 s**, 54.5% of rows) to a peak at **4** (**8.2 s**), then
falls back through 8–9 (**~7 s**) — in the overall trend and every ply tertile. The feature itself
checks out (995 re-enumerated legal moves from 280 in-check FENs, zero violations), and the shape
isn't a disguised game-phase artifact — it's not explained by ply, legal moves, or total pieces
alone, and bootstrap 95% CIs are non-overlapping at every bin past the peak.

#### Nonmonotonicity

**The inverted-U is really two different populations pulling in opposite directions, and material
imbalance is what separates them.** Splitting by the mover's material imbalance:

![checks available × material imbalance — interaction heatmap](../figures/minply15_maxply75/board/png/checks_material_interaction_heatmap.png)

![checks available vs RT, by material band](../figures/minply15_maxply75/board/png/checks_material_band_curves.png)

| band | k=0 | k=4 | k=9 | k=10+ | shape |
|---|---:|---:|---:|---:|---|
| ≤−5 (heavily behind) | 3.18 s | 5.88 s | 6.64 s | **7.26 s** | rises, never falls |
| 0 (even) | 6.28 s | **9.22 s** | 7.98 s | 8.20 s | cleanest inverted-U |
| ≥+5 (heavily ahead) | 5.68 s | 6.36 s | 5.04 s | **4.76 s** | peaks at k=1, falls steadily |

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/checks_mech_behind_k8.svg" width="200"><br>k=8, crushed materially<br>RT 14 s</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/checks_mech_ahead_k9.svg" width="200"><br>k=9, comfortably ahead<br>RT 2 s</td>
</tr></table>

Bootstrap-confirmed at both extremes, and replicated within a single fixed ply band. The pooled
inverted-U is a mixture: the **behind** band never turns over (more checks = a harder search for a
try), the **ahead** band falls almost from the start (more checks = an easier cruise to a win), and
the **even** band — the bulk of the data — shows the cleanest version of the inverted-U itself.
Population composition alone explains a large share of the pooled decline: the "≥+5" share of rows
rises from **0.95%** (k=0) to **23.5%** (k=12) while "≤−5" barely moves (4% → ~6%); holding each
band's own curve fixed at its k=4 shape, that compositional shift alone accounts for **~40%** of the
pooled decline right after the peak, rising to **~80%** by the far tail.

**The remaining within-band decline (both even and ahead) is gated by board thinning, not caused by
material imbalance directly.** Splitting the material-*even* band itself into total-pieces-on-board
tertiles isolates this cleanly, since material is held fixed here by construction:

![checks available vs RT, by total-pieces tertile (material-even band)](../figures/minply15_maxply75/board/png/checks_pieces_tier_curves.png)

The dense tertile rises **monotonically** through 10+, structurally identical to
`n_captures_avail`'s own well-behaved curve — no decline at all. The thinned tertile shows the full
inverted-U (peak ~8.6 s at k=4-5, down to ~7.4 s by k=9-10); the mid tier sits in between. Checks
behave exactly like captures — more options, more to weigh — as long as the board is dense, and only
flip to a decline once it has thinned enough.

Directly comparing mean piece count by band confirms this is a genuine joint effect, not board
thinning by itself: all three bands lose pieces as checks-available rises, but by very different
amounts (ahead: 19.8 → 16.3; even: 25.0 → 18.4; behind: 20.2 → 18.7) — the even band thins out the
*most* in absolute terms, yet only mildly declines, while the ahead band thins less but falls
steadily. Thinning is necessary for the decline to appear at all, but its size and direction depend
on *which* band it happens in — a thinned, materially-ahead position reads as "resolved," a thinned,
even or behind one does not.

**A sharper test of what "thinning" actually means: it's specifically how many replies the
opponent has, not raw material.** Splitting material into its two raw components (self- vs.
opponent-material) doesn't add anything new — they're almost perfectly collinear in this data
(ρ = 0.97; both decline together as a game progresses, so neither carries much information on its
own, unlike their small-variance *difference*, material_imbalance, already tested above). But
`opponent_mobility` — the opponent's own legal-move count, computed directly from the FEN by
flipping the side to move — is a genuinely distinct axis (only ρ = 0.44 with opponent's material)
and splits the curve more cleanly than material does:

| opponent mobility | k=0 | k=4 | k=6 | k=9-10 | shape |
|---|---:|---:|---:|---:|---|
| low | 5.37 s | 7.15 s | 7.39 s | ~5.6-5.7 s | full inverted-U, falls back near baseline |
| high | 6.37 s | 9.05 s | 8.48 s | noisy (n<80) | rises and stays elevated, no clean fall |

When the opponent has few legal replies left, RT rises with checks-available then falls all the way
back down. When the opponent still has plenty of replies, RT rises and *stays up* — having more
checks of your own doesn't make the decision easier if there's still a lot of opposing moves to rule
out. This holds up under control, too: partial Spearman of RT with `opponent_mobility` is +0.09
whether or not opponent-material is partialled out alongside checks-available — it isn't just
riding on material's coattails.

*(Sample-based, not yet full-table: 200K FENs / 211K real move instances, and the "high mobility"
tail past k=6 has n<100. Directionally the cleanest evidence yet for "fewer things to check ⇒
faster," but not yet as solid as the material-band result above — would need a full rerun to
promote to a first-class part of the story.)*

**What specifically drives the ahead band's own decline:** the mover's minor pieces steadily die off
(1.69 → 1.24 mean) while major pieces stay flat (2.42 → 2.34) — the mover ends up disproportionately
likely to still have their queen (73.3% → 96.5%) as the board thins. A direct test — splitting the
band into "mover retains queen" (75.4%) vs. "no queen" (24.6%) — confirms this isn't just explained
away by that shift:

![checks available vs RT, split by queen retention (way ahead)](../figures/minply15_maxply75/board/png/checks_queen_retention_curves.png)

The decline survives almost fully *within* "retains queen" (7.96 s at k=1 → 4.76 s at k=10+); "no
queen" is flat and noisy. So queen-retention is a **gate**, not the whole mechanism — a second,
independent signal confirms there's more to it. `opponent_king_edge_distance` (0 = edge/corner, 3 =
fully central) predicts RT without going through checks-available at all:

![opponent king edge distance vs response time (way ahead)](../figures/minply15_maxply75/board/png/bivariate_opponent_king_edge_distance.png)

A clean, monotonic decline — 6.66 s at the edge to 4.31 s fully central — holding in every ply
tertile. Counterintuitively, the opponent's king moves *toward* the center as checks-available
rises, not into a corner: as the board empties, its pawn/piece shield dissolves and it ends up
exposed wherever it stands. Ply and clock don't already capture this — ply tracks total piece count
well (ρ = −0.73) but queen-retention only weakly (ρ = −0.23), and `mover_has_queen`'s own RT
correlation (ρ = +0.15) is comparable to ply's (ρ = −0.14) — two loosely-coupled signals, not one
subsuming the other.

<table><tr>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/checks_ahead_lowk_example.svg" width="220"><br>k=1, +6 material<br>both queens still on, denser position</td>
<td align="center"><img src="../figures/minply15_maxply75/board/svg/checks_ahead_highk_example.svg" width="220"><br>k=9, +15 material<br>lone queen vs. bare king, exposed center</td>
</tr></table>

> **Result:** Material imbalance is the organizing variable behind the checks-available inverted-U.
> It splits one mixed population into three genuinely different curves (behind: never turns over;
> even: cleanest inverted-U; ahead: falls almost immediately), and that compositional shift alone
> accounts for ~40–80% of the pooled decline. The rest is a within-band effect gated by board
> thinning — dense positions behave exactly like captures-available, monotonic, no decline — whose
> direction depends on which band is thinning: in the ahead band specifically, thinning
> disproportionately leaves the mover with a queen and the opponent's king exposed, and this decline
> survives conditioning on queen-retention directly. In the behind band, the same thinning instead
> keeps RT rising — a harder, must-find-a-try search, never a cruise. (Distinct-checking-piece
> redundancy and "hanging check" exclusions were also tested and move the curve in the same
> direction, but only partially and inconsistently — they're not part of the core account above.)
> What produces the curve's *rise* (0 to the peak) remains open.

### Captures Available

![captures available vs response time](../figures/minply15_maxply75/board/png/bivariate_n_captures_avail.png)

RT rises monotonically from 5.1 s at zero captures available to a plateau around 9.2–9.7 s at 9–13
(r = **+0.131**, key: `board/corr/pearson/n_captures_avail`).

#### Nonmonotonicity

None — no inverted-U, unlike checks. The reason is structural: captures-available correlates
*positively* with total pieces on the board (+0.248 — more captures when the board is busier/denser),
the opposite sign from checks-available (−0.258 — more checks when the board has simplified/opened
up), and it's essentially uncorrelated with |material imbalance| (≈ 0.00, vs. checks' +0.153). Checks
track how far a winning conversion has progressed; captures just track how tactically busy a
position is, which has no special relationship to who's ahead.

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
  `in_check` is not tracked as a standalone feature; its decomposition above uses one-off diagnostic
  queries, not a regenerated dashboard. `prev_move_was_capture` **is** promoted to two supplements
  (`supp_abs_material_imbalance`, `supp_material_imbalance` — signed and absolute), following the
  `supp_n_legal` recipe. The checks-available
  satisficing cross-check uses the engine tree-signal cache
  (`tree_values_cache/vals_befs1cp_md36_100000_7_pwin.parquet`, ~100K human moves matched to a
  search tree — see `engine.py`/[(engine tree analyses)](reference.md)) joined to `board_view` on
  `fen`, not the board-level dashboards themselves. `mover_has_queen`/`opponent_king_edge_distance`
  (the way-ahead-band checks-available mechanism figures, `checks_queen_retention_curves` and
  `bivariate_opponent_king_edge_distance`) are parsed per-FEN with python-chess over the
  `material_imbalance >= 5` subset (~1.5M distinct FENs, `_build_ahead_king_view` in `board.py`) —
  small enough to compute serially inside `run_plot()`, no Slurm shard/merge needed. The two
  `checks_ahead_*.svg` board examples were rendered with `chess.svg.board`, same convention as the
  other hand-picked SVG examples in this report (raw SVG; `cairosvg`/`svglib` aren't installed).
  `checks_pieces_tier_curves` uses only `n_pieces_on_board_inc_pawns` (already a stored column, no
  new per-FEN computation). `n_distinct_checking_pieces` (`checks_diversity_curves`,
  `bivariate_n_distinct_checking_pieces`) is parsed per-FEN with python-chess over a 500K-row plain
  proportional sample (`_build_checks_diversity_view`) — counts DISTINCT origin squares of pieces
  that can deliver check (not distinct piece-types, so two different bishops each giving check count
  as 2; a discovered check is attributed to the stationary checking piece, not the piece that moved
  to unveil it — same distinction `is_hanging_check` draws).
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
