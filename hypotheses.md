# Hypothesis board — why doesn't z_t beat baselines yet?

**Snapshot as of 2026-07-08, late afternoon.** This board is a companion to [`plan.md`](plan.md)
(the live status dashboard — not edited here) and exists to answer one question directly, the way
the investigation's owner actually asks it.

## The core problem, stated plainly

> We have a metacontroller (GNN encoder `z_t` + readout) that is trying to predict when planning
> (continued tree search) is worth it. That's it. Everything else is in service of that. The
> AlwaysHalt / AlwaysContinue / FixedHalt\* (SingleHalt\*) baselines exist to show the problem
> isn't trivially solved — so far it appears not to have been. There's a chasm between "not
> trivial" and "solved," and the two standard diagnostic questions are:
> 1. **Is it the trees?** Too wide, too deep, genuinely uninformative, or the wrong population?
> 2. **Is it training?** Frozen-encoder readouts, end-to-end joint training, explicit hand-crafted
>    features — none of it should be *this* hard to move past the baselines.

Every row below is graded against **one bar that matters for "training-side"**: a
statistically-confirmed (paired-bootstrap 95% CI excluding 0) win over **SingleHalt\*** — not
AlwaysStop, which any non-trivial policy clears and which the investigation has explicitly said is
not a sophisticated result (see `plan.md`, Agent 2 framing). For "tree-side" rows the bar is
whether a tree-shape/value/population change measurably moves regret or headroom, tested the same
rigorous way.

**Status legend:** ✅ confirmed · ❌ ruled out · 🟡 partially tested · ⬜ untested · ⏳ pending
(another agent is actively producing this evidence right now — not blocked on, not guessed at).

---

## Tree-side: is it the trees?

### T-1. Are the trees too wide or too deep for the controller to use well?

| | |
|---|---|
| **Status** | ❌ ruled out as a naive "shape is broken" story — but the finding is real and drove T-3 below |
| **Evidence** | [`plan.md` §T2](plan.md) (n=5,599 validation episodes): median depth=12 (p10=7, p90=18, 0.0% at depth≤2 — no breadth-starvation), median width=709 (p10=408, p90=1433) — **width is ~60× depth**, a real and large asymmetry, not a red flag in isolation. [`plan.md` §T3](plan.md) pruning companion: cutting width with `prune_epsilon=0.1` significantly shrinks median width 1099→425, but does **not** redirect the freed budget into depth as the width/depth imbalance suggested — depth actually drops slightly (8→7) — and yield/noise-proxy/quality show no compounding benefit. **Deploy recommendation in `plan.md`: keep current defaults, no point in the pruning grid showed a significant benefit.** |
| **Reading** | The trees are wide, but "make them narrower" doesn't buy anything on its own — this is a genuine negative result, not an untested lever. See T-1's sanity companion: [`plan.md` §T1](plan.md) — 5 example trees (spanning p10 to max argmax) pass illegal-move/root-dominance/linear-chain checks; the one flagged issue (133 duplicate-FEN pairs, i.e. transpositions, in the single largest outlier tree) is a documented, expected chess phenomenon, not a generation bug. |
| **What it would take to test further** | Nothing more on *this* specific lever — T3's own conclusion is that width-cutting is a dead end for regret/quality. The live successor question is whether search *allocation itself* changes under different tree-generation rules (see T-3, T-5 below), not post-hoc shape-trimming of the existing corpus. |

### T-2. Is planning genuinely not helpful at all in this population (more search = no gain)?

| | |
|---|---|
| **Status** | 🟡 partially tested — the specific levers tried are ruled out; the *general* claim is not fully closed |
| **Evidence** | [`plan.md` §T3](plan.md) quiescence sweep (`sf_search_limit_nodes ∈ {1,10,50}`, 200 trees/grid-point, bootstrapped CIs): **no significant effect on anything** — noise proxy, argmax>2 yield%, depth/width, and generation cost all overlap across the grid. "More per-leaf Stockfish search does not measurably change tree quality in this regime." Pruning (same section): no regret benefit in either direction. |
| **Reading** | Two independent levers for "give the tree more/better information" (deeper leaf search, and reshaping width→depth via pruning) both came back null. That's real evidence against "the trees just need to be bigger/deeper," but it is not the same claim as "planning has zero value in this domain" — the *headroom* question (does any policy have real room above SingleHalt\*?) is answered separately by the `REGIME` track, which is ⏳ pending write-up (`regime_select.md` not yet written; `plan.md` cites the raw grid: 352 points, 102 meaningful, meaningful maintenance-scale region `≲0.005`). A **prior, differently-calibrated** headroom sweep (`labnotebook.md`, 2026-07-07 entry, "Retraining-free headroom sweep") found fraction of achievable regret already recovered by SingleHalt\* was 0.83–0.94 at low maintenance (little headroom) but only 0.68–0.77 at maintenance≈0.1 (real headroom) — this is suggestive that headroom is regime-dependent, not that it's globally absent, but it predates the current Diagnosis-phase corpus/calibration and should not be read as still-current without `REGIME`'s confirmation. |
| **What it would take to test further** | `regime_select.md` landing (⏳ in progress) is the direct next evidence — it will state, on the *current* corpus, whether SingleHalt\* already captures most of what's achievable across the confirmed regime set, or whether there's a real, regime-general gap for a learned policy to close. |

### T-3. Is the signal (tree shape / eventual value) just not learnable from `z_t`?

| | |
|---|---|
| **Status** | ❌ ruled out — the signal is learnable; the frozen child-WDL pretraining objective is the bottleneck, not the architecture |
| **Evidence** | [`plan.md` §E2](plan.md) (job `10831415`, n_test=161,280 steps): the frozen child-WDL-pretrained `z_t` decodes basic tree-shape stats only weakly — MLP R² = 0.068 (n_nodes), 0.131 (height), 0.170 (width), vs a trivial upper bound ≈1.0 and a shuffle floor ≈0. This alone would suggest `z_t` has lost most of the structural signal. **But** the *same encoder architecture*, trained directly against a stats-reconstruction objective instead of the frozen child-WDL objective, reconstructs those same three stats at held-out R² = 0.843 (n_nodes), 0.950 (height), 0.828 (width) — job `10833402` (`diagnosis_s2_stats_regression_pretrain.slurm`), and a structural-only variant (excluding `current_value`) at 0.898/0.945/0.872 — job `10834317` (`diagnosis_s2b_stats_regression_pretrain_structural.slurm`). **Neither job produced a written report** (the S2/S3 track was explicitly marked "dropped, superseded, not revisited" in `plan.md` once end-to-end training (`Z`) was judged to test the same question more directly) — these numbers are pulled directly from the archived SLURM stdout (`slurm/archive/logs/diag-s2-stats-encoder_10833402.out`, `diag-s2b-structural-encoder_10834317.out`), not from a report, and are cited here because that's where the actual evidence lives. |
| **Reading** | [`plan.md` §E3](plan.md) states the synthesis: "objective/architecture mismatch with partial information loss — the child-WDL pretraining objective has hit a real ceiling AND doesn't preserve even basic structural stats a trivial readout gets for free." The 0.07–0.17 vs 0.83–0.95 gap is the load-bearing evidence for that claim: the architecture is not the limiting factor, the *training target* (frozen child-WDL) is. This is exactly why the investigation moved to unfreezing the encoder (see Training-side, row S-2). |
| **What it would take to test further** | Already acted on — this diagnosis directly motivated `Z`/`Z-EXT` (encoder-unfrozen training). No further tree-side test needed; the open question moved to training-side. |

### T-4. Is value-scale saturation a confound masking real headroom?

| | |
|---|---|
| **Status** | 🟡 partially tested — real effect confirmed, but the one test that could fully resolve it (regeneration under corrected values) is still ⏳ pending |
| **Evidence** | [`plan.md` §T3 "Related discovery"](plan.md): 56% of all nodes and 89% of leaf nodes have `\|value\| ≥ 0.99`; the underlying `cp_order` is continuous (median 268) — saturation comes entirely from Stockfish's WDL conversion, a step function around ~300cp (0% saturated below, 94% above). [`cp_recalibration.md`](cp_recalibration.md): a shape-frozen replay (relabel nodes with `tanh(cp_order/T)`, replay PUCT's backup over the *same* tree topology, no new search) at `T ∈ {100, 300, 600}` found a **mixed, not monotonic** result: `T=600` (most fully desaturated, 2.5% saturated vs. 56% baseline) has **significantly worse** headroom (`Δfraction_recovered = −0.1798, CI [−0.2927,−0.0751]`, excludes 0); `T=100` (paradoxically *more* saturated than baseline, 62.8%) has **significantly better** headroom (`+0.0932, CI [+0.0034,+0.1832]`); `T=300` is statistically indistinguishable from baseline (`+0.0232, CI [−0.0234,+0.0757]`, includes 0). `cp_recalibration.md`'s own recommendation: **do not launch a full encoder retrain on this evidence** — the strongest desaturation setting is the worst performer, the opposite of the naive "desaturate → more headroom" story. |
| **Reading** | `cp_recalibration.md`'s best-supported (but explicitly labeled non-mechanistic-proof) hypothesis: saturation may make the halt-reward trajectory *jumpier* (large step-like swings as `cp_order` crosses the saturation threshold), and a single fixed stopping step can't track that per-episode volatility — so desaturating *smooths* the trajectory into something a fixed rule is already good at approximating, which would explain why more desaturation can mean *less* relative headroom for an adaptive policy. This is stated as a hypothesis, not a proven mechanism (the report itself flags this). Crucially, **every one of these results is capped by construction**: the replay only re-labels values on a tree topology that was already grown under the OLD saturating value function — it cannot show what PUCT would have chosen to explore under corrected values from the start. |
| **What it would take to test further** | Exactly what's running now: real regeneration under `tanh(cp_order/300)` at generation time (not post-hoc), plus pruning, end-to-end through the pipeline (Agent 3, `plan.md`). Output will be `cp_regen.md` — **does not exist yet as of this snapshot**; smoke-scale generation (job `10842167`) was running at last check. **Mark this row `⏳ pending` for its decisive test; the replay-only evidence above is real but explicitly non-decisive.** |

### T-5. Is `puctvalue_md36` (our own Stockfish-PUCT corpus) the wrong tree population?

| | |
|---|---|
| **Status** | ⏳ pending — no result yet |
| **Evidence** | None yet specific to this hypothesis. Context: `plan.md`'s Agent 1 track is running the same pipeline against ysagiv's `human_trees` (lc0/Leela search on real 2023 human-game root FENs, ~1.08M trees, independently generated from our Stockfish-driven corpus) as a check on whether "z_t barely beats baselines" is corpus-specific. As of this snapshot: both quality filters (`ysagiv-argmax-filter`, `ysagiv-xaba-filter`) have completed tree-selection and are now in the packing stage (jobs `10841946`/`10841947` running) — no encoder/controller training or evaluation has happened yet, and **no `ysagiv_trees.md` file exists**. |
| **Reading** | Motivated by a real, cited concern (`plan.md`, Agent 1's "Why"): our own corpus was flagged by T-3/T-4 as saturating/possibly lower quality; ysagiv's corpus is a genuinely independent generation pipeline that may not share the pathology. This is a real, live test — just not resolved yet. |
| **What it would take to test further** | Nothing further to *do* — it's already running. Wait for `ysagiv_trees.md`: T1/T2-style sanity panel + the standard frontier/decodability eval at REGIME-confirmed regime(s). |

---

## Training-side: what about training?

### S-1. Frozen encoder + PG-readout-only (production 20-epoch, then +150-epoch, controller)

| | |
|---|---|
| **Status** | ❌ ruled out — genuinely closed, negative result, not just "not yet significant" |
| **Evidence** | [`our_trees_continued.md`, sub-line B](our_trees_continued.md): resumed the real 20-epoch production checkpoint (`mchalt_controller.pt`) and ran 150 more epochs (job `10842054`). Training curve hard-plateaus: val greedy regret 0.11753 (epoch 20) → best-ever 0.11681 (continuation epoch 88) → ends at 0.11757 (epoch 170) — a movement of ~0.0007 at best, not a real trend. Full paired-significance eval (job `10842428`, n_eval=1680, 11 checkpoints × 3 REGIME-confirmed regimes = 33 tests): **zero confirmed wins against SingleHalt\* across all 33 points.** At the primary regime the paired diff sits pinned at ≈0.0000 for the entire continuation. At the nonzero-maintenance regime it does **worse**: SingleHalt\* confirmed-beats this checkpoint at all 11 checkpoints there (e.g. epoch 20: `−0.0215 [−0.0286,−0.0142]`). Stats-Controller confirmed-beats this checkpoint at **every one of the 33 points**. It does reliably clear the low bar (AlwaysStop) at 2 of 3 regimes — explicitly flagged in that report as "not a sophisticated result." |
| **Reading** | This lineage was already converged well before this agent's 150-epoch extension (the plateau reportedly starts ~epoch 15-16 in the *original* run). More epochs of frozen-encoder, head-only training will not move this. |
| **What it would take to test further** | Nothing — this specific lineage (frozen encoder, PG head only) is a closed dead end per its own report's explicit verdict. |

### S-2. Encoder-unfrozen, jointly trained end-to-end (e2e / `Z`, `Z-EXT`)

| | |
|---|---|
| **Status** | 🟡 partially confirmed at 1 epoch; the natural next data point (more epochs) is ⏳ pending |
| **Evidence** | [`sig_significance.md`, SIG-Z](sig_significance.md): after 1 epoch of joint (encoder-unfrozen) training (job `10831440`, ~25 min, `pg_max_episodes=15000` cap), paired bootstrap 95% CI vs. three comparators: **significantly beats the original frozen-encoder `z_t`** (`+0.0046, CI [+0.0000,+0.0091]`, excludes 0) — real, confirmed progress; **does not yet significantly beat SingleHalt\*** (`+0.0042, CI [−0.0004,+0.0090]`, straddles 0, barely); **does not significantly beat or lose to Stats-Controller** (`−0.0024, CI [−0.0079,+0.0030]`). Decodability improved an order of magnitude alongside this (e2e `z_t` decodes `R(t)` at MLP R²=0.111 vs. the frozen encoder's 0.013). `plan.md` explicitly corrects its own earlier framing here: "the point-estimate win over SingleHalt\* was noise, not signal, at 1 epoch." |
| **Reading** | This is the single most promising open lever in the training-side bucket: it already has one *confirmed* win (over the frozen z_t it replaced) and is close-but-not-there on the real bar (SingleHalt\*), with the natural next step (more epochs, ~40 min/epoch measured real throughput) already scoped and dispatched. |
| **What it would take to test further** | Already running: `our_trees_continued.md` sub-line A. **As of this snapshot, sub-line A is an explicit placeholder** — the file's own text reads "(filled in once the continued-training + materialize + eval jobs complete — see status note below if this section is still short when read)," and it is still short. Per `plan.md`'s status dashboard, the original 3-epoch e2e job was cancelled (verified no work lost, contention on `gpu-short`) and re-split into three chained 1-epoch jobs on `gpu-test`; the first step (job `10842808`) was queued at last check. **Do not guess the outcome — mark pending.** |

### S-3. Explicit structural features, no encoder at all (Stats-Controller)

| | |
|---|---|
| **Status** | ✅ confirmed — the one clean, unambiguous positive result in the whole investigation |
| **Evidence** | [`plan.md` §S1](plan.md) (job `10832187`, n_fit=3919/n_eval=1680): Stats-Controller (`{n_nodes, height, width}`, structural only, explicitly excluding `current_value` since that's architecturally unavailable to `z_t`-Controller too — see `plan.md`'s stated rationale) achieves regret 0.1138 [0.1061,0.1218] vs. SingleHalt\* (k=10) 0.1204 [0.1129,0.1286]. [`sig_significance.md`, SIG-S](sig_significance.md): paired bootstrap confirms this is real, not a point-estimate artifact — mean diff `+0.0067, CI [+0.0026,+0.0112]`, excludes 0. |
| **Reading** | This is the load-bearing result for "is the underlying problem solvable at all" — three hand-crafted structural numbers, with no learned representation whatsoever, produce a policy that *significantly* beats the best fixed-threshold baseline. That means real, exploitable position-dependence exists in this population and this cost regime. It reframes the entire investigation: the open question is not "does an edge exist" (yes) but **"why hasn't the learned representation (`z_t`) matched or exceeded three raw counts yet."** |
| **What it would take to test further** | Nothing needed to confirm this further — it's done. The open question it raises (why does `z_t`, which architecturally has strictly more information available, not yet beat this) is what S-1/S-2/T-3/T-4 above are all, in different ways, trying to answer. |

### S-4. Cost-regime dependence — is the story robust, or an artifact of one lucky point?

| | |
|---|---|
| **Status** | ⏳ pending for the authoritative answer; 🟡 partial supporting evidence exists |
| **Evidence** | `plan.md`'s own summary of the `REGIME` track (write-up `regime_select.md` **does not exist yet** as of this snapshot — "write-up pending" per the status dashboard): a 352-point grid over `time_lambda × maintenance_scale`, 102 meaningful (non-degenerate `k*`) points, raw JSON on disk. It root-causes an earlier `maintenance_scale ∈ {0.05, 0.1}` collapse as those values simply being outside the meaningful region — the meaningful region is nonzero-maintenance-friendly but only at `maintenance_scale ≲ 0.005`, much smaller than what was originally swept. Independent corroboration of *a* collapse phenomenon (not proof of the region estimate) comes from `sig_significance.md`'s bonus finding: at `maintenance_scale=0.05` all four methods (SingleHalt\*, Stats, orig `z_t`, e2e `z_t`) collapse to an *identical* regret value (0.0090), and at 0.1 all four collapse to 0.0023 — a clean, independent confirmation that something real and shared is happening in that region, consistent with (but not the same evidence as) `REGIME`'s own diagnosis. `our_trees_continued.md` sub-line B (see S-1) already used 3 REGIME-confirmed regimes and found the story does *not* uniformly hold — sub-line B is confirmed-worse than SingleHalt\* specifically at the nonzero-maintenance regime, better only at the primary/far-lambda regimes, so at least one training-side result **is** regime-dependent in practice. |
| **Reading** | The regime-dependence question already has one concrete data point (S-1's checkpoint doing worse under nonzero maintenance) showing it matters in practice — this is not a purely hypothetical concern. The general characterization (which regimes are "meaningful," how big the achievable-headroom region is) is real work already done (352-point grid) but not yet written up in a form this board can cite precisely. |
| **What it would take to test further** | `regime_select.md` landing is the direct next step — mark this row's headline verdict pending until then. |

---

## Cross-cutting / doesn't fit cleanly

### X-1. Was the tree-generation pipeline itself simply buggy (not a shape/scale/objective question, a correctness question)?

| | |
|---|---|
| **Status** | ✅ resolved (found and fixed), historically important context for reading everything above |
| **Evidence** | [`labnotebook.md`, 2026-07-06 entry](labnotebook.md); [`outputs/reports/reference.md`](outputs/reports/reference.md) ("Tree-gen is poisoned" row, retracting the prior `R-EVALUATE`/`R-BUDGET-COLLAPSE` reports). The BeFS tree-gen corpus this whole line of investigation used to run on had a real correctness bug: unvisited root-move edges defaulted to a fake `q_value=0.0`, so on a zero-centered value scale an untested move could silently beat any move that was actually explored and found merely mediocre — the "best move" trace was an elimination artifact for 63% of the 250K-tree corpus. Compounded by BeFS having no exploration term (only ~1 of 20-38 root children ever got expanded past one ply). Fixed by rewriting the search classes with a principled FPU and switching the live default to PUCT. |
| **Reading** | This is why the current investigation exists on a rebuilt corpus (`puctvalue_md36`) and recalibrated cost regime (`time_lambda` 10.0→0.01, following the `value_feature` switch to win-probability scale) — everything in this board postdates that fix. It is not an open hypothesis anymore, but it is exactly the kind of failure mode ("looks like a modeling problem, is actually a generation bug") worth keeping in view when reading any negative result above. |
| **What it would take to test further** | N/A — closed. |

### X-2. Is training instability / small-network optimization noise inflating some of the negative results, independent of representation?

| | |
|---|---|
| **Status** | 🟡 partially tested, but the evidence predates the current corpus/calibration and has not been re-run under it |
| **Evidence** | [`outputs/reports/normative.md`](outputs/reports/normative.md) (an **earlier, now-superseded** run of this same "does z_t beat baselines" question — see the note in `report.md`'s edit below): its own diagnostic found Frac\* sometimes beats Stats-Controller not because Stats-Controller's feature set is worse (it's a strict superset) but partly because of under-optimization — handing the Stats-Controller a better-scaled centering feature narrowed a significant gap by ~39% without closing it. The same report's own caveats section states PG training is "numerically sensitive" at a single fixed seed, with run-to-run regret swings of ±30-70 on a comparable scale for Stats-Controller specifically. |
| **Reading** | This is a real, documented phenomenon in this general problem class (small PG-trained MLP heads), but the specific numbers above come from `normative.md`, which — see the report.md edit below — reflects an earlier tree corpus and an uncalibrated cost regime, not the current Diagnosis-phase pipeline. It has not been independently re-confirmed on the current corpus. Worth keeping in mind (e.g. as a partial explanation for why `our_trees_continued.md` sub-line B's checkpoints are all so close to identical, or why single-seed PG results throughout this investigation carry unquantified seed variance), but not something this board treats as settled evidence about the *current* results. |
| **What it would take to test further** | A multi-seed replication of at least one of the confirmed results (SIG-S or SIG-Z) on the current pipeline — not yet done anywhere in this investigation. |

---

## Synthesis: what's actually promising vs. a dead end

**Read literally from the evidence above, not speculation:**

- **Most promising, concretely actionable right now:** S-2 (encoder-unfrozen e2e training). It already has one confirmed win (beats the frozen z_t it replaced) and is *close* on the bar that matters (SingleHalt\*, CI straddling 0 by a small margin at just 1 epoch). The next data point (more epochs) is already dispatched and running — this is the single most likely near-term source of a genuinely new confirmed result, but it is **not yet a result**; `our_trees_continued.md` sub-line A is a placeholder as of this snapshot.
- **Most important result already in hand:** S-3 (Stats-Controller). It settles the meta-question the user is actually worried about ("is this solvable at all, or is the signal just not there") — yes, a real, confirmed, exploitable edge over SingleHalt\* exists in this population and cost regime, using nothing but three raw counts. Every negative training-side result (S-1) or partial one (S-2) has to be read against this: the ceiling clearly isn't zero, so the gap is about *how* to extract the edge, not *whether* it exists.
- **Closed dead ends, don't re-pursue:** T-1's raw pruning lever (cutting width doesn't help), T-2's quiescence lever (more per-leaf search doesn't help), and S-1 (more epochs of frozen-encoder PG-head training). All three have specific, cited negative evidence, not just "we didn't get to it."
- **Genuinely open, evidence pending, don't guess:** T-4 (value-scale saturation — the shape-frozen replay is mixed/negative but explicitly can't see search-allocation effects; the real test is running), T-5 (wrong tree population — ysagiv corpus check is running, no data yet), S-4 (regime robustness — the 352-point grid exists but isn't written up yet). All three could plausibly change the picture in either direction; none should be treated as settled by this board.
- **Honest gap in the story:** nobody has yet directly explained *why* `z_t` (with, architecturally, strictly more information available than the 3-number Stats-Controller) hasn't matched or beaten it. E2/S2's contrast (T-3 above) shows the representational capacity is there when trained against the right objective — that motivated unfreezing (S-2), which is the closest thing to a live answer to this exact question, and isn't resolved yet.
