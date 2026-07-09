# ysagiv `human_trees` — Phase 2 Agent 1

Record of Phase 2 Agent 1 (plan.md): does the "z_t barely beats baselines" story from our own
`puctvalue_md36` corpus hold on ysagiv's independently-generated `human_trees` corpus
(Leela/lc0 search, real 2023 human-game root positions), or is it specific to our own
Stockfish-driven generation? Methodology, verified numbers, and an honest bottom line below.

**Status: COMPLETE.** Both filter branches (argmax, xaba) ran the full pipeline through eval at
all 3 REGIME-confirmed regime points (frontier + decodability + paired significance). Job
`10845263` (argmax eval) completed cleanly (exit 0:0, 37m51s) after this document's argmax
section was first drafted as a placeholder — now filled in below with the full table.

## Corpus

`/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/data/human_trees` — 1,078,585 `.pt` files, format
`cts_raw_pretrain_example_v5`. Confirmed by direct load: schema is the same CSR tree format
(`parent_index`, `child_ptr`, `children_index`, `is_expanded`, `is_terminal`, `depth`,
`node_features`/`feature_names` = `value, wdl_win, wdl_draw, wdl_loss, wdl_var, prior`,
`edge_wdl_targets`, `oracle_root_moves`, `oracle_root_q_trace`, `oracle_best_move_index`,
`oracle_final_root_q_values`, `root_position_spec`, `incoming_moves`, `metadata` incl.
`engine_kind: lc0`) our own pipeline's `tree_encoder_feature_schema()` expects (5 of its 6
feature columns are a direct match; no `cp_order`, not needed). `oracle_trace_expansion_counts`
confirms 96 root expansions per tree — same `search_budget=96` as our own corpus, so
`fixed_budget=96` config carries over unchanged.

**Read-only throughout**: confirmed no write access to ysagiv's directory (`touch` test denied,
`drwxr-sr-x ysagiv griffith`). Every artifact this track produced lives under hl4291's own
scratch (`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv*`) or the repo's `outputs/figures/ysagiv/`.

## Corrections applied mid-task (coordinator-directed)

1. **Filter choice reopened.** The task brief named `argmax>2` (our own "thinking helps"
   filter) as the only filter to use, for direct comparability with the rest of this
   investigation. Mid-task the coordinator asked for a **second, independent filter run in
   parallel** — the `exclude_xaba` churn-motif filter (already implemented inside
   `cts.data.preprocess_mc.pack._source_top_level_root_churn_category`, previously only usable
   as an inline flag during `mc_pack`, not as a standalone pre-filter) — so the two filters
   could be compared head-to-head on the identical raw sample, not just one run in isolation.
   New standalone script `src/cts/data/preprocess_mc/filter_xaba.py` mirrors
   `filter_argmax.py`'s CLI/output shape exactly (same include-list contract feeding
   `split.include_list`) so both populations go through the identical downstream
   split→pack→train→eval chain.
2. **All outputs kept under hl4291's own scratch**, never `ysagiv`'s tree (confirmed read-only;
   see above) — new configs (`config_ysagiv_argmax.yaml`, `config_ysagiv_xaba.yaml`) route
   every `split_root`/`packed_dir`/`mc_packed_dir`/etc. to
   `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_{argmax,xaba}/`, distinct from
   `config_minply15_maxply75.yaml` (untouched).
3. **SLURM chaining**: initially triggered each pipeline stage manually after checking the
   previous one via `sacct`. Coordinator redirected to the `submit_all.sh` pattern — the full
   remaining chain submitted upfront with `--dependency=afterok:<parent>` links between stages,
   so SLURM itself advances the DAG without anyone polling.
4. **Resource sizing**: `ysagiv_eval_multi.slurm` was originally sized like production's
   `eval.slurm` (`qos=short`, `1:30:00`) — oversized for this track (smoke-scale populations,
   no separation/maintenance-sweep branch). Coordinator flagged `short` QOS had ~20x more
   queued jobs than `test`; switched to `qos=test`/`0:55:00` once confident of finishing under
   the 1h cap. Both eval jobs cleared in 3-16 min, confirming the resize was safe.
5. **Config import-time bug**: both new configs initially omitted a `human_analysis:` section
   (judged "not needed" — no `filter_and_sample`/`board`/`gen_trees`/`engine_analysis` stages
   run in this track). `analysis.evaluate` transitively imports `analysis.utils.helpers`, which
   unconditionally reads `human_analysis` **at module import time**
   (`db_connection()`'s default argument dereferences `CONFIG["selected_db_default"]` at
   `def`-time) — crashed both eval jobs immediately (`KeyError`). Fixed by adding a minimal
   `human_analysis: {selected_db_default: ${scratch_dir}/personal.db}` stanza to both configs
   (points at the same shared, read-only DB production uses; never actually opened by anything
   in this chain) — verified with a standalone `import analysis.evaluate` before resubmitting.

## Sanity check (T1/T2-style, on a 20,000-tree seeded raw sample)

**Sampling**: `os.listdir` the full 1,078,585-file directory (0.79s), seeded (`seed=42`)
`random.sample` of 20,000 filenames, symlinked into
`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv/raw_sample/` (reads only — symlinks point back
into `human_trees`, nothing is copied or written there). All filter/pack/train work below runs
on this SAME 20K population, not the full 1.08M (this repo's "iteration speed over
completeness" operating principle — smoke-scale first).

**T2 (histograms, n=3,000 of the 20K, seed 0)** — `src/analysis/ysagiv_sanity.py`, reusing
`analysis.tree_diagnostics`'s exact T1/T2 helper functions from our own corpus's check:

| stat | median | p10 | p90 | max | red flag | fired? |
|---|---|---|---|---|---|---|
| depth (final tree) | 6.0 | 4.0 | 9.0 | — | depth≤2 (breadth-starvation) | **no**, 0.0% |
| width (final tree) | 1359.5 | 602.0 | 2217.1 | — | pinned at root branching ceiling | ambiguous, see caveat below |
| argmax (oracle stop step) | **1.0** | 0.0 | 16.0 | 95.0 | argmax≥90 (right-censoring) | **marginal**, 0.17% (5/3000) |

Width-flag caveat: unlike our own corpus (single shared root per validation set), every ysagiv
tree has a DIFFERENT root position, so there's no single true legal-move ceiling to check width
against — the median root branching factor (29) was used as a representative line, and nearly
every tree's width sits above it (expected: width is the max *level* size across the whole
tree, not just the root's immediate branching, and these trees add ALL legal children at every
expansion — see below). Not treated as a real red flag, just not a directly transferable check.

**Headline finding: argmax median = 1.0, vs. our own corpus's median = 6** (T2 in plan.md,
n=5,599). This corpus's oracle "thinking helps" signal is far weaker at the population level —
foreshadowed (and then confirmed) the much lower `argmax>2` filter survival rate below.

**T1 (5 example trees spanning the argmax spread: p10/p50/p75/p90/max)** — all pass illegal-move,
linear-chain, and root-dominance checks (`analysis.tree_diagnostics.illegal_move_check` /
`linear_chain_check` / `root_dominance_check`, identical functions to our own corpus's T1):

| label | argmax | nodes | branching nodes | dup-FEN pairs | illegal moves | linear chain | root dominance |
|---|---|---|---|---|---|---|---|
| p10 | 0 | 3,896 | 96 | 164 | 0 (pass) | pass | pass |
| p50 | 1 | 3,965 | 96 | 35 | 0 (pass) | pass | pass |
| p75 | 1 | 3,965 | 96 | 35 | 0 (pass) | pass | pass |
| p90 | 16 | 2,602 | 96 | 346 | 0 (pass) | pass | pass |
| max | 95 | 1,405 | 96 | 461 | 0 (pass) | pass | pass |

**One consistent flag: duplicate-FEN (transposition) check fails on all 5 examples**, unlike
our own corpus (where only the single most-extreme outlier tree failed it, 133 pairs; the 4
non-extreme examples passed clean). Read as a real, structural difference in HOW these trees
are generated, not a correctness bug: every one of the 96 root expansions in a ysagiv tree adds
**every legal child** as a leaf (branching nodes always have exactly 96 — one per expansion —
regardless of argmax; node counts run into the thousands from only 96 expansions), unlike our
own PUCT-selective generation, which grows more sparsely. A tree with thousands of leaf nodes
spanning many different move orders is simply far more likely to contain legitimate
transpositions by chance. No illegal moves anywhere (0/5, exactly as strong a correctness
signal as our own corpus's T1) — the duplicate-FEN pattern is a generation-style difference, not
evidence of a bug.

Plots: `outputs/figures/ysagiv/png/ysagiv_t2_histograms.png`,
`outputs/figures/ysagiv/trees/ysagiv_t1_tree_{p10,p50,p75,p90,max}.svg`. Raw:
`outputs/figures/ysagiv/ysagiv_t{1,2}_summary.json`.

**Verdict: non-degenerate. Gate cleared** — no illegal moves, no breadth-starvation, no
right-censoring beyond a negligible tail, no linear chains, no root-dominance pathology. The
duplicate-FEN pattern and the much lower argmax median are real, reportable differences from our
own corpus, not correctness bugs — proceeded to real compute.

## Filters (head-to-head, same 20,000-tree raw sample, seed 42)

Per the task brief, the primary filter is **Filter A: `argmax>2`**, our own "thinking helps"
filter (`cts.data.preprocess_mc.filter_argmax`), for direct comparability with every other
regret number in this investigation. **Filter B: `exclude_xaba`** (new standalone
`filter_xaba.py`, see "Corrections" above) was added mid-task as a second, independent run for
head-to-head comparison, not a replacement.

| filter | raw kept / 20,000 | survival | notes |
|---|---|---|---|
| **A: argmax>2** | 3,750 | **18.8%** | vs. our own corpus's ~24.7% (900-FEN pilot) — lower, consistent with T2's lower argmax median |
| **B: exclude X\*AB\*A** | 13,140 | **65.7%** | categories: `X*A`=12,406 (62.0%), `X*AB*A`=6,860 (34.3%, excluded), `A`=734 (3.7%) |

**A real, notable finding, not just a survival-rate difference**: after the include-list filter,
`mc_pack`'s own baked-in `min_halt_reward_range=0.05` trivial-tree filter (same threshold both
branches share, matching production) interacts very differently with each population:

| branch | split (train/val) | mc_pack episodes (train/val) | additional attrition at mc_pack |
|---|---|---|---|
| argmax | 3,000 / 750 | 2,978 / 745 | **0.9% / 0.7%** (27 trees total) |
| xaba | 10,512 / 2,628 | 8,293 / 2,093 | **21.1% / 20.4%** (2,754 trees total) |

Argmax>2 already selects for trees whose halt-reward genuinely moves across the trajectory (by
construction — a nontrivial oracle argmax requires the reward to change), so almost nothing is
lost downstream. `exclude_xaba` selects on a completely different axis (root-best-move
stability), so a large fraction of its survivors — plausibly disproportionately the `A`
(never-churns) and stable `X*A` motif trees — turn out to have a nearly flat halt-reward curve
and get dropped by the SAME trivial-tree filter anyway. **Net trainable population**: argmax
≈3,723/20,000 (18.6%, barely different from raw), xaba ≈10,386/20,000 (51.9%, down from 65.7%
raw but still ~2.8x argmax's trainable population).

## Pipeline

Reused the existing scripts unmodified (`slurm/pipeline/{pack_trees,train_encoder,pack_root,
pack_root_merge,train_readout_pg}.slurm`), driven by the two new configs via
`CONFIG=config_ysagiv_{argmax,xaba}.yaml sbatch ...` — same architecture, same hyperparameters
as `config_minply15_maxply75.yaml` end to end (`d_embed=32`, encoder 6 epochs, PG readout 20
epochs, `pg_episode_batch=1024`, `time_mode=linear`), only paths/population differ. Both encoder
runs (child-WDL pretraining) show the same real, non-trivial loss decrease seen on our own
corpus (argmax: 0.7776→0.7713 over 6 epochs on 3,000/750 train/val trees; not a degenerate flat
line).

**Eval** at 3 REGIME-confirmed meaningful cost-regime points (`outputs/figures/minply15_maxply75/
diagnosis/regime_select_results.json`, filtered for `meaningful=true`), not just the single
`m=0/λ=0.01` point used everywhere else in this investigation:

1. `time_lambda=0.01, maintenance_scale=0.0` — the point used everywhere else (our corpus: k*=10)
2. `time_lambda=0.001, maintenance_scale=0.001` — a nonzero-maintenance regime (our corpus: k*=8)
3. `time_lambda=0.0005, maintenance_scale=0.0` — a much cheaper-search regime (our corpus: k*=56)

New script `src/analysis/ysagiv_sig.py` generalizes `evaluate_sig_s.py`'s paired-bootstrap
methodology (`cts.stats.bootstrap_ci`, percentile, `n_boot=2000`, never normal-theory) to all
three pairwise diffs (SingleHalt\* vs Stats, SingleHalt\* vs z_t, Stats vs z_t), reusing
`analysis.evaluate`'s `_load_assessment_data`/`_fit_stop_controllers`/`_regret_at` directly — same
70/30 fit/eval episode split (on the VALIDATION split only, `shuffle=False`), same regime
machinery as every other paired-significance result in this investigation.

## Results — xaba branch (COMPLETE)

n_fit=1,465 / n_eval=628 (70/30 split of 2,093 validation episodes) at every regime (same
episodes reused across all 3 regime evaluations — only the cost function changes).

| regime | k* | SingleHalt\* | Stats | z_t | SH−Stats (CI) | SH−zt (CI) | Stats−zt (CI) |
|---|---|---|---|---|---|---|---|
| λ=0.01, m=0 | 3 | 0.0834 | 0.0718 | 0.0805 | +0.0116 [+0.0035,+0.0210] **Stats wins** | +0.0029 [−0.0037,+0.0108] ns | −0.0087 [−0.0183,+0.0004] ns (barely) |
| λ=0.001, m=0.001 | 4 | 0.0533 | 0.0462 | 0.0515 | +0.0072 [−0.0026,+0.0175] ns | +0.0019 [−0.0031,+0.0078] ns | −0.0053 [−0.0147,+0.0042] ns |
| λ=0.0005, m=0 | 95 | 0.0396 | 0.0360 | **0.0217** | +0.0036 [+0.0001,+0.0060] **Stats wins** (barely) | **+0.0179 [+0.0148,+0.0202] z_t WINS** | **+0.0143 [+0.0104,+0.0185] z_t WINS** |

(diffs are `a − b`; a positive, CI-excludes-0 value means `b` significantly beats `a` — i.e.
lower regret.)

**Headline finding, genuinely different from our own corpus's story: at the λ=0.0005 (cheap-
search) regime, z_t-Controller significantly beats BOTH SingleHalt\* AND Stats-Controller on the
xaba-filtered ysagiv population** — the clearest positive result for a learned z_t readout
anywhere in this whole investigation (stronger than the e2e z_t result on our own corpus, which
only significantly beat the *old frozen* z_t, not SingleHalt\* or Stats). At the other two
regimes z_t is statistically indistinguishable from the other two methods (not a loss, just not
a confirmed win) — this is regime-specific, not a universal win.

**Caveat on interpretation**: `k*=95` at this regime (out of a 96-expansion ceiling) means
SingleHalt\*'s best FIXED stop is "almost never stop early" — cheap search makes continuing
almost always worth it in expectation. That a single global k must commit to nearly the full
budget on every tree, while z_t can condition on per-episode state and stop MUCH earlier on the
(frequent, per T2's low argmax median) trees where continuing genuinely doesn't help, is a
plausible mechanism for the win — z_t isn't beating a hard problem, it's exploiting exactly the
kind of per-episode heterogeneity a fixed k structurally cannot use. This reading is consistent
with, not independently proven by, the data collected here (same "no ad-hoc mechanism claims"
caveat as `cp_recalibration.md`).

**Decodability** (regime-independent, `n≈2093` validation episodes, R² predicting `R(t)`):
steps alone=0.105, tree-stats alone=0.238, **z_t alone=0.339**, all combined=0.531.
Orthogonality: stats predicts steps at R²=0.771 (heavily entangled with trajectory position),
z_t predicts steps at only R²=0.098 (far more independent signal). **z_t's decodability here
(0.339) is dramatically higher than our own corpus's frozen z_t (0.013) and even higher than our
own corpus's e2e-trained z_t (0.111)** — on this filtered ysagiv population, the frozen
child-WDL encoder alone already carries much more usable structural signal than the same
architecture does on our own corpus.

Plots: `outputs/figures/ysagiv/xaba/{regime_lambda0p01_m0,regime_lambda0p001_m0p001,
regime_lambda0p0005_m0}/{pdf,png}/frontier.*`, `outputs/figures/ysagiv/xaba/{pdf,png}/
decodability.*`. Raw: `outputs/figures/ysagiv/xaba/ysagiv_sig_regime_*.json`.

## Results — argmax branch (COMPLETE)

n_fit=522 / n_eval=223 (70/30 split of 745 validation episodes) at every regime.

| regime | k* | SingleHalt\* | Stats | z_t | SH−Stats (CI) | SH−zt (CI) | Stats−zt (CI) |
|---|---|---|---|---|---|---|---|
| λ=0.01, m=0 | 8 | 0.2676 | 0.2701 | 0.3879 | −0.0025 [−0.0298,+0.0245] ns | **−0.1203 [−0.1637,−0.0790] SingleHalt\* WINS** | **−0.1178 [−0.1599,−0.0775] Stats WINS** |
| λ=0.001, m=0.001 | 8 | 0.1889 | 0.2120 | 0.4249 | −0.0231 [−0.0465,+0.0026] ns (barely) | **−0.2359 [−0.2943,−0.1814] SingleHalt\* WINS** | **−0.2129 [−0.2704,−0.1599] Stats WINS** |
| λ=0.0005, m=0 | 95 | 0.0310 | 0.0310 | 0.0310 | +0.0000 [+0.0000,+0.0000] **degenerate collapse** | +0.0000 [+0.0000,+0.0000] **degenerate collapse** | +0.0000 [+0.0000,+0.0000] **degenerate collapse** |

(diffs are `a − b`; a positive, CI-excludes-0 value means `b` significantly beats `a`.)

**Headline: on the argmax-filtered population, z_t-Controller is significantly WORSE than both
SingleHalt\* and Stats-Controller at both of the two regimes that produce a real (non-collapsed)
comparison** — the opposite direction from the xaba branch. This is the same qualitative pattern
this whole investigation has repeatedly found on our OWN corpus (z_t struggling to clear the
fixed-stop bar) — so on argmax specifically, ysagiv's corpus does **not** rescue z_t; if
anything it's a starker loss here (z_t's absolute regret is 1.4-2.2x SingleHalt\*'s) than
anything recorded on our own corpus.

**The λ=0.0005 regime is a degenerate collapse, not a genuine three-way tie** — SingleHalt\*,
Stats, and z_t produce byte-identical regret (0.030950090386274165 for all three, confirmed to
full float precision in the raw JSON) on every one of the 223 eval episodes, hence a paired diff
of exactly 0.0 with a zero-width CI. This is the same "all methods collapse to an identical
value" pathology `SIG-Z`'s bonus finding first documented for `maintenance_scale` on our own
corpus (`sig_significance.md`) — here it's `time_lambda`-driven instead, on the smaller,
more class-imbalanced argmax population (`k*=95`, 223 eval episodes) rather than genuine
evidence the three methods are equivalent. **Contrast with xaba**: the identical regime
(`λ=0.0005, m=0`, also `k*=95`) does NOT collapse on the xaba population (SingleHalt\*=0.0396,
Stats=0.0360, z_t=0.0217, all distinct) — the same near-ceiling cost regime behaves completely
differently depending on which filter's population it's evaluated against, most plausibly
because argmax's much smaller eval set (223 vs. 628 episodes) and its selection for large-argmax
trees leave less room for episode-level heterogeneity once search is this cheap.

**Decodability** (regime-independent, `n≈745` validation episodes, R² predicting `R(t)`): steps
alone=0.222, tree-stats alone=0.251, z_t alone=0.448, all combined=0.568. Orthogonality: stats
predicts steps at R²=0.740, z_t predicts steps at R²=0.376 (more entangled with trajectory
position than on the xaba branch, where z_t-predicts-steps was only 0.098). z_t's raw
decodability (0.448) is actually the HIGHEST of any branch/corpus in this whole investigation —
yet the same z_t is the branch's worst-performing controller by regret. **This is itself a
notable finding**: high R(t)-decodability does not translate into good stopping decisions here,
a reminder (consistent with E2/E3's original diagnosis on our own corpus) that decodability and
control performance are related but not the same question — a readout can carry a lot of
information about R(t) and still make worse stop/continue calls than a much simpler tree-stats
readout or even a single fixed threshold.

Plots: `outputs/figures/ysagiv/argmax/{regime_lambda0p01_m0,regime_lambda0p001_m0p001,
regime_lambda0p0005_m0}/{pdf,png}/frontier.*`, `outputs/figures/ysagiv/argmax/{pdf,png}/
decodability.*`. Raw: `outputs/figures/ysagiv/argmax/ysagiv_sig_regime_*.json`.

## Bottom line

- **Corpus is non-degenerate**: T1/T2 sanity check clears every red flag except a benign,
  explicable duplicate-FEN pattern tied to this corpus's full-width-per-expansion generation
  style (not a bug).
- **The two filters behave very differently on this corpus, in a way that matters beyond raw
  survival rate**: argmax>2 has lower raw survival (18.8% vs 65.7%) but almost no additional
  loss at the trivial-tree filter stage; xaba's higher raw survival is substantially eroded
  (65.7%→51.9% net) by the same downstream filter, because it selects on a different axis
  (move-stability, not reward-range) that doesn't guarantee a trainable trajectory.
- **z_t's story is filter-dependent, not a fixed property of the ysagiv corpus as a whole — this
  directly answers the motivating question.** On **xaba**, z_t significantly beats BOTH
  baselines at the λ=0.0005 regime (the strongest positive z_t result anywhere in this
  investigation, stronger than the e2e win on our own corpus) and ties (not loses) at the other
  two. On **argmax**, z_t is significantly WORSE than both baselines at both of the two
  non-degenerate regimes — a starker loss than anything seen on our own corpus. **Reconciling
  the two**: the same corpus, evaluated at the same 3 cost regimes with the same architecture
  and training recipe, produces opposite verdicts on whether a learned z_t readout beats fixed
  or hand-crafted baselines, purely as a function of which "thinking helps" quality filter
  selected the training/eval population. This means the original question ("is z_t's weakness
  our-corpus-specific?") was underspecified — the answer depends at least as much on population
  selection as on which raw corpus is used.
- **The one clean win (xaba, λ=0.0005) carries a real caveat the numbers themselves flag**:
  `k*=95` is a near-ceiling regime where a fixed stop is structurally disadvantaged (must commit
  to nearly the full budget for every episode), and the *identical* regime collapses to a
  degenerate three-way tie on the argmax population — i.e. near-ceiling regimes are exactly
  where this whole setup is most prone to either a real per-episode-heterogeneity win (xaba) or
  a spurious collapse (argmax), depending on population size/composition. This is not a reason to
  discount the xaba win (it was paired-significance tested on 628 real held-out episodes, not an
  artifact of the collapse pathology), but it does mean the win should not yet be read as "z_t is
  reliably good whenever search is cheap" without checking whether it holds across a finer sweep
  of `time_lambda`, not just the single `λ=0.0005` point sampled here. **Forward pointer**: the
  coordinator has dispatched a follow-up (Agent 4) to sweep `time_lambda` on the xaba branch
  specifically to test whether this win is real/general or an artifact of sitting close to the
  right-censoring boundary — not duplicated here.
- **Decodability is not a reliable proxy for control quality**: argmax's z_t has the highest raw
  R²(R(t)) of any branch (0.448) yet is the worst-performing controller on that same branch by
  regret — echoes E2/E3's original diagnosis on our own corpus that structural information and
  good stopping decisions are related but distinct questions.
- **Honest net read**: this track does NOT deliver a clean "ysagiv trees fix z_t" result. It
  delivers a real, filter-dependent split — one population/regime combination where z_t is the
  best documented result in this whole investigation, and one population where z_t is the worst.
  Given the task's mandate to prefer an honest partial/mixed result over an overstated one, this
  is reported as exactly that: a genuine, population-dependent split, not a discovery that
  ysagiv's corpus resolves the "z_t barely beats baselines" story either way.
