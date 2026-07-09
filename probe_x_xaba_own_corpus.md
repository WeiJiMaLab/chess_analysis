# Probe X — `exclude_xaba` on our own `puctvalue_md36` corpus

Record of the "Probe X" node dispatched from `plan.md`'s "External expert input (Yotam Sagiv,
2026-07-08 evening)" section: Yotam, architect of the `exclude_xaba` filter, noted it was
"deliberately engineered to make the stopping decision nontrivial" and flagged that we had never
applied it to our own corpus (only to his `human_trees`, via Agent 1's `ysagiv_xaba_filter.slurm`
branch). This probe runs it on our own `puctvalue_md36` corpus, in **two independent, non-stacked
conditions**, mirroring Agent 1's own argmax-vs-xaba head-to-head design:

- **Condition A** — `exclude_xaba` **stacked on top of** our existing argmax>2 "thinking helps"
  filter (the population every other number in this investigation, e.g. `sig_significance.md`,
  is expressed against).
- **Condition B** — `exclude_xaba` used **alone**, entirely **replacing** argmax>2 (no argmax
  pre-filter at all).

Both are compared against the **`argmax_only` baseline** (production population, no xaba) at the
same 3 REGIME-confirmed regime points, using the same paired-bootstrap significance methodology
(`cts.stats.bootstrap_ci`, percentile, 2000 resamples, never normal-theory) used throughout this
investigation.

## Scope confirmation (before any repack)

Read `src/cts/data/preprocess_mc/pack.py` (the `exclude_xaba: bool` field + CLI `--exclude-xaba`)
and `slurm/pipeline/{pack_trees,pack_root,mcpack_only}.slurm` directly. Confirmed:
`exclude_xaba` is a field of `PackControllerEpisodesConfig`, checked once, first, in
`_build_packed_tree_result` (`preprocess_mc/pack.py:750`) — i.e. **at the controller-level
packing stage only**. It is entirely absent from `src/cts/data/preprocess_gnn/pack.py` (the
encoder-pretraining-data packer) and from `preprocess_gnn/split.py`. `pack_trees.slurm`'s own
comment ("`split.include_list` is null, so every generated tree is packed (no pre-filter)")
additionally confirmed that skipping argmax filtering is simply a matter of not setting
`split.include_list` — no separate code path needed.

**Conclusion: no encoder retrain needed for either condition.** Both conditions reuse the
already-trained frozen child-WDL encoder (`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/packed/tiny_encoder.pt`,
byte-identical file referenced by both new configs and by the base production config) — this is a
moderate-cost probe (repack + re-materialize + refit-and-eval the controller/readouts), not a
full pipeline rerun from tree generation.

## Methodology

**Isolation.** Two new, additive, standalone configs (`config_probe_x_xaba.yaml` for condition A,
`config_probe_x_xaba_condb.yaml` for condition B) — neither touches
`config_minply15_maxply75.yaml` or any file another agent was using. Both reuse the base run's
already-trained encoder and (for condition A) the base run's already-built `split/` directory
read-only; all new outputs live under fresh, uniquely-named scratch trees
(`/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/probe_x_xaba/` and
`.../probe_x_xaba_condb/`) and figures land at
`outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/probe_x_xaba_on_own_corpus.*`.

**Condition A** reuses the base run's `split/` (train/validation partition of the argmax>2
subset, 22,838 train / 5,710 validation trees) and reruns only `mc_pack` with
`exclude_xaba=true` (mirrors `slurm/mcpack_only.slurm`'s "repack episodes only" pattern), then
materializes the validation split against the frozen encoder (single CPU worker; small enough not
to need `pack_root.slurm`'s array).

**Condition B** needed a fresh `split` (no `include_list`, i.e. no argmax filter at all) —
running the split stage over the full 400,000 generated trees would have taken a single-CPU-worker
materialize step upward of 6 hours (extrapolated from condition A's measured ~0.38s/episode rate),
so **`split.include_list` was repurposed as a pure compute-tractability random subsample**: 30,000
tree filenames drawn uniformly (seed 0, no quality/argmax criterion whatsoever) from the full
400,000-tree corpus, split 80/20 into 24,000 train / 6,000 validation trees. `exclude_xaba=true`
is then the **only** filter applied on top of that random subsample. This is analogous to Agent
1's own precedent of subsampling the 1.08M-tree `human_trees` corpus down to a tractable 20K
sample rather than processing it in full.

**Regime points.** Same 3 points used throughout: `λ=0.01, m=0` (the primary validated default),
`λ=0.0005, m=0` (looser time pressure, no maintenance), `λ=0.01, m=0.001` (small nonzero
maintenance, REGIME's confirmed "meaningful" region for the `argmax_only`/condition-A
populations). `REGIME`'s meaningfulness criterion was fit *on the `argmax_only` population*; as
noted below, it does not automatically transfer to condition B's very different population.

**Significance test.** `src/analysis/evaluate_probe_x_xaba.py` (new), which calls
`analysis.ysagiv_sig.compute_pairwise_significance` **unmodified** — the same script Agent 1 uses
on the ysagiv corpus — once per (population, regime) cell. That function fits SingleHalt*/
Stats-Controller/z_t-Controller independently on a 70/30 fit/eval split of each population's own
validation set (grouped by source tree, never leaking a tree's episodes across the split) and
paired-bootstraps the three pairwise regret differences on the shared eval episodes. All raw
numbers below are pulled directly from
`outputs/figures/minply15_maxply75/diagnosis/probe_x_xaba_results.json`.

## Survival rates

| population | train accepted / total | train survival | validation accepted / total | validation survival |
|---|---|---|---|---|
| `argmax_only` (baseline, no xaba) | 22,338 / 22,838 | **97.81%** | 5,599 / 5,710 | **98.06%** |
| `argmax_and_xaba` (condition A) | 17,040 / 22,838 | **74.61%** | 4,241 / 5,710 | **74.27%** |
| `xaba_only` (condition B, on random 30K subsample) | 3,043 / 24,000 | **12.68%** | 763 / 6,000 | **12.72%** |

**Condition A**: stacking `exclude_xaba` on top of argmax>2 drops an *additional* ~23-24
percentage points of trees (97.8%→74.6% train, 98.1%→74.3% validation) — a substantial,
non-trivial filter, consistent with Yotam's description of it as deliberately aggressive.

**Condition B**: applied alone to a random (non-argmax-filtered) sample, `exclude_xaba` drops
**~87%** of trees — roughly 3.5x more aggressive than when it's stacked on the argmax>2 subset.
There is no matched "no-filter-at-all" control on the exact same random 30K sample to cleanly
attribute 100% of this gap to xaba specifically (the pre-existing `min_halt_reward_range=0.05`
trivial-tree filter also fires, though it accounts for only ~2% of drops when measured on the
argmax-filtered baseline). **Hypothesis, not a verified mechanism**: the X*AB*A churn-and-revisit
motif appears to be far more common among trees that argmax>2 already screens *out* (i.e. "thinking
doesn't help" positions, where a shallow search's best-move estimate is more likely to wobble
before any real signal emerges) than among the argmax>2 subset (positions where extended search
demonstrably changes the answer, which apparently correlates with a *more* stable final trace).
This is a plausible reading of the ~74% vs. ~13% gap, not something this probe verifies mechanistically.

## Results

### `argmax_only` (baseline, established reference — reproduced here for direct comparison)

| regime | SingleHalt* | Stats | z_t | SingleHalt*−Stats (paired, 95% CI) | verdict |
|---|---|---|---|---|---|
| λ=0.01, m=0 (k*=10, n_eval=1680) | 0.1204 [0.1129,0.1286] | 0.1137 [0.1061,0.1216] | 0.1207 [0.1130,0.1287] | +0.0067 [+0.0026,+0.0112] | **Stats CONFIRMED beats SingleHalt\*** |
| λ=0.0005, m=0 (k*=56, n_eval=1680) | 0.0334 [0.0298,0.0375] | 0.0304 [0.0277,0.0334] | 0.0311 [0.0281,0.0347] | +0.0030 [+0.0002,+0.0064] | **Stats CONFIRMED beats SingleHalt\*** |
| λ=0.01, m=0.001 (k*=6, n_eval=1680) | 0.1154 [0.1075,0.1233] | 0.1146 [0.1073,0.1225] | 0.1152 [0.1081,0.1225] | +0.0008 [−0.0035,+0.0049] | not significant |

Point estimates match the previously-established numbers almost exactly (SingleHalt\*=0.1204,
Stats≈0.1137, z_t=0.1207 at the primary regime — see `sig_significance.md`), a good consistency
check that this run's `argmax_only` population and pipeline are the same production artifacts, not
a re-derivation. Stats significantly beats SingleHalt\* at 2 of 3 regimes here (it was already
known to at the primary regime; this run additionally confirms it at the looser-λ regime, and
shows the maintenance regime doesn't reach significance either way).

### Condition A — `argmax_and_xaba` (xaba stacked on argmax>2)

| regime | SingleHalt* | Stats | z_t | SingleHalt*−Stats | SingleHalt*−z_t | verdict |
|---|---|---|---|---|---|---|
| λ=0.01, m=0 (k*=10, n_eval=1272) | 0.1210 [0.1124,0.1301] | 0.1166 [0.1086,0.1248] | 0.1207 [0.1129,0.1291] | +0.0044 [−0.0009,+0.0099] | +0.0003 [−0.0056,+0.0060] | not significant (either) |
| λ=0.0005, m=0 (k*=49, n_eval=1272) | 0.0328 [0.0274,0.0387] | 0.0293 [0.0257,0.0337] | 0.0279 [0.0245,0.0319] | +0.0035 [−0.0003,+0.0079] | **+0.0049 [+0.0005,+0.0101]** | **z_t CONFIRMED beats SingleHalt\*** |
| λ=0.01, m=0.001 (k*=7, n_eval=1272) | 0.1127 [0.1054,0.1206] | 0.1105 [0.1028,0.1186] | 0.1108 [0.1033,0.1191] | +0.0022 [−0.0028,+0.0076] | +0.0019 [−0.0015,+0.0054] | not significant |

**Headline for condition A: the story changes at one of the three regimes.** Stats-Controller's
significant win over SingleHalt\* at the primary regime — the investigation's headline confirmed
result on the `argmax_only` population — **loses significance** under xaba-filtering (CI
[−0.0009, +0.0099], now straddles 0), likely a smaller-n effect (n_eval drops 1680→1272) as much
as a population-shape effect. But a *new* significant result appears that the `argmax_only`
population never showed anywhere in this investigation: **at the looser-λ regime, z_t-Controller
significantly beats SingleHalt\*** (+0.0049, CI [+0.0005, +0.0101], excludes 0) — the first
confirmed SingleHalt\*-beating result for z_t (as opposed to Stats) anywhere in this whole session
(cf. `sig_significance.md`'s SIG-Z, where e2e z_t got close but never confirmed a win over
SingleHalt\* on the unfiltered population). This is exactly the kind of result Yotam's framing
predicted: a filter engineered to make the stopping decision nontrivial surfacing a real edge for
the learned (z_t) representation that a trivial population was masking.

### Condition B — `xaba_only` (xaba alone, no argmax; n≈229, small-sample caveat below)

| regime | SingleHalt* | Stats | z_t | SingleHalt*−Stats | SingleHalt*−z_t | verdict |
|---|---|---|---|---|---|---|
| λ=0.01, m=0 (k*=4, n_eval=229) | 0.0917 [0.0727,0.1136] | 0.1056 [0.0782,0.1379] | **0.3051 [0.2584,0.3487]** | −0.0139 [−0.0401,+0.0106] | **−0.2134 [−0.2587,−0.1687]** | z_t **catastrophically loses** to both |
| λ=0.0005, m=0 (k*=90, n_eval=229) | 0.0292 [0.0273,0.0312] | 0.0302 [0.0283,0.0321] | 0.0298 [0.0279,0.0317] | −0.0010 [−0.0021,+0.0005] | −0.0005 [−0.0017,+0.0010] | **z_t CONFIRMED beats Stats** by a tiny margin (stats−zt = +0.0004, CI [+0.0002,+0.0006]); SingleHalt* vs. either: n.s. |
| λ=0.01, m=0.001 (k*=**0**, n_eval=229) | 0.0869 [0.0599,0.1179] | 0.0869 (identical) | **0.1641 [0.1027,0.2372]** | 0.0000 (degenerate) | **−0.0772 [−0.1440,−0.0218]** | **degenerate regime** (k*=0 ⇒ SingleHalt*≡Stats trivially); z_t confirmed loses |

**Headline for condition B: z_t-Controller fails badly on this population, at the two regimes
where the comparison is non-degenerate-for-SingleHalt\*-vs-Stats.** At the primary regime it loses
by a huge, confirmed margin (regret 0.305 vs. SingleHalt\*'s 0.092 — nearly 3.3x worse, CI entirely
below 0). At the maintenance regime `k*=0`: SingleHalt\* and Stats become byte-identical trivial
"always stop immediately" controllers (a genuinely degenerate cell, not a real test of Stats vs.
SingleHalt\* — REGIME's "meaningful" classification was fit on the `argmax_only` population and
does not transfer automatically to this very different, much smaller, much more aggressively
filtered population), and z_t still loses badly there too (−0.077, CI excludes 0). Only at the
looser-λ regime does z_t's behavior look sane, and there it actually **edges out Stats** (not
SingleHalt\*) by a real but minuscule, confirmed margin (0.0298 vs. 0.0302 regret, +0.0004, CI
[+0.0002,+0.0006] — tiny relative to the ~0.03 scale of this regime).

**Caveat, stated plainly per the coordinator's flag**: `n_eval=229` (and `n_fit=534`) is by far
the smallest population anywhere in this investigation — roughly 5.5x smaller than condition A's
(n_eval=1272) and 7.3x smaller than `argmax_only`'s (n_eval=1680). The `z_t`-Controller readout has 33 input dims
(`d_embed=32 + T_t`); fitting a PG-trained MLP head on 534 fit episodes is a real
undertraining/overfitting risk that the other two (cheaper, lower-dimensional) controllers are far
less exposed to — SingleHalt\* is a 1-parameter grid search, Stats-Controller's readout is only
4-dimensional. z_t's catastrophic loss here is very plausibly a small-n instability artifact of
the readout, not evidence that the frozen z_t representation is intrinsically bad on
xaba-filtered-without-argmax trees. This result should be read as **directional, not
conclusive** — re-running condition B at a larger raw-sample scale (closer to the full 400K, with
a properly parallelized array materialize) is the natural follow-up if this track is picked back
up, and is the single most important open item this probe leaves behind.

## Plot

![Probe X: unfiltered vs. xaba-filtered, three populations x three regimes](outputs/figures/minply15_maxply75/diagnosis/png/probe_x_xaba_on_own_corpus.png)

Top row per regime panel: marginal regret (95% CI) for SingleHalt*/Stats/z_t, grouped by
population (`argmax_only` solid, `argmax_and_xaba` hatched, `xaba_only` dotted — method identity
stays color-coded consistently with every other plot in this investigation; population is a
texture/alpha secondary encoding). Bottom row: the SingleHalt\*−Stats paired-diff 95% CI per
population (filled marker = confirmed win) — the actual significance test, not just eyeballing
overlapping marginal bars. PDF: `outputs/figures/minply15_maxply75/diagnosis/pdf/probe_x_xaba_on_own_corpus.pdf`.

## Bottom line

**Does xaba-filtering our own corpus change the story? Yes, in both directions, depending on how
it's applied — and no single verdict generalizes:**

1. **Condition A (stacked on argmax>2, ~74% survival)**: the previously-confirmed Stats-beats-
   SingleHalt\* result at the primary regime loses significance (smaller n), but a genuinely new
   result appears — **z_t-Controller significantly beats SingleHalt\*** at the looser-λ regime,
   the first such confirmed result anywhere in this investigation. This is real, if modest,
   support for Yotam's claim that the filter surfaces a real edge for the learned representation.
2. **Condition B (xaba alone, replacing argmax>2, ~13% survival on the tractability subsample)**:
   z_t catastrophically underperforms both baselines at 2 of 3 regimes, most plausibly a
   small-n readout-fitting artifact (n_eval=229) rather than a real representational failure —
   **not a confident conclusion**, flagged as needing a larger-scale rerun before it's trusted
   either way.
3. **Survival rates alone are informative regardless of downstream results**: xaba's exclusion
   rate is highly population-dependent (~24% additional drop stacked on argmax>2, vs. ~87% drop
   applied to a random/unfiltered sample) — the churn motif is evidently concentrated in
   positions our argmax>2 filter already screens out.

Per Yotam's own recommended honest framing: this is a mixed, not a clean, result. It does not
uniformly validate or refute the "z_t is worthless" prior — it depends on exactly how the
population is filtered, and condition B's headline number needs a bigger sample before it should
move anyone's confidence much.

## Artifacts

- Configs: `config_probe_x_xaba.yaml` (condition A), `config_probe_x_xaba_condb.yaml` (condition B)
- Condition B's tractability sample list: `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/probe_x_xaba_condb/random_sample_include_list.txt`
- SLURM: `slurm/pipeline/mcpack_only.slurm` (condition A repack, `CONFIG` override), `slurm/probe_x_xaba_materialize.slurm`, `slurm/probe_x_xaba_condb_split_pack.slurm`, `slurm/probe_x_xaba_condb_materialize.slurm`, `slurm/probe_x_xaba_eval.slurm`
- Eval script: `src/analysis/evaluate_probe_x_xaba.py` (reuses `analysis.ysagiv_sig.compute_pairwise_significance` unmodified)
- Raw results: `outputs/figures/minply15_maxply75/diagnosis/probe_x_xaba_results.json`
- Plot: `outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/probe_x_xaba_on_own_corpus.*`
- Packed/materialized data (read-only after this point): `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/probe_x_xaba/`, `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/minply15_maxply75/probe_x_xaba_condb/`
- SLURM job IDs: condition A `10845255` (mc_pack) → `10845256` (materialize); condition B `10845867` (split+mc_pack) → `10845868` (materialize); combined eval `10845873`. All `COMPLETED`, exit 0:0.
