# Probe Y — apples-to-apples eval of the subtree-weighted encoder

Resolves plan.md's "External expert input (Yotam Sagiv...)" item 5: Sagiv recalled his own
`oracle96`/subtree-weighted controller lineage as "much better than the one trained on human
trees," citing a historical "greedy regret" of 0.024 for a fitted-Q controller
(`[z_t, T_t]` inputs) built on top of `tree_encoder_child_wdl_async_k1_subtree_weighted.pt`
(encoder pretrained with edge cross-entropy weighted by child subtree size). That number
comes from a different era's pipeline (a fitted-Q controller, not the PG-trained advantage
head this investigation now uses everywhere) and the same lineage's own labnotebook shows
"greedy regret" at wildly different scales across entries over time — the same instability
class as this investigation's own `time_lambda` miscalibration history. This probe does NOT
try to reconcile the 0.024 number directly (that is a separate, parallel line of git
archaeology into the historical pipeline's own cost-scale/regret-definition, tracked
independently — see the coordinator's note below). Instead it asks a cleaner question:
loaded fresh and measured under THIS investigation's current, calibrated methodology, does
the subtree-weighted **encoder** (not the old controller checkpoints) produce a z_t that
gives a real, statistically-confirmed edge over SingleHalt*/Stats-Controller — and if so, is
that edge bigger than the edge our own production encoder gets over the same baselines?

**Framing constraint (per direct instruction, important — read before the Results section):**
this is a **within-population** comparison only. All three methods (SingleHalt*,
Stats-Controller, subtree-weighted z_t-Controller) are fit and scored on the *same* held-out
`human_trees` episodes. The question resolved here is whether the subtree-weighted z_t beats
the other two *there*, and by how much relative to Stats-Controller's own margin over
SingleHalt* on the *same* population — **not** whether its absolute regret number is smaller
or larger than our own `puctvalue_md36` corpus's ~0.11–0.12 range. Different tree populations
(different branching, different position-difficulty distribution, different search-budget
provenance — `human_trees` is lc0/Leela search over real human-game positions, our corpus is
Stockfish-driven PUCT) can have structurally different regret *magnitudes* for reasons that
have nothing to do with methodology, even under perfectly consistent calibration. An
absolute cross-corpus number would conflate population differences with methodology
differences. This report deliberately does not draw that comparison.

## Checkpoint and architecture (confirmed, not assumed)

Loaded `tree_encoder_child_wdl_async_k1_subtree_weighted.pt` directly (read-only,
`/scratch/gpfs/GRIFFITHS/ysagiv/chess/CTS/checkpoints/`) and inspected its own embedded
metadata (`checkpoint["metadata"]["encoder_architecture"]`) rather than assuming it matches
our production encoder's shape:

```
{'k': 1, 'd_embed': 128, 'd_message': 128, 'n_heads': 4, 'd_att': 32,
 'node_embed_hidden': 128, 'node_feat': 5}
```

**`d_embed=128`, not 32** — 4× our production encoder's embedding width. `node_feat=5`
matches `cts.core.schema.TREE_ENCODER_FEATURE_NAMES`'s 5 baseline features, confirming this
checkpoint is compatible with the same packed node-feature schema our own pipeline (and
Agent 1's `human_trees` pipeline) uses — this is a real load-and-run compatibility, not a
format-conversion job. `pretrain_objective` in the metadata is
`search_consolidated_edge_wdl_v1`. Per `labnotebook.md`'s "Encoder pretrain" section
(2026-05-20 entry): 1000 epochs from fresh init, cross-entropy weighted per edge by child
subtree size, trading ~3× worse leaf-cell KL for ~21× lower KL on the large-subtree cells the
weighting targeted.

**Validation population**: this encoder was pretrained on
`pretrain_split_oracle96_trace_filtered_rerun` (per `labnotebook.md`), i.e. the same
`human_trees` corpus (lc0/Leela search over human-game root FENs) our own Agent 1 track is
using — not a disjoint corpus. Evaluated here on Agent 1's `human_trees`, argmax>2-filtered
("thinking helps" criterion, `min_argmax=2`, matching `filter_argmax.py`'s criterion used
throughout this investigation), reused read-only rather than re-derived.

## Infrastructure reused vs. built

- **Reused, read-only**: Agent 1's `mc_packed` directory at
  `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_argmax/mc_packed/` — the argmax>2-filtered
  `human_trees` controller-episode packing (train: 2,978 episodes across 6 shards;
  validation: 745 episodes across 2 shards; `cts_budgeted_controller_episode_manifest_v4`
  format, packed at `time_lambda=0.01, maintenance_scale=0.0` — cost-independent at pack
  time, re-appliable at analysis time per `analysis.evaluate`'s design). Confirmed stable
  (last modified well before this probe started, no in-flight writers into that specific
  subdirectory) before pointing at it. Never wrote into this directory or anything under
  `ysagiv_argmax/`.
- **Built new**: everything downstream of the packed episodes is this probe's own, under
  `/scratch/gpfs/GRIFFITHS/hl4291/lmcos/probe_y_subtree_weighted/` — never under
  `/scratch/gpfs/GRIFFITHS/ysagiv/` (read-only source) or Agent 1's own output tree.
  - `config_probe_y_subtree.yaml` (repo root) — new, additive config; does not touch
    `config_minply15_maxply75.yaml` or `config_ysagiv_{argmax,xaba}.yaml`.
  - `slurm/probe_y_materialize.slurm` — array CPU job (16 workers): encodes both splits'
    episodes through the frozen subtree-weighted encoder (`cts.data.preprocess_mc.materialize`,
    architecture auto-read from the checkpoint). Mirrors `slurm/pipeline/pack_root.slurm`'s
    array/worker pattern — an initial single-worker design was tried first and abandoned once
    a smoke test showed materialize is genuinely slow on this encoder (see "Throughput
    finding" below).
  - `slurm/probe_y_materialize_merge.slurm` — stitches the array workers' shards into
    `{train,validation}_cache.pt`, chained via `--dependency=afterok`.
  - `slurm/probe_y_eval.slurm` — chained via `--dependency=afterok`, runs the eval script
    below.
  - `src/analysis/evaluate_probe_y.py` — new script, reuses `analysis.evaluate`'s private
    helpers directly (`_load_assessment_data`, `_return_curves`, `_fit_stop_controllers`,
    `_frontier_panel`/`_draw_zoom_indicator`/`_padded_range`/`_deconflict_points`,
    `cts.stats.bootstrap_ci`) rather than reimplementing them — same pattern
    `evaluate_sig_s.py`/`evaluate_e2e_z3.py` already established for one-off comparisons that
    don't fit `analysis.evaluate`'s hardcoded 3-series shape. Fits SingleHalt* / Stats-Controller
    / subtree-weighted z_t-Controller ONCE (shared fit/eval split and cost curves — not two
    separate expensive PG-training calls), then paired-bootstraps two significance tests:
    SingleHalt* − z_t and Stats-Controller − z_t, on the same held-out episodes.

**Incidental fix required, noted for the record**: importing `analysis.evaluate` transitively
imports `analysis.utils.helpers`, which eagerly runs `load_config_section("human_analysis")`
at module-import time against whatever `$CONFIG` is active, and one of its module-level
default-argument expressions (`db_connection`'s `database=CONFIG["selected_db_default"]`)
evaluates immediately too — both raise if the active config lacks those exact keys. This is a
generic footgun for any new, additive config (it independently crashed Agent 1's own eval
jobs on `config_ysagiv_argmax.yaml`, per the coordinator). `config_probe_y_subtree.yaml`
includes a minimal `human_analysis:` stub (`figures_dir`, `selected_db_default`) purely to
satisfy the import — no code in this probe reads any field from that section.

## Methodology

Identical to `SIG-S`/`SIG-Z` (`sig_significance.md`): 70/30 fit/eval split partitioned by
source TREE (never by raw episode index, to avoid leaking a tree's other truncation-depth
episodes across the boundary — `analysis.evaluate._split`), SingleHalt* fit by grid-search
minimizing fit-split regret, Stats-Controller and the subtree-weighted z_t-Controller both
PG-trained advantage heads (`cts.train.pg_controller_train.fit_readout_pg`, 200 epochs,
lr=1e-3 — the exact deployed trainer, no bespoke loop), all three scored on the same held-out
eval episodes. Paired significance: percentile bootstrap 95% CI (`cts.stats.bootstrap_ci`,
`n_boot=2000`, never normal-theory, per this repo's fixed convention) on the per-episode
regret **difference** (baseline − subtree-weighted z_t). CI entirely > 0 ⇒ subtree-weighted
z_t significantly wins; CI entirely < 0 ⇒ it significantly loses; CI straddling 0 ⇒ not
distinguishable at this sample size.

**Regime**: primary point `time_mode=linear, time_lambda=0.01, maintenance_scale=0.0` —
`REGIME`'s confirmed-meaningful point on our OWN corpus (`k*=10, k_frac=0.105,
meaningful=True` in `regime_select_results.json`), used here as a starting reference only,
per direct instruction not to assume it transfers unchanged to a different corpus.
**Sanity-checked directly on `human_trees`**: `evaluate_probe_y.py` fits SingleHalt* on this
population at this regime and flags degeneracy if the fit collapses to `k=0` (Always Stop) or
`k=k_max` (Always Continue) — see Results for the actual `k*` found here.

## Results

**Status: complete.** Retry array `10846533` (all 4 indices), merge `10846537`, and eval
`10846538` all finished cleanly (`sacct`: `COMPLETED`, exit `0:0`). Raw numbers:
`outputs/figures/minply15_maxply75/diagnosis/probe_y_subtree_weighted_result.json`. Figure:
`outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/probe_y_subtree_weighted_eval.*`.

### Regime sanity check (on `human_trees`, not assumed from our own corpus)

SingleHalt* fit at `time_lambda=0.01, maintenance_scale=0.0` (the same primary point
`REGIME` confirmed meaningful on `puctvalue_md36`) lands at **k\*=8** out of a 95-step
ceiling (`k_frac=0.084`) — a real interior optimum, not collapsed to `k=0` (Always Stop) or
the ceiling (Always Continue). **Not degenerate on this population** — the regime transfers
reasonably, at least as a starting point; no re-grid was needed to get a meaningful primary
point here.

### Point estimates (n=223 held-out episodes, 70/30 fit/eval split by source tree, n_fit=522)

| method | regret | 95% CI | avg. stop step |
|---|---|---|---|
| Always Stop (k=0) | 0.3460 | [0.3143, 0.3790] | 0 |
| SingleHalt\* (k=8) | 0.2676 | [0.2378, 0.2971] | 8.0 |
| **subtree-weighted $z_t$-Controller** | **0.2625** | [0.2301, 0.2960] | 17.5 |
| Stats-Controller | 0.2701 | [0.2408, 0.3010] | 23.5 |
| Always Continue (k=max) | 0.6905 | [0.6646, 0.7161] | 95 |

Point estimate: the subtree-weighted z_t-Controller has the LOWEST regret of the three real
controllers — beating both SingleHalt\* and Stats-Controller on this reading. But (see next
section) neither margin survives a paired significance test at this sample size.

### Paired significance (percentile bootstrap, n_boot=2000, on the SAME 223 eval episodes)

| comparison | mean diff (baseline − z_t) | 95% CI | verdict |
|---|---|---|---|
| SingleHalt\* − subtree z_t | +0.0051 | [−0.0292, +0.0385] | **NOT significant** (straddles 0) |
| Stats-Controller − subtree z_t | +0.0076 | [−0.0313, +0.0473] | **NOT significant** (straddles 0) |

**Neither paired comparison excludes 0.** The point-estimate win visible in the table above
does not survive pairing — the per-episode regret differences are noisy enough (CI half-widths
~0.03–0.05, roughly an order of magnitude wider than `SIG-S`'s on `puctvalue_md36`, where
n_eval was ~1,680 vs. this population's 223) that "subtree z_t is genuinely better" and
"the three methods are indistinguishable here" are both consistent with the data. This is the
SAME verdict shape `SIG-Z` found for e2e z_t vs. SingleHalt\* on our own corpus (a promising
point estimate that a proper paired test could not confirm) — not evidence of a mechanism,
just a reminder that point estimates alone are not a result in this investigation's convention.

### Honest verdict

1. **Within-population**: the subtree-weighted encoder does NOT show a statistically
   confirmed edge over SingleHalt\* or Stats-Controller on this `human_trees` argmax-filtered
   validation split, at this regime, at this sample size. It shows a directionally favorable
   point estimate, nothing more. This is the same "trains, earns more reward than these
   baselines but not others, work remains to be done" shape Sagiv himself recommended
   reporting honestly (plan.md item 4) — except here even the "beats AlwaysStop" floor is
   the only fully solid claim; the harder bar (beats SingleHalt\*/Stats-Controller) is not
   cleared with confidence.
2. **The 0.024 figure does not replicate**, not even approximately, under this investigation's
   honest, current methodology on the SAME encoder and the SAME corpus family the historical
   number was reported on: this run measures the subtree-weighted z_t-Controller's regret at
   **0.2625**, roughly **11× the historical 0.024**. This report does not attempt a unit
   reconciliation between the old fitted-Q pipeline's regret definition/cost-scale and this
   investigation's (that is explicitly the separate, parallel git-archaeology track's job —
   see below) — so this is not a claim that 0.024 is *provably wrong* on its own terms. But
   the gap is large enough, and directionally consistent with plan.md's independent finding
   that this exact lineage's own labnotebook shows "greedy regret" at wildly different scales
   across entries over time, that the practical conclusion is the same either way: **0.024 is
   not a number that describes what this encoder actually delivers when measured today,
   under calibrated cost regimes and paired significance testing.** The more defensible
   summary of what this encoder does is the within-population result above — a real but
   statistically unconfirmed edge, not a 5-10x improvement over hand-crafted baselines.
3. **Relative to our own production encoder** (which does not significantly beat
   SingleHalt\*/Stats-Controller on `puctvalue_md36` either — `SIG-Z`): the subtree-weighted
   encoder is in the same boat, not a clearly stronger alternative. Both show a real but
   unconfirmed edge over simple baselines; neither clears the bar this investigation set.
   (Absolute regret magnitudes are NOT compared here across corpora, per this report's framing
   constraint — the point is the pattern of results, not the numbers themselves.)

**Throughput finding that changed the job shape**: an initial single-worker local smoke test
(8 episodes, `episode_batch_size=8`) took 79s — roughly 10s/episode. The subtree-weighted
encoder materializes EVERY snapshot (tree-growth step) within each episode, not just the
final tree, so an episode at the `search_budget=96` root-expansion budget costs ~96 forward
passes. Extrapolated to the full corpus (3,723 episodes), a single-worker job would take
~10 hours — far past a first attempt's 1.5h budget (that attempt, job `10845225`/`10845226`,
was cancelled before wasting the time slot). Redesigned to match
`slurm/pipeline/pack_root.slurm`'s existing array/worker pattern (this repo's own established
calibration for materialize jobs of comparable shape) instead of a single-worker job:

- `slurm/probe_y_materialize.slurm` (`--array=0-15`, 16 workers, originally `mem=6G qos=short`)
  → job **10845893**. **12/16 tasks completed; 4 (indices 1, 7, 11, 12) were OOM-killed**
  (`sacct`: `OUT_OF_MEMORY`, exit `0:125`) — caught and root-caused by the coordinator, not
  by this session's own monitoring. `sacct` `MaxRSS` showed even the 12 tasks that COMPLETED
  were riding at 6286–6292 MB against a 6 GB request — essentially zero margin cluster-wide,
  not a fluke a bare retry would fix. Fixed by bumping `--mem` to 12G (real headroom) and
  switching `--qos short → test` (each task only takes 4–10 min, well under the 1h test-queue
  ceiling, and `test` is far less contended on this cluster). Re-ran ONLY the 4 failed
  indices — the 12 that already succeeded were left as-is (their worker shard dirs confirmed
  present and valid) — via `NWORKERS=16 sbatch --array=1,7,11,12
  slurm/probe_y_materialize.slurm` (`NWORKERS=16` pins the dataset-chunk boundaries to the
  ORIGINAL 16-way split so the re-run's slices line up exactly with the 12 already-completed
  workers') → retry job **10846533**.
- `slurm/probe_y_materialize_merge.slurm` → job **10846537**,
  `--dependency=afterok:10846533` (depends only on the retry job, not the original
  partially-failed `10845893` — `afterok` on an array job requires EVERY task in it to
  succeed, so a dependency on `10845893` itself can never resolve now that 4 of its tasks
  are terminally FAILED; that exact trap stranded the first merge attempt, job `10845909`,
  in `PENDING (DependencyNeverSatisfied)` — cancelled once root-caused, never ran).
- `slurm/probe_y_eval.slurm` → job **10846538**, `--dependency=afterok:10846537`. All three
  legs of the corrected chain confirmed `COMPLETED` via `sacct`.
- Output: `outputs/figures/minply15_maxply75/diagnosis/probe_y_subtree_weighted_result.json`
  (raw numbers) and `outputs/figures/minply15_maxply75/diagnosis/{pdf,png}/probe_y_subtree_weighted_eval.*`
  (the frontier-style comparison figure, both panels, with the paired-diff verdicts in the
  figure's own title).

**Plot rendering bug hit and fixed after the eval job produced numbers** (worth recording —
not a data/methodology bug, purely cosmetic, but the task requires the plot to actually be
legible): the first render put the full long-form finding sentence (~350 characters) into
`fig.suptitle(..., wrap=True)`, which wrapped to enough lines at the given fontsize/figure
width to visually overlap the right (zoomed) panel's y-axis — matplotlib does not reflow
axes to make room for a long suptitle, and `bbox_inches="tight"` only expands the saved
canvas, it does not prevent already-positioned artists from overlapping. Confirmed by
inspecting the first rendered PNG directly. Fixed in `evaluate_probe_y.py`'s `_render` by
(a) using a short, fixed-line-count title (one line per baseline comparison, compact Δ/CI
notation, the long-form sentence kept in the printed log / JSON instead) and (b)
`fig.subplots_adjust(top=0.83)` called BEFORE placing the suptitle (calling
`fig.tight_layout(rect=...)` AFTER `suptitle()` was tried first and silently dropped the
title from the saved bbox-tight render entirely — also confirmed by inspection, not just
inferred). Re-rendered locally from the already-saved `probe_y_subtree_weighted_result.json`
(no need to re-run the SLURM chain — `_render` is a pure function of the saved data, same
`_compute`/`_render` split convention `analysis.evaluate`'s own `replot_saved` uses)
and confirmed visually correct on the second pass.

**Two config footguns hit and fixed while setting this up** (both worth knowing about for
any future new, additive config in this repo):
1. `analysis.evaluate`'s import chain eagerly evaluates `analysis.utils.helpers.CONFIG =
   load_config_section("human_analysis")` and a module-level default argument
   (`db_connection`'s `database=CONFIG["selected_db_default"]`) at IMPORT time — any config
   missing a `human_analysis:` section with at least `selected_db_default` crashes on
   `import analysis.evaluate`, independent of whether that code path is ever exercised. Fixed
   with a minimal stub in `config_probe_y_subtree.yaml` (this also crashed Agent 1's
   `config_ysagiv_argmax.yaml` independently — not fixed here, not our file to touch).
2. `slurm/helpers/setup_env.sh` (shared by every pipeline config) unconditionally queries
   `globals.config_dir` and `encoder.output_checkpoint` regardless of which stages a given
   run actually needs — a config that (correctly, per this probe's scope) omits `encoder:`
   entirely because it loads an existing checkpoint rather than pretraining one still needs
   stub values for these two keys or `setup_env.sh` itself fails before the run even starts.

## On the historical 0.024 figure

Addressed directly above (see "Honest verdict," point 2): it does not replicate, not even
approximately — this run measures ~0.26 against the historical ~0.024, roughly 11×. A full
unit reconciliation between the old fitted-Q pipeline's cost-scale/regret-definition and this
investigation's is out of scope here — that reconciliation is a separate, parallel
investigation (run concurrently, per the coordinator), doing git/labnotebook archaeology into
the *original* fitted-Q pipeline's own cost-scale and regret definition, to establish whether
0.024 is trustworthy even on its own historical terms. That is a different, narrower question
than this probe's
(whether the same *encoder* shows a real relative edge under current, honest, apples-to-apples
measurement) and the two lines of evidence are kept separate deliberately: a clean relative
result here does not vindicate 0.024's absolute value, and a debunking of 0.024's own
methodology does not by itself tell us whether the encoder is any good. Once both land, they
should be read side by side, not merged into one number.
