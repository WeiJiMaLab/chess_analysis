# Assessing the `z_t` halting head: does the learned embedding buy anything?

**Ref:** `R-EVALUATE` · [Index](reference.md)

> **Status:** 📝 active — how to *assess* the MCHalt meta-controller's pure-`z_t` head, and whether it has
> learned anything useful. Headline: **yes — in an adequate cost regime the `z_t` head significantly beats the
> tree-stats head and the best fixed stop** (up to −132 regret, paired CI excludes 0). The shipped power-law cost
> regime is *degenerate* (all tiers tie — a poorly-set-up profile, not a model failure); switch to a constant-cost
> profile that isolates **R** (the value of continued search) and the tier ladder separates as predicted. A
> decodability probe shows why: R is estimable only from `z_t`. Prototype-grade, single training seed. Figures +
> mechanism: `outputs/figures/minply15_maxply75/mchalt_diagnosis/` ← produced by `src/analysis/evaluate.py`.

## The problem

The deployed controller reads a pure `z_t` head (`controller_inputs=['z_t','T_t']`); the tree-stats baseline reads
`[height, width, n_nodes]`. Two things made the head *look* worthless — both turned out to be **assessment
artifacts**, not model failures:

1. **The PG readout froze** — the soft stop policy `σ(A_t)` saturated to 0/1 in epoch 1, killing the gradient.
   Fixed by a `stop_temperature` anneal 4→1, LR 3e-4, a minimal 32×2 head; soft `E[regret]` now trains 696→209.
2. **After the fix, every readout tied NeverStop** — but only because the *cost regime* was degenerate and the
   `z_t` fit was undertrained (both resolved below).

## The assessment framework

1. **Make the halt problem non-degenerate.** The oracle stop / regret is a function of the cost config; the packed
   `halt_rewards`/`tree_sizes` are cost-*independent*, so any regime is evaluated at analysis time (no repacking).
   In the shipped regime (power-law time cost, ~free until the budget runs out) NeverStop is near-optimal → nothing
   to learn. Only a regime where the optimal stop is interior is *assessable*.
2. **Fixed-stop frontier in (search-effort, regret) space.** For each "stop at step `k`" rule plot
   `(mean stop step, mean regret)`; its endpoints are exactly AlwaysStop (k=0) and NeverStop (k=max_len) — asserted
   in code (KGRID must span the longest episode or the frontier is truncated and lies). The minimum is the best
   *position-independent* policy.
3. **Beat-the-frontier test.** A learned readout is one point `(mean stop, regret)`; below the frontier = lower
   regret for the same search effort, i.e. position-dependence a fixed rule can't buy.
4. **Representational edge.** Whether `z_t` *can* beat tree-stats reduces to: does it predict the eventual child
   reward (the encoder's target) better than structure does?

## The stopping-value decomposition

Each tier can estimate one term of the stopping value, so each *should* buy a specific edge:

| term | who can estimate it | expected edge |
|---|---|---|
| **C-step** (time cost ∝ steps) | fixed-k (sees step) | beats always/never |
| **C-maint** (∝ n_nodes) | stats readout | beats best-fixed *iff tree size varies at fixed step* |
| **R** (value of continued search) | `z_t` only (needs eventual value) | beats stats — *the* edge |

Design note (baked into `evaluate.py`): the readouts see **steps-taken, not budget**, and all fitting is on the
validation split (70/30 episode split) because the train materialized cache is `shuffle=True` and its `z_t` rows
don't align to episodes; validation is `shuffle=False` and does (alignment self-checked).

## Findings

**R is estimable — and only from `z_t`.** Target `R(t) = max_{s≥t} halt_reward(s) − halt_reward(t)` (cost-free
gain still achievable). Held-out MLP R²: steps 0.027, stats 0.033, **`z_t` 0.139**, `z_t`+stats 0.149, shuffle 0.0.
So R lives ~4× more in the embedding than in structure/steps, and stats add nothing on top. It is *nonlinear*
(Ridge R² for `z_t` is only 0.016) — which is exactly why a linear/undertrained head can't use it.

![R-decodability: R(t)=value of continuing is decodable ~4× better from z_t than from tree-stats/steps, and only nonlinearly](../figures/minply15_maxply75/mchalt_diagnosis/png/decodability.png)

**In an adequate regime the ladder separates.** The shipped **power-law regime is degenerate** (all four tiers =
148, every paired gap exactly 0). Switching to a **constant-cost (`time_mode=linear`) profile that isolates R**
(15k val, budget-out/steps-in, proper nonlinear fits, paired CIs):

| regime | never | best-fixed | stats | `z_t` | `z_t` − stats (paired 95% CI) |
|---|---:|---:|---:|---:|---|
| power-law (shipped) | 148 | 148 | 148 | 148 | **0** (degenerate) |
| linear λ=5 | 289 | 261 | 256 | **195** | **−61 [−85, −38]** ✔ |
| linear λ=10 | 432 | 327 | 338 | **207** | **−132 [−163, −102]** ✔ |
| linear λ=40 | 1295 | 377 | 384 | **225** | **−159 [−202, −117]** ✔ |

`z_t`≻stats is significant in 5/6 linear regimes; `fixed`≻`never` (C-step) is significant everywhere. In the
frontier below (linear λ=10), the `z_t` point sits well *below* the fixed-stop frontier while tree-stats sits *on*
it (a fixed rule in disguise):

![Adequate regime (linear cost, λ=10): the z_t head sits below the fixed-stop frontier and the tree-stats point](../figures/minply15_maxply75/mchalt_diagnosis/png/frontier.png)

The tier-separation view — the reward readout relative to the fixed baseline across regimes. The `z_t`−stats
(R edge) drops significantly below 0 in the linear regimes; `stats`−best-fixed (C-maint edge) only bites when a
small per-node cost is added (`maintenance_scale`≈0.05), and even then it is weak; the power-law base is flat:

![Tier separation vs cost regime: z_t−stats (R edge) significant in the linear regimes; C-maint only with maintenance; power-law degenerate](../figures/minply15_maxply75/mchalt_diagnosis/png/delta_mean_regret.png)

**The representational substrate is real** (encoder-vs-structure probe, 2.49M held-out edges). Reducible child-WDL
KL = 0.854 nats: a structure-only MLP explains **0.055 nats** beyond a constant (~6%); the encoder embedding
explains **0.491 nats beyond structure = 57.5%** (encoder loss_gap 0.31 vs structure 0.80). The eventual-reward
signal genuinely lives in `z`, not in structure, and the encoder learned most (not all) of it.

**`z_t` needs the nonlinear head + adequate training.** Fitting lr 1e-2/30 ep leaves it near AlwaysStop (41%
stop-at-0 — the old "collapse"); lr 1e-3/200 ep surfaces the edge. The deployed `pg_controller_train` (LR 3e-4,
20 ep) sits in the collapse range — retuning it is the cheapest lever on the real controller.

## Verdict

**The `z_t` head has learned something real and useful.** Assessed correctly — non-degenerate regime, fixed-stop
frontier as the bar, adequate nonlinear training — it **significantly** beats the tree-stats head and the best
fixed stop, the R-decodability probe supplies the mechanism (R is estimable only from `z`), and the encoder probe
supplies the substrate (57.5% of child-WDL KL). The earlier "worthless / within noise" readings were a *degenerate
cost profile* plus an *undertrained fit*. The honest boundary: this is a proof-of-concept in a constructed
R-isolating regime, single training seed.

## Caveats & open work

- **The adequate regime is constructed, not the deployed one.** `linear` constant cost isolates R for the PoC; the
  shipped power-law profile is degenerate. The open *design* question is the right deployable cost profile (and its
  shadow-price λ — below), not whether the head can learn.
- **C-maint edge is weak and regime-fiddly.** `n_nodes` is *not* purely `f(steps)` — step alone predicts only
  R²≈0.79 of it (within-step CV≈0.24), so ~21% varies at a fixed step. A maintenance sweep gives stats a *small*
  significant edge over best-fixed at some scales (−27 @ 0.05, −11 @ 0.30) but not others (≈0 @ 0.10, reversed @
  1.0): cranking maintenance up doesn't monotonically help. It is real but ~5× weaker than the R edge.
- **Single training seed.** Paired CIs are over eval episodes, not fit seeds; multi-seed replication would fully
  nail significance (effects are large and consistent across 5 λ, so robust).
- **Per-game budget / opportunity cost.** The per-move cost λ is really the *shadow price* of a per-game time
  budget (water-filling: equalize marginal R across a game's moves). λ can be *derived*, not tuned — the quantile
  of the R distribution matching a target think-time — turning "stop when R(t) < λ" into a principled rule.
- **Encoder only partially fits child-WDL** (loss_gap 0.31) — larger `d_embed`/`k` may raise the R ceiling.

## Reproduce

All three figures are produced by the canonical module (mechanism version-controlled in `src/analysis/evaluate.py`;
the old five-tier ladder was removed — only these trusted plots remain):

```
python -m analysis.evaluate --which all \
  --packed-root <mc_packed> --cache <validation_cache.pt> \
  --out-dir outputs/figures/minply15_maxply75/mchalt_diagnosis
```

or the pipeline step `sbatch slurm/pipeline/eval.slurm`. `--which {frontier,decodability,separation}` runs one;
`--time-mode`/`--time-lambda` set the frontier's cost regime (default `linear` λ=10, the adequate regime). Figures
use bundled Helvetica Neue (`fonts/`). Supporting one-off probes (scratch `/scratch/.../_proto/`): the
encoder-vs-structure child-WDL probe and the maintenance-edge sweep. Full-split topology decodability from `z_t`
(a separate question) lives in `python -m cts.analysis.zt_probe`.
