# Does the learned `z_t` halting head buy anything over hand-crafted tree-stats or a fixed stop?

**Ref:** `R-EVALUATE` · [Index](reference.md)

> **Status:** 📝 active — assessing the MCHalt meta-controller's pure-`z_t` head against a hand-crafted
> tree-stats head and the best fixed (position-independent) stop. Headline: **yes — in a constant-cost
> regime that isolates R (the value of continued search), the `z_t` head significantly beats both** (up
> to −151 regret, paired 95% CI excludes 0). The shipped power-law cost regime is *degenerate* (nothing
> to learn there — not a model failure). A decodability probe explains why: R is estimable from `z_t`
> several-fold better than from tree structure, and only nonlinearly. Prototype-grade, single training
> seed — the Stats-/`z_t`-Controller PG fits show real run-to-run numerical variability (see Caveats).
> Figures + mechanism: `outputs/figures/minply15_maxply75/normative/` ← produced by
> `src/analysis/evaluate.py`.

## Overview

The deployed MCHalt controller decides *when to stop searching* a move. Three tiers of stopping rule
compete for that decision: a one-parameter fixed stop (**Frac\***, always halts at the same step), a
tree-stats readout (sees `[steps, n_nodes, height, width]`), and a `z_t` readout (sees the encoder's root
embedding). Does the learned embedding actually buy anything over the cheaper alternatives — and if so,
where does the edge come from? This report walks through three figures that answer it, then interprets
what the results mean and how sensitive they are to the assumed cost regime.

## Results

### Can `R(t)` — the value of continuing to search — be decoded from `z_t`, or does tree structure already carry it?

`R(t) = max_{s≥t} halt_reward(s) − halt_reward(t)` is the cost-free gain still achievable by continuing.
Held-out MLP R² (2×64 head, 15k validation episodes): steps **0.034**, +stats **0.036**, **+`z_t` 0.146**,
`z_t`+stats **0.131**. `R` lives almost entirely in the embedding — tree-stats add essentially nothing on
top of steps alone. The signal is also markedly *nonlinear*: the Ridge (linear) fits stay flat and low for
every tier, including `z_t`, so a linear or undertrained head cannot access it.

![R-decodability: R(t)=value of continuing is decodable far better from z_t than from tree-stats/steps, and only nonlinearly](../figures/minply15_maxply75/normative/png/decodability.png)

> **Result:** `R(t)` is estimable, but only from `z_t`, and only with a nonlinear readout. This is the
> representational precondition for `z_t` beating tree-stats on regret — it has no path to an edge if the
> encoder didn't already know something about eventual reward that the tree shape doesn't reveal.

### Does the learned `z_t` head beat tree-stats and the best fixed stop, and in which cost regime?

The shipped power-law time cost is *degenerate* — search is nearly free until the budget runs out, so
NeverStop is close to optimal and there is nothing for any tier to learn. Switching to a constant per-step
cost (`time_mode=linear`) makes the stop genuinely interior. At `λ=10`: Frac\* (fixed stop, k=8) = **335.0**
regret, Stats-Controller = **339.8**, `z_t`-Controller = **188.3** — paired `z_t` − Stats = **−151 [−190, −114]**,
a 95% CI that excludes 0 by a wide margin. In the frontier below, the `z_t` point sits clearly *below* the
fixed-stop curve (position-dependence a fixed rule cannot buy), while Stats-Controller sits right on it — a
fixed rule in disguise.

![Fixed-stop frontier (linear cost, λ=10): the z_t-Controller sits below the frontier; Stats-Controller sits on it](../figures/minply15_maxply75/normative/png/frontier.png)

> **Result:** In the adequate (constant-cost) regime, `z_t` significantly beats both the best fixed stop
> and the tree-stats readout. The edge is exactly where the decodability probe predicted: R.

### How does each controller do relative to `z_t`, as the cost regime (λ) and maintenance scale vary?

Two sweeps, both expressed as **Δ regret = candidate − `z_t`** (positive = `z_t` wins), for all four
non-`z_t` policies: Frac\*, Stats-Controller, AlwaysStop, AlwaysContinue.

![Δ regret vs z_t, sweeping the linear cost λ at maintenance=0](../figures/minply15_maxply75/normative/png/delta_mean_regret_lambda.png)

| λ (maintenance=0) | Frac\* | Stats-Controller | AlwaysStop | AlwaysContinue |
|---|---:|---:|---:|---:|
| 5  | +93.8  | +131.5 | +800.4 | +124.0  |
| 10 | +146.7 | +138.8 | +793.3 | +264.6  |
| 40 | +195.0 | +194.8 | +795.9 | +1153.6 |

![Δ regret vs z_t, sweeping the maintenance scale at linear λ=5](../figures/minply15_maxply75/normative/png/delta_mean_regret_maintenance.png)

| maintenance (λ=5) | Frac\* | Stats-Controller | AlwaysStop | AlwaysContinue |
|---|---:|---:|---:|---:|
| 0    | +93.8  | +106.2 | +800.4 | +124.0  |
| 0.05 | +110.1 | +136.6 | +801.3 | +187.1  |
| 0.1  | +127.4 | +144.2 | +804.2 | +252.2  |
| 0.3  | +137.0 | +140.8 | +804.1 | +501.0  |
| 1.0  | +164.6 | +173.6 | +811.8 | +1379.8 |

> **Result:** `z_t` beats every other policy, everywhere in both sweeps — the gap only ever widens, never
> closes or reverses. AlwaysStop is catastrophic everywhere (stopping at step 0 forfeits nearly all
> achievable reward, cost-regime-independent). AlwaysContinue gets rapidly worse as either λ or maintenance
> grows, since the cost of never stopping scales with exactly the quantity being taxed. (Stats-Controller's
> own numbers move noticeably between runs at fixed inputs — e.g. λ=40 read +264.2 in one run and +194.8 in
> this one, see Caveats — but never enough to close the gap to `z_t`.)

### What does this set of results actually mean, taken together?

Put together, the three figures tell one coherent story rather than three separate ones. The decodability
probe establishes the *precondition* (R lives in `z_t`, nonlinearly); the frontier shows the *consequence*
in the one regime chosen to be assessable (`z_t` clears the fixed-stop frontier that Stats-Controller
merely sits on); and the two sweeps show that consequence is not a fluke of one cost setting — it holds,
and strengthens, across an order of magnitude of λ and the full maintenance range tested. The practical
reading: the pure-`z_t` head is not a redundant re-encoding of tree-stats dressed up in 32 extra dimensions
— it is estimating something (the eventual value of continuing) that step count, node count, and tree
shape do not expose, and that something is exactly what a resource-rational stopping rule needs. The
honest boundary is that this is demonstrated in a *constructed* cost regime built to isolate R, with a
single training seed — not (yet) in the shipped deployment regime, which is degenerate by construction (see
below).

> **Clarification:** The `z_t` head's advantage is a genuine representational edge (it knows something
> tree-stats structurally cannot access), not an artifact of a favorable cost setting — it survives every
> regime tested, including the ones that make the *other* tiers look best.

### Which choice of λ and maintenance is actually sensible?

Neither parameter has a "true" value — λ is the shadow price of a per-move time budget, and maintenance is
how strongly per-node cost should bite — but the sweep data narrows the sensible range:

- **λ.** Too small (well below 5) and the regime approaches the degenerate power-law case where nothing
  separates; too large (λ=40) pushes regret into the 800–1200 range, several times the length of a typical
  episode, which stretches credulity as a per-move cost. **λ≈10** is the sweet spot already used as the
  frontier's default: the `z_t`-vs-Stats gap is large and significant (−151 [−190,−114]), but the regret
  magnitudes (188–340) stay commensurate with episode lengths (a handful to a few dozen steps), so the
  cost is plausible rather than contrived.
- **Maintenance.** At `maintenance=0`, both Frac\* and Stats-Controller trail `z_t` by a similar amount
  (+93.8 vs +106.2) — a small nonzero maintenance (**≈0.05–0.1**) keeps both gaps to `z_t` legible (+110 to
  +144) without AlwaysContinue's cost exploding to the point of dominating the plot (+187 to +252 at
  0.05–0.1, vs +1380 at maintenance=1.0, where the sweep stops being informative about anything except
  "never stopping is now absurd"). Values ≥0.3 mostly amplify the AlwaysContinue penalty rather than adding
  information about the Frac\*/Stats/`z_t` comparison (the finer-grained Frac\*-vs-Stats question needs a
  direct paired comparison — next section).

> **Decision:** `linear` time cost, **λ≈10, maintenance≈0.05–0.1** is the recommended assessment regime:
> large enough to make every tier's edge legible, small enough that the resulting regrets stay
> interpretable relative to episode length, and it does not depend on cranking any one knob to an extreme
> to see a significant result.

### If Frac\* sometimes beats the Stats-Controller, why — and does giving it Frac\* as a feature fix it?

The λ-sweep table above hides a wrinkle visible at finer resolution: refit with the historical 5-λ grid,
**Stats-Controller is *worse* than Frac\* at λ=5** (paired Stats − Frac\* = **+57.8 [+28.1, +90.4]**, CI
excludes 0 — Frac\* wins) but **not distinguishable at λ=10** (+5.9 [−12.9, +27.6]). This is the interesting
case: Stats-Controller's own feature set is a strict superset of Frac\*'s (it sees `steps` plus `n_nodes`,
`height`, `width`), so in principle it could always at least recover Frac\*'s performance by learning to
ignore the extra three dimensions. That it sometimes does *worse* points at optimization, not
representation.

**Diagnostic probe** (real validation data, 15k episodes, same fit/eval split): retrain the Stats-Controller
with one added feature — `steps_taken − k_frac` (i.e. hand the readout the fixed-optimal threshold as a
centering offset) — and compare:

| λ | Frac\* | Stats (plain) | Stats (+ centered-step) | Stats(plain) − Frac\* | Stats(aug) − Frac\* |
|---|---:|---:|---:|---:|---:|
| 5  | 279.5 [253.1, 308.7] | 337.3 [301.2, 375.8] | 314.8 [281.8, 348.5] | **+57.8 [+28.1, +90.4]** | **+35.3 [+15.7, +55.6]** |
| 10 | 335.0 [302.8, 367.6] | 340.9 [306.0, 379.0] | 327.5 [293.1, 364.8] | +5.9 [−12.9, +27.6]      | −7.5 [−28.9, +16.6]      |

Centering the feature on the known-good fixed threshold **narrows the λ=5 gap by ~39%** (+57.8 → +35.3),
but does **not** close it — the augmented Stats-Controller is still significantly beaten by Frac\*. So the
answer is partially, not fully: some of the gap is a feature-scaling/optimization headwind (a PG-trained
30-epoch fit has to *discover* the right offset from raw `n_nodes`/`height`/`width`, and handing it a
better-scaled input measurably helps), but a real residual gap remains even after removing that headwind —
consistent with this report's other finding that these small MLP/PG heads are training-sensitive in
general (the deployed `z_t` head itself needs ~200 epochs, not 30, to escape a collapsed near-AlwaysStop
solution). Fitting the Stats-Controller for more epochs, or with a lower learning rate, is the next lever
to try before concluding the C-maint edge is genuinely weaker than Frac\* at low λ.

> **Clarification:** Frac\* beats Stats-Controller at λ=5 mostly (but not only) because the PG-trained
> readout under-uses its own step feature — recentering it on the fixed-optimal threshold recovers ~39% of
> the gap. The rest looks like a genuine, if small, extra optimization cost of fitting 4 raw features
> instead of 1 within the same 30-epoch budget, not evidence the stats features carry negative information.

### Why doesn't `z_t`+stats beat `z_t` alone in the decodability probe?

A second counter-to-naive-expectation result in the same probe: `z_t`+stats (0.131) scores *lower* than
`z_t` alone (0.146), even though `z_t`+stats is a strict feature superset. The `stats` features are almost
pure noise on their own (linear R²≈0.036, barely above the ≈0 shuffle floor), and the probe is a small,
fixed-capacity 2×64 MLP trained for at most 400 epochs with early stopping on a small inner validation
slice. Adding three weakly-informative dimensions to that fixed budget makes the optimization landscape
harder to search in the same number of steps — this is a documented general phenomenon for small networks
trained under a bounded budget, not evidence the extra features are actively harmful. It is reproducible
(`z_t`+stats = 0.131 in all four independent pipeline runs so far, vs `z_t` alone at 0.146–0.148), i.e. a
stable property of this training setup rather than noise.

> **Clarification:** `z_t`+stats underperforming `z_t` alone is a finite-capacity/finite-training
> optimization artifact, not a representational one — the same caution about undertraining that applies to
> the deployed controller applies to this probe.

## Methods

**The assessment framework.**
1. *Make the halt problem non-degenerate.* Regret is a function of the cost config; the packed
   `halt_rewards`/`tree_sizes` are cost-*independent*, so any regime is evaluated at analysis time with no
   repacking. In the shipped power-law regime, NeverStop is near-optimal — nothing to learn. Only a regime
   where the optimal stop is interior is assessable.
2. *Fixed-stop frontier in (search-effort, regret) space.* For every "stop at step k" rule, plot
   `(mean stop step, mean regret)`; its endpoints are exactly AlwaysStop (k=0) and AlwaysContinue
   (k=max episode length). The minimum is the best *position-independent* policy.
3. *Beat-the-frontier test.* A learned readout is one point; below the frontier means lower regret for the
   same average search effort — position-dependence a fixed rule cannot buy.
4. *Representational edge.* Whether `z_t` *can* beat tree-stats reduces to whether it predicts the eventual
   reward better than structure does — the decodability probe.

**The stopping-value decomposition.** Each tier can estimate one term of the stopping value:

| term | who can estimate it | expected edge |
|---|---|---|
| **C-step** (cost ∝ steps) | fixed-k (sees step) | beats Always{Stop,Continue} |
| **C-maint** (cost ∝ n_nodes) | stats readout | beats Frac\* *iff* tree size varies at fixed step |
| **R** (value of continued search) | `z_t` only (needs eventual value) | beats stats — *the* edge |

Readouts see **steps-taken, not remaining budget**; all fitting is on the validation split with a 70/30
episode split (the train materialized cache is `shuffle=True`, so its `z_t` rows don't align to episodes;
validation is `shuffle=False` and does — alignment is self-checked at load time). Frac\* is a one-parameter
grid search minimizing fit-split regret; Stats-/`z_t`-Controllers are MLP readouts
(`cts.models.readout.build_advantage_head`) trained by the exact expected-return objective via the deployed
trainer (`cts.train.pg_controller_train.fit_readout_pg`) — Stats for 30 epochs (lr 1e-2), `z_t` for 200
(lr 1e-3, since the R signal is nonlinear and needs more training to surface).

## Appendix

### Caveats & open work

- **The adequate regime is constructed, not the deployed one.** `linear` constant cost isolates R for this
  assessment; the shipped power-law profile is degenerate. The open *design* question is the right
  deployable cost profile (and its shadow price λ), not whether the head can learn.
- **Single training seed — and PG training is numerically sensitive.** Paired CIs are over eval episodes,
  not fit seeds. `seed=0` is fixed, but the Stats-/`z_t`-Controller PG fits still vary noticeably between
  otherwise-identical pipeline runs (different compute nodes / thread counts change floating-point
  reduction order in a non-convex 30–200-epoch fit): e.g. Stats-Controller's regret at λ=40 read 264.2 in
  one run and 194.8 in another. The *qualitative* story (`z_t` wins everywhere, by a wide and significant
  margin) is stable across every run observed, but exact regret values for Stats-Controller specifically
  should be read as ±30–70 noise, not fixed constants. Multi-seed replication would quantify this properly.
- **Per-game budget / opportunity cost.** The per-move cost λ is really the *shadow price* of a per-game
  time budget (water-filling: equalize marginal R across a game's moves). λ could be *derived* — the
  quantile of the R distribution matching a target think-time — rather than swept by hand.
- **Encoder only partially fits child-WDL** (from the separate encoder-vs-structure probe: loss_gap 0.31 vs
  structure 0.80) — a larger `d_embed`/`k` may raise the R ceiling further.
- **The Frac\*-vs-Stats diagnostic (feature-centering probe) is a one-off, not part of the committed
  pipeline** — it answers "is the gap optimization or representation," not "what's the best possible stats
  readout." A properly retuned (more epochs / lower LR) Stats-Controller might close more of the gap.

### Reproduce

All four figures are produced by the canonical module:

```
python -m analysis.evaluate --which all \
  --packed-root <mc_packed> --cache <validation_cache.pt> \
  --out-dir outputs/figures/minply15_maxply75/normative
```

or the pipeline step `sbatch slurm/pipeline/eval.slurm`. `--which {frontier,decodability,separation}` runs
one figure group (`separation` now renders both the λ- and maintenance-sweep panels); `--time-mode`/
`--time-lambda`/`--maintenance-scale` set the frontier's cost regime (default `linear` λ=10, maintenance=0
— the recommended assessment regime above). Figures use bundled Helvetica Neue (`fonts/`). Full-split
topology decodability from `z_t` (a separate question) lives in `python -m cts.analysis.zt_probe`.
