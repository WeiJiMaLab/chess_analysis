# lmcos_tiny — do learned halt policies beat blind stopping on the SF-2000 budgeted oracle?

**Ref:** `R-LMCOS-TINY` · [Index](reference.md)

## The question

Given a **cost-aware stopping problem** — at each search expansion you may halt (lock in the
current best move) or pay a per-step cost and continue — can a **learned readout** that decides
when to stop beat the trivial and one-knob baselines? And does a **learned GNN representation**
(`z_t`) beat **hand-crafted tree statistics** (height, width, node count) at this decision?

Trees are Stockfish Elo-2000 searches on 50K human-game root FENs (the `lmcos_tiny` pipeline).
The oracle (`cts.data.preprocess_mc.oracle`) solves the per-episode halt/continue MDP by backward
DP, giving the optimal stop step **OSS** and the optimal return `oracle_value`. For a policy that
stops at step `s`, the **return** is the cost-adjusted `g(s) = value(move@s) − Σ_{t<s} cost(t)`,
and **regret = oracle_value − g(s)** (the gap to the cost-adjusted peak — *not* a pure value gap;
over-searching is penalised because cost outruns the marginal value).

> **Result:** PG-trained readouts clearly beat the blind baselines — but the **hand-crafted
> tree-stats readout beats the learned GNN-z controller**: mean regret **0.104 [0.101, 0.107]**
> vs **0.154 [0.151, 0.158]** (non-overlapping bootstrap 95% CIs), at *less* compute (16.4 vs
> 22.8 expansions). The tiny child-WDL `z_t` embedding is a **worse stopping signal than raw tree
> structure** — the learned representation does not earn its keep for the halting decision.
> *(Single rung elo2000, single GNN-z run — see caveats.)*

## The five policies

All five emit a per-step advantage `A_t` and use ONE decision rule — stop at the first step with
`A_t ≤ 0` — so the comparison isolates the *advantage signal*:

| tier | policy | inputs | fit |
|---|---|---|---|
| 1 | Always-Stop | — | none (stop@0) |
| 2 | Never-Stop | — | none (full budget) |
| 3 | Fraction-θ | `θ·B` | one scalar θ on train regret |
| 4 | Readout(tree-stats) | `[height, width, n_nodes, T_t]` | **policy gradient** |
| 5 | Readout(GNN-z) | `[z_t, T_t]` | **policy gradient** (checkpoint) |

Tiers 4 and 5 are trained by the **same** objective (below) and both read the per-step remaining
budget `T_t`, so **4-vs-5 isolates the representation** (hand-crafted stats vs learned `z_t`).

## Training: policy gradient on the exact expected return

The deployed objective — regret of the greedy `first A≤0` rule — is a *threshold-crossing* of the
advantage trace and has no usable gradient in the head weights. The earlier readout fit a
**differentiable surrogate** (advantage-MSE + a sign-BCE), then selected the operating point on
regret. Here we instead optimise regret **directly**: treat the controller as a stochastic stop
policy (continue w.p. `σ(A_t)`); because the full halt-reward trace is known offline, the stop-step
distribution and expected return are written in **closed form** (the policy gradient with the stop
step marginalised analytically — no REINFORCE sampling):

```
P(stop@s) = (∏_{u<s} σ(A_u))·(1 − σ(A_s));   E[return] = Σ_s P(stop@s)·g(s);   loss = E[regret]
```

This is differentiable (the soft stop distribution replaces the hard argmax); checkpoints are still
**selected on the hard greedy val regret**, so the numbers are directly comparable to the baselines.
Implementation: `cts.train.pg_controller_train` (GNN-z) and the inline tier-4 fit in
`cts.analysis._budgeted.evaluate` (`--stats-objective pg`), both vectorised (batched closed form).

## Results (validation split, n = 23,380 episodes)

| model | mean regret | 95% CI | P(stop=OSS) | mean expansions | stop bias |
|---|---|---|---|---|---|
| Always-Stop (θ=0) | 0.497 | [0.489, 0.505] | 0.47 | 0.0 | −11.2 |
| Fraction θ\*=0.25 | 0.196 | [0.192, 0.200] | 0.17 | 8.0 | −3.2 |
| **Readout(tree-stats, PG)** | **0.104** | **[0.101, 0.107]** | 0.22 | 16.4 | +5.3 |
| Readout(GNN-z, PG) | 0.154 | [0.151, 0.158] | 0.28 | 22.8 | +11.6 |
| Never-Stop (θ=1) | 1.379 | [1.370, 1.387] | 0.08 | 30.1 | +19.0 |

![Regret vs compute](../figures/lmcos_tiny/regret_vs_compute.png)

The Fraction-θ family traces a U in (compute, regret); its optimum (the ★) is θ\*≈0.25. Both learned
readouts sit **below** the curve (better than blind fraction at their compute), and **tree-stats is
furthest below** — lower regret *and* less compute than GNN-z.

**Reading it:**
- **PG beats the surrogate.** The PG GNN-z controller (0.154) beats the earlier advantage-MSE
  controller (~0.20) and the Fraction-θ optimum (0.196) — directly optimising regret pulls the
  stop earlier (over-search bias +11.6 vs the MSE controller's ~+14) and lowers regret.
- **Hand-crafted features beat the learned embedding.** tree-stats (0.104) dominates GNN-z (0.154)
  on mean (non-overlapping CIs), tail (median 0.016 vs 0.077; p90 0.265 vs 0.334), *and* compute.
- **Why (hypotheses, not settled):** the stop decision is largely "how far have I searched / what
  shape is the tree", which `height/width/n_nodes` encode directly, whereas a *value*-pretrained
  `z_t` may not; and the encoder is a tiny POC (k=1, d_embed=32) that may lack capacity for a
  stopping-relevant signal. GNN-z also over-searches more (+11.6 vs +5.3).

## Caveats

- **Single rung (elo2000), single GNN-z training run.** Before treating "tree-stats > GNN-z" as
  settled: re-run GNN-z across seeds and check 1800/2200.
- The encoder is the tiny child-WDL POC; a larger / structure-aware encoder might change the verdict.
- Direct test of the "z_t lacks tree-size" hypothesis: give the GNN-z readout `n_nodes` explicitly.

## Methods / reproduce

Pipeline `lmcos_tiny/` (`pipeline/submit_all.sh`): treegen → PUCT∩monotone filter → split/gnn_pack/
mc_pack → encoder → materialize z_t cache → **PG readout** (`3_train_readout_pg.slurm`) → eval.
Eval (regenerates the table + figures + per-model 95% CIs):

```bash
python -m cts.analysis._budgeted.evaluate \
  --packed-root /scratch/.../sf_mc_packed/elo2000 --out-dir figures/lmcos_tiny \
  --stats-objective pg --controller-device cpu \
  --controller-checkpoint /scratch/.../sf_packed/elo2000/sf_mchalt_pg.pt \
  --materialized-validation-cache /scratch/.../sf_mc_materialized/elo2000/validation_cache.pt \
  --results-json figures/lmcos_tiny/sf_elo2000_results.json
```
