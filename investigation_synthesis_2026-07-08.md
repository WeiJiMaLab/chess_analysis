# Can a learned metacontroller beat fixed-stop baselines? — synthesis as of 2026-07-08 evening

Paused for review at the user's request. This pulls together every verified result from this
investigation's diagnosis phase and Phase 2 (the overnight multi-agent fan-out). Everything below
was independently checked against the underlying files/JSON/SLURM logs before being included here
— nothing is relayed from an agent's self-report without verification. Individual track write-ups
are linked throughout; this document is the cross-track synthesis, not a replacement for them.

## The question

Does a GNN-embedding-based controller (`z_t`), trained to decide when to stop tree search, ever
reliably beat: (a) SingleHalt\* (the best single fixed stopping step, fit in hindsight), or
(b) Stats-Controller (a 3-number tree-shape heuristic, architecturally *less* informed than z_t)?
Both are the "real bar" — beating AlwaysStop/AlwaysContinue is a much lower, less interesting bar
that most methods clear trivially.

## Bottom line

**Yes, but only in one specific, now well-characterized setting** — and it is not yet the setting
our own production corpus or pipeline uses by default. Everywhere the win shows up, it is on the
**ysagiv `human_trees` corpus, filtered by `exclude_xaba`** (a churn-motif filter that drops trees
whose root-best-move trace thrashes and revisits an earlier move). Nowhere else — not on our own
corpus under any filter tried, not on ysagiv under the argmax filter, not via more/longer training
of the frozen-encoder PG head — did z_t confirm a win over SingleHalt\* that survived a second look.

## Track-by-track results

| Track | Verdict | Detail |
|---|---|---|
| **Diagnosis phase** (`SIG`, `CP`, `REGIME`, `G`) | Closed | Stats-Controller confirmed beats SingleHalt\* on our corpus (+0.0067, CI [+0.0026,+0.0112]). z_t does not, at baseline. `REGIME` established the valid cost-regime band for our corpus (`maintenance_scale` ≤0.005, `time_lambda` roughly [0.0001,0.017] at m=0) — used throughout everything downstream. |
| **Agent 1 — ysagiv filter comparison** | Done | Same corpus, two filters, opposite outcomes. **xaba-filtered**: z_t ties SingleHalt\*/Stats at 2 of 3 regimes, wins clearly at the third (see Agent 4 below for why that's not an artifact). **argmax-filtered** (our own filter, applied to ysagiv): z_t loses significantly at both non-degenerate regimes (−0.12 to −0.24 vs. both baselines), and the shared λ=0.0005 regime collapses to a degenerate 3-way tie rather than showing xaba's genuine separation. Filter choice, not just corpus, determines the outcome. [`ysagiv_trees.md`](ysagiv_trees.md) |
| **Agent 4 — ysagiv xaba λ-regime sweep** | Done — **the strongest result in the investigation** | Tested whether the original xaba win (at a regime where SingleHalt\* sits right at a 96-expansion right-censoring ceiling, k\*=95) was a real effect or a comparison-against-a-degenerate-baseline artifact. It's real: z_t significantly beats both baselines not only there but at λ=0.0015 (k\*=47, dead-center of the meaningful cost-regime band, margin **+0.0672** vs. SingleHalt\* — *stronger* than at the ceiling) and λ=0.002 (k\*=7, interior, +0.0464). The effect only fades to a tie once k\*≤4. Determinism-checked (reproduced Agent 1's numbers exactly on re-run). [`ysagiv_xaba_regime_sweep.md`](ysagiv_xaba_regime_sweep.md) |
| **Agent 2 — push our own corpus further** | Done | Two sub-lines. **Frozen-encoder PG head, 20→170 epochs**: clean negative, zero wins across 33 checkpoint×regime points, plateaued immediately. **e2e (encoder unfrozen), 1→4 epochs**: 1 confirmed win out of 12 checkpoint×regime points (epoch 2 only), did not replicate at epochs 3-4 — indistinguishable from a false positive at uncorrected α=0.05. One real, replicated finding: e2e training reliably beats the *old* frozen z_t representation (2/2 epochs checked) — genuine representational progress that hasn't yet translated into beating SingleHalt\*. [`our_trees_continued.md`](our_trees_continued.md) |
| **Agent 3 — real CP regeneration + pruning** | Done — negative | Regenerating trees under `tanh(cp_order/300)` value saturation + pruning was tested as an alternative corpus. Negative at both regimes tested: every controller loses headroom vs. the WDL baseline, Stats-Controller totally degenerates onto AlwaysStop (2/2 regimes), and z_t's one win (over the now-degenerate Stats) didn't replicate at a second regime. Explicit recommendation: don't scale this recipe up. [`cp_regen.md`](cp_regen.md) |
| **Probe X — xaba filter on our own corpus** | Done | Three populations, same 3 regimes each. **argmax_only** (n=1680, our standard baseline): no z_t win anywhere, matches `SIG` exactly. **argmax_and_xaba** (n=1272, xaba stacked on our filter, 74.3-74.6% survival): a confirmed z_t win at the cheap regime, `+0.0049 [+0.0005,+0.0101]` — small but real, and the *only* confirmed z_t-over-SingleHalt\* win anywhere on our own corpus that isn't a since-failed-to-replicate one-off. **xaba_only** (n=229, our filter dropped entirely, 30K tractability subsample, only 12.7% survival — ~3.5x more aggressive than stacked, suggesting the churn motif concentrates where argmax\>2 already screens): z_t performs badly at 2 of 3 regimes (confirmed losses, one nearly 3.3x worse than SingleHalt\*), plausibly a small-n readout-overfitting artifact (n_eval=229, 4-dim controller) rather than a real representational failure — flagged as directional, not conclusive. [`probe_x_xaba_own_corpus.md`](probe_x_xaba_own_corpus.md) |
| **Probe Y — subtree-weighted encoder (ysagiv's own lineage)** | Done | Apples-to-apples test of Yotam's `oracle96` encoder (`d_embed=128`, pretrained with subtree-size-weighted cross-entropy) on `human_trees`: directionally favorable point estimate (0.2625 vs. SingleHalt\*'s 0.2676, Stats' 0.2701) but neither paired diff reaches significance. **The historical "0.024 greedy regret" figure does not replicate** — this measurement is ~11x higher under current methodology, corroborating the git-archaeology finding below. [`probe_y_subtree_weighted_eval.md`](probe_y_subtree_weighted_eval.md) |
| **Git archaeology — the 0.024 figure** | Done | Traced to commit `4bca905` (2026-05-22): `time_lambda=18.537` (~1854x today's calibrated value), power-law cost shape, unfiltered population — not comparable to current numbers on any axis, and the repo's own historical labnotebook already flagged that λ as likely degenerate at the time. Not a valid benchmark. [`git_archaeology_greedy_regret.md`](git_archaeology_greedy_regret.md) |
| **Agent 5 — ysagiv xaba encoder capacity sweep** | **Paused, incomplete** | Testing whether the tiny production GNN (`d_embed=32,k=1`) is capacity-limited, via a wider (`d_embed=64`) and a deeper (`k=2`) arm. Both arms got equally far before cancellation: `train_encoder` completed cleanly for both (valid checkpoints on disk), then both `pack_root` arrays were killed mid-run. Zero eval numbers exist for either arm. Resume recipe documented (no need to redo encoder training — just re-run `pack_root → pack_root_merge → train_readout_pg → eval` against the existing configs). [`ysagiv_xaba_capacity_sweep.md`](ysagiv_xaba_capacity_sweep.md) (baseline table only) |
| **Agent 6 — e2e training on ysagiv xaba** | **Paused, incomplete** | Testing whether e2e (joint encoder+head) training improves on the already-strong frozen-encoder xaba result, the way it weakly did on our own corpus. Cancelled mid-run: epoch 1's training job was killed before producing a checkpoint. Zero epochs completed, zero eval numbers exist. [`ysagiv_xaba_e2e.md`](ysagiv_xaba_e2e.md) (methodology only) |

## What this adds up to

**The xaba filter is the one lever that has produced a real, robustness-checked win**, and it's
specific to the ysagiv `human_trees` corpus in its strongest form — Agent 4's sweep shows the
effect holds and even strengthens across a >10x range of cost regimes, so it isn't a one-off or a
right-censoring artifact. On our own corpus, xaba (stacked on our existing filter) produced one
small but real win; xaba alone (replacing our filter, small-n subsample) looked bad, though that
arm's small sample size and a degenerate SingleHalt\* at one of its regimes make it hard to trust
as-is.

Every other lever tried — more training epochs (frozen or e2e), a different value function
(CP regeneration), width/depth of the encoder (unfinished, paused before any signal) — either came
back negative or didn't get far enough to have an answer.

**Open, unresolved by anything run so far**: why does the xaba effect fade once `k*≤4`? Is encoder
capacity actually a factor (Agent 5, paused)? Does e2e training add anything on top of xaba's
already-strong frozen baseline (Agent 6, paused)? Does xaba-only (no argmax) genuinely hurt on our
corpus, or is that an artifact of the small n=229 subsample (Probe X, condition B)?

## What's cleanly closed vs. still open

**Closed, trustworthy, no further work needed**: diagnosis phase, Agent 1, Agent 2, Agent 3,
Agent 4, Probe Y, git archaeology.

**Needs a short write-up pass, not new compute**: Probe X (numbers exist, doc doesn't — the table
above has everything from the raw log).

**Genuinely incomplete, would need fresh jobs to finish**: Agent 5 (capacity sweep — 0/2 arms have
any eval numbers), Agent 6 (e2e-on-xaba — 0/2 epochs completed).
