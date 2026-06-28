# Human move time — does a normative search model match it?

**Ref:** `R-MOVETIME-MODEL` · [Index](reference.md)

## Does a normative search model reproduce human think time?

Humans deliberate longest when the decision is *wide* (see the board-features inquiry in the
index). Here we ask whether quantities read off an **lc0 search tree on the same position** —
Gain (value of computation), move-quality, and a **greedy stopping step** (how many expansions until the
search locks onto its best move) — track human think time, and where model and humans part ways.

> **Result:** The engine's value-search quantities (GSS, Gain, MQ) track human RT only **weakly**,
> and the LMCOS normative oracle stops on *value-convergence* while humans deliberate on
> *structural complexity* — the model captures the direction of the effects, not the dominant driver.

### Do lc0-search quantities track human RT?

On the lc0-tree subset (≈211K human moves whose position has a generated tree):

| Metric | Definition (lc0 tree) | r with log RT |
|---|---|---|
| **GSS** — greedy stopping step | first expansion the eventual-best move is found (greedy, zero-cost) | **+0.110** |
| **Gain** — value of computation | `final_Q(best @ 96 exp.) − final_Q(best @ 1 exp.)` ≥ 0 | **+0.085** |
| **MQ** — move quality | `final_Q(played) − final_Q(best)` ≤ 0 | **−0.151** |
| **Action gap** | top1 − top2 of children's 1-ply value-head backup | **−0.058** |

![Gain vs response time](../figures/gain_vs_rt.png)

Gain is the value of computation, read from a **single growing oracle tree** at two budgets. The
intuition, in five steps:

1. Take the **shallow tree** (after 1 expansion) and the **deep tree** (after 96).
2. Read the **shallow action** — the best move in the shallow tree (the move the search expands first).
3. On the deep tree, read the **deep action** (`argmax final_Q`) and its value.
4. Look up the **shallow action's value in the deep tree** (its converged `final_Q`).
5. `Gain = value(deep action) − value(shallow action)`.

In one line: **how much value does the full search find beyond its own first guess?** Because the
shallow action is read *from the shallow tree*, it was necessarily expanded there, so it is always
present in the deep tree with a real value — **no filtering is needed at any level** (the earlier
bugs came from taking the shallow action from a *different* source — a 1-ply value head, or a
prior-driven recommendation — that named a move the search never expanded, whose deep value was the
uninitialized 0.0). The coupling is **positive and monotone** — more value-of-computation, longer
thinks — the direction Russek-style accounts predict. Gain is zero in ~⅔ of positions (the first
expansion already lands on the search-best move).

> **Definition fix (supersedes two buggy ones).** Two earlier definitions took the shallow action
> from *outside* the shallow tree — a 1-ply value-head best, or a prior-driven recommendation — which
> in **won** positions named a move the search never expanded, so its `final_Q` read the uninitialized
> **0.0** and `Gain` was spuriously pinned at **≈ 1.0** (2.7% of moves), producing a *pathological RT
> dip* at the top bin (humans play obvious wins fast). Reading the shallow action *from the growing
> tree itself* (the first move the search expands) removes the bug at the source — it is always a real,
> visited move — and needs no filtering. The dip is gone, Gain rises monotonically, and the RT coupling
> is **r = +0.085** (vs +0.106 for the intermediate patch — lower but honest: value of search over the
> policy's first pick, on one converged evaluation).

![MQ vs RT — global, by ply tertile, and by GSS difficulty stratum](../figures/mq_vs_rt.png)

MQ is the move-quality of the human's *played* move, plotted as the **outcome** of think time
(the same MQ-vs-log-RT relationship, segmented three ways: global, by ply tertile, and by GSS
difficulty stratum). The correlation is **negative**: longer thinks land on worse moves. The
engine-best-move rate falls **0.62 → 0.49** across move-time deciles, and the rank correlation
(Spearman ρ ≈ −0.12 on the 1M-move audit) survives de-meaning by ply and by player.

The standing read was that this is purely a **difficulty confound** — long thinks land on harder
positions, where even a deliberating human plays the engine-best move less often. We tested that
directly by segmenting the MQ↔RT slope **within difficulty strata**, using GSS (engine search
effort to find the best move) as the difficulty proxy — the **third panel** above:

| Conditioning | Spearman ρ(MQ, log RT) |
|---|---|
| overall (pooled) | **−0.232** |
| within GSS 0–1 (easy) / 2–31 (med) / 32–95 (hard) | −0.235 / −0.202 / −0.188 |
| partial \| GSS | **−0.213** |
| partial \| legal moves | −0.160 |
| partial \| GSS + legal moves | **−0.150** |

> **Correction:** The negative MQ↔RT slope is **not** a GSS-difficulty confound and only *partly*
> a legal-moves one. It stays clearly negative inside **every** GSS stratum (−0.19 to −0.24, barely
> moved from the pooled −0.232), and conditioning on GSS removes only ~7% of the association.
> Worse, **GSS is a poor difficulty proxy here**: its easiest stratum (GSS 0–1 — forced recaptures
> / only-moves the engine fixes instantly) has the *worst* mean MQ, because humans who deviate
> there lose a lot — GSS is engine search-effort, not human-perceived difficulty. The **legal-move
> count** is the stronger difficulty axis (ρ(legal moves, MQ) = −0.25, ρ(legal moves, RT) = +0.34) and explains
> more, but the joint partial still leaves **−0.150 — about ⅔ of the effect intact**. So a real
> residual "longer thinks → worse moves" survives both difficulty controls, pointing to
> *selection / uncertainty* (people deliberate precisely when unsure, and subjective uncertainty
> predicts errors beyond any objective difficulty index) rather than to objective difficulty alone.
> The sign also holds within every ply tertile.

![greedy stop step vs RT](../figures/gss_vs_rt.png)
![action gap vs RT](../figures/actiongap_vs_rt.png)

GSS (one point + SEM per integer value) **rises monotonically** with RT across the full range
(no cap needed); the action gap is flat-to-weak.

> **Result:** Every engine value-search quantity tracks human RT only faintly (|r| ≲ 0.17).
> The model is in the right direction but explains little of the variance in human think time.

### Where do the model and humans diverge?

The clearest signal is *which features* drive the oracle versus the human. On 39,668 lc0 trees
the same four features point the same way for the oracle's stop step and human RT (**4/4
directions agree**) — but the emphasis is opposite:

| Feature | r(oracle stop) | r(human RT) |
|---|---|---|
| legal moves | +0.014 | **+0.195** |
| material | +0.011 | +0.039 |
| gain_depth (value gained) | **+0.233** | +0.096 |
| action gap (toptwo) | **−0.290** | −0.064 |

The oracle halts on **value-landscape** features (gain_depth, action gap) and ignores the legal-move
count; humans deliberate on **structural complexity** (legal moves dominate) and barely track Gain. On
matched human-FEN trees the oracle stop step still correlates positively with log(RT) (smoke
r = **+0.091**, n = 497), but gain_depth predicts the *oracle's* halt strongly (+0.463) while
being near-zero for *humans* (+0.020).


> **Result:** The oracle and humans agree on the sign of every feature but not the mechanism:
> the oracle stops when the value gap is decided; humans deliberate when the move set is wide.

### How do the model variables relate?

![Spearman correlation matrix — lc0 metrics](../figures/correlation_matrix.png)

A rank (Spearman) correlation matrix over the lc0 metrics, board structure, and log(RT) — rank,
because Gain / MQ / action gap are zero-inflated and skewed, so Pearson understates (and can flip
the sign of) their monotone relationships. The strongest tie to log(RT) is the **legal-move count**
(still stronger than any engine metric); **MQ ↔ RT** is the difficulty-*plus-residual* effect above;
**GSS** ties to the Gain / action-gap *value-convergence* cluster, not to the legal-move count.

The matrix also carries **H(π)** — the entropy of lc0's *policy prior* over the root's legal moves
(prior uncertainty over the argmax; a policy-side width measure, **no search**). It is the **strongest
*tree-derived* predictor of log RT (+0.24)**, ~3× any value-of-search quantity, and it survives controls
for Gain/GSS — but it is a *width* signal (ρ +0.59 with the legal-move count) that the **raw legal-move
count subsumes** (partial ρ(H(π), RT | legal moves) = +0.05; +0.03 controlling for legal moves + Gain +
GSS jointly, n ≈ 204K). So the policy entropy adds essentially **nothing over the raw legal-move count**:
lc0's over-confident policy makes it a weaker proxy than the count itself (see the legal-moves report's
P1). The operative width variable is the **raw legal-move count**, not any policy-weighted refinement of it.

> **Result:** The legal-move count is the connective tissue between human RT and position structure; the
> engine value-search metrics form a separate, weakly-expressed value-convergence cluster; and the policy
> entropy H(π) is a third — *width-side* — correlate that the legal-move count subsumes. The driver of human
> deliberation is decision **width**, not realized value-of-computation.

## What can we conclude?

| Claim | Evidence |
|---|---|
| Decision width drives human deliberation — more than engine Gain | legal-moves r ≈ +0.20/+0.33 ≫ Gain; oracle ignores the legal-move count, humans don't |
| The normative model captures direction, not the dominant driver | 4/4 feature directions agree, but oracle halts on value-convergence, humans on structure |
| Engine value-of-computation tracks RT, but weakly | Gain r = +0.085 (growing-tree def: best @ 96 vs @ 1 expansion); GSS r = +0.110 |
| "More time → worse moves" is **not just** a difficulty confound | MQ r = −0.151; survives partialling GSS + legal moves (ρ = −0.150, ~⅔ of the effect) |

> **Result:** Humans look *resource-rational about the width of the decision*; the value-search
> model explains the easy direction but misses what most strongly paces human thought. The
> MQ↔RT slope is only ~⅓ objective-difficulty; the residual points to selection / uncertainty.
> Next: a strength-matched (SF-2000) oracle on the same FENs to test whether the model–human
> mismatch is a strength artifact, and isolating the selection vs. blunder-after-long-think drivers.

## Methods

### Definitions

| Quantity | Definition | Source |
|---|---|---|
| **MQ** | `final_Q(played) − final_Q(best)` (≤ 0; 0 = engine-best played) | lc0 tree (replaces the retired Stockfish-d5 `e_win_taken − e_win_best`) |
| **Gain** | `final_Q(best @ 96 exp.) − final_Q(best @ 1 exp.)` (≥ 0), both on the converged 96-exp. Q | lc0 tree; from the growing oracle search trace (`oracle_root_q_trace`) |
| **GSS** | first expansion `oracle_best_move_index` reaches its final value (greedy, zero-cost) | lc0 tree |
| **Action gap** | top1 − top2 of root children's 1-ply value-head backup (`−child.value`) | lc0 tree |
| **H(π)** | `−Σ π(a) log π(a)` over the root's legal moves; π = lc0 policy-head prior (`node_features[:, prior]`) | lc0 tree (policy head; **no search**) |

> **Decision:** GSS is the **greedy** stopping step (the budgeted oracle's stop at *zero* cost = the
> first expansion the best move is found), **not** the cost-aware DP `optimal_stop_step`, which under
> the default time cost bails almost immediately and isn't an interpretable difficulty proxy. Gain is
> read from the **single growing oracle tree**: best @ 96 expansions vs best @ 1 expansion, both scored
> on the converged 96-expansion Q, with the 1-expansion move taken from the search trace's first-visit
> (so it is always a visited move — never an unvisited 0.0). This supersedes two earlier definitions that mixed a
> 1-ply value-head "shallow" against the deep Q and read an unvisited `final_Q == 0.0`, which pinned a
> spurious ~1.0 spike. MQ is the LC0 final-Q loss (Stockfish-d5 MQ retired); the tree FEN is
> normalized to 4 fields for the human join.

### Data and pipeline

- **lc0-tree subset:** ~199K trees from the canonical `human_trees` set (lc0 search on 2023 human-game
  root FENs), joined to ~211K human RTs on the **4-field** FEN. MQ is matched to the human's played UCI
  (98.0% match). (An earlier lexicographic 150K slice of the full FEN universe gave a weaker,
  unrepresentative GSS↔RT — a different FEN *population*, not a generation bug; same engine/config/net.)
- `human_analytics/tree_values_analysis.py` derives GSS/Gain/Action-Gap/MQ, **H(π)** (root policy-prior
  entropy), and the lc0 Spearman
  matrix; per-tree values are **cached to parquet** (deterministic in trees/n_trees/seed),
  so plot iterations reload the cache locally in seconds. One-time compute via
  `slurm/tree_values.slurm`. All panels use **K=10 tie-safe quantile bins** with per-bin SEM in
  `utils/analysis.py` (tie-safe keeps a repeated integer in one bin, so discrete GSS doesn't split
  across edges; low-cardinality x collapses to ≤10 integer points). For the continuous Gain / MQ /
  action-gap the value point-mass (e.g. Gain's 55%-at-0) is isolated as its own point, and **Gain
  is plotted on a √ x-axis** (Russek 2025: move time fits √ΔUC better than linear, and √ handles
  the zero-mass that log cannot). (Gain is the `voc` column; the figure x is `sqrt(voc)`.)
- **MQ-by-difficulty:** the 3rd panel of `mq_vs_rt.png` is a second `Analyzer` on the same `mq_rt`
  table built with `segment_column="gss"` — the *identical* MQ-vs-RT panel, segmented by GSS stratum
  instead of ply tertile (so it differs from the by-ply panel only in its legend).
  `human_analytics/mq_vs_rt_by_gss.py` prints the within-stratum and partial Spearman ρ (on GSS /
  legal moves), straight from the cache.
- **Oracle (Tier A/B):** `lmcos` `pack.build_compact_trajectory` → `oracle.compute_budgeted_oracle`
  → `analysis/human_oracle_comparison.py`, which validates each tree's FEN against the manifest.
  Tier A: 39,668 lc0 trees (budget 96). Tier B: smoke on 497 clean human-FEN trees (1K job hit the
  1-hour SLURM wall); 10K is the power target.

## Appendix (logs)

### Status

| Step | Status |
|------|--------|
| lc0 GSS/Gain/Action-Gap/MQ + Spearman matrix, ~199K trees | ✅ done (cached) |
| Gain redefined: best @ 96 vs @ 1 expansion on one growing tree | ✅ supersedes the 1-ply/unvisited-0.0 defs; dip gone, r = +0.085 |
| P1: H(π) policy-prior entropy vs log RT + partials (n ≈ 204K) | ✅ +0.234; subsumed by raw legal-move count (see legal-moves report) |
| MQ difficulty-confound audit (N = 1M, Spearman + controls) | ✅ done |
| MQ↔RT segmented by GSS (difficulty-residualized partials) | ✅ survives GSS (−0.213) & GSS + legal moves (−0.150); not a pure confound |
| Oracle Tier A (4/4 directions, 39,668 trees, 72 invariant tests) | ✅ done |
| Oracle Tier B: 10K human FENs as 20 shards | ✅ submitted; oracle+join+plots ⬜ after jobs |
| SF-2000 strength-matched oracle (CP→WDL) on same FENs | ⬜ gated on Tier B 10K r |
| E[ΔUC] over top-5 depth-1 candidates (Russek Figures 4–5) | ⬜ open |

### Invalid runs (do not cite)

- Tier A initial (5K trees, budget 43, 2026-06-03): two bugs (evolving-Q halt rewards;
  mixed-regime gain_depth) — superseded by the corrected results above.
- Tier A glob `filtered_shard_0000?`: missed shards 00010–00019 — fixed to `filtered_shard_*`.

### Reproduce (lc0 metrics)

```bash
# one-time compute (populates the parquet cache), on the cluster:
sbatch human_analytics/slurm/tree_values.slurm
# iterate on plots/matrix later, locally, straight from cache (seconds):
PYTHONPATH=human_analytics python human_analytics/tree_values_analysis.py
```

*Merges the former R-VOC-MQ (human Gain/MQ vs move time) and R-ORACLE-RT (oracle stop step vs
human RT) reports, folding the R-THEORY validation-tier framework.*
