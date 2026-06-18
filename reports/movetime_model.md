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

On the lc0-tree subset (≈119K human moves whose position has a generated tree):

| Metric | Definition (lc0 tree) | r with log RT |
|---|---|---|
| **GSS** — greedy stopping step | first expansion the eventual-best move is found (greedy, zero-cost) | **+0.115** |
| **Gain** — value of computation | `final_Q(deep best) − final_Q(1-ply best)` ≥ 0 | **+0.073** |
| **MQ** — move quality | `final_Q(played) − final_Q(best)` ≤ 0 | **−0.154** |
| **Action gap** | top1 − top2 of children's 1-ply value-head backup | **−0.064** |

![Gain vs response time](../figures/gain_vs_rt.png)

Gain is the deep-search improvement over the shallow (1-ply value-head) choice. The coupling is
**weakly positive** — more value-of-computation, longer thinks — the direction Russek-style
accounts predict. Gain is zero in ~⅔ of positions (search confirms the 1-ply pick).

![MQ vs response time](../figures/mq_vs_rt.png)

MQ is the move-quality of the human's *played* move, plotted as the **outcome** of think time.
The correlation is **negative**: longer thinks land on worse moves. The engine-best-move rate
falls **0.62 → 0.49** across move-time deciles, and the rank correlation (Spearman ρ ≈ −0.12 on
the 1M-move audit) survives de-meaning by ply and by player.

> **Clarification:** The negative MQ↔RT correlation is a **difficulty confound, not a sign
> bug** — long thinks land on objectively harder positions, where even a deliberating human
> plays the engine-best move less often. The sign holds within every ply tertile.

![greedy stop step vs RT](../figures/gss_vs_rt.png)
![action gap vs RT](../figures/actiongap_vs_rt.png)

GSS (binned in groups of 5) **rises monotonically** with RT across the full range (no cap needed);
the action gap is flat-to-weak.

> **Result:** Every engine value-search quantity tracks human RT only faintly (|r| ≲ 0.17).
> The model is in the right direction but explains little of the variance in human think time.

### Where do the model and humans diverge?

The clearest signal is *which features* drive the oracle versus the human. On 39,668 lc0 trees
the same four features point the same way for the oracle's stop step and human RT (**4/4
directions agree**) — but the emphasis is opposite:

![oracle stop step vs human RT — feature correlations](../lmcos/analysis/figures/oracle_stop_step_vs_human_rt.png)

| Feature | r(oracle stop) | r(human RT) |
|---|---|---|
| branching | +0.014 | **+0.195** |
| material | +0.011 | +0.039 |
| gain_depth (value gained) | **+0.233** | +0.096 |
| action gap (toptwo) | **−0.290** | −0.064 |

The oracle halts on **value-landscape** features (gain_depth, action gap) and ignores branching;
humans deliberate on **structural complexity** (branching dominates) and barely track Gain. On
matched human-FEN trees the oracle stop step still correlates positively with log(RT) (smoke
r = **+0.091**, n = 497), but gain_depth predicts the *oracle's* halt strongly (+0.463) while
being near-zero for *humans* (+0.020).

![same-FEN oracle vs human RT](../lmcos/analysis/figures/human_oracle_rt_comparison.png)

> **Result:** The oracle and humans agree on the sign of every feature but not the mechanism:
> the oracle stops when the value gap is decided; humans deliberate when the move set is wide.

### How do the model variables relate?

![Spearman correlation matrix — lc0 metrics](../figures/correlation_matrix.png)

A rank (Spearman) correlation matrix over the lc0 metrics, board structure, and log(RT) — rank,
because Gain / MQ / action gap are zero-inflated and skewed, so Pearson understates (and can flip
the sign of) their monotone relationships. The strongest tie to log(RT) is **branching** (still
stronger than any engine metric); **MQ ↔ RT** is the difficulty confound; **GSS** ties to the
Gain / action-gap *value-convergence* cluster, not to branching.

> **Result:** Branching is the connective tissue between human RT and position structure; the
> engine metrics form a separate value-convergence cluster that humans only weakly express.

## What can we conclude?

| Claim | Evidence |
|---|---|
| Decision width drives human deliberation — more than engine Gain | branching r ≈ +0.20/+0.30 ≫ Gain; oracle ignores branching, humans don't |
| The normative model captures direction, not the dominant driver | 4/4 feature directions agree, but oracle halts on value-convergence, humans on structure |
| Engine value-of-computation tracks RT, but weakly | Gain r = +0.073; GSS r = +0.115 |
| "More time → worse moves" is a difficulty confound, not a paradox | MQ r = −0.154, negative within every ply tertile |

> **Result:** Humans look *resource-rational about the width of the decision*; the value-search
> model explains the easy direction but misses what most strongly paces human thought. Next:
> difficulty-residualized MQ partials, and a strength-matched (SF-2000) oracle on the same FENs
> to test whether the mismatch is a strength artifact.

## Methods

### Definitions

| Quantity | Definition | Source |
|---|---|---|
| **MQ** | `final_Q(played) − final_Q(best)` (≤ 0; 0 = engine-best played) | lc0 tree (replaces the retired Stockfish-d5 `e_win_taken − e_win_best`) |
| **Gain** | `final_Q(deep best) − final_Q(1-ply best)` (≥ 0) | lc0 tree; shallow = 1-ply value-head best |
| **GSS** | first expansion `oracle_best_move_index` reaches its final value (greedy, zero-cost) | lc0 tree |
| **Action gap** | top1 − top2 of root children's 1-ply value-head backup (`−child.value`) | lc0 tree |

> **Decision:** GSS is the **greedy** stopping step (the budgeted oracle's stop at *zero* cost = the
> first expansion the best move is found), **not** the cost-aware DP `optimal_stop_step`, which under
> the default time cost bails almost immediately and isn't an interpretable difficulty proxy. Gain's
> shallow baseline is the **1-ply value-head best** (not the all-zero step-0 trace, which once inflated
> Gain to a spurious ~1.0 mass). MQ is the LC0 final-Q loss (Stockfish-d5 MQ retired); the tree FEN is
> normalized to 4 fields for the human join.

### Data and pipeline

- **lc0-tree subset:** ~114K trees from the canonical `human_trees` set (lc0 search on 2023 human-game
  root FENs), joined to human RT on the **4-field** FEN. MQ is matched to the human's played UCI (≈98%
  match). (An earlier lexicographic 150K slice of the full FEN universe gave a weaker, unrepresentative
  GSS↔RT — a different FEN *population*, not a generation bug; same engine/config/net.)
- `human_analytics/tree_values_analysis.py` derives GSS/Gain/Action-Gap/MQ and the lc0 Spearman
  matrix; per-tree values are **cached to parquet** (deterministic in trees/n_trees/seed),
  so plot iterations reload the cache locally in seconds. One-time compute via
  `slurm/tree_values.slurm`. Binning lives in `utils/analysis.py` (integer bins for GSS, tie-safe
  + zero-lump for Gain/MQ/gap, `min_bin_count` to drop sparse tails). (Gain is the `voc` column.)
- **Oracle (Tier A/B):** `lmcos` `pack.build_compact_trajectory` → `oracle.compute_budgeted_oracle`
  → `analysis/human_oracle_comparison.py`, which validates each tree's FEN against the manifest.
  Tier A: 39,668 lc0 trees (budget 96). Tier B: smoke on 497 clean human-FEN trees (1K job hit the
  1-hour SLURM wall); 10K is the power target.

## Appendix (logs)

### Status

| Step | Status |
|------|--------|
| lc0 GSS/Gain/Action-Gap/MQ + Spearman matrix, 100K trees | ✅ done (cached) |
| MQ difficulty-confound audit (N = 1M, Spearman + controls) | ✅ done |
| Oracle Tier A (4/4 directions, 39,668 trees, 72 invariant tests) | ✅ done |
| Oracle Tier B: 10K human FENs as 20 shards | ✅ submitted; oracle+join+plots ⬜ after jobs |
| Difficulty-residualized MQ partials | ⬜ open |
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
