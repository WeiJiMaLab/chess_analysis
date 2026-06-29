# Value-pruning: changing the cost profile so stopping becomes optimal

**Ref:** `R-PRUNING` · [Index](reference.md) · parent thread [(R-TREESEARCH)](treesearch.md)

> **Status:** 📝 proposal — the plan + the open decisions. The thesis: the only **live lever** on *when* it is
> worth stopping is the **cost profile**, and **value-pruning** is how we move it. This report stages the
> experiment and pins the decisions still open (the interview at the bottom).

## Why prune? Because the reward knobs are dead ends *by construction*

In [(R-TREESEARCH)](treesearch.md) every reward-side manipulation was a no-op for the RT story: evaluator
strength (**SF-1 ≈ SF-100** on every RT-correlation), `UCI_Elo` (bit-identical), hindsight-vs-causal (halter ≈
oracle). **In hindsight this is forced.** We **filtered the trees for reward to planning** — positions where
searching deeper improves the value (roughly monotone value-gain with depth). On that set:

- the **reward profile is held ~fixed and monotone**, so the optimal stop is pinned by the **cost**, not the
  reward — moving the reward (strength, Elo, hindsight) barely shifts *when* to stop;
- therefore the only way to move *where stopping is optimal* is to change the **cost profile**.

**Value-pruning is a cost-profile lever.** Today the tree enumerates **every** legal child of every expanded node
as a leaf, so the leaf-cost ∝ raw branching (`n_total`/`n_leaf` ↔ RT = +0.323, but 0.857-collinear with
legal-moves — a *re-description* of decision width). If instead we keep only the **plausible** leaves (within a
value threshold of the best), the leaf-cost reflects the **value landscape**, not raw count — and *that* is where
satisfaction / sharpness could enter, beyond the bare floor.

> **Decision:** stop pulling reward knobs; the experiment is to vary the **cost** via value-pruning.

## What is value-pruning, exactly?

At each expanded node, when enumerating its legal children, **drop** (do not attach as a leaf, do not consider
for expansion) any child whose **prior/leaf-eval value** is more than **ε** below that node's best child. The
freed expansion budget then drives **deeper** on the surviving lines → a **narrower, taller** tree. The pruned
leaf-count becomes the cost. Two locked-in design choices (from R-TREESEARCH §9):

- prune by the **prior** (early/heuristic value), **not** hindsight `final_Q` — you prune with your gut before
  calculating;
- prune **relative to the node's best** ("could this plausibly be best"), not an absolute win-prob floor.

> **Clarification:** pruning **changes the tree's shape**, so a post-hoc prune of an existing tree is *not* the
> tree a pruned search builds (the freed budget redeploys deeper). The fit needs **re-generated** trees. The
> post-hoc prune is still useful — but only to **choose** the threshold (next section).

## The staged plan

1. **Pick the ε grid cheaply (no regen).** On the existing **250K n=1 trees** (`sf_trees/elo2000_n1`), compute the
   *proxy* pruned leaf-count — post-hoc prune the completed tree at a sweep of ε — and see which ε makes the
   pruned leaf-count best track RT. Use this only to choose **4 scalar levels: none / low / medium / high** (so
   the regen grid stays tractable). *No-pruning is already known good (+0.32); it should only improve.*
2. **Regenerate with pruning in the search** (the real test) at the chosen level(s). Add an `ε` (prune-threshold)
   parameter to `build_tree`'s PUCT child-enumeration; regenerate at **n=1** (strength doesn't matter).
3. **Lock in** the best-fitting ε (an admittedly arbitrary but *fixed* choice), then **fit the normative model**
   (the satisficing leaf-cost / budgeted-oracle with cost ∝ pruned leaves) and show it **replicates**
   `size − satisfaction + sharpness` — the P-FIT test from R-TREESEARCH §9.

## Scale & feasibility (measured)

| quantity | number | note |
|---|---|---|
| n=1 throughput | **1.36 trees/s/process** | from the dumb-eval array logs |
| 750K wall time | **~1.5–3 h** | 50–100 concurrent array tasks |
| per-tree size | **~220 KB** | identical n1/n100 (N changes values, not shape); pruned trees are *smaller* |
| 750K storage | **~165 GB** (less, pruned) | `/scratch` is 95% full, **1.7 TB free** — tight |
| FEN supply | **110M** (`lmcos/fens.txt`) | sample 750K roots; no blocker |
| already on disk | **250K n=1 trees** | reuse for the ε-picking proxy |
| reclaimable | **~55 GB** (`sf_filtered/elo1350`) | UCI_Elo duplicate, safe to delete [[sf-uci-elo-noop-for-eval]] |

> **Result (feasibility):** 750K @ n=1 is a **half-day** job, ~165 GB — fine *if* we free the `elo1350` duplicate
> and keep the pruned-regen grid small (1–2 levels, not 4× full-scale).

## Open questions — the interview

These need a decision before a confident rollout (see the questions asked alongside this report):

1. **Pruning rule** — relative-to-best by leaf-eval (recommended) vs absolute win-prob floor vs top-k by prior.
2. **Regen grid scale** — pick ε on the 250K proxy then regen **750K at one locked level**, or regen a smaller set
   at 2–3 levels first to confirm the proxy picked right?
3. **Target count** — is **750K** the *generated* count (filter for reward-to-planning keeps ~25% → ~190K
   analyzed) or the *filtered/analyzed* target (→ generate ~3M)?
4. **The normative model + success criterion** — which model do we lock-and-fit (satisficing consideration-set
   meta-MDP vs budgeted-oracle step\* with pruned-leaf cost), and what counts as "replicates the findings"
   (recovers the size−satisfaction+sharpness signs? beats the bare ±0.31 floor? by how much?).
5. **Filter** — keep the same reward-to-planning filter at scale, or relax it (since at 750K we can afford a
   broader, more representative set)?
