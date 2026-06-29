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
   *proxy* pruned leaf-count — post-hoc prune the completed tree at a sweep of ε (relative-to-best by leaf-eval
   value) — and see which ε makes the pruned leaf-count best track RT. Use this only to choose **2–3 scalar levels
   (low / medium / high)** to actually re-generate. *No-pruning is already known good (+0.32); it should only
   improve.*
2. **Regenerate 2–3 ε levels at ~250K each** (the real test, and the proxy-validation). Add an `ε`
   (prune-threshold) parameter to `build_tree`'s PUCT child-enumeration; at each node drop children whose
   leaf-eval value is >ε below the sibling-best; regenerate at **n=1**. Compare the *regenerated* pruned-leaf-cost
   ↔ RT against the proxy's pick (regen changes tree shape, so this confirms the proxy chose right) and against
   the +0.32 floor. Pick the **winner ε**.
3. **Scale the winner to 750K generated** (~190K after the reward-to-planning filter) at the locked ε, n=1.
4. **Lock in** that (arbitrary but fixed) ε, then **fit the normative model — oracle first, then meta-MDP**:
   - **(a) budgeted-oracle step\*** with cost ∝ pruned leaves (reuse the existing DP; a fast go/no-go on whether
     the pruned cost moves step\*↔RT past the floor);
   - **(b) the satisficing consideration-set meta-MDP** (include-next-move / stop, reward = value(chosen) −
     c·#included, ~1–2 params) — the publishable model, fit only if (a) moves.
   - **Success** = recovers the `size − satisfaction + sharpness` signs **and** beats the bare ±0.31 floor.

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

## Locked decisions (2026-06-29)

| Decision | Choice |
|---|---|
| **Pruning rule** | **Relative to best, by leaf-eval value** — drop a child if its leaf-eval value is >ε below the sibling-best ("could this plausibly be best"); ε is a value gap. |
| **Regen grid** | **Regen 2–3 ε levels at ~250K first** to validate the proxy + test the shape change, *then* scale the winner to 750K. |
| **Target count** | **750K generated** (~190K analyzed after the reward-to-planning filter — 3× today's 64K). Keep the filter. |
| **Normative model** | **Oracle first, then meta-MDP** — step\* with pruned-leaf cost as the go/no-go; the satisficing meta-MDP as the publishable model if the oracle moves. |
| **Success** | Recovers `size − satisfaction + sharpness` signs **and** beats the bare ±0.31 floor. |

> **Storage note:** the grid (2–3 × ~250K, *pruned* so < 55 GB each) + the 750K winner ≈ **250–330 GB peak** on a
> `/scratch` with 1.7 TB free — feasible if we **free `elo1350`** (~55 GB) and delete the losing grid levels
> after picking ε.

## Next concrete step

Add the `prune_epsilon` parameter to `build_tree`'s child-enumeration (relative-to-best by leaf-eval value), and
write the proxy script (`lmcos_tiny/analysis/prune_proxy.py`) that post-hoc prunes the existing 250K n=1 trees
across an ε sweep and reports pruned-leaf-cost ↔ RT — to fix the 2–3 levels to re-generate.
