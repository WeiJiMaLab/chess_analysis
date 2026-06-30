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

## Step 1 result — the proxy (2026-06-29)

`prune_proxy.py` on all **250K n=1 trees**, joined to **262,286 human moves**:

| ε | pruned/legal | ρ(pruned, RT) | partial \| legal |
|---|---|---|---|
| *floor: legal_moves* | — | **+0.341** | — |
| *n_total (unpruned)* | 96 | +0.319 | +0.112 |
| 0.0  | 0.17 | −0.163 | −0.085 |
| 0.05 | 1.7  | −0.152 | −0.078 |
| 0.1  | 7    | −0.142 | −0.074 |
| 0.2  | 17   | −0.124 | −0.070 |
| 0.3  | 23   | −0.107 | −0.067 |
| 0.5  | 32   | −0.070 | −0.059 |
| 1.0  | 58   | +0.094 | −0.019 |

> **Result:** ε is a **dial along the size ↔ satisfaction axis**. Aggressive pruning (ε→0) makes the
> pruned-count a **satisfaction** signal (−0.16: many near-best moves → faster); no pruning (ε→2) makes it the
> **size** signal (+0.32); the correlation crosses zero at ε≈0.9. **No single ε beats the bare legal-moves floor
> (+0.34)** in magnitude — but the **partial-controlling-for-legal-moves** is non-zero at both ends (−0.085
> satisfaction, +0.112 deep-branching), so the pruned count *does* carry information beyond raw width.
>
> **Clarification:** this is the **loose** proxy (post-hoc prune of the *completed* tree, final values, no budget
> redeploy, no stopping dynamics). It cannot show pruning's value-add — it only confirms ε is the right knob and
> locates the action at **low ε**. The regen is the real test.

## Chosen regen levels & next step

The proxy puts the satisfaction signal at **low ε** (the cognitively plausible regime: ε=0.05→~58 nodes,
ε=0.1→~240, ε=0.3→~790; vs ~3370 unpruned). Regen grid (step 2): **ε ∈ {0.05, 0.2, 0.5}** — aggressive / mid /
loose — spanning the transition, with the existing no-prune trees as the size baseline.

**Implemented:** `prune_epsilon` threaded through `BuildTreeConfig → TeacherSearchConfig`; in the PUCT loop a
node's children are value-pruned at depth≥1 (relative-to-best by leaf-eval value, negamax best=min; root keeps all
legal moves). Pruned children are never attached → the freed budget redeploys deeper.

## Step 2 — the depth-36 correction (the binding constraint)

The smoke test exposed a latent bug: `max_depth` was **4**, which *capped the freed budget* — pruned forced
positions hit the depth wall and stopped at <96 expansions instead of thinking deeper. Fixed to **36**
(`configs/core.yaml`, both `treegen` and `mc_pack`). Effect (same FENs, n=1):

| tree | n_total md4 / md36-none / md36-ε0.1 | max-depth md4 / md36-none / md36-ε0.1 |
|---|---|---|
| 1 | 3685 / 3671 / **220** | 4 / 5 / **19** |
| 0 | 3632 / 4713 / 2446 | 4 / 6 / 4 |

> **Result:** the depth lift matters **only for pruned trees** — uniform-prior PUCT is breadth-leaning, so
> unpruned search barely used depth 4. With pruning + depth 36 the budget drives **deep** on forcing lines (tree
> 1: depth **19**, avg expand depth 7.9, full 96 expansions) — the faithful model of human deep calculation.
> Unpruned md36 ≠ md4 for some positions, so the **baseline is regenerated too**.

**Ran (full 250K each, n=1, md36):** `n1md36` (baseline) + `n1md36_eps{0.05,0.1,0.3}` via `prune_regen.slurm`.

## Step 2 result — raw pruned node-count does NOT beat the floor

`prune_regen_analyze.py`, 262,286 human moves, all md36:

| set | med n_total | ρ(n_total, RT) | partial \| legal |
|---|---|---|---|
| *floor: legal moves* | — | **+0.341** | — |
| none (unpruned) | 2858 | +0.319 | +0.099 |
| ε=0.05 | 776 | −0.045 | −0.038 |
| ε=0.1 | 866 | −0.053 | −0.048 |
| ε=0.3 | 1171 | −0.008 | −0.058 |

> **Result:** with the depth-36 budget free to **redeploy deep**, an aggressively-pruned position no longer yields
> a *small* tree (≈800 deep-narrow nodes, not the proxy's tiny sets), so the pruned `n_total` **washes out to
> ≈0** and is nowhere near the +0.34 floor. The proxy's satisfaction signal (−0.15) was a **depth-4 truncation
> artifact** (pruned ⇒ small tree); with proper depth it disappears. Stable at 262K moves — a clean null, not
> noise; **more trees will not move it.**

> **Implication:** raw node-count is *not* the cost that matches RT. The value-add, if any, must come from the
> **satisficing STOP** — the oracle step\* / a 1–2-param consideration-set model on the pruned trace (the pruned
> trees show oss→0: prune to near-best ⇒ nothing to gain ⇒ decide immediately), **not** the leaf-count. That is
> the live step-4 test, runnable on the existing 250K (no more trees).

## Stage 2d — the engine-derived (no-refit) analyses, staged

`lmcos_tiny/pipeline/2d_data_analysis.slurm` runs the param-free chain (budgeted-oracle VOC signals +
softmax-VOC τ sweep) on a tree SET over the **unfiltered** population → `sf_analysis/<SET>/`; `2d_figures.slurm`
makes the plots. Launched on all 5 md36 sets; figures regenerate on `n1md36`. The halt-policy zoo + hindsight
halter stay in `3_train`/`4_eval` (they need training).
