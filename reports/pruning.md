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
   - **Success** = a resource-rational planner (per-operation cost, value = decision quality) **reproduces** the
     `size − satisfaction + sharpness` curves with *sensible* cost parameters. **Legal-moves is the
     *explanandum*, not a rival to beat** — collinearity with it is the goal, because the win is *explaining why
     it is resource-rational to spend effort ∝ your options*, not finding an orthogonal signal. (We spent a long
     time scoring "beats the +0.34 floor"; that was the wrong target — a bare count is not a normative model.)

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
| **Success** | A resource-rational planner **reproduces** the `size − satisfaction + sharpness` curves with sensible cost params. **NOT** "beats the floor" — legal-moves is the thing being *explained*, not a rival. |

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

## Step 3 — the pruning-rule grid: absolute / rank / extended relative (2026-06-30)

Step 2 tested the *relative* rule. To check whether the null is specific to that rule or general, we regenerated
two new families + an extended relative range (md36, n=1, ~50K roots each) and re-ran `prune_regen_analyze.py`:

- **absolute** `abs{0.3,0.5,0.7}` — drop a child whose leaf-eval value is below an **absolute** win-prob floor
  (the rule R-TREESEARCH §9 *rejected* on principle — kept here as a falsification probe);
- **rank** `rank{2,4,8}` — keep only the top-k children (value-**blind**: rank, not value gap);
- **extended relative** `rel{0.5,0.75,1.0,1.5}` — the looser end of the original dial.

> **Floor caveat:** the new sets are a different 50K-root sample → legal-floor **ρ(legal,RT)=+0.250** (n=57,087),
> *not* the +0.341 of the 250K relative sets. Each cost is judged against its **own** floor.

| set (n=57,087, floor +0.250) | rule | med n_total | med max_depth | ρ(n_total,RT) | **partial \| legal** | max_depth partial\|legal |
|---|---|---|---|---|---|---|
| rank2 | top-k, value-blind | 223 | 9 | +0.241 | −0.037 | +0.075 |
| rank4 | top-k, value-blind | 410 | 7 | +0.187 | −0.071 | +0.084 |
| rank8 | top-k, value-blind | 780 | 6 | +0.116 | −0.072 | +0.089 |
| rel0.5 | rel-to-best | 1438 | 6 | −0.016 | −0.076 | +0.051 |
| rel1.0 | rel-to-best | 2017 | 5 | +0.059 | −0.069 | +0.010 |
| rel1.5 | rel-to-best | 2410 | 5 | +0.069 | −0.094 | +0.010 |
| **abs0.3** | **absolute floor** | 1192 | 10 | −0.028 | **−0.138** | **+0.106** |
| abs0.5 | absolute floor | 554 | 15 | +0.017 | −0.023 | +0.061 |
| abs0.7 | absolute floor | 357 | 25 | +0.059 | +0.049 | −0.032 |

All ρ/partial carry bootstrap 95% CIs (in the job log) tight to ±0.01 — these are stable, not noise.

> **Result — the leaf-count null is rule-general.** Across **every** new mode and level, the raw pruned
> `n_total` (= `n_frontier`, since `n_expanded` is ≈const) **does not beat its legal-moves floor**. The closest,
> `rank2` at +0.241, sits *below* its +0.250 floor and has partial\|legal ≈ 0 — it just re-tracks width through a
> fixed-fan geometry. Rank pruning *cannot* carry satisfaction by construction (it is value-blind), and indeed
> doesn't. The step-2 conclusion holds for absolute and rank pruning too: **raw node-count is the wrong cost.**

> **The one new signal — `abs0.3`.** Absolute pruning at 0.3 yields the **largest beyond-width partial in the
> whole experiment**: n_total partial\|legal = **−0.138** [−0.146,−0.129] and max_depth partial = **+0.106**
> [+0.097,+0.113]. The negative count-partial is a *satisfaction* channel (more moves clearing an absolute bar →
> decide faster), the positive depth-partial a *sharpness/forcing* channel (narrower ⇒ deeper search ⇒ longer
> think). **But this is the rule we rejected on principle:** an absolute floor entangles the surviving-count with
> the position's **absolute evaluation** (winning-ness), a known RT correlate that is *not* the satisficing
> consideration-set mechanism. The −0.138 is suggestive but **suspect as a construal measure** until the
> winning-ness confound is partialled out — that check gates any downstream use.

> **The robust new pattern — depth, not count.** `max_depth` / `mean_leaf_depth` partial\|legal is **positive
> across nearly every mode** (rank4 +0.084, rank8 +0.089, abs0.3 +0.106, abs0.5 +0.061): wherever pruning frees
> the budget to **go deep on forcing lines**, *how deep it goes* tracks RT beyond width — more consistently than
> any leaf-count. The project's RT decomposition never used **calculation depth** as a cost; this grid says it
> may be the better one. *(Carried as a candidate cost for the step-4 oracle, alongside the pruned leaf-cost.)*

## Stage 2d — the engine-derived (no-refit) analyses, staged

`lmcos_small/pipeline/2d_data_analysis.slurm` runs the param-free chain (budgeted-oracle VOC signals +
softmax-VOC τ sweep) on a tree SET over the **unfiltered** population → `sf_analysis/<SET>/`; `2d_figures.slurm`
makes the plots. Launched on all 5 md36 sets; figures regenerate on `n1md36`. The halt-policy zoo + hindsight
halter stay in `3_train`/`4_eval` (they need training).
