# Minimal MC plan — diagnose why GNN/MC loses to its own nested baselines, then rebuild small & transparent

**Status:** Interview answered (2026-06-23). Resolutions below; §8 retains your inline answers.

## 0. Resolutions (from your answers)
- **Q1 → build BOTH controller variants, both fit/selected on REGRET, same split, same eval:**
  (a) the **full GNN/MC** head (`z_t, T_t`) trained to directly minimize regret; (b) a **1-parameter
  budget-only head** — advantage computed from the budget scalar alone (ignores `z_t`). (b) is the
  nested lower bound (a) must beat; if (a) ≯ (b), the root state adds nothing for stopping. This is
  THE scientific test.
- **Q2 → decoy heads through the EXISTING controller greedy-eval harness** (that's the harness we're
  trying to trust). First action.
- **Q3 → trivial hand-crafted root feature first**, then a tiny GNN.
- **Q4 → YES**: `f*` and GNN/MC fit on the same objective + same split, evaluated on the same held-out.
- **Q5 → GPU fine; CPU for the small models.** No SLURM/overnight for Phases 0–2.
- **Q6 → random subset of the CURRENT filtered tree set.**
- **Q7 → keep current artifacts** for reference (now documented in `mc_pipeline.md`).
- **Q8 → optimize Phases 0–2 for iteration speed + transparency**, chase numbers only in Phase 3.
- **Q9 → overnight re-pack chain SHELVED** (never launched).
- **Provenance complaint → `mc_pipeline.md` created** as the single authoritative lineage record;
  going forward, no GNN/MC instance is "kept" without a lineage entry.

---

## 1. Goal (restated)

Train a **normative meta-controller** — GNN encoder (root state `z_t`) + MC readout head — that
says **when a deliberator *ought* to stop**, given (a) what it can see of the current search tree
and (b) the time budget it knows in advance. The benchmark is **regret vs the budgeted oracle**
(`regret = oracle_value − return_for_stop_step`), lower is better, plus P(stop == OSS).

## 2. The mystery, reframed by your nested-model argument

Always-Stop, Never-Stop, and Fraction-of-Budget are **strict special cases** of the GNN/MC
hypothesis class. The MC head maps `(z_t, T_t) → advantage → {STOP, CONT}`:

- **Never-Stop** = output CONT always (ignore `z_t`, `T_t`).
- **Always-Stop** = output STOP always.
- **Fraction-of-Budget(f)** = STOP iff `N_t ≥ f·B` — a threshold on the budget only (ignore `z_t`).

So the controller can represent every baseline by **ignoring the root state**. With a consistent
eval harness and a regret-aligned objective, its achievable regret is therefore **≤ min(baselines)**.
**Any** case where it loses is structurally suspect. Overfitting cannot plausibly produce a 3.7×
gap (0.077 → 0.21–0.29) for a 1-D-ish decision head selected on validation regret.

**The one legitimate, non-bug way it can still lose:** the controller is **not trained on the metric
it is judged by.** Fraction-`f*` is fit by *directly minimizing regret* on train; the GNN/MC is fit
on a **surrogate** (advantage-MSE + sign-BCE) and then has its stop decision read off a **fixed,
untuned threshold** (`advantage ≤ 0`). Minimizing the surrogate ≠ minimizing regret — especially
where the advantage target is ≈ 0 (the subagent found ~50% of targets have |adv| < 0.01, i.e. the
decision boundary is exactly where the signal is weakest). A head that predicts adv ≈ 0 to minimize
MSE makes near-random STOP/CONT calls at the boundary. **This is my leading hypothesis: objective /
decision-rule mismatch, not capacity or overfitting.**

## 3. Ranked hypotheses (what Phase 0 will confirm/refute)

1. **H-OBJ (objective/threshold mismatch).** Surrogate loss + fixed `adv≤0` threshold ≠ regret
   minimization. Most likely; fully consistent with "nested yet worse."
2. **H-HARNESS (apples-to-oranges eval).** The GNN/MC greedy-eval and the baseline eval compute
   regret on subtly different episodes / stop-step semantics / return conventions. Must be ruled out
   *first* — if present, every number so far is untrustworthy.
3. **H-TARGET (advantage targets aren't the optimal-stopping Q).** If targets are myopic one-step
   `continue−halt` rather than the oracle's global DP values, the `adv≤0` rule is myopic, not OSS.
4. **H-SCALE (feature scaling).** Raw `T_t`/`N_t` swamp `z_t`; real, already patched, but you
   doubt it explains the whole gap. Keep as contributor, not prime suspect.
5. **H-OPT (undertraining).** 3 epochs, sign barely learned — but cannot by itself make the model
   *worse than Always-Stop*, so it's downstream of H-OBJ/H-HARNESS.

## 4. Plan (phased)

### Phase 0 — Harness sanity via **decoy heads** (your idea; do FIRST; ~1–2 h, my work)
The decisive, cheap test that separates H-HARNESS from H-OBJ/H-TARGET:

- Build three **decoy MC heads** that *ignore `z_t`* and implement Always / Never / Fraction(f)
  exactly, then run them **through the GNN/MC greedy-eval harness** (same code path the trained
  controller uses).
- **If** their regrets reproduce the standalone `alt_models_eval` baselines → the harness is
  consistent (H-HARNESS refuted); the gap is H-OBJ/H-TARGET, and we've *proven* the class contains
  a 0.077-regret solution that training failed to find.
- **If** they differ → H-HARNESS confirmed: the comparison was never fair; fix the harness before
  anything else.
- In parallel: a **line-by-line audit** of the two regret computations (episode set, `oracle_value`,
  stop-step indexing, `return_for_stop_step`) and of the **advantage-target construction**
  (myopic vs DP-Q) — addresses H-TARGET.

**Deliverable:** a one-page "where the gap comes from" verdict with numbers.

### Phase 1 — **Minimal, transparent, ~30-min** GNN+MC (the real ask)
A single, readable script (`minimal_mc.py`) — no SLURM, no 7-stage pipeline — that runs end-to-end
on a **small tree subset** and logs every number's provenance:

1. Load a small subset (proposed ~3–5k trees) → build episodes.
2. **Root features:** start with the GNN frozen *or* (proposed) a **trivial hand-crafted root
   vector** first, to isolate the MC-head/objective logic from the encoder entirely; then swap in a
   **tiny GNN** (d_embed≈32, 1 layer). (See Q3.)
3. Train the MC head (small MLP) — **and** fit `f*` — on the **same train episodes**.
4. Evaluate **all** models (Always/Never/Fraction/GNN-MC/decoys) through **one** regret function on
   **one** held-out set.
5. Print a single comparison table + save the figure. Target wall-clock **≤ 30 min**, ideally CPU.

This *is* the documentation: one file = the whole end-to-end process, runnable and inspectable.

### Phase 2 — Fair comparison + objective alignment (two-head design, per Q1)
- Put **every** model behind one eval entrypoint on one split (kills H-HARNESS permanently).
- Train/select **on regret directly** (empirical regret minimization over the head's class →
  nestedness guarantees ≥ baselines up to overfit), for **both**:
  - **(b) 1-param budget-only head** — advantage = g(budget; one scalar θ). This is Fraction-as-a-head;
    its fitted regret should ≈ `f*` and is the nested floor.
  - **(a) full GNN/MC head** — advantage = MLP(`z_t, T_t`). Must satisfy regret(a) ≤ regret(b);
    the gap regret(b) − regret(a) **is the value of the root state** for normative stopping.
- Re-check H-TARGET: confirm the advantage targets are the oracle's optimal-stopping (post-DP) values
  — already partly confirmed (`continue_value` is post-DP, `mc_pipeline.md` §8).

### Phase 3 — Scale back up
Only once the minimal GNN/MC **provably ≥ baselines**, re-introduce the full tree set / bigger
encoder, reusing the *same* eval entrypoint. No scaling before the small case is correct.

## 5. Provenance / documentation deliverable
- `minimal_mc.py` = the single source of truth for the small pipeline (config + run log emitted).
- A short `mc_pipeline.md` recording the **exact** end-to-end lineage of any GNN/MC instance we
  keep (tree set → encoder → materialize → controller → eval), with commands + commit hashes, so
  no number is ever orphaned again. This directly answers the opacity complaint.

## 6. Parallelism & where things run
- **Parallel now:** Phase 0 decoy-head test ∥ Phase 0 code audit ∥ scaffolding `minimal_mc.py`.
- Phase 1 depends on the Phase 0 verdict only for *where* to focus (objective vs harness), not to
  start coding.
- Everything in Phases 0–2 is small enough to run **interactively** (CPU or 1 GPU); no SLURM,
  no overnight. Phase 3 returns to the cluster.

## 7. Concerns & assumptions
- **C1.** H-OBJ likely requires changing the *training/selection objective* (surrogate → regret-
  aligned). That's a design change, not a bugfix — I want your explicit buy-in (Q1).
- **C2.** I cannot currently *guarantee* the two existing eval paths are identical; the minimal
  pipeline's single eval function is how I make fairness structural rather than hoped-for.
- **C3.** The advantage target is ≈0 exactly at the decision boundary; even a correct objective may
  need either a margin/decision-focused loss or a regret-direct objective. (Note: `min_decision_margin`
  is NOT ysagiv's — they used 0.0 — so I will not reach for it without your say-so.)
- **A1.** `regret = oracle_value − return_for_stop_step` on the held-out split is THE metric for all
  models. **A2.** The budgeted-oracle OSS/`oracle_value` is the achievable ground truth. **A3.**
  Overfitting is not the prime cause. **A4.** A small tree subset preserves the qualitative result.

## 8. Outstanding questions (please answer line-by-line)

- **Q1 (the big one — objective).** Do you want the meta-controller **trained/selected to directly
  minimize regret** (guarantees ≥ baselines by nestedness), or must we keep the **fitted-Q
  advantage** surrogate and instead only **tune its decision threshold on regret**? Or keep the
  surrogate exactly as-is and treat any residual gap as the scientific finding? *(My rec: at least
  tune the threshold on regret in Phase 2; strongly consider a regret-aligned selection.)*

  Okay I think we ought to have both -- one that is directly fitted on the regret and the other which is literally a 1-parameter scalar that operates on the budget ONLY to compute the advantage.


- **Q2 (decoy heads).** Confirm Phase 0's decoy-head test through the **existing** controller_train
  greedy-eval is the right first move (vs building it only inside the new minimal harness). *(Rec:
  existing harness — that's what we're trying to trust.)*

  Build it through the existing harness! That's an obvious one.


- **Q3 (encoder in the minimal pipeline).** Start the minimal run with a **trivial hand-crafted root
  feature** to isolate the MC-head/objective, then add a **tiny GNN** — or insist on the GNN from the
  start? *(Rec: trivial-feature first; it's the fastest way to localize the bug.)*

  Trivial feature first, I guess.


- **Q4 (fairness).** Agree that `f*` and the GNN/MC must be fit on the **same objective and same
  train split**, evaluated on the **same held-out split**, for the nested comparison to be valid?

  YES


- **Q5 (compute / "30 min").** CPU-interactive, 1-GPU-interactive, or SLURM for the minimal loop?
  *(Rec: 1 GPU interactive, or CPU if the encoder is tiny.)*

  GPU is fine. For the smaller models you can use CPU


- **Q6 (tree subset).** Use a fresh small random sample of the **filtered** trees (~3–5k), or a
  specific subset you have in mind? Size preference?

  Use a random subset from our current tree set! Obviously.


- **Q7 (artifacts).** Keep the current full-pipeline artifacts (`encoder_best.pt`, `packed/mc`,
  `controller_filtered.pt`, the broken figures) for reference while we build the minimal one — yes?

  Yes!



- **Q8 (scope of "minimal").** Confirm we optimize Phases 0–2 for **iteration speed + transparency
  over performance** (tiny encoder, subset), and only chase numbers in Phase 3.
  
  Yes



- **Q9 (the overnight run).** I have NOT launched the overnight re-pack/normalization chain. Confirm
  you want it **shelved** in favor of this minimal-first plan. *(Rec: shelve it.)*

  Shelved

- **BROADER POINT** This entire pipeline that has been used to generate the current GNN / MC trees has been an absolute disaster from a provenance standpoint. A million tiny decisions have been made that have made it nearly impossible to trace the source of errors. First, I need a *COMPREHENSIVE* account of where the data came from, each step in which it was filtered, each step that went into training, what the architecture was, and what was done / altered in the packing, and then what went into the cost calculation when training the meta-controller. Forgive me for my bluntness but why did you choose to do it this way? It's neither conscientious nor interpretable. Why is there no place that records all these changes step by step? Why don't we have a record that lays out exactly how this present iteration of the GNN / MC was trained, what datasets, etc. for reproducibility? Why are they strewn about in a million different .yaml files? I am completely at a loss for why this was the chosen way to go about this. 
