# R-VOC-MECH — VOC mechanism: what the subtree-weighted controller reads

**Ref:** `R-VOC-MECH` · [Index](README.md) · Thread: LMCOS · Related: [(R-LMCOS-STAGE4)](lmcos-stage4-ablation.md), [(R-ARCH-LMCOS)](archive-lmcos-notebook-legacy.md)

## Summary

| | |
|---|---|
| **Description** | Behavioral + constructed-tree diagnostics on the subtree-weighted-encoder controller (`fittedq_subtree_weighted_zt_tt`, inputs `[z_t, T_t]`, greedy regret 0.024). The rerun-encoder `[z_t, T_t]` controller (`fittedq_rerun_encoder_zt_tt_ablation`) is the negative control throughout. |
| **Rationale** | Decide whether the controller does genuine metacontrol — its stop decision causally reads the child-WDL value landscape encoded in `z_t` — versus a search-progress shortcut (the "z encodes `N_t`" null, a step-counter heuristic). |
| **Expectation** | If metacontrol: stopping depends on the value landscape even with tree size held fixed. If a shortcut: stopping tracks `N_t` and is insensitive to value once size is controlled. |
| **Finding** | ✅ **Metacontrol.** On constructed trees with value ⊥ size, corr(advantage, best) = **−0.60..−0.78**, corr(advantage, size) ≈ **−0.1**; `N_t` is decodable from `z_t` at **R²=0.68** but **unused**. Rule: stop when budget is low, else continue while the field of near-best candidates is broad. The rerun control reads value **incoherently** (wrong-signed), so subtree-weighting pretraining is what makes the value structure usable. ❌ A linear-perturbation probe gave a wrong "reads progress" result (artifact), overturned by the constructed-tree probe. ⏳ Corrected response-surface + halt-trajectory reruns pending (Della maintenance). |

## Procedure

| Step | Status |
|------|--------|
| WDL-subspace ablation — greedy-regret differential (`wdl_subspace_ablation.py`, 9 unit tests) | ✅ done |
| Per-snapshot readout table + decode check (`value_readout_table.py`) | ✅ done |
| Descriptive advantage-vs-feature curves (`value_readout_summarize.py`) | ✅ done |
| Linear value-perturbation probe (`value_perturbation.py`) | ❌ wrong result — superseded |
| Matched-snapshot conditional (`value_conditional.py`) | ⬜ built, not run — superseded by constructed trees |
| Constructed-tree probe, value ⊥ size (`synthetic_trees.py`) | ✅ done — **decisive** |
| Response-surface — single-ε candidate sweep (`value_response_surface.py`) | ⏳ pending rerun |
| Halt-trajectory — on-distribution trigger, halt ≥ 5 (`halt_trajectory.py`) | ⏳ pending rerun |
| Network-internals — head weights / hidden activations | ⬜ todo |

---

## WDL-subspace ablation (2026-06-01)

Causal greedy-regret instrument. SVD the child-WDL decoder's root readout (`mlp.0.weight[:, :d_embed]`) → orthonormal basis **V** of `z_t` directions ordered by decoder singular value. For a k-sweep: KEEP-ONLY (project `z_t` onto `span(V[:, :k])`) and ABLATE (the orthogonal complement); re-run the frozen advantage head over the materialized cache and score with the trainer's exact stop-when-advantage≤0 rule (greedy regret + snapshot sign accuracy). Run as a **differential**: subtree-weighted vs rerun encoder under the same `N_t`-free `[z_t, T_t]` pipeline.

Integration check: baseline (no projection) reproduces the recorded best-val regret exactly — subtree-weighted **0.0230**, rerun **0.2186** — over 953,562 snapshots / 30,630 episodes.

**Functional result (the positive evidence):**
- **Subtree-weighted** — ablating the top-k WDL directions monotonically degrades regret (0.023 → 0.050 @k=8 → 0.061 @k=16 → 0.145 @k=64) and the halt/continue sign bit (0.920 → 0.821 → 0.797 → 0.690). KEEP-ONLY the WDL subspace rebuilds **sub-floor** competence: the top-64 directions alone give regret **0.099**, well below the **0.218** `T_t`-only floor. Signal is distributed (top-1 alone is useless: regret 0.450).
- **Rerun (control)** — baseline 0.219 *is* the `T_t`-only floor (sign 0.580 ≈ chance); KEEP-ONLY never breaks the floor (best 0.273 @k=64). No sub-floor competence to attribute.
- Full ablation (k=128, zeroing all `z_t`) blows up for both (0.804 / 0.403) — off-distribution for a head trained on full `z_t`, not a WDL-specific effect; read the small/mid-k range.

**Structural result (the counter-evidence):** the head's `z_t`-weight energy in the top-k WDL subspace sits **~at chance for both** controllers (subtree-weighted 1.05–1.25× chance; rerun 1.16–1.27×). Decoder singular spectra are flat (subtree-weighted top-8 = 18.2 → 11.2), so the "WDL subspace" is the k most decoder-relevant directions of a near-isotropic readout, not a sharp low-dimensional competition subspace.

The two instruments disagree informatively: functional ablation shows the subtree-weighted controller has genuine sub-floor competence carried by decoder-read `z_t` directions that the rerun controller lacks; the static weight-alignment probe does not separate the two encoders. This left the `N_t`-proxy null **not yet refuted** — motivating the readout-characterization arc.

---

## Readout characterization arc (2026-06-03)

Goal: pin the function the controller computes from `z_t` — value landscape (metacontrol) vs search-progress proxy (z ≈ `N_t`) — and the form of the rule.

1. **Per-snapshot table + decode check** (`value_readout_table`). Re-encode the validation episodes; record advantage + decoded root-child values (W−L per child), 953,562 snapshots. Decode validated: corr(best-child value, halt_reward) = **0.92** (subtree-weighted), **0.84** (rerun) — the value primitives are recoverable from `z`.

2. **Descriptive** (`value_readout_summarize`). Within `T_t` bins, advantage correlates with best/spread more than the top-two margin, growing with budget (subtree-weighted best corr −0.55 at the top budget). But the rerun budget-heuristic controller reproduced most of the pattern (best −0.44) → the bulk is value↔progress collinearity, not proven value use.

3. **Linear perturbation — recorded as a cautionary WRONG result** (`value_perturbation`). Ridge-extract linear value/progress directions in `z`, move `z` along the progress-orthogonalized value direction, re-run the head. R²(`N_t`|`z`) = **0.68** (subtree) vs 0.14 (rerun). The perturbation showed the subtree controller barely responding to value and I concluded "reads progress, not value." **Artifact** — linear directions on a nonlinearly-encoded GNN embedding, value/progress collinear so the orthogonalized direction was near-degenerate, plus head near-saturation (stops 83%). Overturned by (5).

4. **Matched real snapshots** (`value_conditional`). Built (condition on true `N_t`, `T_t`; test the within-cell value effect); validated on synthetic data (matched stop-gap +0.01 for a step-counter vs +0.23 for a value-user at 0.75 collinearity). Not run on real data — superseded by the constructed-tree approach.

5. **Constructed-tree probe — decisive** (`synthetic_trees`). Build trees with the value landscape and node count set **independently** (value is a node feature), run through the real encoder→controller, decode to confirm. corr(decoded best, size) ≈ 0.05/0.01 — decorrelated by construction.
   - **Subtree-weighted:** corr(advantage, best) = **−0.60..−0.78** across budgets, corr(advantage, size) ≈ **−0.1** — uses value, correctly signed (winning → stop), barely uses size.
   - **Rerun:** corr(advantage, best) **wrong-signed** (+0.58 at high budget), leans on size more (−0.2..−0.4).
   - So the subtree controller reads the value landscape, not the step count; `N_t`, though decodable at R²=0.68, goes **unused** (decodable ≠ used).

6. **Response surface — the form** (`value_response_surface`). Sweep best-value / top-two margin / candidate-count separately at fixed size, per budget, x-axis = decoded value. Subtree-weighted: (a) budget gates — at `T_t`=4, advantage ≈ −0.25 flat (stop regardless); (b) **candidate-count is the largest value lever** — going from 1 to 5 comparable children raises advantage by +0.12/+0.24/+0.44/+0.48 at `T_t`=4/12/30/60 (more comparable candidates → continue); (c) best-value is a smaller, decreasing term (swing ~0.3); (d) **the top-two margin has no effect** (flat at every budget). Rerun: best-value wrong-signed, candidate-count flat.

**Conclusion.** The subtree-weighted controller does metacontrol, not step-counting. Its rule: **stop when budget is low; otherwise continue while several moves sit at comparable value (the candidate field is broad) and stop once it narrows**, plus a minor "more winning → stop" term; the top-two margin is unused. That is a rough value-of-computation proxy (a broad candidate field ≈ an unresolved decision ≈ more search worthwhile), not a refined expected-improvement estimate. The same architecture on the rerun encoder reads value incoherently, so the subtree-weighting pretraining is what made the value structure usable downstream.

**Methodological note.** The linear-perturbation analysis (3) gave the wrong answer; the decisive evidence came from constructed inputs that set value and progress independently. Do not cite (3) as a finding.

---

## Open

- **Response-surface rerun** — corrected single-ε candidate sweep (one ε=0.15; extras placed `candidate_gap`=0.05 < ε below best; counts 0–6, best fixed), replacing the earlier 0.3-gap tail test. ⏳ blocked on Della maintenance.
- **Halt-trajectory rerun** — on-distribution per-step probe aligned at the controller halt, restricted to episodes that searched ≥ 5 steps (median halt = 1 makes the trajectory features ill-defined otherwise). Tests whether the candidate-field collapse is the feature that moves at real halts. ⏳ blocked on Della maintenance.
- **Network-internals** — nothing above inspects the network itself; all of it is behavioral (input→output function mapping). Planned: capture the 3-layer head's hidden activations over validation snapshots, regress each decoded feature (candidate-count, best-value) on the activations per layer, and localize the candidate-count signal. ⬜ todo.

---

## Artifacts

**Code** (this repo, `lmcos/analysis/`): `wdl_subspace_ablation.py`, `value_readout_table.py`, `value_readout_summarize.py`, `value_perturbation.py`, `value_conditional.py`, `synthetic_trees.py`, `value_response_surface.py`, `halt_trajectory.py`; test `lmcos/tests/test_wdl_subspace_ablation.py`. The slurm launchers and per-run configs are **not** in the repo — they hardcode ysagiv-specific Della paths.

**Checkpoints** (ysagiv read-only): controllers `fittedq_subtree_weighted_zt_tt.pt`, `fittedq_rerun_encoder_zt_tt_ablation.pt`; paired decoders `tree_encoder_child_wdl_async_k1_{subtree_weighted,rerun}_decoder.pt`; manifest `controller_packed_combined_nomaint_no_xaba/validation_manifest.json`.

**Outputs** (ysagiv scratch, `…/CTS/analysis/voc_mechanism/`): `wdl_subspace_ablation_{subtree_weighted,rerun}.json`, `value_readout_table_{subtree_weighted,rerun}.npz`, `value_readout_summary_{subtree_weighted,rerun}.json`, `value_perturbation_{subtree_weighted,rerun}.json`, `synthetic_trees_{subtree_weighted,rerun}.{npz,json}`, `value_response_surface_{subtree_weighted,rerun}.json`. PDFs `[pending]` (matplotlib CXXABI/libstdc++ mismatch in the Python-3.14 CTS env).

> Note: this report was ported from the pre-migration `lmcos/LAB_NOTEBOOK.md` (2026-06-01 and 2026-06-03 entries) after the notebook was migrated to repo-root `labnotebook.md`.
