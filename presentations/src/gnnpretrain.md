---
theme: default
title: GNN encoder pretraining
info: Child-WDL supervised pretraining of the tree-GNN encoder.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: text-left
mdc: true
math: katex
---

# GNN encoder pretraining

<div class="mt-4 text-lg opacity-80">Child-WDL supervised pretraining of the tree-GNN encoder</div>

<div class="mt-6 text-sm opacity-60 max-w-2xl">
Source report: <code>reports/gnn-pretrain.md</code> (R-PRETRAIN)
</div>

---
layout: default
---

# The question & the result

<div class="mt-6 grid grid-cols-2 gap-8 text-sm">
<div>

**Q.** Can the tree-GNN encoder be supervised-pretrained on the child-edge-WDL objective, and does
the loop actually work?

</div>
<div>

**A.** **Yes.** Smoke run on A100 — loss descends, no bugs. Dataset decision is *forced*; the
headline caveat is **timing** (must pre-pack before the full run).

</div>
</div>

---
layout: default
---

# Smoke: loss descends, loop is functioning

<div class="mt-4 text-base opacity-90">

256 train / 64 val, weighted child-WDL CE, Adam lr=1e-2, A100:

| epoch | train loss_gap | val loss_gap |
|---|---|---|
| 1 | 0.748 | 0.495 |
| 3 | 0.400 | 0.351 |
| 10 (best val) | **0.291** | **0.306** |
| 15 | 0.336 | 0.425 |

</div>

<div class="mt-4 text-sm opacity-80">

- Train loss_gap **0.748 → 0.291** vs a ≈0.17-nat entropy floor — **functioning, no bugs.**
- Val diverges after ep ~10 — **expected & healthy** (lr=1e-2 overfits 256 trees). For the full run
  drop to lr ≈ 1e-3.

</div>

<div class="mt-3 text-xs opacity-50">
Figure: <code>lmcos/analysis/figures/pretrain_smoke_loss.png</code>
</div>

---
layout: default
---

# Dataset decision is forced

<div class="mt-4 text-sm opacity-90">

| Dataset | Count | `edge_wdl_targets`? |
|---|---|---|
| ysagiv `human_trees` | 1,135,085 | ❌ **none** (only scalar `node_targets`) |
| **`lc0_trees`** | **150,003** | ✅ shape (E,3), sums to 1, no NaN |

</div>

<div class="mt-6 text-sm opacity-80">

- `ChildWdlPretrainer` **hard-requires** edge targets; the node-target path was removed (`69d024e`).
- ysagiv trees cannot drive the current loss without new code (different semantics).
- → **Train on `lc0_trees`.** No reason to re-derive targets we already have.

</div>

---
layout: default
---

# The timing caveat (load-bearing)

<div class="mt-4 text-sm opacity-90">

- Measured **56 s/epoch** on 256 trees — **CPU-tensorization-bound, not GPU-bound** (raw loader,
  `num_workers=0`).
- Naive scale to 150K → **~9 h/epoch. Do NOT run raw.**

</div>

<div class="mt-6 text-base opacity-90">

**Full-run plan**

1. **Pre-pack** the 150K trees once (`preprocess_gnn.pack` → tensorized shards), as ysagiv does.
2. Measure **one packed epoch** for the real per-epoch wall-clock (expect 1–2 orders faster).
3. Launch: lr ≈ 1e-3, weighted CE, resume + bucketed-KL on, `qos=gpu-short/medium`.

</div>

<div class="mt-4 text-sm opacity-70">

#epochs-to-converge must be read off the **real run's val loss_gap plateau**, not this overfit smoke.

</div>
