---
theme: default
title: GNN encoder pretraining
info: Child-WDL supervised pretraining of the tree-GNN encoder.
addons:
  - "@/shared/slidev-addon-base"
css: ./style.css
class: mdl-cover
mdc: true
math: katex
---

<span class="mdl-kicker">Meta-control of tree search</span>

# GNN encoder pretraining

<div class="mt-2 text-lg opacity-80">Child-WDL supervised pretraining of the tree-GNN encoder</div>

<div class="mt-4 text-sm opacity-50">Source: <code>reports/gnn-pretrain.md</code> (R-PRETRAIN)</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Overview</span>The question &amp; the result</div>

<div class="mdl-content">
  <div class="mdl-two">
    <div class="mdl-card mdl-card--neutral">
      <p class="mdl-card-h">Question</p>
      <p class="body">Can the tree-GNN encoder be supervised-pretrained on the child-edge-WDL objective, and does the loop actually work?</p>
    </div>
    <div class="mdl-card mdl-card--accent">
      <p class="mdl-card-h">Answer</p>
      <p class="body"><strong>Yes.</strong> Smoke run on A100 — loss descends, no bugs.</p>
      <div class="mdl-found">Dataset decision is <em>forced</em>; the headline caveat is <strong>timing</strong> — must pre-pack before the full run.</div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig mdl-titlefig--wide">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Smoke run</span>Loss descends, the loop works</div>
      <table class="mdl-table mdl-table--wide">
        <thead>
          <tr><th>epoch</th><th class="num">train loss_gap</th><th class="num">val loss_gap</th></tr>
        </thead>
        <tbody>
          <tr><td>1</td><td class="num">0.748</td><td class="num">0.495</td></tr>
          <tr><td>3</td><td class="num">0.400</td><td class="num">0.351</td></tr>
          <tr class="hl"><td>10 (best val)</td><td class="num">0.291</td><td class="num">0.306</td></tr>
          <tr class="dim"><td>15</td><td class="num">0.336</td><td class="num">0.425</td></tr>
        </tbody>
      </table>
      <div class="mdl-text" style="margin-top:0.7rem">256 train / 64 val, weighted child-WDL CE, Adam lr = 1e-2, A100.</div>
    </div>
    <div class="mdl-text" style="align-self:center">
      <ul>
        <li>Train loss_gap <strong>0.748 → 0.291</strong> vs a ≈0.17-nat entropy floor — <strong>functioning, no bugs.</strong></li>
        <li>Val diverges after ep ~10 — <strong>expected &amp; healthy</strong> (lr = 1e-2 overfits 256 trees). For the full run drop to lr ≈ 1e-3.</li>
      </ul>
      <div class="text-xs opacity-50 mt-3">Figure: <code>lmcos/analysis/figures/pretrain_smoke_loss.png</code></div>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-content">
  <div class="mdl-titlefig mdl-titlefig--wide">
    <div class="mdl-tf-left">
      <div class="mdl-title"><span class="mdl-kicker">Data</span>The dataset decision is forced</div>
      <table class="mdl-table mdl-table--wide">
        <thead>
          <tr><th>Dataset</th><th class="num">Count</th><th><code>edge_wdl_targets</code>?</th></tr>
        </thead>
        <tbody>
          <tr class="dim"><td>ysagiv <code>human_trees</code></td><td class="num">1,135,085</td><td>❌ none (scalar only)</td></tr>
          <tr class="hl"><td><code>lc0_trees</code></td><td class="num">150,003</td><td>✅ (E,3), sums to 1, no NaN</td></tr>
        </tbody>
      </table>
    </div>
    <div class="mdl-text" style="align-self:center">
      <ul>
        <li><code>ChildWdlPretrainer</code> <strong>hard-requires</strong> edge targets; the node-target path was removed (<code>69d024e</code>).</li>
        <li>ysagiv trees can't drive the current loss without new code (different semantics).</li>
        <li>→ <strong>Train on <code>lc0_trees</code>.</strong> No reason to re-derive targets we already have.</li>
      </ul>
    </div>
  </div>
</div>

---
layout: default
class: mdl-slide
---

<div class="mdl-title"><span class="mdl-kicker">Caveat</span>The timing caveat (load-bearing)</div>

<div class="mdl-content">
  <div class="mdl-found danger" style="margin:0 0 1.1rem">
    Measured <strong>56 s/epoch</strong> on 256 trees — CPU-tensorization-bound, not GPU-bound (raw loader, <code>num_workers=0</code>). Naive scale to 150K → <strong>~9 h/epoch. Do NOT run raw.</strong>
  </div>
  <div class="mdl-steps">
    <div class="mdl-step">
      <div class="mdl-step-h"><span class="mdl-step-n">1</span><span class="mdl-step-t">Pre-pack</span></div>
      <ul><li>Pack the 150K trees once (<code>preprocess_gnn.pack</code> → tensorized shards), as ysagiv does.</li></ul>
    </div>
    <div class="mdl-step">
      <div class="mdl-step-h"><span class="mdl-step-n">2</span><span class="mdl-step-t">Measure</span></div>
      <ul><li>One packed epoch for the real per-epoch wall-clock (expect 1–2 orders faster).</li></ul>
    </div>
    <div class="mdl-step">
      <div class="mdl-step-h"><span class="mdl-step-n">3</span><span class="mdl-step-t">Launch</span></div>
      <ul><li>lr ≈ 1e-3, weighted CE, resume + bucketed-KL on, <code>qos=gpu-short/medium</code>.</li></ul>
    </div>
  </div>
  <div class="text-xs opacity-60 mt-4">#epochs-to-converge must be read off the <strong>real run's</strong> val loss_gap plateau, not this overfit smoke.</div>
</div>
