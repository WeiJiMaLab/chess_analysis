---
theme: default
title: Chess Meta-Control (CMC)
info: |
  A didactic breakdown of the CMC architecture: Control Flow · Representation · GNN Mechanics · Decision · Training.
class: text-left
mdc: true
font:
  sans: 'Inter'
  mono: 'Fira Code'
---

<style>
.slidev-layout.default,
.slidev-layout.section,
.slidev-layout.center,
.slidev-layout.two-columns {
  padding: 3.5rem 5rem !important;
}
h1 {
  @apply text-3xl font-bold mb-6 text-slate-800 tracking-tight;
}
h2 {
  @apply text-xl font-semibold mb-4 text-slate-600;
}
p, li {
  @apply text-lg text-slate-700 leading-relaxed;
}
.caption {
  @apply text-sm text-slate-500 italic mt-4;
}
.takeaway {
  @apply mt-6 p-4 bg-slate-50 border-l-4 border-indigo-500 text-slate-700 font-medium rounded-r-lg;
}
</style>

# Chess Meta-Control (CMC)

**Optimizing the Economy of Thought in Complex Search**

<div class="mt-40 text-sm opacity-60">
`lmcos/demos/understanding.md` · `project.md`
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-indigo-600 font-bold uppercase tracking-widest text-xs mb-2">Section I</div>
    <h1 class="text-4xl text-slate-800">Part 1 — The Problem</h1>
  </div>
</div>

---

# Why Meta-Control?

<v-clicks>

- **Fixed Budgets are Wasteful:** Standard engines spend the same time on a "forced" move as a complex tactical blunder.
- **Human Intuition:** Skilled players know *when* to stop thinking—a stopping problem we can formalize.
- **The Trade-off:** Is the move-quality I’m about to discover worth the computational "electricity" (time/tokens) I’m about to spend?

</v-clicks>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-indigo-600 font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl text-slate-800">Part 2 — The Methods</h1>
  </div>
</div>

---

# 1. Control Flow: The Loop

The Meta-Controller sits on top of a standard MCTS planner (like Leela), deciding at each step whether to expand further or act.

<div class="mt-8">
  <LeelaSearchLoop />
</div>

---

# 2. Representation & Architecture

<div class="mt-8">
  <MetaControllerZoom />
</div>

---

# 3. Learning: Bidirectional Sweeps

The GNN performs representation learning over the tree using two sequential passes.

<div class="mt-4">
  <GnnTwoSweeps />
</div>

<v-clicks>

- **Initialization:** Every node begins with its state set to its feature vector: $h^{(0)} = x$.
- **Upward (Evidence Funnel):** Children aggregate discovery into the parent via Attention + GRU.
- **Downward (Global Broadcast):** The root's summary is broadcast back down to give branches global context.

</v-clicks>

---

# 4. Decision: The Halt Controller

We frame search depth as a learnable policy $\pi_\theta$. The **Halt Controller** solves a stopping problem by maximizing the net value of computation.

**The Economy of Thought**
The agent seeks an optimal depth $k^*$ that maximizes the expected reward:
$$ R(k) = \mathbb{E} \left[ \text{Value}(T_k) \right] - C \cdot k $$

**Mechanistic Readout**
- **Input:** The current root hidden state $\mathbf{h}_{root}$.
- **Output:** A halting probability $P(\text{halt} | \mathbf{h}_{root})$ predicted by an MLP.
- **Optimization:** Trained via Policy Gradients ($\nabla_\theta J$) against a DP Oracle to learn the inflection point of the "Thinking Curve."

---

# 5. Training: The Supervised Split

We use a two-stage pipeline to build an "Intuitive Engine."

<v-clicks>

1. **Pre-training (GNN Backbone):** 
   - **Task:** Predict the future (Oracle WDL) from partial trees.
   - **Signal:** Every edge in the tree provides a training signal (**ChildWDL**).

2. **Policy Tuning (Halt Head):**
   - **Task:** Solve the "Economy of Thought."
   - **Signal:** Trained via **PPO** on live traces using the **DP Oracle** (Thinking Curve peak).

</v-clicks>

---

# 1. Move times are heavy-tailed

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/move_time_distribution.png" />
  
  <div class="takeaway border-slate-300 bg-slate-50/50 text-base py-6">
    <b>Key Takeaway:</b><br><br>
    Most moves are near-instant, but the "long tail" of deep thinks dominates variance. 
    <br><br>
    Log-transforming to <b>$\ln(T)$</b> is required to stabilize variance and isolate the behavioral signal.
  </div>
</div>

---

# 2a. Elasticity: The Naive Aggregate

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/attempt1_naive_trend.png" />
  
  <div class="takeaway border-blue-500 bg-blue-50/50 text-base py-6">
    <b>Attempt 1 (Primary Confound): Opening Theory</b><br><br>
    The raw data shows a shallow positive trend ($\beta \approx 0.09$). 
    <br><br>
    However, this is corrupted by opening moves where players have maximum clocks but move instantly due to preparation.
  </div>
</div>

---

# 2b. Elasticity: The Ply Paradox

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/attempt2_ply_wise_trend.png" />
  
  <div class="takeaway border-red-500 bg-red-50/50 text-base py-6">
    <b>Attempt 2 (Secondary Confound): Selection Bias</b><br><br>
    Controlling for ply reveals a paradox: while aggregate bins look positive, <b>within-ply slopes are negative</b>. 
    <br><br>
    Faster players (who maintain higher clocks) dominate the high-clock buckets, masking the true relation.
  </div>
</div>

---

# 2c. Elasticity: The Resolution

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/clock_move_analysis/attempt3_controlled_trend.png" />
  
  <div class="takeaway border-emerald-500 bg-emerald-50/50 text-base py-6">
    <b>Attempt 3 (The Thinking Hypothesis): Fixed-Effect Control</b><br><br>
    Accounting for <b>both</b> player identity and game stage resolves the paradox.
    <br><br>
    <span class="text-emerald-700 font-bold text-xl">$\beta \approx 0.54$</span><br>
    A 10% increase in clock time leads to a ~5.4% increase in the thinking budget.
  </div>
</div>

---

# 3. Value of Computation (The Demand)

<div class="grid grid-cols-2 gap-12 mt-12 items-start">
  <img class="w-full object-contain" src="/figures/voc_analysis/voc_quad_view.png" />
  
  <div class="takeaway text-base py-6">
    <b>Key Takeaway:</b><br><br>
    <b>VOC</b> measures the potential gain from deep engine search over a shallow read.
    <br><br>
    <b>$\ln(T) \propto \sqrt{VOC}$</b>: Humans spend the most "thought-capital" on positions where depth matters most.
  </div>
</div>
