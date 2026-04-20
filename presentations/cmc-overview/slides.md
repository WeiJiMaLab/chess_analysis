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
  @apply text-2xl font-bold mb-4 text-slate-800;
}
h2 {
  @apply text-lg font-semibold mb-2 text-slate-600;
}
p, li {
  @apply text-sm text-slate-700 leading-relaxed;
}
.caption {
  @apply text-xs text-slate-500 italic mt-2;
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
layout: section
---

# Part 3 — Sanity Check: Human Data

---
layout: two-cols
layoutClass: gap-8
---

# Move times are heavy-tailed

Humans exhibit highly variable thinking times. Most moves are quick, while a long tail of "deep thinks" dominates.

We work in **$\ln(T)$** to stabilize variance; effects read as multiplicative changes (%).

::right::

<img class="rounded border border-slate-200" src="/figures/clock_move_analysis/move_time_distribution.png" alt="Raw and log move time distributions" />

---
layout: two-cols
layoutClass: gap-8
---

# The "Hump" in the Middle

Thinking demand is not flat. Openings and endgames are often fast; the tactically rich **Mid-game** requires the most depth (VOC).

::right::

<img class="rounded border border-slate-200" src="/figures/ply_analysis/ply_impact_comparison.png" alt="Log move time vs ply" />

---
layout: two-cols
layoutClass: gap-8
---

# Value of Computation (VOC)

**VOC** is the gain in win probability from deep search over a shallow read.

People spend significantly more time in positions where $\sqrt{\text{VOC}}$ is high—a behavioral proof of Meta-Control.

::right::

<img class="rounded border border-slate-200" src="/figures/voc_analysis/voc_quad_view.png" alt="VOC quad view" />

---

# Summary: Machine vs. Human

| Feature | Engine Meta-Control | Human Behavior |
| :--- | :--- | :--- |
| **Budget** | Linear Expansion Cost ($C$) | Clock Time Left |
| **Demand** | Value Gain / Convergence | Value of Computation (VOC) |
| **Signal** | GNN Tree Summary | Board Intuition |
| **Goal** | Peak Net Reward $R(k)$ | Winning under Clocks |
