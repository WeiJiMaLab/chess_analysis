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

**Optimizing the Economy of Thought**

<div class="mt-20 text-sm opacity-60">

`lmcos/demos/understanding.md` · `project.md`

</div>

---
layout: section
---

# Part 1 — The Problem

---

# Why Meta-Control?

<v-clicks>

- **Fixed Budgets are Wasteful:** Standard engines spend the same time on a "forced" move as a complex tactical blunder.
- **Human Intuition:** Skilled players know *when* to stop thinking—a stopping problem we can formalize.
- **The Trade-off:** Is the move-quality I’m about to discover worth the computational "electricity" (time/tokens) I’m about to spend?

</v-clicks>

---
layout: section
---

# Part 2 — The Methods


---
---
# 1. Control Flow: The Loop

The Meta-Controller sits on top of a standard MCTS planner (like Leela), deciding at each step whether to expand further or act.

<div class="mt-8">
  <LeelaSearchLoop />
</div>

---
layout: two-cols
---

# 2. Representation & Architecture

To learn on a tree, we must first map board states into a fixed-size vector space ($x$) which feeds into our modular controller.

**Under the Hood**
- **Serialization:** Board $\to$ Leela $\to$ MLP $\to$ $x$.
- **GNN Backbone:** Processes the vectorized tree.
- **Readout Head:** Taps the root state to decide.

::right::

<div class="ml-4">
  <SearchTreeSnapshot />
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
layout: center
class: text-center
---

# 4. Decision: The Halt Controller

Once the tree is summarized at the **Root**, we attach a **HaltController** (Policy Head).

$$ R(k) = \text{Value}(T_k) - C \cdot k $$

It maps the Root Hidden State $h_{root}$ to a binary choice:
**CONTINUE** (Expand more) vs **HALT** (Play now).

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
