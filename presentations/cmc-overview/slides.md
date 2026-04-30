---
theme: default
title: Learned Meta-control of Tree Search
info: |
  A didactic breakdown of the CMC architecture: Control Flow · Representation · GNN Mechanics · Decision · Training.
css: ./style.css
class: text-left
mdc: true
math: katex
---

<ChessBackground />

<div class="absolute bottom-12 left-14 z-10 text-white">
  <h1 class="m-0" style="line-height: 1.0; font-size: 3rem; color: white !important;">
    Learned Meta-control<br>
    of Tree Search
  </h1>
  
  <div class="mt-2 text-lg opacity-80">
    Yotam Sagiv & Jordan Lei
  </div>

  <div class="mt-6 text-[10px] font-bold uppercase tracking-widest opacity-40">
    Mechanistic Interpretability · Resource Rationality · Chess Search
  </div>
</div>

---

# Paper Sketch

<div class="h-full flex flex-col justify-start mt-2">
  <PaperSketch />
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section I</div>
    <h1 class="text-4xl">Part 1 — The Problem</h1>
  </div>
</div>

---

# Why Meta-Control?

<v-clicks>

- **Fixed Budgets are Wasteful:** Standard engines spend the same time on a "forced" move as a complex tactical blunder.
- **The Trade-off:** Is the move-quality I’m about to discover worth the computational "electricity" (time/tokens) I’m about to spend?
- **Real-World Constraints:** Tokens cost money, but more importantly, **real-time decisions have a physical cost**. Most LLM tasks are sufficiently time-intensive that every token of "thought" must justify itself.
- **Human Intuition:** Skilled players know *when* to stop thinking—a stopping problem we can formalize.

</v-clicks>

---

# Part 2: High-Level Architecture

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <LeelaSearchLoop />
  
  <div class="mt-8 p-3 bg-neutral-soft border-l-2 border-accent italic text-[11px] opacity-80">
    Implementation details (GNN sweeps, DP Oracle, Halt Controller) are in the Appendix.
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Section II</div>
    <h1 class="text-4xl">Part 2 — Validation & Alignment</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      How do we know the meta-controller works as intended? Humans serve as an <b>existence proof</b> and a <b>target distribution</b> for efficient search.
    </div>
  </div>
</div>

---

# Behavioral data: how we chose these games

<div class="grid grid-cols-1 gap-4 mt-4 text-sm leading-relaxed max-w-5xl">
  <p>
    All timing plots use the same DuckDB sample: merged <b><code>selected_moves</code></b>, then analysis table <b><code>_selected_moves_nonzero_T</code></b> from <code>preprocess_data.py preprocess</code> (same notion as <code>movetime_analysis.py</code>).
  </p>
  <ul class="list-disc pl-6 space-y-2">
    <li>
      <b>Game list:</b> We restrict to game IDs listed in the <code>selected_games</code> table (in that same database). The exact inclusion rule (Elo, time control, etc.) is whatever that table encodes.
    </li>
    <li>
      <b>Extracting moves (<code>src/slurm/scripts/preprocess_data.py process_shard</code>):</b> Run from a Slurm array (<code>preprocess_shard.sbatch</code>), each job walks parquet under the archive root scoped to <code>selected_games</code>, and by default <b>drops an entire game</b> if any row has negative <code>move_time</code>. Shards merge into table <code>selected_moves</code>.
    </li>
    <li>
      <b>Launcher (<code>src/slurm/_preprocess.sh</code>):</b> Fresh temp dir → <code>select_games</code> → array <code>process_shard</code> → <code>merge</code> → <code>berserk</code> id pass → <code>preprocess</code> (<code>_selected_moves</code> / <code>_selected_moves_nonzero_T</code>).
    </li>
    <li>
      <b>Downstream analysis:</b> <code>preprocess_data.py preprocess</code> materializes <code>_selected_moves</code> (with <code>game_phase</code>, clocks, raw <code>move_time</code>, <b><code>n_possible_moves</code></b>). Histograms / dashboards (<code>move_time_summary.py</code>, <code>movetime_analysis.py</code>) use SQL for ln <i>T</i> and bins on top of <code>_selected_moves_nonzero_T</code> where appropriate.
    </li>
  </ul>
</div>

---

# 1. Move times: raw and log (deliberation only)

<div class="text-xs opacity-60 mb-2 -mt-2">Excludes premoves / <code>move_time = 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/move_time_summary/combined.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">move_time_summary</b><br><br>
    Histograms of T and ln T (SQL equal-width bins). Most mass is very fast; the tail is why we often use log time downstream.
  </div>
</div>

---

# 2. Move times: raw and log (with premoves)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--include_zeroT</code> — including zero-time moves.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/move_time_summary/combined_include_zeroT.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">vs §1</b><br><br>
    Strips the origin spike; “deliberation-only” when paired with the same filter in other plots.
  </div>
</div>

---

# 3. Game stage: ply vs. think time

<div class="text-xs opacity-60 mb-2 -mt-2">Excludes <code>move_time = 0</code> (same sample as dashboards in <code>movetime_analysis.py</code>).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/combined.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">ply_movetime</b><br><br>
    Left: mean ln T by integer ply. Right: by ply decile. Game stage, not clock.
  </div>
</div>

---

# 4. Remaining clock vs. think time — dashboard

<div class="text-xs opacity-60 mb-2 -mt-2">Player clock; excludes <code>move_time = 0</code>. Four-panel figure from <code>movetime_analysis.py</code> (saved separately from the heatmap).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">clock_movetime</b><br><br>
    2×2: binned log clock vs ln T, scatter + OLS, per-ply β of ln T on your log remaining clock.
  </div>
</div>

---

# 5. Remaining clock vs. think time — quantile heatmap

<div class="text-xs opacity-60 mb-2 -mt-2">Same sample as §4. Joint <code>ntile</code> bins of player clock × move ply; cell color = mean ln <i>T</i>, opacity = mass (see figure colorbar).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <div class="slide-quantile-heatmap-wrap min-w-0">
    <!-- <img src="/figures/clock_movetime/combined_quantile_heatmap.png" alt="" /> -->
  </div>
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">clock × ply</b><br><br>
    Reads like the static dashboards in <code>exploratory/heatmap_clock_ply.py</code>, but built inside the same <code>Analyzer</code> pipeline as §4.
  </div>
</div>

---

# 6. Intuition: per-ply β in the opening

<div class="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6 mt-2 items-start">
  <div class="text-sm leading-relaxed max-w-md space-y-3">
    <p>
      <b>Opening (~ply &lt; 50):</b> per-ply β is often <b>negative</b> — more time left, <i>shorter</i> thinks.
    </p>
    <p class="text-slate-600">
      <b>Pace</b> confounds the axes: long thinkers are low on clock; fast movers bank time. Sketch: <i>x</i> = your remaining clock, <i>y</i> = this move (e.g. ln <i>T</i>).
    </p>
    <p class="text-xs opacity-75">First few plies: sign can flip (premoves / edge noise).</p>
  </div>
  <div class="shrink-0 w-full max-w-[300px] mx-auto lg:mx-0 p-3 rounded-lg border border-slate-200 bg-slate-50/80 text-[11px] leading-snug">
    <div class="text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1">Schematic: early plies</div>
    <svg viewBox="0 0 220 200" class="w-full h-auto" aria-label="Scatter sketch: slow pace in upper left, fast pace in lower right">
      <!-- Axes -->
      <line x1="36" y1="20" x2="36" y2="168" stroke="#334155" stroke-width="1.2"/>
      <line x1="36" y1="168" x2="200" y2="168" stroke="#334155" stroke-width="1.2"/>
      <!-- Y: ln T -->
      <text x="8" y="100" class="text-[8px] fill-slate-600" transform="rotate(-90 8 100)" style="font-size: 8px;">longer think →</text>
      <text x="20" y="24" class="text-[7px] fill-slate-500" style="font-size: 7px;">high</text>
      <text x="20" y="162" class="text-[7px] fill-slate-500" style="font-size: 7px;">low</text>
      <!-- X: clock -->
      <text x="108" y="192" class="text-[8px] fill-slate-600" text-anchor="middle" style="font-size: 8px;">← less time left &nbsp;·&nbsp; more time left →</text>
      <text x="48" y="180" class="text-[7px] fill-slate-500" style="font-size: 7px;">low</text>
      <text x="188" y="180" class="text-[7px] fill-slate-500" text-anchor="end" style="font-size: 7px;">high</text>
      <!-- Diagonal / cloud -->
      <line x1="56" y1="48" x2="188" y2="148" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4 3"/>
      <!-- Cloud dots sparse -->
      <circle cx="62" cy="58" r="3" fill="#6366f1" opacity="0.35"/>
      <circle cx="72" cy="52" r="3" fill="#6366f1" opacity="0.35"/>
      <circle cx="180" cy="140" r="3" fill="#0d9488" opacity="0.45"/>
      <circle cx="170" cy="150" r="3" fill="#0d9488" opacity="0.45"/>
      <circle cx="175" cy="135" r="2.5" fill="#0d9488" opacity="0.35"/>
      <circle cx="68" cy="64" r="2.5" fill="#6366f1" opacity="0.3"/>
      <!-- Labels -->
      <text x="50" y="44" class="text-[9px] font-semibold fill-slate-800" style="font-size: 9px;">“slow” pace</text>
      <text x="50" y="55" class="text-[7px] fill-slate-600" style="font-size: 7px;">long T, little clock</text>
      <text x="50" y="64" class="text-[7px] fill-slate-500" style="font-size: 7px;">(upper left)</text>
      <text x="128" y="132" class="text-[9px] font-semibold fill-slate-800" style="font-size: 9px;">“fast” pace</text>
      <text x="128" y="143" class="text-[7px] fill-slate-600" style="font-size: 7px;">short T, lots of clock</text>
      <text x="128" y="152" class="text-[7px] fill-slate-500" style="font-size: 7px;">(lower right)</text>
    </svg>
  </div>
</div>

---

# 7. Opponent clock vs. think time — dashboard

<div class="text-xs opacity-60 mb-2 -mt-2"><code>opp</code> variant: x = their remaining clock; excludes <code>move_time = 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_opp.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">clock_movetime (opponent)</b><br><br>
    Same 2×2 structure as §4, but the predictor is opponent clock. Contrast to player clock.
  </div>
</div>

---

# 8. Opponent clock vs. think time — quantile heatmap

<div class="text-xs opacity-60 mb-2 -mt-2">Joint bins of <b>opponent</b> remaining clock × move ply; same encoding as §5.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain max-h-[400px]" src="/figures/clock_movetime/combined_opp_quantile_heatmap.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">opp clock × ply</b><br><br>
    Side-by-side with §5 highlights whose budget pressure matters once ply is held in quantile space.
  </div>
</div>

---

# 9. Branching: legal moves vs. think time — dashboard

<div class="text-xs opacity-60 mb-2 -mt-2">x = <code>n_possible_moves</code>; y = log move time (<code>ln(move_time + ε)</code>), same convention as ply/clock dashboards. Excludes <code>move_time = 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/npossiblemoves_movetime/combined.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">npossiblemoves_movetime</b><br><br>
    2×2: branching vs log <i>T</i>, OLS, per-ply β. Matches ply/clock y-axis scaling.
  </div>
</div>

---

# 10. Branching: legal moves vs. think time — quantile heatmap

<div class="text-xs opacity-60 mb-2 -mt-2">Joint bins of legal-move count × move ply; cell color = mean log <i>T</i> (same transform as the main branching dashboard).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain max-h-[400px]" src="/figures/npossiblemoves_movetime/combined_quantile_heatmap.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">branching × ply</b><br><br>
    Where complexity (width) and stage interact after marginalizing the main dashboards in §9.
  </div>
</div>

---

# 11. Interaction: Clock & Ply (3D)

<div class="grid grid-cols-1 gap-4 h-full -mt-6">
  <div class="h-[340px]">
    <SurfPlot3D 
      csvPath="/data/heatmap_quantile.csv" 
      title="Thinking Topology: Quantile Interaction" 
      :zScale="6.0"
    />
  </div>
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">The Thinking Landscape</b><br><br>
    Interactive readout of resource allocation. The "peak" represents the mid-game where complexity is highest, modulated by remaining budget.
  </div>
</div>

---

<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Appendix</div>
    <h1 class="text-4xl">Technical Implementation Details</h1>
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

---

# 4. Decision: The Halt Controller

We frame search depth as a learnable policy $\pi_\theta$. The **Halt Controller** solves a stopping problem by maximizing the net value of computation.

**The Economy of Thought**
The agent seeks an optimal depth $k^*$ that maximizes the expected reward:

$$
R(k) = \mathbb{E} \left[ \text{Value}(T_k) \right] - C \cdot k
$$

**Mechanistic Readout**
- **Input:** The current root hidden state $h_{root}$.
- **Output:** A halting probability $P(\text{halt} | h_{root})$ predicted by an MLP.
- **Optimization:** Trained via Policy Gradients ($\nabla_\theta J$) against a DP Oracle to learn the inflection point of the "Thinking Curve."

---

# 5. Training Stage 1: Representation

<div>
  <GnnPretrainDiagram />
</div>


---

# ChildWDL: Calibrating the Thinking Curve

<div class="flex flex-col items-center justify-start mt-4">
  <div class="w-4/5">
    <ChildWdlDiagram />
  </div>
</div>

---

# 6. Training Stage 2: Policy (The DP Oracle)

Identifying the **"Economy of Thought"** inflection point. The model learns to halt when the expected **net reward $R$** begins to drop, rather than just maximizing raw value $V$.

<div class="mt-8">
  <PolicyPretrainDiagram />
</div>

---

# Intuition: The DP Oracle

<div class="h-full flex flex-col items-center justify-start mt-4">
  <div class="w-full max-w-4xl">
    <DpOracleDiagram />
  </div>
</div>

<div class="grid grid-cols-3 gap-6 mt-8 text-sm px-8">
  <div>
    <b class="text-accent block mb-2">Why DP?</b>
    Optimal search is recursive. To know if a move is right, we must reason <b>backwards</b> from terminal leaf outcomes.
  </div>
  <div>
    <b class="text-accent block mb-2">Why does it work?</b>
    We solve for $V^*(s) = \max(V, \mathbb{E}[V^*_{child}] - C)$. This defines the <b>mathematical optimum</b> for halting.
  </div>
  <div>
    <b class="text-accent block mb-2">How do we train?</b>
    We use the DP result as <b>Ground Truth</b>. The GNN's $h_i$ is mapped to this "perfect" binary decision.
  </div>
</div>
