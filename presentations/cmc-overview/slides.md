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
    All timing plots use the same DuckDB sample: <b>move-level rows in <code>selected_moves</code></b> inside the project personal database, joined to analysis tables built by the Python scripts.
  </p>
  <ul class="list-disc pl-6 space-y-2">
    <li>
      <b>Game list:</b> We restrict to game IDs listed in the <code>selected_games</code> table (in that same database). The exact inclusion rule (Elo, time control, etc.) is whatever that table encodes.
    </li>
    <li>
      <b>Extracting moves (<code>src/utils/load_data.py</code>):</b> A Slurm array walks Lichess parquet shards under the archive root, keeps only those <code>gid</code>s, and by default <b>drops an entire game</b> if any row has a negative <code>move_time</code> (bad timestamps). Staging <code>.parquet</code> files are merged into <code>selected_moves</code>.
    </li>
    <li>
      <b>Launch script (<code>src/slurm/script_load_moves.sh</code>):</b> Reserves a clean temp directory, submits the load-shards array job, and submits a <b>merge</b> job that runs only after the array succeeds, repopulating <code>selected_moves</code> from the combined parquet.
    </li>
    <li>
      <b>Downstream analysis:</b> The plotting code calls <code>preprocess()</code> to add <code>ln_move_time</code>, clock features, <b><code>n_possible_moves</code></b> (legal-move count at the position) and quantile bins, on top of <code>selected_moves</code>.
    </li>
  </ul>
</div>

---

# 1a. Move times: raw and log (default)

<div class="text-xs opacity-60 mb-2 -mt-2">All preprocessed rows after <code>preprocess</code> — includes <code>move_time = 0</code> (premoves / instant plays).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/move_time_summary/combined.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">From <code>move_time_summary.py</code></b><br><br>
    <b>Equal-width SQL bins</b> of <code>move_time</code> and <code>ln_move_time</code>, side by side, with the same <code>preprocess</code> table as the other analyses. 
    <br><br>
    Most moves are very fast; the <b>long right tail</b> of slow moves dominates variance in the raw scale, which is one reason we work in log time for the regressions.
  </div>
</div>

---

# 1b. Move times: raw and log (nonzero_T)

<div class="text-xs opacity-60 mb-2 -mt-2">Same script with <code>--nonzero_T</code> → <code>_selected_moves_nonzero_T</code> (drops rows with <code>move_time = 0</code>).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/move_time_summary/combined_nonzero_T.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">Why compare</b><br><br>
    The spike at the origin in the raw histogram is largely <b>premoves and instant replies</b>. Filtering them reweights the sample toward “deliberate” plies; regression panels below use the same <code>--nonzero_T</code> option when you want that definition of think time.
  </div>
</div>

---

# 2a. Game stage: ply vs. mean log think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2">Default <code>preprocess</code> table — all move times including zeros.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/combined.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">From <code>ply_movetime.py</code></b><br><br>
    <b>Left:</b> For each integer ply, mean and dispersion of <code>ln_move_time</code> in raw ply space. 
    <br><br>
    <b>Right:</b> The same response summarized by <b>decile bins of move ply</b> (cross-sectional rank), to smooth irregular early/late plies. 
    <br><br>
    Together they show how average thinking time evolves across the game, without conflating that with clock budget.
  </div>
</div>

---

# 2b. Game stage: ply vs. mean log think time (nonzero_T)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--nonzero_T</code>: only rows with <code>move_time &gt; 0</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/ply_movetime/combined_nonzero_T.png" />
  
  <div class="takeaway border-primary bg-primary-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">Impact of the filter</b><br><br>
    Per-ply means shift toward the <b>active-thinking</b> distribution; the mass at very low <code>ln_move_time</code> from true zeros is removed. Use this slide next to 2a to show how much of the “fast” end is zero-time moves.
  </div>
</div>

---

# 3a. Remaining clock vs. think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2">Default sample — includes <code>move_time = 0</code> rows in aggregates and OLS.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">From <code>clock_movetime.py</code> (2×2)</b><br><br>
    <b>Top-left & top-right:</b> Average <code>ln_move_time</code> vs log remaining clock and vs <b>decile bins of player clock</b> (mean clock in each bin on the x-axis). 
    <br><br>
    <b>Bottom-left:</b> Subsampled scatter of log clock vs log move time with a single <b>global OLS</b> line in log–log space. 
    <br><br>
    <b>Bottom-right:</b> <b>Per-ply</b> OLS slope of <code>ln_move_time</code> on <code>ln_player_clock_time</code>—how “elasticity” of think time to the clock varies by stage.
  </div>
</div>

---

# 3b. Remaining clock vs. think time (nonzero_T)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--nonzero_T</code>: OLS and bins exclude premoves / 0s.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_nonzero_T.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">Impact of the filter</b><br><br>
    Slopes and binned means focus on <b>positive</b> think times; the global OLS and per-ply <code>β</code> can move materially versus 3a. Pair these slides to show how sensitive clock–time relationships are to including instant moves.
  </div>
</div>

---

# 4a. Opponent clock vs. your think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2">Same 2×2 as §3, but <code>clock_movetime.py --opp</code>: <b>x-axis is the other player’s</b> remaining clock before your move (from <code>opponent_clock_time</code> in <code>preprocess</code>).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_opp.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">From <code>clock_movetime.py --opp</code></b><br><br>
    <b>Top-left & top-right:</b> Mean <code>ln_move_time</code> (your time on the clock to move) vs log <b>opponent</b> clock and vs opponent clock deciles. 
    <br><br>
    <b>Bottom-left:</b> Scatter + global OLS of your log think time on log <b>opponent</b> remaining time. 
    <br><br>
    <b>Bottom-right:</b> Per-ply OLS slopes for that relationship—do you spend longer when <i>they</i> are low on time? Compare directly to §3a (player clock on the x-axis).
  </div>
</div>

---

# 4b. Opponent clock vs. your think time (nonzero_T)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--nonzero_T --opp</code>: same opponent-clock analysis, premoves / <code>move_time = 0</code> removed.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/clock_movetime/combined_nonzero_T_opp.png" />
  
  <div class="takeaway border-success bg-success-soft text-sm py-4">
    <b class="text-success uppercase tracking-wider text-xs">Pair with 4a</b><br><br>
    As in §3a vs §3b, removing instant moves can shift slopes and binned means. This is the opponent-clock analogue for the <b>active-thinking</b> subsample; compare to <code>combined_nonzero_T.png</code> (player clock) to separate own-budget and rival-budget effects.
  </div>
</div>

---

# 5a. Branching: legal moves vs. raw think time (default)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>npossiblemoves_movetime.py</code> — y-axis is <b>raw</b> <code>move_time</code> (seconds), not log; x is the number of legal moves at the position (from <code>preprocess</code>).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/npossiblemoves_movetime/combined.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">From <code>npossiblemoves_movetime.py</code> (2×2)</b><br><br>
    Same layout idea as the clock deck, but measures <b>position complexity via branching</b> instead of time pressure. 
    <br><br>
    <b>Top-left & top-right:</b> Mean <code>move_time</code> by integer <code>n_possible_moves</code> and by <b>decile bins of legal-move count</b> (<code>n_possible_moves_qbin</code>). 
    <br><br>
    <b>Bottom-left:</b> Subsampled scatter of legal moves vs raw think time with a <b>global OLS</b> in the original scale. 
    <br><br>
    <b>Bottom-right:</b> Per-ply OLS slope of <code>move_time</code> on <code>n_possible_moves</code>—how the branching–time relationship changes by game stage. (No separate “opponent” branch; this is player-side only.)
  </div>
</div>

---

# 5b. Branching: legal moves vs. raw think time (nonzero_T)

<div class="text-xs opacity-60 mb-2 -mt-2"><code>--nonzero_T</code>: only rows with <code>move_time &gt; 0</code>; otherwise identical pipeline.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/npossiblemoves_movetime/combined_nonzero_T.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">Pair with 5a</b><br><br>
    Dropping premoves and instant replies reweights the sample toward plies with positive deliberation, which can change slopes and the per-ply <code>β</code> panel. Compare to §1–§2 (log time) to separate <b>heavy tails</b> and <b>stage</b> effects from <b>branching</b> in raw seconds.
  </div>
</div>

---

# 6. Value of Computation (The Demand)

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-8 items-start">
  <img class="w-full object-contain" src="/figures/voc_movetime/standard_voc_single_sample.png" />
  
  <div class="takeaway bg-neutral-soft text-sm py-4">
    <b class="text-primary uppercase tracking-wider text-xs">Key Takeaway</b><br><br>
    <b>VOC</b> measures the potential gain from deep engine search over a shallow read.
    <br><br>
    $\log(T) \propto \sqrt{\text{VOC}}$: Humans spend the most "thought-capital" where depth matters most. 
    <br><br>
    <b>This is the target alignment for our Meta-Controller.</b>
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
