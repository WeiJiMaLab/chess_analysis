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
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Background</div>
    <h1 class="text-4xl">Related work</h1>
    <div class="mt-3 text-sm opacity-60 max-w-xl mx-auto">
      Adaptive “when to think,” imagination-based control, and learned tree search — with citations on-slide.
    </div>
  </div>
</div>

---

# Why meta-control (in one minute)

<div class="text-sm leading-relaxed max-w-3xl space-y-3 mt-1">
  <ul class="list-disc pl-5 space-y-2">
    <li>Inference cost (tokens, rollouts, MCTS nodes) is <b>budget</b> you want to spend where it helps.</li>
    <li>A <b>meta-controller</b> that only decides whether to continue can keep an <b>outside view</b>; baking the same choice into the planner often <b>entangles</b> “when” with “how.”</li>
    <li>The same tradeoff is <b>fast vs. slow</b> reasoning: model-free heuristics vs. model-based lookahead (e.g. Kahneman, 2011; Daw, Niv &amp; Dayan, 2005 on MB/MF RL).</li>
  </ul>
</div>

---

# Adaptive Computation Time (ACT)

<div class="text-xs opacity-80 mb-2">Alex Graves — <i>Adaptive Computation Time for Recurrent Neural Networks</i>, NeurIPS 2016; arXiv:1603.08983. Figures reproduced from the paper (Fig. 1–2).</div>

<div class="grid grid-cols-2 gap-5 items-start mt-1">
  <figure class="m-0">
    <img class="w-full object-contain max-h-52" src="/figures/lit/act-fig1-rnn.png" alt="Standard RNN computation graph" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 1 — Standard RNN (two input steps).</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-52" src="/figures/lit/act-fig2-act.png" alt="RNN with ACT" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 2 — Same graph with variable intermediate “ponder” steps and halting.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  <b>Idea:</b> sigmoidal <b>halting unit</b> + <b>ponder cost</b> $\tau$ in the loss so depth-per-input is learned. Precursors: self-delimiting nets (SLIMs) with hard thresholds.
</p>

---

# Imagination-Based Planner (IBP)

<div class="text-xs opacity-80 mb-2">R. Pascanu, Y. Li, O. Vinyals, N. Heess, L. Buesing, S. Racanière, D. Reichert, T. Weber, D. Wierstra &amp; P. Battaglia — <i>Learning model-based planning from scratch</i>, arXiv:1707.06170. Figures reproduced from the paper (Fig. 1–2); raster assets extracted from the arXiv PDF.</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/ibp-fig1.png" alt="IBP schematic" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 1 — Manager imagines vs. acts; memory aggregates the trace.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/ibp-fig2.png" alt="IBP imagination strategies" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 2 — 1-step / <i>n</i>-step / tree imagination strategies over imagined states.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  <b>For us:</b> explicit <i>when-to-imagine</i>, but <b>end-to-end</b> training and an <b>RNN</b> trace → strong on <i>order</i>, weak on explicit <b>tree topology</b> in the representation.
</p>

---

# Thinker

<div class="text-xs opacity-80 mb-2">S. Chung, I. Anokhin &amp; D. Krueger — <i>Thinker: Learning to Plan and Act</i>, arXiv:2307.14993. Figures reproduced from the paper (Fig. 2–3); source SVGs from the arXiv package.</div>

<div class="grid grid-cols-2 gap-4 items-start">
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/thinker-fig2-stage.svg" alt="Thinker stage" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 2 — One stage: <i>K − 1</i> imaginary model steps, then one real env step.</figcaption>
  </figure>
  <figure class="m-0">
    <img class="w-full object-contain max-h-64" src="/figures/lit/thinker-fig3-tree.svg" alt="Thinker tree traversal" />
    <figcaption class="mt-1 opacity-70 text-[11px]">Fig. 3 — Rollout / reset dynamics under a stage budget and max depth.</figcaption>
  </figure>
</div>

<p class="mt-3 text-sm max-w-4xl">
  <b>Progress:</b> planning and acting separated via augmented actions. <b>Limits here:</b> fixed stage geometry and a learned model-specific interface — not a separate halting policy over arbitrary search trees.
</p>

---

# Learned tree search (fixed budget)

<div class="text-sm leading-relaxed max-w-3xl space-y-3 mt-1">
  <p>
    <b>AlphaZero</b> (Silver et al., 2018, <i>Science</i>): policy + value coupled to MCTS — strong structure, but simulations per move are set externally, not a per-node “worth another expansion?” learner.
  </p>
  <p>
    <b>MCTSnets</b> (Guez et al., 2018, ICML; arXiv:1802.04697): backups and visit patterns through learned embeddings — topology in the net, still not an independent meta-controller for compute.
  </p>
</div>

---

# Gap → this deck

<div class="grid grid-cols-2 gap-6 mt-2 text-sm leading-relaxed">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider">Meta-control</b>
    <p class="mt-1">Good on <i>when</i> to stop / imagine; often sequential summaries of the trace.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary">
    <b class="text-secondary text-xs uppercase tracking-wider">Tree search</b>
    <p class="mt-1">Good on <i>how</i> to search; budget usually fixed, not a trained halt policy on the tree.</p>
  </div>
</div>

<p class="mt-5 text-sm max-w-3xl">
  <b>Our thread:</b> halting / allocation using a <b>structural</b> view of the search tree (later slides), trainable <b>beside</b> a fixed engine — with chess oracle + human timing as targets.
</p>

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

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-3">
  <p>
    Every timing slide uses the <b>same</b> human move pool in DuckDB. Code: <code>analysis/slurm/scripts/preprocess.py</code> (<code>get_games</code> → Slurm <code>shard</code> → <code>merge</code> → <code>process_moves</code>); orchestration: <code>analysis/slurm/preprocess.sh</code>. Bad-game filtering (negative <code>move_time</code>, berserk, grant-more-time) is applied during <code>shard</code>; <code>merge</code> builds <code>moves</code>; <code>process_moves</code> builds <code>processed_moves</code> and <code>processed_moves_nonzero</code>.
  </p>
  <ul class="list-disc pl-5 space-y-2">
    <li><b>Which games</b> — Strong rapid by default: <b>10+0</b>, <b>both players 2000+ Elo</b>, <b>Oct 2023–Jan 2024</b> (filters in <code>preprocess.py</code> <code>config</code>).</li>
    <li><b>Extract &amp; merge</b> — Slurm array runs <code>preprocess.py shard</code> on Lichess parquet joined to table <code>games</code>; <code>merge</code> loads <code>moves</code>. Whole games with any negative <code>move_time</code> (or berserk / grant-more-time under the shard SQL) are dropped.</li>
    <li><b>Feature tables</b> — After <code>merge</code> + <code>process_moves</code>, <code>processed_moves</code> and <code>processed_moves_nonzero</code> add <code>ply_tertiles</code>, board counts (<code>n_pieces_on_board_*</code>, <code>n_self_pieces_exc_pawns</code>, <code>n_opp_pieces_exc_pawns</code>), partial <code>fen</code>, and raw <code>move_time</code> (same feature set as legacy <code>_selected_moves</code> / <code>_selected_moves_nonzero_T</code>).</li>
    <li><b>Plots</b> — Unless noted, positive think time only (no premoves; same sample as <code>movetime_analysis.py</code> on <code>processed_moves_nonzero</code>, via <code>utils.selected_db.TABLE_PROCESSED_MOVES_NONZERO</code>). Histograms / dashboards add ln&nbsp;<i>T</i>, <code>ntile</code> bins, and heatmaps in analysis SQL—not as extra columns frozen at ingest.</li>
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
  <img class="w-full object-contain max-h-[340px]" src="/figures/clock_movetime/combined_quantile_heatmap.png" />
  
  <div class="takeaway border-accent bg-accent-soft text-sm py-4">
    <b class="text-accent uppercase tracking-wider text-xs">clock × ply</b><br><br>
    Same idea as optional standalone clock×ply heatmaps in <code>exploratory/</code>, but emitted from the shared <code>Analyzer</code> path as §4.
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
  <img class="w-full object-contain max-h-[340px]" src="/figures/clock_movetime/combined_opp_quantile_heatmap.png" />
  
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
  <img class="w-full object-contain max-h-[340px]" src="/figures/npossiblemoves_movetime/combined_quantile_heatmap.png" />
  
  <div class="takeaway border-secondary bg-neutral-soft text-sm py-4">
    <b class="text-secondary uppercase tracking-wider text-xs">branching × ply</b><br><br>
    Where complexity (width) and stage interact after marginalizing the main dashboards in §9.
  </div>
</div>

---

# 11. Material: non-pawn pieces vs. think time — dashboard

<div class="text-xs opacity-60 mb-2 -mt-2">x = <code>n_pieces_on_board_exc_pawns</code> (pieces on board excluding pawns, from SQL on <code>board_position</code>); y = <code>ln(move_time + ε)</code>. Excludes <code>move_time = 0</code>. <code>movetime_analysis.py --only pieces_exc</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/n_pieces_exc_pawns_movetime/combined.png" />
  
  <div class="takeaway border-sky-500 bg-sky-50/90 text-sm py-4">
    <b class="text-sky-700 uppercase tracking-wider text-xs">n_pieces_exc_pawns_movetime</b><br><br>
    2×2: material vs log <i>T</i>, same structure as clock/branching; bottom row splits by global ply tertiles (<code>ply_tertiles</code>).
  </div>
</div>

---

# 12. Material: non-pawn pieces vs. think time — quantile heatmap

<div class="text-xs opacity-60 mb-2 -mt-2">Joint <code>ntile</code> bins of non-pawn piece count × move ply; cell color = mean log <i>T</i> (same transform as §11).</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain max-h-[340px]" src="/figures/n_pieces_exc_pawns_movetime/combined_quantile_heatmap.png" />
  
  <div class="takeaway border-sky-500 bg-sky-50/90 text-sm py-4">
    <b class="text-sky-700 uppercase tracking-wider text-xs">material × ply</b><br><br>
    Endgame-rich counts vs opening ply, analogous to §10 after conditioning on width in §9.
  </div>
</div>

---

# 13. Material (self): own non-pawn pieces vs. think time — dashboard

<div class="text-xs opacity-60 mb-2 -mt-2">x = <code>n_self_pieces_exc_pawns</code> (moving player's non-pawn count from FEN case + <code>player_white</code>); y = <code>ln(move_time + ε)</code>. <code>movetime_analysis.py --only self_pieces_exc</code>.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain" src="/figures/n_self_pieces_exc_pawns_movetime/combined.png" />

  <div class="takeaway border-amber-500 bg-amber-50/90 text-sm py-4">
    <b class="text-amber-900 uppercase tracking-wider text-xs">n_self_pieces_exc_pawns_movetime</b><br><br>
    Same layout as §11, but predictor is <b>your</b> remaining officers (not both sides total). Contrasts total material in §11–12.
  </div>
</div>

---

# 14. Material (self): own non-pawn pieces vs. think time — quantile heatmap

<div class="text-xs opacity-60 mb-2 -mt-2">Joint <code>ntile</code> bins of own non-pawn count × move ply; same encoding as §12.</div>

<div class="grid grid-cols-[65fr_35fr] gap-10 mt-4 items-start">
  <img class="w-full object-contain max-h-[340px]" src="/figures/n_self_pieces_exc_pawns_movetime/combined_quantile_heatmap.png" />

  <div class="takeaway border-amber-500 bg-amber-50/90 text-sm py-4">
    <b class="text-amber-900 uppercase tracking-wider text-xs">own material × ply</b><br><br>
    Stage-conditioned view of how “how much of my army is left” relates to think time (cf. total board count in §12).
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
