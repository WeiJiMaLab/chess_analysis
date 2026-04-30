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
<div class="h-full flex items-center justify-center text-center">
  <div>
    <div class="text-accent font-bold uppercase tracking-widest text-xs mb-2">Behavioral results</div>
    <h1 class="text-4xl">Human move timing & figures</h1>
    <div class="mt-4 text-sm opacity-60 max-w-xl mx-auto">
      How do we know the meta-controller works as intended? Humans serve as an <b>existence proof</b> and a <b>target distribution</b> for efficient search.
    </div>
  </div>
</div>

---

# Behavioral data: how we chose these games

<div class="mt-4 text-sm leading-relaxed max-w-3xl space-y-3">
  <p>
    Every timing slide uses the <b>same</b> human move pool in DuckDB. Code: <code>src/slurm/scripts/preprocess_data.py</code>; end-to-end orchestration: <code>src/slurm/_preprocess.sh</code> (<code>select_games</code> → Slurm <code>process_shard</code> → <code>merge</code> → <code>berserk</code> → <code>preprocess</code>).
  </p>
  <ul class="list-disc pl-5 space-y-2">
    <li><b>Which games</b> — Strong rapid by default: <b>10+0</b>, <b>both players 2000+ Elo</b>, <b>Oct–Dec 2023</b> (filters in <code>preprocess_data.py select_games</code>).</li>
    <li><b>Extract &amp; merge</b> — Array job <code>preprocess_shard.sbatch</code> runs <code>preprocess_data.py process_shard</code> on Lichess parquet, restricted to <code>selected_games</code>; shards load into <code>selected_moves</code>. By default, drop a whole game if <i>any</i> move has negative <code>move_time</code>.</li>
    <li><b>Feature tables</b> — <code>preprocess_data.py preprocess</code> writes <code>_selected_moves</code> and <code>_selected_moves_nonzero_T</code> (e.g. <code>ply_tertiles</code>, clocks, <code>n_possible_moves</code>, board counts <code>n_pieces_on_board_exc_pawns</code> / <code>n_pieces_on_board_inc_pawns</code> from FEN placement, raw <code>move_time</code>), excluding berserk / grant-more-time games per the SQL in that step.</li>
    <li><b>Plots</b> — Unless noted, positive think time only (no premoves; same sample as <code>movetime_analysis.py</code> on <code>_selected_moves_nonzero_T</code>). Histograms / dashboards (<code>move_time_summary.py</code>, <code>movetime_analysis.py</code>, …) add ln&nbsp;<i>T</i>, <code>ntile</code> bins, and heatmaps in analysis SQL—not as extra columns frozen at ingest.</li>
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
    Same idea as the static surfaces in <code>exploratory/heatmap_clock_ply.py</code>, but emitted from the shared <code>Analyzer</code> path as §4.
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

<div class="text-sm leading-snug max-w-3xl space-y-2 mt-1">
  <ul class="list-disc pl-5 space-y-1 m-0">
    <li>Inference cost (tokens, rollouts, MCTS nodes) is <b>budget</b> you want to spend where it helps.</li>
    <li>A <b>meta-controller</b> that only decides whether to continue can keep an <b>outside view</b>; baking the same choice into the planner often <b>entangles</b> “when” with “how.”</li>
    <li>The same tradeoff is <b>fast vs. slow</b> reasoning: model-free heuristics vs. model-based lookahead (e.g. Kahneman, 2011; Daw, Niv &amp; Dayan, 2005 on MB/MF RL).</li>
  </ul>
</div>

---

<div class="lit-ref-header">
<h1>Adaptive Computation Time (ACT)</h1>
<p class="lit-ref-cite">Graves (2016) <i>Adaptive Computation Time for Recurrent Neural Networks</i>. NeurIPS.</p>
</div>

<div class="pt-2 mt-0">
<div class="grid grid-cols-2 gap-3 items-start text-sm">
  <figure class="m-0">
    <div class="bg-white rounded-md p-1.5 shadow-sm">
      <img class="w-full object-contain max-h-56" src="/figures/lit/act-fig1-rnn.png" alt="Standard RNN computation graph" />
    </div>
    <figcaption class="mt-1 opacity-80 leading-tight text-sm"><b>Fig. 1</b> — One state update per input tick.</figcaption>
  </figure>
  <figure class="m-0">
    <div class="bg-white rounded-md p-1.5 shadow-sm">
      <img class="w-full object-contain max-h-56" src="/figures/lit/act-fig2-act.png" alt="RNN with ACT" />
    </div>
    <figcaption class="mt-1 opacity-80 leading-tight text-sm"><b>Fig. 2</b> — Multiple internal updates per input; halting ends the mini-loop.</figcaption>
  </figure>
</div>
</div>

<div class="grid grid-cols-2 gap-2 mt-1 text-sm leading-tight">
  <div class="pl-2 pr-4 border-l-2 border-accent/40 opacity-90 flex flex-col gap-0 mt-5">
    <div class="text-accent font-bold uppercase tracking-wide text-xs leading-none m-0 p-0">What it does</div>
    <p class="m-0 p-0 leading-tight">Stops the RNN early or late per input; halt + ponder cost $\tau$ trades error vs. compute (cf. SLIMs: discrete thresholds).</p>
  </div>
  <div class="pl-2 pr-4 border-l-2 border-secondary/40 opacity-90 flex flex-col gap-0 mt-5">
    <div class="text-secondary font-bold uppercase tracking-wide text-xs leading-none m-0 p-0">What’s missing</div>
    <p class="m-0 p-0 leading-tight">Still a <b>sequence</b> model — no representation of <b>search-tree topology</b> for “where to expand.”</p>
  </div>
</div>

---

<div class="lit-ref-header">
<h1>Imagination-Based Planner (IBP)</h1>
<p class="lit-ref-cite">Pascanu et al. (2017) <i>Learning model-based planning from scratch</i>. arXiv.</p>
</div>

<div class="pt-2 mt-0">
<div class="grid grid-cols-2 gap-3 items-start text-sm">
  <figure class="m-0">
    <div class="bg-white rounded-md p-1.5 shadow-sm">
      <img class="w-full object-contain max-h-[138px]" src="/figures/lit/ibp-fig1.png" alt="IBP schematic" />
    </div>
    <figcaption class="mt-1 opacity-80 leading-tight text-sm"><b>Fig. 1</b> — Manager: <i>imagine</i> (model) vs <i>act</i> (world); memory updates each tick.</figcaption>
  </figure>
  <figure class="m-0">
    <div class="bg-white rounded-md p-1.5 shadow-sm">
      <img class="w-full object-contain max-h-[185px]" src="/figures/lit/ibp-fig2.png" alt="IBP imagination strategies" />
    </div>
    <figcaption class="mt-1 opacity-80 leading-tight text-sm"><b>Fig. 2</b> — Where to imagine from: <b>1-step</b>, <b><i>n</i>-step</b> chain, or <b>tree</b> over prior imagined nodes.</figcaption>
  </figure>
</div>
</div>

<div class="grid grid-cols-2 gap-2 mt-1 text-sm leading-tight">
  <div class="pl-2 pr-4 border-l-2 border-accent/40 opacity-90 flex flex-col gap-0 mt-5">
    <div class="text-accent font-bold uppercase tracking-wide text-xs leading-none m-0 p-0">What it does</div>
    <p class="m-0 p-0 leading-tight">Manager learns <b>imagine vs act</b> and <b>where</b> to branch imagination (1-step / chain / tree).</p>
  </div>
  <div class="pl-2 pr-4 border-l-2 border-secondary/40 opacity-90 flex flex-col gap-0 mt-5">
    <div class="text-secondary font-bold uppercase tracking-wide text-xs leading-none m-0 p-0">What’s missing</div>
    <p class="m-0 p-0 leading-tight">Trained <b>end-to-end</b>; plan state is an <b>RNN over time</b> — not a controller that <i>sees</i> the imagined graph as such.</p>
  </div>
</div>

---

<div class="lit-ref-header">
<h1>Thinker</h1>
<p class="lit-ref-cite">Chung et al. (2023) <i>Thinker: Learning to Plan and Act</i>. NeurIPS.</p>
</div>

<div class="pt-2 mt-0">
<div class="grid grid-cols-2 gap-3 items-start text-sm">
  <figure class="m-0">
    <div class="bg-white rounded-md p-1.5 shadow-sm">
      <img class="w-full object-contain max-h-[220px]" src="/figures/lit/thinker-fig2-stage.svg" alt="Thinker stage" />
    </div>
    <figcaption class="mt-1 opacity-80 leading-tight text-sm"><b>Fig. 2</b> — <i>K − 1</i> steps on the <b>model</b>, then <b>one</b> real env step per stage.</figcaption>
  </figure>
  <figure class="m-0">
    <div class="bg-white rounded-md p-1.5 shadow-sm">
      <img class="w-full object-contain max-h-[220px]" src="/figures/lit/thinker-fig3-tree.svg" alt="Thinker tree traversal" />
    </div>
    <figcaption class="mt-1 opacity-80 leading-tight text-sm"><b>Fig. 3</b> — Plan steps: <b>advance</b> in the imagined tree or <b>reset</b> to root; depth cap <i>L</i> forces reset.</figcaption>
  </figure>
</div>
</div>

<div class="grid grid-cols-2 gap-2 mt-1 text-sm leading-tight">
  <div class="pl-2 pr-4 border-l-2 border-accent/40 opacity-90 flex flex-col gap-0 mt-5">
    <div class="text-accent font-bold uppercase tracking-wide text-xs leading-none m-0 p-0">What it does</div>
    <p class="m-0 p-0 leading-tight">Splits <b>model rollouts</b> and <b>real env</b> via augmented actions; rollouts stay <b>inspectable</b>.</p>
  </div>
  <div class="pl-2 pr-4 border-l-2 border-secondary/40 opacity-90 flex flex-col gap-0 mt-5">
    <div class="text-secondary font-bold uppercase tracking-wide text-xs leading-none m-0 p-0">What’s missing</div>
    <p class="m-0 p-0 leading-tight"><i>K</i>, <i>L</i> fix the <b>stage script</b> — not a learned <b>halt / where-to-expand</b> policy on an arbitrary explicit tree.</p>
  </div>
</div>

---

# Learned tree search (fixed budget)

<div class="text-sm leading-snug max-w-3xl space-y-2 mt-1">
  <p class="m-0">
    <b>AlphaZero</b> (Silver et al., 2018, <i>Science</i>): policy + value coupled to MCTS — strong structure, but simulations per move are set externally, not a per-node “worth another expansion?” learner.
  </p>
  <p class="m-0">
    <b>MCTSnets</b> (Guez et al., 2018, ICML; arXiv:1802.04697): backups and visit patterns through learned embeddings — topology in the net, still not an independent meta-controller for compute.
  </p>
</div>

---

# Gap → this deck

<div class="grid grid-cols-2 gap-4 mt-2 text-sm leading-snug">
  <div class="p-3 bg-neutral-soft border-l-2 border-accent">
    <b class="text-accent text-xs uppercase tracking-wider block leading-none mb-1">Meta-control</b>
    <p class="m-0">Good on <i>when</i> to stop / imagine; often sequential summaries of the trace.</p>
  </div>
  <div class="p-3 bg-neutral-soft border-l-2 border-secondary">
    <b class="text-secondary text-xs uppercase tracking-wider block leading-none mb-1">Tree search</b>
    <p class="m-0">Good on <i>how</i> to search; budget usually fixed, not a trained halt policy on the tree.</p>
  </div>
</div>

<p class="mt-3 text-sm max-w-3xl leading-snug m-0">
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

---

# High-Level Architecture

<p class="architecture-preset-caption">Original — Illustrator export (<code>diagram-01.svg</code>), unchanged</p>

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <ArchitectureDiagram />
</div>

---

# High-Level Architecture

<p class="architecture-preset-caption">Schematic · minimal — light panels, mono labels</p>

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <ArchitectureSchematic variant="minimal" />
</div>

---

# High-Level Architecture

<p class="architecture-preset-caption">Schematic · blueprint — dark field, Plex Mono, in-figure header</p>

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <ArchitectureSchematic variant="blueprint" />
</div>

---

# High-Level Architecture

<p class="architecture-preset-caption">Schematic · editorial — serif + mono mix, caption block</p>

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <ArchitectureSchematic variant="editorial" />
</div>

---

# High-Level Architecture

<p class="architecture-preset-caption">Schematic · cards — floating modules, Grotesk titles, soft shadow</p>

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <ArchitectureSchematic variant="cards" />
</div>

---

# High-Level Architecture

<p class="architecture-preset-caption">Schematic · swiss — grid poster, heavy rules, figure legend</p>

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <ArchitectureSchematic variant="swiss" />
</div>

---

# Training

<div class="h-full flex flex-col items-center justify-center bg-transparent">
  <img
    class="w-full max-h-[min(78vh,620px)] object-contain"
    src="/diagrams/diagram-02.svg"
    alt="Training: tree consolidation, backward values V(a_t), halt targets, cross-entropy"
  />
</div>
