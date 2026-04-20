<script setup lang="ts">
/**
 * High-Level Horizontal Control Loop.
 * Board -> Leela -> Meta-Controller -> Halt/Continue.
 */
</script>

<template>
  <div class="loop-container">
    <svg viewBox="0 0 600 180" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow-high" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#94a3b8" />
        </marker>
        <marker id="arrow-high-indigo" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#6366f1" />
        </marker>
      </defs>

      <!-- 1. Board State -->
      <rect x="20" y="70" width="100" height="40" rx="6" class="box hollow" />
      <text x="70" y="95" text-anchor="middle" class="label">Board (s)</text>

      <line x1="120" y1="90" x2="160" y2="90" class="edge" marker-end="url(#arrow-high)" />

      <!-- 2. Leela (Planner) -->
      <rect x="160" y="60" width="140" height="60" rx="8" class="box filled-slate" />
      <text x="230" y="90" text-anchor="middle" class="label white bold">Leela</text>
      <text x="230" y="105" text-anchor="middle" class="tiny white opacity-70">(MCTS Planner)</text>

      <line x1="300" y1="90" x2="340" y2="90" class="edge" marker-end="url(#arrow-high)" />

      <!-- 3. Meta-Controller -->
      <rect x="340" y="55" width="160" height="70" rx="10" class="box fill-indigo" />
      <text x="420" y="90" text-anchor="middle" class="label white bold">Meta-Controller</text>
      <text x="420" y="105" text-anchor="middle" class="tiny white opacity-70">(GNN Evaluator)</text>

      <!-- Decision Points -->
      <!-- Halt (Success) -->
      <line x1="500" y1="90" x2="540" y2="90" class="edge" marker-end="url(#arrow-high)" />
      
      <rect x="540" y="70" width="40" height="40" rx="20" class="box success" />
      <text x="560" y="95" text-anchor="middle" class="label white bold">Act</text>

      <!-- Continue (Loop back) -->
      <path d="M420 125 L420 160 L230 160 L230 120" fill="none" class="edge loop-path" marker-end="url(#arrow-high-indigo)" />
      <text x="325" y="155" text-anchor="middle" class="path-label">CONTINUE SEARCH</text>
    </svg>
  </div>
</template>

<style scoped>
.loop-container {
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
  padding: 1rem;
}
svg {
  width: 100%;
  height: auto;
  font-family: 'Inter', sans-serif;
}
.box { stroke-width: 1.5; }
.hollow { fill: #fff; stroke: #cbd5e1; }
.filled-slate { fill: #475569; stroke: none; }
.fill-indigo { fill: #6366f1; stroke: #4f46e5; }
.success { fill: #10b981; stroke: none; }
.label { font-size: 11px; fill: #1e293b; }
.white { fill: #fff; }
.bold { font-weight: 700; }
.tiny { font-size: 9px; }
.edge { stroke: #94a3b8; stroke-width: 1.5; fill: none; }
.loop-path { stroke: #6366f1; stroke-dasharray: 4; }
.path-label { font-size: 9px; font-weight: 700; fill: #6366f1; text-transform: uppercase; }
.opacity-70 { opacity: 0.7; }
</style>
