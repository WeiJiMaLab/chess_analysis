<script setup lang="ts">
/**
 * Didactic flow chart for the CMC control loop.
 * Board -> Planner -> Tree -> Meta-Controller -> [Loop or Act]
 * SHRUNK vertically and simplified.
 */
</script>

<template>
  <div class="loop-container">
    <svg viewBox="0 0 400 320" xmlns="http://www.w3.org/2000/svg">
      <!-- Markers -->
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#94a3b8" />
        </marker>
        <marker id="arrow-indigo" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#6366f1" />
        </marker>
      </defs>

      <!-- 1. Board State -->
      <rect x="135" y="10" width="130" height="25" rx="6" class="box hollow" />
      <text x="200" y="27" text-anchor="middle" class="label">Board State (s)</text>

      <line x1="200" y1="35" x2="200" y2="55" class="edge" marker-end="url(#arrow)" />

      <!-- 2. Planner -->
      <rect x="110" y="55" width="180" height="35" rx="6" class="box filled" />
      <text x="200" y="77" text-anchor="middle" class="label white">MCTS Planner (Leela)</text>
      
      <line x1="200" y1="90" x2="200" y2="110" class="edge" marker-end="url(#arrow)" />

      <!-- 3. Search Tree -->
      <rect x="135" y="110" width="130" height="25" rx="6" class="box hollow" />
      <text x="200" y="127" text-anchor="middle" class="label">Search Tree (T)</text>

      <line x1="200" y1="135" x2="200" y2="155" class="edge" marker-end="url(#arrow)" />

      <!-- 4. Meta-Controller -->
      <rect x="110" y="155" width="180" height="45" rx="10" class="box highlight" />
      <text x="200" y="177" text-anchor="middle" class="label white font-bold">Meta-Controller (GNN)</text>
      <text x="200" y="190" text-anchor="middle" class="sub white opacity-80">Economy of Thought</text>

      <!-- Decision Arrows -->
      <!-- Branch 1: Continue (Loop Back) -->
      <path d="M110 177 L60 177 L60 72 L110 72" fill="none" class="edge loop-path" marker-end="url(#arrow-indigo)" />
      <text x="75" y="115" text-anchor="middle" class="path-label" transform="rotate(-90 75 115)">CONTINUE</text>

      <!-- Branch 2: Halt (Act) -->
      <line x1="200" y1="200" x2="200" y2="240" class="edge" marker-end="url(#arrow)" />
      <text x="210" y="225" text-anchor="start" class="path-label">HALT (Play)</text>

      <!-- 5. Best Move -->
      <rect x="120" y="240" width="160" height="35" rx="6" class="box success" />
      <text x="200" y="262" text-anchor="middle" class="label white">Execute Best Move</text>
    </svg>
  </div>
</template>

<style scoped>
.loop-container {
  width: 100%;
  max-width: 320px;
  margin: 0 auto;
}
svg {
  width: 100%;
  height: auto;
  font-family: 'Inter', sans-serif;
}
.box { stroke-width: 1.5; }
.hollow { fill: #fff; stroke: #94a3b8; }
.filled { fill: #475569; stroke: #475569; }
.highlight { fill: #6366f1; stroke: #4f46e5; }
.success { fill: #10b981; stroke: #059669; }
.label { font-size: 11px; font-weight: 500; fill: #1e293b; }
.white { fill: #fff; }
.font-bold { font-weight: 700; }
.sub { font-size: 9px; }
.edge { stroke: #94a3b8; stroke-width: 1.5; }
.loop-path { stroke: #6366f1; stroke-dasharray: 4; }
.path-label { font-size: 8px; font-weight: 700; fill: #6366f1; text-transform: uppercase; }
.opacity-80 { opacity: 0.8; }
</style>
