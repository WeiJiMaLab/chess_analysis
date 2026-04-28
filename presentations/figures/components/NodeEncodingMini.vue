<script setup lang="ts">
/**
 * Tiny diagram showing how individual nodes are encoded.
 * Board (s) -> Leela/MLP -> x (Input Vector).
 */
</script>

<template>
  <div class="encoding-mini">
    <svg viewBox="0 0 320 60" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arr-mini" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
        </marker>
        <marker id="arr-mini-i" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#6366f1" />
        </marker>
      </defs>

      <!-- Label -->
      <text x="0" y="32" class="tag uppercase">Enc:</text>

      <!-- Board -->
      <rect x="35" y="15" width="60" height="25" rx="4" class="box hollow" />
      <text x="65" y="31" text-anchor="middle" class="txt bold">Board (s)</text>

      <line x1="95" y1="28" x2="125" y2="28" class="edge" marker-end="url(#arr-mini)" />
      <text x="110" y="22" text-anchor="middle" class="tiny-label">Leela</text>

      <!-- MLP -->
      <rect x="125" y="15" width="60" height="25" rx="4" class="box fill-slate" />
      <text x="155" y="31" text-anchor="middle" class="txt white bold">MLP</text>

      <line x1="185" y1="28" x2="215" y2="28" class="edge" marker-end="url(#arr-mini-i)" />
      
      <!-- Vector x -->
      <rect x="215" y="15" width="60" height="25" rx="4" class="box hollow-indigo" />
      <text x="245" y="31" text-anchor="middle" class="txt indigo-txt bold">Vector (x)</text>
    </svg>
  </div>
</template>

<style scoped>
.encoding-mini {
  width: 100%;
  max-width: 320px;
  background: #f8fafc;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  margin: 1rem 0;
}
svg { width: 100%; height: auto; font-family: 'Inter', sans-serif; }
.box { stroke-width: 1.25; }
.hollow { fill: #fff; stroke: #cbd5e1; }
.hollow-indigo { fill: #f5f3ff; stroke: #6366f1; stroke-width: 1.5; }
.fill-slate { fill: #475569; stroke: none; }
.white { fill: #fff; }
.bold { font-weight: 700; }
.txt { font-size: 8px; fill: #1e293b; }
.indigo-txt { fill: #6366f1; }
.tiny-label { font-size: 6px; font-weight: 800; fill: #94a3b8; text-transform: uppercase; }
.tag { font-size: 8px; font-weight: 900; fill: #6366f1; }
.edge { stroke: #94a3b8; stroke-width: 1; fill: none; }
.uppercase { text-transform: uppercase; }
</style>
