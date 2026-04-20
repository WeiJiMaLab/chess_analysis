<script setup lang="ts">
/**
 * Unified Unrolled Mechanistic View of the GNN Sweeps.
 * Labeled children nodes and external h(0)=x initialization.
 * Shrunk and shifted for better ergonomics.
 */
</script>

<template>
  <div class="gnn-unified">
    <div class="header-row">
      <div class="ph-label upward pl-20">Phase 1: Upward (Evidence)</div>
      <div class="ph-label downward pl-5">Phase 2: Downward (Context)</div>
    </div>
    
    <svg viewBox="0 0 820 280" xmlns="http://www.w3.org/2000/svg" class="gnn-svg">
      <defs>
        <marker id="arr-u" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#4f46e5" />
        </marker>
        <marker id="arr-d" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#059669" />
        </marker>
        <marker id="arr-prev" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#10b981" />
        </marker>
      </defs>

      <!-- 1. Initialization (Far Left) -->
      <g transform="translate(10, 85)">
        <rect x="0" y="0" width="70" height="35" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-dasharray="2" />
        <text x="35" y="14" text-anchor="middle" class="tiny-label black-txt italic">Initial h</text>
        <text x="35" y="27" text-anchor="middle" class="txt-small black-txt bold">h(0) = x</text>
      </g>

      <!-- 2. Entry Arrow: h(t) -->
      <path d="M85 102 L140 102" fill="none" class="edge prev-state" marker-end="url(#arr-prev)" />
      <text x="110" y="94" text-anchor="middle" class="tiny-label emerald-txt bold">h(t)</text>

      <!-- MAIN GNN DIAGRAM (Shifted Right) -->
      <g transform="translate(140, 0)">
        <!-- PHASE 1: UPWARD -->
        <g transform="translate(0, 0)">
          <!-- Phase 1 GRU -->
          <rect x="0" y="80" width="160" height="40" rx="6" class="box fill-indigo" />
          <text x="80" y="105" text-anchor="middle" class="txt bold white">GRU (Upward Update)</text>

          <!-- Upward Arrow to Parent with Intercept -->
          <line x1="80" y1="80" x2="80" y2="35" class="edge up" marker-end="url(#arr-u)" />
          
          <!-- INTERCEPT Tap -->
          <circle cx="80" cy="55" r="2.5" fill="#4f46e5" />
          <path d="M80 55 L325 55 L325 140 L395 140" fill="none" class="edge flow-dash" marker-end="url(#arr-u)" />
          <text x="230" y="50" class="tiny-label indigo-txt">h(t+1/2)</text>

          <!-- Parent (Upward target) -->
          <rect x="20" y="10" width="120" height="25" rx="4" class="box hollow-vsmall" />
          <text x="80" y="26" text-anchor="middle" class="txt-vsmall">Parent Node</text>

          <!-- Attention (MHA) -->
          <rect x="20" y="145" width="120" height="30" rx="6" class="box fill-indigo" />
          <text x="80" y="165" text-anchor="middle" class="txt bold white">Attention (MHA)</text>
          <line x1="80" y1="145" x2="80" y2="120" class="edge up" marker-end="url(#arr-u)" />
          <text x="85" y="138" class="tiny-label">h_att</text>

          <!-- Children (Sources) -->
          <path d="M-20 230 L40 175" class="edge up" marker-end="url(#arr-u)" />
          <path d="M80 230 L80 175" class="edge up" marker-end="url(#arr-u)" />
          <path d="M180 230 L120 175" class="edge up" marker-end="url(#arr-u)" />

          <rect x="-40" y="230" width="40" height="25" rx="4" class="box hollow-thin" />
          <text x="-20" y="247" text-anchor="middle" class="txt-tiny">Child 1</text>

          <rect x="60" y="230" width="40" height="25" rx="4" class="box hollow-thin" />
          <text x="80" y="247" text-anchor="middle" class="txt-tiny">Child 2</text>

          <rect x="160" y="230" width="40" height="25" rx="4" class="box hollow-thin" />
          <text x="180" y="247" text-anchor="middle" class="txt-tiny">Child 3</text>
        </g>

        <!-- PHASE 2: DOWNWARD -->
        <g transform="translate(395, 0)">
          <!-- Phase 2 GRU -->
          <rect x="0" y="120" width="160" height="40" rx="6" class="box fill-emerald" />
          <text x="80" y="145" text-anchor="middle" class="txt bold white">GRU (Downward Update)</text>

          <!-- Downward Arrow to Children with Intercept -->
          <line x1="80" y1="160" x2="80" y2="185" class="edge down" />
          
          <!-- INTERCEPT Tap -->
          <circle cx="80" cy="178" r="2.5" fill="#10b981" />
          <line x1="80" y1="178" x2="270" y2="178" class="edge next-state" marker-end="url(#arr-prev)" />
          <text x="210" y="173" class="tiny-label emerald-txt">h(t+1)</text>

          <!-- Parent (Source) -->
          <rect x="20" y="10" width="120" height="25" rx="4" class="box hollow-vsmall" />
          <text x="80" y="26" text-anchor="middle" class="txt-vsmall">Parent Node</text>
          
          <!-- MLP -->
          <rect x="20" y="60" width="120" height="30" rx="6" class="box fill-emerald" />
          <text x="80" y="80" text-anchor="middle" class="txt bold white">Downward MLP</text>
          <line x1="80" y1="35" x2="80" y2="60" class="edge down" marker-end="url(#arr-d)" />
          <line x1="80" y1="90" x2="80" y2="120" class="edge down" marker-end="url(#arr-d)" />
          <text x="85" y="110" class="tiny-label">h_msg</text>

          <!-- Children (Targets) -->
          <path d="M80 185 L80 195 L-5 195 L-5 210" fill="none" class="edge down" marker-end="url(#arr-d)" />
          <path d="M80 185 L80 195 L80 210" fill="none" class="edge down" marker-end="url(#arr-d)" />
          <path d="M80 185 L80 195 L165 195 L165 210" fill="none" class="edge down" marker-end="url(#arr-d)" />

          <rect x="-30" y="210" width="50" height="25" rx="4" class="box hollow-thin" />
          <text x="-5" y="227" text-anchor="middle" class="txt-tiny">Child 1</text>

          <rect x="55" y="210" width="50" height="25" rx="4" class="box hollow-thin" />
          <text x="80" y="227" text-anchor="middle" class="txt-tiny">Child 2</text>

          <rect x="140" y="210" width="50" height="25" rx="4" class="box hollow-thin" />
          <text x="165" y="227" text-anchor="middle" class="txt-tiny">Child 3</text>
        </g>

        <!-- Separation Line -->
        <line x1="335" y1="10" x2="335" y2="260" stroke="#f1f5f9" stroke-width="2" stroke-dasharray="4" />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.gnn-unified {
  width: 100%;
  padding: 0.25rem 0.5rem;
  background: white;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  font-family: 'Inter', sans-serif;
}
.header-row {
  display: flex;
  justify-content: flex-start;
  margin-bottom: -0.5rem;
}
.pl-20 { padding-left: 170px; }
.pl-5 { padding-left: 170px; }
.ph-label {
  font-size: 13px;
  font-weight: 800;
  text-align: center;
}
.upward { color: #4f46e5; }
.downward { color: #059669; }
.box {
  stroke-width: 1.25;
}
.hollow-vsmall {
  fill: #fdfdfd;
  stroke: #64748b;
  stroke-width: 1;
}
.hollow-thin {
  fill: #f8fafc;
  stroke: #94a3b8;
  stroke-width: 0.75;
}
.fill-indigo {
  fill: #6366f1;
  stroke: #4f46e5;
}
.fill-emerald {
  fill: #10b981;
  stroke: #059669;
}
.white { fill: #fff; }
.bold { font-weight: 700; }
.txt {
  font-size: 9px;
  fill: #1e293b;
}
.txt-small { font-size: 11px; }
.txt-vsmall { font-size: 8px; fill: #475569; }
.txt-tiny { font-size: 8px; fill: #64748b; }
.black-txt { fill: #1e293b; }
.edge {
  fill: none;
  stroke-width: 1.25;
}
.up { stroke: #6366f1; }
.down { stroke: #10b981; }
.prev-state, .next-state {
  stroke: #10b981;
  stroke-dasharray: 2;
}
.flow-dash {
  stroke: #4f46e5;
  stroke-dasharray: 4;
}
.tiny-label {
  font-size: 7px;
  font-weight: 600;
  fill: #94a3b8;
  font-family: ui-monospace, monospace;
}
.indigo-txt { fill: #4f46e5; }
.emerald-txt { fill: #059669; }
svg {
  width: 100%;
  height: auto;
  margin: 0 auto;
  display: block;
}
</style>
