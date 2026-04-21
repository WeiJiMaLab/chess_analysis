<script setup lang="ts">
/**
 * Unified Unrolled Mechanistic View of the GNN Sweeps.
 * Initialization: h(0)=x (Outside) -> Black Arrow -> 
 * Main Figure: h(t) (Green, Inside) -> Phase 1 -> Phase 2.
 * Manual SVG bounding box for perfect vertical alignment.
 */
</script>

<template>
  <div class="gnn-container">
    <svg viewBox="0 0 820 300" xmlns="http://www.w3.org/2000/svg" class="gnn-svg">
      <defs>
        <!-- Arrow Markers -->
        <marker id="arr-u" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="text-accent" />
        </marker>
        <marker id="arr-d" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="text-success" />
        </marker>
        <marker id="arr-secondary" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="text-secondary" />
        </marker>
        <marker id="arr-prev" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="text-success" />
        </marker>
      </defs>

      <!-- 1. Bounding Box (Manual Figure Border) -->
      <rect x="120" y="5" width="690" height="285" rx="12" fill="white" stroke="var(--border-subtle)" stroke-width="1.5" />
      
      <!-- Separation Line -->
      <line x1="470" y1="35" x2="470" y2="250" stroke="var(--bg-neutral)" stroke-width="2" stroke-dasharray="4" />

      <!-- 2. Initialization -->
      <g transform="translate(10, 85)">
        <rect x="0" y="0" width="70" height="35" rx="4" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-dasharray="2" />
        <text x="35" y="14" text-anchor="middle" class="tiny italic">Initial h</text>
        <text x="35" y="27" text-anchor="middle" class="txt-small bold">h(0) = x</text>
      </g>

      <!-- Secondary arrow pointing INTO the bounding box -->
      <line x1="80" y1="102.5" x2="120" y2="102.5" stroke="var(--color-secondary)" stroke-width="1.5" marker-end="url(#arr-secondary)" />

      <!-- 3. Internal Mechanistic Logic -->
      <g transform="translate(130, 0)">
        <!-- h(t) arrow (Inside) -->
        <path d="M0 102.5 L60 102.5" fill="none" stroke="var(--color-success)" stroke-dasharray="2" marker-end="url(#arr-prev)" />
        <text x="30" y="94" text-anchor="middle" class="tiny text-success bold">h(t)</text>

        <!-- PHASE 1: UPWARD -->
        <g transform="translate(60, 0)">
          <!-- Phase 1 GRU -->
          <rect x="0" y="82.5" width="160" height="40" rx="6" class="box bg-accent" />
          <text x="80" y="107.5" text-anchor="middle" class="txt bold text-accent">GRU (Upward Update)</text>

          <!-- Upward Arrow to Parent -->
          <line x1="80" y1="82.5" x2="80" y2="35" class="edge edge-accent" marker-end="url(#arr-u)" />
          
          <!-- INTERCEPT Tap -->
          <circle cx="80" cy="55" r="2.5" fill="var(--color-accent)" />
          <path d="M80 55 L315 55 L315 142.5 L385 142.5" fill="none" class="edge flow-dash" marker-end="url(#arr-u)" />
          <text x="215" y="50" class="tiny text-accent">h(t+1/2)</text>

          <!-- Parent -->
          <rect x="20" y="10" width="120" height="25" rx="4" class="box hollow-secondary" />
          <text x="80" y="26" text-anchor="middle" class="txt-vsmall">Parent Node</text>

          <!-- Attention -->
          <rect x="20" y="147.5" width="120" height="30" rx="6" class="box bg-accent" />
          <text x="80" y="167.5" text-anchor="middle" class="txt bold text-accent">Attention (MHA)</text>
          <line x1="80" y1="147.5" x2="80" y2="122.5" class="edge edge-accent" marker-end="url(#arr-u)" />

          <!-- Children -->
          <path d="M-20 230 L40 177.5" class="edge edge-accent" marker-end="url(#arr-u)" />
          <path d="M80 230 L80 177.5" class="edge edge-accent" marker-end="url(#arr-u)" />
          <path d="M180 230 L120 177.5" class="edge edge-accent" marker-end="url(#arr-u)" />

          <rect x="-40" y="230" width="40" height="24" rx="4" class="box hollow-thin" />
          <text x="-20" y="246" text-anchor="middle" class="txt-tiny">Child 1</text>
          <rect x="60" y="230" width="40" height="24" rx="4" class="box hollow-thin" />
          <text x="80" y="246" text-anchor="middle" class="txt-tiny">Child 2</text>
          <rect x="160" y="230" width="40" height="24" rx="4" class="box hollow-thin" />
          <text x="180" y="246" text-anchor="middle" class="txt-tiny">Child 3</text>
        </g>

        <!-- PHASE 2: DOWNWARD -->
        <g transform="translate(415, 0)">
          <!-- Phase 2 GRU -->
          <rect x="0" y="122.5" width="160" height="40" rx="6" class="box bg-success" />
          <text x="80" y="147.5" text-anchor="middle" class="txt bold text-success">GRU (Downward Update)</text>

          <!-- Intercept h(t+1) -->
          <circle cx="80" cy="180.5" r="2.5" fill="var(--color-success)" />
          <line x1="80" y1="180.5" x2="250" y2="180.5" class="edge next-state" marker-end="url(#arr-prev)" />
          <text x="180" y="175" class="tiny text-success bold">h(t+1)</text>

          <!-- Parent (Source) -->
          <rect x="20" y="10" width="120" height="25" rx="4" class="box hollow-secondary" />
          <text x="80" y="26" text-anchor="middle" class="txt-vsmall">Parent Node</text>
          
          <!-- MLP -->
          <rect x="20" y="62.5" width="120" height="30" rx="6" class="box bg-success" />
          <text x="80" y="82.5" text-anchor="middle" class="txt bold text-success">Downward MLP</text>
          <line x1="80" y1="35" x2="80" y2="62.5" class="edge edge-success" marker-end="url(#arr-d)" />
          <line x1="80" y1="92.5" x2="80" y2="122.5" class="edge edge-success" marker-end="url(#arr-d)" />

          <!-- Children (Targets) -->
          <path d="M80 162.5 L80 210" class="edge edge-success" marker-end="url(#arr-d)" />
          <path d="M80 188 L-10 188 L-10 210" fill="none" class="edge edge-success" marker-end="url(#arr-d)" />
          <path d="M80 188 L170 188 L170 210" fill="none" class="edge edge-success" marker-end="url(#arr-d)" />

          <rect x="-35" y="210" width="50" height="25" rx="4" class="box hollow-thin" />
          <text x="-10" y="227" text-anchor="middle" class="txt-tiny">Child 1</text>
          <rect x="55" y="210" width="50" height="25" rx="4" class="box hollow-thin" />
          <text x="80" y="227" text-anchor="middle" class="txt-tiny">Child 2</text>
          <rect x="145" y="210" width="50" height="25" rx="4" class="box hollow-thin" />
          <text x="170" y="227" text-anchor="middle" class="txt-tiny">Child 3</text>
        </g>
      </g>

      <!-- Fig Footers -->
      <text x="310" y="278" text-anchor="middle" class="ph-label upward uppercase letter-spacing">Phase 1: Upward Update</text>
      <text x="640" y="278" text-anchor="middle" class="ph-label downward uppercase letter-spacing">Phase 2: Downward Update</text>
    </svg>
  </div>
</template>

<style scoped>
.gnn-container {
  width: 100%;
  padding: 0.5rem;
}
.gnn-svg {
  width: 100%;
  height: auto;
  display: block;
  font-family: 'Inter', sans-serif;
}
.ph-label { font-size: 11px; font-weight: 800; }
.upward { fill: var(--color-accent); }
.downward { fill: var(--color-success); }
.uppercase { text-transform: uppercase; }
.letter-spacing { letter-spacing: 0.05em; }
.box { stroke-width: 1.25; }
.hollow-secondary { fill: var(--bg-neutral); stroke: var(--color-secondary); stroke-width: 1; }
.hollow-thin { fill: var(--bg-neutral); stroke: var(--border-subtle); stroke-width: 0.75; }
.fill-accent { fill: var(--color-accent); stroke: var(--color-accent); }
.fill-success { fill: var(--color-success); stroke: var(--color-success); }
.white { fill: #fff; }
.bold { font-weight: 700; }
.txt { font-size: 10px; }
.txt-small { font-size: 11px; }
.txt-vsmall { font-size: 8px; fill: var(--color-primary); }
.txt-tiny { font-size: 8px; fill: var(--color-secondary); }
.tiny { font-size: 8px; fill: var(--color-secondary); }
.edge { fill: none; stroke-width: 1.25; }
.edge-accent { stroke: var(--color-accent); }
.edge-success { stroke: var(--color-success); }
.next-state { stroke: var(--color-success); stroke-dasharray: 2; }
.flow-dash { stroke: var(--color-accent); stroke-dasharray: 4; }
.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
</style>
