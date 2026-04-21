<script setup lang="ts">
/**
 * Unified Unrolled Mechanistic View of the GNN Sweeps.
 * Phase 1: Upward Update (Accent Dark) -> h(t+1/2)
 * Phase 2: Downward Update (Accent Light) -> h(t+1)
 * Connection: Upward result h(t+1/2) flows into Downward GRU.
 */
</script>

<template>
  <div class="gnn-container">
    <svg viewBox="0 0 850 340" xmlns="http://www.w3.org/2000/svg" class="gnn-svg">
      <defs>
        <marker id="arr-up" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent-dark" />
        </marker>
        <marker id="arr-down" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent-light" />
        </marker>
        <marker id="arr-secondary" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="var(--color-secondary)" />
        </marker>
      </defs>

      <!-- 1. Main Bounding Box -->
      <rect x="130" y="5" width="700" height="320" rx="16" fill="white" stroke="var(--border-subtle)" stroke-width="1.5" />
      
      <!-- Vertical Split -->
      <line x1="480" y1="40" x2="480" y2="280" stroke="var(--bg-neutral)" stroke-width="2" stroke-dasharray="4" />

      <!-- 2. Initialization -->
      <g transform="translate(10, 100)">
        <rect x="0" y="0" width="100" height="45" rx="8" class="bg-neutral" stroke-dasharray="2" />
        <text x="50" y="15" text-anchor="middle" class="tiny-i text-secondary">Initial h</text>
        <text x="50" y="34" text-anchor="middle" class="label-bold">h(0) = x</text>
      </g>

      <line x1="110" y1="122.5" x2="130" y2="122.5" stroke="var(--color-secondary)" stroke-width="1.5" marker-end="url(#arr-secondary)" />

      <!-- 3. Internal Logic -->
      <g transform="translate(140, 0)">
        <!-- Previous state arrow -->
        <path d="M0 122.5 L60 122.5" fill="none" stroke="var(--color-accent-light)" stroke-dasharray="2" marker-end="url(#arr-down)" />
        <text x="30" y="114" text-anchor="middle" class="tiny text-accent-light bold">h(t)</text>

        <!-- PHASE 1: UPWARD -->
        <g transform="translate(60, 0)">
          <rect x="0" y="95" width="180" height="50" rx="8" class="bg-accent-dark" />
          <text x="90" y="125" text-anchor="middle" class="label-bold text-accent-dark">GRU Upward</text>

          <line x1="90" y1="95" x2="90" y2="55" class="edge-accent-dark" marker-end="url(#arr-up)" />
          <rect x="30" y="25" width="120" height="30" rx="6" class="bg-neutral" stroke="var(--color-secondary)" />
          <text x="90" y="44" text-anchor="middle" class="tiny text-primary bold">Parent Node</text>

          <line x1="90" y1="175" x2="90" y2="145" class="edge-accent-dark" marker-end="url(#arr-up)" />
          <rect x="30" y="175" width="120" height="40" rx="8" class="bg-accent-dark" />
          <text x="90" y="200" text-anchor="middle" class="label-bold text-accent-dark">Attention</text>

          <!-- Children -->
          <g transform="translate(90, 255)">
            <rect x="-105" y="0" width="60" height="30" rx="6" class="bg-neutral" />
            <text x="-75" y="19" text-anchor="middle" class="tiny">Child 1</text>
            <rect x="-30" y="0" width="60" height="30" rx="6" class="bg-neutral" />
            <text x="0" y="19" text-anchor="middle" class="tiny">Child 2</text>
            <rect x="45" y="0" width="60" height="30" rx="6" class="bg-neutral" />
            <text x="75" y="19" text-anchor="middle" class="tiny">Child 3</text>
            
            <path d="M-75 0 L-25 -40" class="edge-accent-dark" marker-end="url(#arr-up)" />
            <path d="M0 0 L0 -40" class="edge-accent-dark" marker-end="url(#arr-up)" />
            <path d="M75 0 L25 -40" class="edge-accent-dark" marker-end="url(#arr-up)" />
          </g>

          <!-- h(t+1/2) FLOW INTO PHASE 2 -->
          <path d="M180 120 L270 120 L270 170 L360 170" fill="none" stroke="var(--color-accent-dark)" stroke-width="2.5" stroke-dasharray="4" marker-end="url(#arr-up)" />
          <text x="260" y="112" text-anchor="middle" class="tiny text-accent-dark bold">h(t+1/2)</text>
        </g>

        <!-- PHASE 2: DOWNWARD -->
        <g transform="translate(420, 0)">
          <rect x="0" y="145" width="180" height="50" rx="8" class="bg-accent-light" />
          <text x="90" y="175" text-anchor="middle" class="label-bold text-accent-light">GRU Downward</text>

          <rect x="30" y="25" width="120" height="30" rx="6" class="bg-neutral" stroke="var(--color-secondary)" />
          <text x="90" y="44" text-anchor="middle" class="tiny text-primary bold">Parent Node</text>

          <line x1="90" y1="55" x2="90" y2="85" class="edge-accent-light" marker-end="url(#arr-down)" />
          <rect x="30" y="85" width="120" height="40" rx="8" class="bg-accent-light" />
          <text x="90" y="110" text-anchor="middle" class="label-bold text-accent-light">Downward MLP</text>
          <line x1="90" y1="125" x2="90" y2="145" class="edge-accent-light" marker-end="url(#arr-down)" />

          <!-- Children Targets -->
          <g transform="translate(90, 255)">
            <rect x="-105" y="0" width="60" height="30" rx="6" class="bg-neutral" />
            <text x="-75" y="19" text-anchor="middle" class="tiny">Child 1</text>
            <rect x="-30" y="0" width="60" height="30" rx="6" class="bg-neutral" />
            <text x="0" y="19" text-anchor="middle" class="tiny">Child 2</text>
            <rect x="45" y="0" width="60" height="30" rx="6" class="bg-neutral" />
            <text x="75" y="19" text-anchor="middle" class="tiny">Child 3</text>
            
            <path d="M0 -60 L0 -15" class="edge-accent-light" marker-end="url(#arr-down)" />
            <path d="M0 -40 L-75 -40 L-75 -15" fill="none" class="edge-accent-light" marker-end="url(#arr-down)" />
            <path d="M0 -40 L75 -40 L75 -15" fill="none" class="edge-accent-light" marker-end="url(#arr-down)" />
          </g>

          <!-- FINAL STATE h(t+1) -->
          <path d="M180 170 L250 170" stroke="var(--color-accent-light)" stroke-width="2.5" stroke-dasharray="4" marker-end="url(#arr-down)" />
          <text x="215" y="162" text-anchor="middle" class="tiny text-accent-light bold">h(t+1)</text>
        </g>
      </g>

      <!-- Labels -->
      <text x="330" y="310" text-anchor="middle" class="ph-label text-accent-dark uppercase">Phase 1: Upward Update</text>
      <text x="690" y="310" text-anchor="middle" class="ph-label text-accent-light uppercase">Phase 2: Downward Update</text>
    </svg>
  </div>
</template>

<style scoped>
.gnn-container { width: 100%; padding: 0.5rem; }
.gnn-svg { width: 100%; height: auto; display: block; font-family: var(--font-body); overflow: visible; }
.ph-label { font-size: 13px; font-weight: 800; font-family: var(--font-header); letter-spacing: 0.05em; }
.label-bold { font-size: 14px; font-weight: 700; }
.tiny { font-size: 11px; }
.tiny-i { font-size: 11px; font-style: italic; }

.edge-accent-dark { stroke: var(--color-accent-dark); stroke-width: 2; fill: none; }
.edge-accent-light { stroke: var(--color-accent-light); stroke-width: 2; fill: none; }

.uppercase { text-transform: uppercase; }
.text-accent-dark { fill: var(--color-accent-dark); }
.text-accent-light { fill: var(--color-accent-light); }
.text-secondary { fill: var(--color-secondary); }
.text-primary { fill: var(--color-primary); }

.bg-accent-dark { fill: var(--bg-accent); stroke: var(--color-accent-dark); stroke-width: 2; }
.bg-accent-light { fill: var(--bg-accent); stroke: var(--color-accent-light); stroke-width: 2; }
</style>
