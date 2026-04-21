<script setup lang="ts">
/**
 * High-Level Horizontal Control Loop.
 * Board -> Leela -> Meta-Controller -> 0/1 Decision -> Act/Loop.
 */
</script>

<template>
  <div class="loop-container">
    <svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" class="loop-svg">
      <defs>
        <marker id="arr-std" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-primary" />
        </marker>
        <marker id="arr-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent" />
        </marker>
      </defs>

      <!-- 1. Board State -->
      <g transform="translate(20, 75)">
        <rect x="0" y="0" width="100" height="45" rx="8" class="bg-neutral" />
        <text x="50" y="28" text-anchor="middle" class="label-bold">Board (s)</text>
        <line x1="100" y1="22.5" x2="140" y2="22.5" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 2. Leela (Planner) -->
      <g transform="translate(160, 65)">
        <rect x="0" y="0" width="120" height="65" rx="10" class="bg-primary" />
        <text x="60" y="32" text-anchor="middle" class="label-large text-primary">Leela</text>
        <text x="60" y="52" text-anchor="middle" class="tiny text-secondary">(Planner)</text>
        <line x1="120" y1="32.5" x2="160" y2="32.5" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 3. Meta-Controller -->
      <g transform="translate(320, 60)">
        <rect x="0" y="0" width="160" height="75" rx="12" class="bg-accent" />
        <text x="80" y="35" text-anchor="middle" class="label-large text-accent">Meta-Controller</text>
        <text x="80" y="55" text-anchor="middle" class="tiny text-accent">(GNN Evaluator)</text>
        
        <line x1="160" y1="37.5" x2="200" y2="37.5" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 4. Decision Point (0/1) -->
      <g transform="translate(520, 75)">
        <circle cx="22" cy="22" r="22" class="bg-neutral" stroke-width="2" />
        <text x="22" y="28" text-anchor="middle" class="label-bold">0/1</text>
        
        <!-- Halt (1) -> Act -->
        <line x1="44" y1="22" x2="80" y2="22" class="edge" marker-end="url(#arr-std)" />
        <text x="62" y="15" text-anchor="middle" class="tiny bold">1</text>
      </g>

      <!-- 5. Act Box -->
      <g transform="translate(600, 75)">
        <rect x="0" y="0" width="80" height="45" rx="8" class="bg-success" />
        <text x="40" y="28" text-anchor="middle" class="label-bold text-success uppercase">Act</text>
      </g>

      <!-- 6. Continue Loop (0) -->
      <path d="M542 120 L542 170 L220 170 L220 130" fill="none" class="edge-accent loop-path" marker-end="url(#arr-accent)" />
      <text x="381" y="185" text-anchor="middle" class="ph-label text-accent uppercase">0: Continue Search (Expand T)</text>
    </svg>
  </div>
</template>

<style scoped>
.loop-container { width: 100%; max-width: 800px; margin: 0 auto; padding: 1rem; background: white; border-radius: 1rem; }
.loop-svg {
  width: 100%;
  height: auto;
  font-family: var(--font-diagram);
  overflow: visible;
}

.label-bold { font-size: 14px; font-weight: 700; }
.label-large { font-size: 16px; font-weight: 800; font-family: var(--font-diagram); }
.tiny { font-size: 11px; }
.ph-label { font-size: 12px; font-weight: 800; font-family: var(--font-diagram); letter-spacing: 0.05em; }

.edge { stroke: var(--color-primary); stroke-width: 2; fill: none; }
.edge-accent { stroke: var(--color-accent); stroke-width: 2; fill: none; }
.loop-path { stroke-dasharray: 4; }

.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }
.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.uppercase { text-transform: uppercase; }
.bold { font-weight: 700; }

.bg-neutral { fill: var(--bg-neutral); stroke: var(--border-subtle); stroke-width: 1.5; }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); stroke-width: 1.5; }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 1.5; }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 1.5; }

.fill-primary { fill: var(--color-primary); }
.fill-accent { fill: var(--color-accent); }
</style>
