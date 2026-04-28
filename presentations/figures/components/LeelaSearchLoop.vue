<template>
  <div class="loop-container">
    <svg viewBox="-28 0 828 872" class="loop-svg">
      <defs>
        <marker id="arr-std" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" class="fill-primary" />
        </marker>
        <marker id="arr-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" class="fill-accent" />
        </marker>
      </defs>

      <g class="diagram-shift" transform="translate(-54, 0)">
      <!-- 1. Board State -->
      <g transform="translate(270, 24)">
        <rect x="0" y="0" width="264" height="68" rx="12" class="bg-neutral" />
        <text x="132" y="42" text-anchor="middle" class="label-box bold">Board State (s)</text>
        <line x1="132" y1="68" x2="132" y2="84" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 2. Leela (Planner) -->
      <g transform="translate(269, 112)">
        <rect x="0" y="0" width="266" height="120" rx="16" class="bg-primary" />
        <text x="133" y="52" text-anchor="middle" class="label-large text-primary">Leela MCTS</text>
        <text x="133" y="86" text-anchor="middle" class="label-sub text-secondary uppercase bold">(Base Planner)</text>
        <line x1="133" y1="120" x2="133" y2="172" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 3. Meta-Controller -->
      <g transform="translate(202, 272)">
        <rect x="0" y="0" width="400" height="140" rx="20" class="bg-accent" />
        <text x="200" y="56" text-anchor="middle" class="label-xl text-accent">Meta-Controller</text>
        <text x="200" y="98" text-anchor="middle" class="label-sub text-accent uppercase bold">(Stop/Go Policy)</text>

        <line x1="200" y1="140" x2="200" y2="184" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 4. Decision Point (0/1) -->
      <g transform="translate(367, 456)">
        <circle cx="35" cy="35" r="35" class="bg-neutral circle-node" />
        <text x="35" y="44" text-anchor="middle" class="label-large bold">0/1</text>

        <!-- Halt (1) -> Act -->
        <line x1="35" y1="70" x2="35" y2="142" class="edge" marker-end="url(#arr-std)" />
        <text x="38" y="116" text-anchor="start" class="label-sub text-danger bold">1: HALT</text>
      </g>

      <!-- 5. Act Box -->
      <g transform="translate(336, 600)">
        <rect x="0" y="0" width="132" height="64" rx="12" class="bg-success" />
        <text x="66" y="40" text-anchor="middle" class="label-box text-success bold uppercase">Act</text>
      </g>

      <!-- 6. Continue search (branch 0) -->
      <path d="M 367 491 L 180 491 L 180 176 L 269 176" fill="none" class="edge-accent loop-path anim-dash" marker-end="url(#arr-accent)" />

      <g class="continue-search-slot">
        <text x="0" y="0" text-anchor="middle" class="label-sub text-accent bold uppercase continue-search-text">0: CONTINUE SEARCH</text>
      </g>
      </g>
    </svg>
  </div>
</template>

<script setup>
</script>

<style scoped>
.loop-container {
  box-sizing: border-box;
  width: 100%;
  min-height: 0;
  flex: 1 1 auto;
  padding: 0.25rem 0.35rem 0.25rem 0;
  background: transparent;
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
}

/* Sizing delegated to global .figure-container.figure-leela svg (wins specificity over .figure-container svg) */
.loop-svg {
  width: auto;
  height: auto;
  max-width: 100%;
  font-family: var(--font-diagram);
  overflow: visible;
}

.label-xl { font-weight: 800; letter-spacing: -0.02em; }
.label-large { font-weight: 800; }
.label-box { font-weight: 700; }
.label-sub { letter-spacing: 0.05em; }

/* Match rect stroke; thicker ring for decision node */
.circle-node {
  stroke: var(--border-subtle);
  stroke-width: 3;
  paint-order: stroke fill;
}

.edge { stroke: var(--color-primary); stroke-width: 3.5; fill: none; }
.edge-accent { stroke: var(--color-accent); stroke-width: 4; fill: none; }
.loop-path { stroke-dasharray: 8; }

.anim-dash { animation: dash 20s linear infinite; }
@keyframes dash { to { stroke-dashoffset: -160; } }

.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }
.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }

.bg-neutral { fill: white; stroke: var(--border-subtle); stroke-width: 2.5; }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); stroke-width: 2.5; }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2.5; }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 2.5; }

.fill-primary { fill: var(--color-primary); }
.fill-accent { fill: var(--color-accent); }
.bold { font-weight: 700; }
.uppercase { text-transform: uppercase; }

.continue-search-slot {
  font-size: var(--font-size-sub);
  transform: translate(calc(182px - 1em), 334px) rotate(-90deg);
}

.continue-search-text {
  dominant-baseline: central;
}
</style>
