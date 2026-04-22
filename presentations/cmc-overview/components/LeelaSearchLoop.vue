<template>
  <div class="loop-container">
    <svg viewBox="0 0 1100 400" class="loop-svg">
      <defs>
        <marker id="arr-std" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" class="fill-primary" />
        </marker>
        <marker id="arr-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" class="fill-accent" />
        </marker>
      </defs>

      <!-- 1. Board State -->
      <g transform="translate(30, 120)">
        <rect x="0" y="0" width="160" height="70" rx="12" class="bg-neutral" />
        <text x="80" y="44" text-anchor="middle" class="label-box bold">Board State (s)</text>
        <line x1="160" y1="35" x2="210" y2="35" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 2. Leela (Planner) -->
      <g transform="translate(240, 100)">
        <rect x="0" y="0" width="200" height="110" rx="16" class="bg-primary" />
        <text x="100" y="55" text-anchor="middle" class="label-large text-primary">Leela MCTS</text>
        <text x="100" y="82" text-anchor="middle" class="label-sub text-secondary uppercase bold">(Base Planner)</text>
        <line x1="200" y1="55" x2="290" y2="55" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 3. Meta-Controller -->
      <g transform="translate(530, 90)">
        <rect x="0" y="0" width="260" height="130" rx="20" class="bg-accent" />
        <text x="130" y="60" text-anchor="middle" class="label-xl text-accent">Meta-Controller</text>
        <text x="130" y="95" text-anchor="middle" class="label-sub text-accent uppercase bold">(Stop/Go Policy)</text>
        
        <line x1="260" y1="65" x2="310" y2="65" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 4. Decision Point (0/1) -->
      <g transform="translate(840, 120)">
        <circle cx="35" cy="35" r="35" class="bg-neutral" stroke-width="3" />
        <text x="35" y="44" text-anchor="middle" class="label-large bold">0/1</text>
        
        <!-- Halt (1) -> Act -->
        <line x1="70" y1="35" x2="150" y2="35" class="edge" marker-end="url(#arr-std)" />
        <text x="110" y="22" text-anchor="middle" class="label-sub text-danger bold">1: HALT</text>
      </g>

      <!-- 5. Act Box -->
      <g transform="translate(1000, 120)">
        <rect x="0" y="0" width="100" height="70" rx="12" class="bg-success" />
        <text x="50" y="44" text-anchor="middle" class="label-box text-success bold uppercase">Act</text>
      </g>

      <!-- 6. Continue Loop (0) -->
      <path d="M 875 190 L 875 300 L 340 300 L 340 210" fill="none" class="edge-accent loop-path anim-dash" marker-end="url(#arr-accent)" />
      <rect x="420" y="285" width="280" height="30" rx="15" fill="var(--bg-accent)" stroke="var(--color-accent)" stroke-width="2" />
      <text x="560" y="305" text-anchor="middle" class="label-sub text-accent bold uppercase">0: CONTINUE SEARCH (LOOP)</text>
    </svg>
  </div>
</template>

<style scoped>
.loop-container { width: 100%; padding: 0.5rem; background: transparent; display: flex; justify-content: center; }
.loop-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.label-xl { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
.label-large { font-size: 19px; font-weight: 800; }
.label-box { font-size: 16px; font-weight: 700; }
.label-sub { font-size: 13px; letter-spacing: 0.05em; }

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
</style>
