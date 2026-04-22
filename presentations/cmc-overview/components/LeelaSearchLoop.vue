<template>
  <div class="loop-container">
    <svg viewBox="0 0 850 320" class="loop-svg">
      <defs>
        <marker id="arr-std" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" class="fill-primary" />
        </marker>
        <marker id="arr-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L8,3 L0,6 Z" class="fill-accent" />
        </marker>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
          <feOffset dx="2" dy="2" result="offsetblur" />
          <feComponentTransfer><feFuncA type="linear" slope="0.1" /></feComponentTransfer>
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <!-- 1. Board State -->
      <g transform="translate(30, 100)" filter="url(#shadow)">
        <rect x="0" y="0" width="120" height="60" rx="10" class="bg-neutral" />
        <text x="60" y="38" text-anchor="middle" class="label-box bold">Board State (s)</text>
        <line x1="120" y1="30" x2="160" y2="30" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 2. Leela (Planner) -->
      <g transform="translate(190, 85)" filter="url(#shadow)">
        <rect x="0" y="0" width="160" height="90" rx="14" class="bg-primary" />
        <text x="80" y="45" text-anchor="middle" class="label-large text-primary">Leela MCTS</text>
        <text x="80" y="68" text-anchor="middle" class="label-sub text-secondary uppercase bold">(Base Planner)</text>
        <line x1="160" y1="45" x2="210" y2="45" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 3. Meta-Controller -->
      <g transform="translate(400, 75)" filter="url(#shadow)">
        <rect x="0" y="0" width="200" height="110" rx="16" class="bg-accent" />
        <text x="100" y="50" text-anchor="middle" class="label-xl text-accent">Meta-Controller</text>
        <text x="100" y="80" text-anchor="middle" class="label-sub text-accent uppercase bold">(Stop/Go Policy)</text>
        
        <line x1="200" y1="55" x2="250" y2="55" class="edge" marker-end="url(#arr-std)" />
      </g>

      <!-- 4. Decision Point (0/1) -->
      <g transform="translate(650, 100)" filter="url(#shadow)">
        <circle cx="30" cy="30" r="30" class="bg-neutral" stroke-width="2.5" />
        <text x="30" y="38" text-anchor="middle" class="label-large bold">Decision</text>
        
        <!-- Halt (1) -> Act -->
        <line x1="60" y1="30" x2="110" y2="30" class="edge" marker-end="url(#arr-std)" />
        <text x="85" y="20" text-anchor="middle" class="label-sub text-danger bold">HALT (1)</text>
      </g>

      <!-- 5. Act Box -->
      <g transform="translate(760, 100)" filter="url(#shadow)">
        <rect x="0" y="0" width="90" height="60" rx="10" class="bg-success" />
        <text x="45" y="38" text-anchor="middle" class="label-box text-success bold uppercase">Act</text>
      </g>

      <!-- 6. Continue Loop (0) -->
      <path d="M 680 160 L 680 250 L 270 250 L 270 175" fill="none" class="edge-accent loop-path anim-dash" marker-end="url(#arr-accent)" />
      <rect x="350" y="235" width="230" height="30" rx="15" fill="var(--bg-accent)" stroke="var(--color-accent)" stroke-width="1.5" />
      <text x="465" y="255" text-anchor="middle" class="label-sub text-accent bold uppercase">0: Expand Tree / Search (Loop)</text>
    </svg>
  </div>
</template>

<style scoped>
.loop-container { width: 100%; padding: 2rem; background: transparent; }
.loop-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.label-xl { font-size: 19px; font-weight: 800; letter-spacing: -0.02em; }
.label-large { font-size: 17px; font-weight: 800; }
.label-box { font-size: 15px; font-weight: 700; }
.label-sub { font-size: 12px; letter-spacing: 0.05em; }

.edge { stroke: var(--color-primary); stroke-width: 3; fill: none; }
.edge-accent { stroke: var(--color-accent); stroke-width: 3.5; fill: none; }
.loop-path { stroke-dasharray: 6; }

.anim-dash { animation: dash 20s linear infinite; }
@keyframes dash { to { stroke-dashoffset: -100; } }

.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }
.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }

.bg-neutral { fill: white; stroke: var(--border-subtle); stroke-width: 2; }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); stroke-width: 2; }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2; }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 2; }

.fill-primary { fill: var(--color-primary); }
.fill-accent { fill: var(--color-accent); }
.bold { font-weight: 700; }
.uppercase { text-transform: uppercase; }
</style>
