<template>
  <div class="dp-tree-container bg-white p-6 rounded-3xl border border-slate-100 shadow-sm">
    <svg viewBox="0 0 900 380" class="w-full h-auto" font-family="var(--font-body)">
      <!-- Background / Horizon Area -->
      <rect x="680" y="20" width="200" height="340" fill="var(--bg-neutral)" rx="12" />
      <text x="780" y="45" text-anchor="middle" class="txt-horizon uppercase">Horizon Limit</text>
      
      <!-- Connection Edges -->
      <g fill="none" stroke-width="3">
        <!-- 63 -> Act -->
        <path d="M 180 180 L 180 100" stroke="var(--color-accent)" marker-end="url(#arrow-accent)" />
        <!-- 63 -> Continue -->
        <path d="M 180 180 L 480 180" stroke="var(--color-success)" marker-end="url(#arrow-success)" />
        
        <!-- 64 -> Act -->
        <path d="M 480 180 L 480 100" stroke="var(--color-accent)" marker-end="url(#arrow-accent)" />
        <!-- 64 -> Continue (Horizon) -->
        <path d="M 480 180 L 730 180" stroke="var(--color-danger)" marker-end="url(#arrow-danger)" />
      </g>

      <!-- Edge Labels -->
      <text x="190" y="130" class="txt-edge text-accent">Halt (Act)</text>
      <text x="330" y="170" text-anchor="middle" class="txt-edge text-success">Continue</text>
      
      <text x="490" y="130" class="txt-edge text-accent">Halt (Act)</text>
      <text x="605" y="170" text-anchor="middle" class="txt-edge text-danger">Continue</text>

      <!-- Nodes -->
      <g class="nodes">
        <circle cx="180" cy="180" r="28" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text x="180" y="188" text-anchor="middle" class="label-large text-primary">63</text>
        <text x="180" y="235" text-anchor="middle" class="label-bold text-secondary">Depth k=63</text>

        <circle cx="480" cy="180" r="28" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text x="480" y="188" text-anchor="middle" class="label-large text-primary">64</text>
        <text x="480" y="235" text-anchor="middle" class="label-bold text-secondary">Depth k=64</text>
      </g>

      <!-- Terminal Value Boxes -->
      <!-- V(act, 63) -->
      <rect x="110" y="50" width="140" height="45" rx="8" class="bg-accent" stroke-width="2" />
      <text x="180" y="78" text-anchor="middle" class="label-bold text-accent">V(act, 63)</text>

      <!-- V(act, 64) -->
      <rect x="410" y="50" width="140" height="45" rx="8" class="bg-accent" stroke-width="2" />
      <text x="480" y="78" text-anchor="middle" class="label-bold text-accent">V(act, 64)</text>

      <!-- Horizon Penalty -->
      <rect x="705" y="160" width="150" height="45" rx="8" class="bg-danger" stroke-width="2" />
      <text x="780" y="188" text-anchor="middle" class="label-bold text-danger">Reward = 0</text>

      <!-- DP Equations Overlay -->
      <g transform="translate(480, 275)">
        <rect x="-145" y="0" width="290" height="85" rx="12" fill="white" stroke="var(--border-subtle)" stroke-width="2" />
        <text x="-130" y="25" class="formula-label text-accent">Step 1: Solve Horizon</text>
        <text x="-130" y="50" class="formula-main text-secondary">V(continue, 64) = 0</text>
        <text x="-130" y="72" class="formula-main font-bold text-primary">V(64) = P·V(act, 64) + (1-P)·0</text>
      </g>

      <g transform="translate(180, 275)">
        <rect x="-145" y="0" width="290" height="85" rx="12" fill="white" stroke="var(--border-subtle)" stroke-width="2" />
        <text x="-130" y="25" class="formula-label text-success">Step 2: Back Out</text>
        <text x="-130" y="50" class="formula-main text-secondary">V(continue, 63) = V(64)</text>
        <text x="-130" y="72" class="formula-main font-bold text-primary">V(63) = P·V(act, 63) + (1-P)·V(64)</text>
      </g>

      <!-- DP Backing Out Arrow -->
      <path d="M 340 315 L 320 315" fill="none" stroke="var(--color-secondary)" stroke-width="3" stroke-dasharray="4 4" marker-end="url(#arrow-secondary)" class="anim-backout" />

      <defs>
        <marker id="arrow-accent" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="var(--color-accent)" /></marker>
        <marker id="arrow-success" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="var(--color-success)" /></marker>
        <marker id="arrow-danger" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="var(--color-danger)" /></marker>
        <marker id="arrow-secondary" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="var(--color-secondary)" /></marker>
      </defs>
    </svg>
  </div>
</template>

<style scoped>
.dp-tree-container { 
  width: 100%;
  --bg-neutral: #f8fafc;
  --bg-accent: #f5f3ff;
  --color-accent: #6366f1;
  --bg-success: #f0fdfa;
  --color-success: #0d9488;
  --bg-danger: #fff1f2;
  --color-danger: #e11d48;
}
.txt-edge { font-size: 13px; font-weight: 600; }
.txt-horizon { font-size: 12px; font-weight: 800; font-family: var(--font-header); letter-spacing: 0.05em; fill: var(--color-secondary); }

.bg-neutral { fill: var(--bg-neutral); }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); }
.bg-danger { fill: var(--bg-danger); stroke: var(--color-danger); }

.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }
.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }

.formula-label { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }
.formula-main { font-size: 13px; font-family: 'Fira Code', monospace; }

.anim-backout {
  animation: flowBack 3s infinite linear;
}

@keyframes flowBack {
  0% { stroke-dashoffset: 16; opacity: 0; }
  20%, 80% { opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0; }
}
</style>
