<template>
  <div class="dp-tree-container bg-white p-6 rounded-3xl border border-slate-100 shadow-sm">
    <svg viewBox="0 0 900 360" class="w-full h-auto" font-family="Inter, sans-serif">
      <!-- Background / Horizon Area -->
      <rect x="680" y="20" width="200" height="320" fill="var(--bg-neutral)" rx="12" />
      <text x="780" y="45" text-anchor="middle" class="txt-horizon uppercase tracking-widest text-xs font-bold text-secondary">Horizon Limit</text>
      
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
      <text x="190" y="130" class="txt-edge font-bold text-accent">Halt (Act)</text>
      <text x="330" y="170" text-anchor="middle" class="txt-edge font-bold text-success">Continue</text>
      
      <text x="490" y="130" class="txt-edge font-bold text-accent">Halt (Act)</text>
      <text x="605" y="170" text-anchor="middle" class="txt-edge font-bold text-danger">Continue</text>

      <!-- Nodes -->
      <g class="nodes">
        <circle cx="180" cy="180" r="24" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text x="180" y="186" text-anchor="middle" class="text-lg font-black text-primary">63</text>
        <text x="180" y="225" text-anchor="middle" class="text-xs font-bold text-secondary">Depth k=63</text>

        <circle cx="480" cy="180" r="24" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text x="480" y="186" text-anchor="middle" class="text-lg font-black text-primary">64</text>
        <text x="480" y="225" text-anchor="middle" class="text-xs font-bold text-secondary">Depth k=64</text>
      </g>

      <!-- Terminal Value Boxes -->
      <!-- V(act, 63) -->
      <rect x="120" y="50" width="120" height="40" rx="6" class="bg-accent" stroke-width="2" />
      <text x="180" y="75" text-anchor="middle" class="text-sm font-black text-accent">V(act, 63)</text>

      <!-- V(act, 64) -->
      <rect x="420" y="50" width="120" height="40" rx="6" class="bg-accent" stroke-width="2" />
      <text x="480" y="75" text-anchor="middle" class="text-sm font-black text-accent">V(act, 64)</text>

      <!-- Horizon Penalty -->
      <rect x="710" y="160" width="140" height="40" rx="6" class="bg-danger" stroke-width="2" />
      <text x="780" y="185" text-anchor="middle" class="text-sm font-black text-danger">Reward = 0</text>

      <!-- DP Equations Overlay -->
      <!-- Step 1: k=64 -->
      <g transform="translate(480, 260)">
        <rect x="-130" y="0" width="260" height="70" rx="8" fill="white" stroke="var(--border-subtle)" stroke-width="2" />
        <text x="-115" y="22" class="text-xs font-bold text-accent">Step 1: Solve Horizon</text>
        <text x="-115" y="42" class="text-[10.5px] font-mono text-secondary">V(continue, 64) = 0</text>
        <text x="-115" y="60" class="text-[10.5px] font-mono font-bold text-primary">V(64) = P(act)·V(act, 64) + (1-P)·0</text>
      </g>

      <!-- Step 2: k=63 -->
      <g transform="translate(180, 260)">
        <rect x="-130" y="0" width="260" height="70" rx="8" fill="white" stroke="var(--border-subtle)" stroke-width="2" />
        <text x="-115" y="22" class="text-xs font-bold text-success">Step 2: Back Out</text>
        <text x="-115" y="42" class="text-[10.5px] font-mono text-secondary">V(continue, 63) = V(64)</text>
        <text x="-115" y="60" class="text-[10.5px] font-mono font-bold text-primary">V(63) = P(act)·V(act, 63) + (1-P)·V(64)</text>
      </g>

      <!-- DP Backing Out Arrow -->
      <path d="M 340 295 L 320 295" fill="none" stroke="var(--color-secondary)" stroke-width="3" stroke-dasharray="4 4" marker-end="url(#arrow-secondary)" class="anim-backout" />

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
  --bg-accent: #e0e7ff;
  --color-accent: #6366f1;
  --bg-success: #ecfdf5;
  --color-success: #10b981;
  --bg-danger: #fff1f2;
  --color-danger: #f43f5e;
}
.txt-edge { font-size: 11px; }

.bg-neutral { fill: var(--bg-neutral); }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); }
.bg-danger { fill: var(--bg-danger); stroke: var(--color-danger); }

.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }

.anim-backout {
  animation: flowBack 3s infinite linear;
}

@keyframes flowBack {
  0% { stroke-dashoffset: 16; opacity: 0; }
  20%, 80% { opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0; }
}
</style>
