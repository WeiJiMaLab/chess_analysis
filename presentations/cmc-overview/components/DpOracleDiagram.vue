<template>
  <div class="dp-tree-container">
    <svg viewBox="0 0 920 400" class="dp-svg" font-family="var(--font-diagram)">
      <!-- Background / Horizon Area -->
      <rect x="700" y="20" width="200" height="340" fill="var(--bg-neutral)" rx="12" />
      <text x="800" y="45" text-anchor="middle" class="txt-horizon uppercase">Horizon Limit</text>
      
      <!-- Connection Edges -->
      <g fill="none" stroke-width="3.5">
        <!-- 63 -> Act -->
        <path d="M 160 180 L 160 100" stroke="var(--color-accent)" marker-end="url(#arrow-accent)" />
        <!-- 63 -> Continue -->
        <path d="M 160 180 L 520 180" stroke="var(--color-success)" marker-end="url(#arrow-success)" />
        
        <!-- 64 -> Act -->
        <path d="M 520 180 L 520 100" stroke="var(--color-accent)" marker-end="url(#arrow-accent)" />
        <!-- 64 -> Continue (Horizon) -->
        <path d="M 520 180 L 750 180" stroke="var(--color-danger)" marker-end="url(#arrow-danger)" />
      </g>

      <!-- Edge Labels -->
      <text x="170" y="130" class="txt-edge text-accent bold">Halt (Act)</text>
      <text x="340" y="170" text-anchor="middle" class="txt-edge text-success bold">Continue</text>
      
      <text x="530" y="130" class="txt-edge text-accent bold">Halt (Act)</text>
      <text x="635" y="170" text-anchor="middle" class="txt-edge text-danger">Continue</text>

      <!-- Nodes -->
      <g class="nodes">
        <circle cx="160" cy="180" r="32" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text x="160" y="190" text-anchor="middle" class="label-large text-primary bold">63</text>
        <text x="160" y="240" text-anchor="middle" class="label-node-desc text-secondary bold">Depth k=63</text>

        <circle cx="520" cy="180" r="32" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text x="520" y="190" text-anchor="middle" class="label-large text-primary bold">64</text>
        <text x="520" y="240" text-anchor="middle" class="label-node-desc text-secondary bold">Depth k=64</text>
      </g>

      <!-- Terminal Value Boxes -->
      <rect x="80" y="45" width="160" height="50" rx="8" class="bg-accent" stroke-width="2.5" />
      <text x="160" y="78" text-anchor="middle" class="label-box-text text-accent bold">V(act, 63)</text>

      <rect x="440" y="45" width="160" height="50" rx="8" class="bg-accent" stroke-width="2.5" />
      <text x="520" y="78" text-anchor="middle" class="label-box-text text-accent bold">V(act, 64)</text>

      <rect x="725" y="160" width="150" height="50" rx="8" class="bg-danger" stroke-width="2.5" />
      <text x="800" y="192" text-anchor="middle" class="label-box-text text-danger bold">Reward = 0</text>

      <!-- DP Equations Overlay -->
      <g transform="translate(520, 285)">
        <rect x="-160" y="0" width="320" height="95" rx="12" fill="white" stroke="var(--border-subtle)" stroke-width="2.5" />
        <text x="-145" y="28" class="formula-label text-accent uppercase">Step 1: Solve Horizon</text>
        <text x="-145" y="55" class="formula-main text-secondary">V(continue, 64) = 0</text>
        <text x="-145" y="80" class="formula-main bold text-primary">V(64) = P·V(act, 64) + (1-P)·0</text>
      </g>

      <g transform="translate(160, 285)">
        <rect x="-160" y="0" width="320" height="95" rx="12" fill="white" stroke="var(--border-subtle)" stroke-width="2.5" />
        <text x="-145" y="28" class="formula-label text-success uppercase">Step 2: Back Out</text>
        <text x="-145" y="55" class="formula-main text-secondary">V(continue, 63) = V(64)</text>
        <text x="-145" y="80" class="formula-main bold text-primary">V(63) = P·V(act, 63) + (1-P)·V(64)</text>
      </g>

      <!-- DP Backing Out Arrow -->
      <path d="M 360 330 L 320 330" fill="none" stroke="var(--color-secondary)" stroke-width="3" stroke-dasharray="4 4" marker-end="url(#arrow-secondary)" class="anim-backout" />

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
.dp-tree-container { width: 100%; padding: 1rem 0; background: white; }
.dp-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.txt-edge { font-size: 16px; font-weight: 700; }
.txt-horizon { font-size: 14px; font-weight: 800; letter-spacing: 0.05em; fill: var(--color-secondary); }

.label-large { font-size: 22px; font-weight: 800; }
.label-node-desc { font-size: 15px; font-weight: 700; }
.label-box-text { font-size: 15px; font-weight: 800; }

.formula-label { font-size: 14px; font-weight: 800; letter-spacing: 0.05em; }
.formula-main { font-size: 15px; font-family: var(--font-mono); }

.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }
.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }

.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); }
.bg-danger { fill: var(--bg-danger); stroke: var(--color-danger); }

.anim-backout { animation: flowBack 3s infinite linear; }
@keyframes flowBack {
  0% { stroke-dashoffset: 16; opacity: 0; }
  20%, 80% { opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0; }
}
.bold { font-weight: 700; }
.uppercase { text-transform: uppercase; }
</style>
