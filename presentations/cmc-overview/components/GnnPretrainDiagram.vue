<template>
  <div class="pretrain-container">
    <svg viewBox="0 0 1000 440" xmlns="http://www.w3.org/2000/svg" class="pretrain-svg">
      <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" class="text-secondary" />
        </marker>
        <marker id="arrowhead-success" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" class="text-success" />
        </marker>
        <marker id="arrowhead-accent" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" class="text-accent" />
        </marker>
      </defs>

      <!-- STAGE 1: INPUT PARTIAL TREE -->
      <g transform="translate(40, 80)">
        <text x="100" y="-30" text-anchor="middle" class="label-main text-accent">1. PARTIAL TREE (T)</text>
        <rect x="0" y="0" width="200" height="220" rx="12" class="box-input" />
        
        <g transform="translate(100, 50) scale(0.8)">
          <path d="M0,0 L-50,70 M0,0 L50,70 M-50,70 L-80,140 M-50,70 L-20,140" stroke="var(--color-accent)" stroke-width="4" stroke-linecap="round" />
          <circle cx="0" cy="0" r="16" fill="white" stroke="var(--color-accent)" stroke-width="4" />
          <text y="5" text-anchor="middle" class="txt-id">v₁</text>
          <circle cx="-50" cy="70" r="16" fill="white" stroke="var(--color-accent)" stroke-width="4" />
          <text x="-50" y="75" text-anchor="middle" class="txt-id">v₂</text>
          <circle cx="50" cy="70" r="14" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-width="2" />
          <circle cx="-80" cy="140" r="16" fill="white" stroke="var(--color-accent)" stroke-width="4" />
          <text x="-80" y="145" text-anchor="middle" class="txt-id">v₃</text>
          <circle cx="-20" cy="140" r="12" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-width="2" />
        </g>
        <text x="100" y="200" text-anchor="middle" class="label-sub">Initial State xᵢ</text>
      </g>

      <!-- STATIC DATA FLOW -->
      <g transform="translate(250, 180)">
        <path d="M0,-30 C40,-30 40,-100 80,-100" fill="none" stroke="var(--border-subtle)" stroke-width="2" marker-end="url(#arrowhead)" />
        <path d="M0,30 C40,30 40,100 80,100" fill="none" stroke="var(--border-subtle)" stroke-width="2" marker-end="url(#arrowhead)" />
      </g>

      <!-- STAGE 2: ENGINES -->
      <!-- Oracle Search: Expansion from Partial Tree -->
      <g transform="translate(330, 30)">
        <rect x="0" y="0" width="220" height="180" rx="12" class="box-process-success" />
        <text x="110" y="25" text-anchor="middle" class="label-box text-success">Oracle Search</text>
        
        <g transform="translate(110, 50) scale(0.48)">
           <!-- Partial Tree Duplicate (Blue) -->
           <path d="M0,0 L-50,70 M0,0 L50,70 M-50,70 L-80,140 M-50,70 L-20,140" stroke="var(--color-accent)" stroke-width="4" stroke-linecap="round" opacity="0.6" />
           <circle cx="0" cy="0" r="14" fill="white" stroke="var(--color-accent)" stroke-width="3" />
           <circle cx="-50" cy="70" r="14" fill="white" stroke="var(--color-accent)" stroke-width="3" />
           <circle cx="50" cy="70" r="10" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-width="2" />
           <circle cx="-80" cy="140" r="14" fill="white" stroke="var(--color-accent)" stroke-width="3" />
           <circle cx="-20" cy="140" r="10" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-width="2" />

           <!-- Search Expansion (Emerald) -->
           <g class="mcts-expansion">
              <!-- Sequential Path 1 -->
              <path d="M-80,140 L-120,220" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" class="grow-branch-seq1" />
              <circle cx="-120" cy="220" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" class="pop-node-seq1" />

              <!-- Sequential Path 2 -->
              <path d="M-80,140 L-40,220" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" class="grow-branch-seq2" />
              <circle cx="-40" cy="220" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" class="pop-node-seq2" />

              <!-- Sequential Path 3 -->
              <path d="M50,70 L100,140" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" class="grow-branch-seq3" />
              <circle cx="100" cy="140" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" class="pop-node-seq3" />

              <!-- Sequential Path 4 -->
              <path d="M100,140 L80,210" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" class="grow-branch-seq4" />
              <circle cx="80" cy="210" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" class="pop-node-seq4" />

              <!-- Sequential Path 5 -->
              <path d="M100,140 L120,210" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" class="grow-branch-seq5" />
              <circle cx="120" cy="210" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" class="pop-node-seq5" />
           </g>
        </g>
      </g>

      <!-- GNN -->
      <g transform="translate(330, 260)">
        <rect x="0" y="0" width="220" height="120" rx="12" class="box-process-accent" />
        <text x="110" y="30" text-anchor="middle" class="label-box text-accent">GNN Forward</text>
        <rect x="50" y="55" width="120" height="25" rx="4" fill="var(--color-accent)" opacity="0.1" />
        <text x="110" y="72" text-anchor="middle" style="font-size: 10px; font-weight: 800" fill="var(--color-accent)">GNN(T, x)</text>
        <text x="110" y="110" text-anchor="middle" class="label-sub">(Latent Summary)</text>
      </g>

      <!-- OUTPUT FLOW -->
      <g transform="translate(550, 115)">
        <path d="M0,0 L70,0" fill="none" stroke="var(--color-success)" stroke-width="3" marker-end="url(#arrowhead-success)" />
      </g>
      <g transform="translate(550, 320)">
        <path d="M0,0 L70,0" fill="none" stroke="var(--color-accent)" stroke-width="3" marker-end="url(#arrowhead-accent)" />
      </g>

      <!-- STAGE 3: WDL VECTORS -->
      <g transform="translate(640, 30)">
        <text x="80" y="-15" text-anchor="middle" class="label-main text-success">ACTUAL WDL (Y)</text>
        <rect x="0" y="0" width="160" height="130" rx="10" fill="var(--bg-neutral)" stroke="var(--color-success)" stroke-dasharray="4 4" />
        <g transform="translate(20, 20)">
          <text x="-15" y="15" class="txt-id">v₁</text>
          <rect x="0" y="0" width="40" height="20" fill="var(--color-success)" rx="2" />
          <rect x="42" y="0" width="40" height="20" fill="#f59e0b" rx="2" opacity="0.3" /> <!-- Amber for draw is fine -->
          <rect x="84" y="0" width="40" height="20" fill="var(--color-danger)" rx="2" opacity="0.1" />
          <text x="-15" y="50" class="txt-id">v₂</text>
          <rect x="0" y="35" width="20" height="20" fill="var(--color-success)" rx="2" />
          <rect x="22" y="35" width="80" height="20" fill="#f59e0b" rx="2" />
          <rect x="104" y="35" width="20" height="20" fill="var(--color-danger)" rx="2" />
          <text x="-15" y="85" class="txt-id">v₃</text>
          <rect x="0" y="70" width="10" height="20" fill="var(--color-success)" rx="2" opacity="0.1" />
          <rect x="12" y="70" width="20" height="20" fill="#f59e0b" rx="2" opacity="0.3" />
          <rect x="34" y="70" width="90" height="20" fill="var(--color-danger)" rx="2" />
        </g>
      </g>

      <g transform="translate(640, 250)">
        <text x="80" y="-15" text-anchor="middle" class="label-main text-accent">PREDICTED WDL (Ŷ)</text>
        <rect x="0" y="0" width="160" height="130" rx="10" fill="var(--bg-neutral)" stroke="var(--color-accent)" />
        <g transform="translate(20, 20)">
          <text x="-15" y="15" class="txt-id">v₁</text>
          <rect x="0" y="0" width="30" height="20" fill="var(--color-accent)" rx="2" />
          <rect x="32" y="0" width="50" height="20" fill="var(--color-accent)" rx="2" opacity="0.6" />
          <rect x="84" y="0" width="40" height="20" fill="var(--color-accent)" rx="2" opacity="0.2" />
          <text x="-15" y="50" class="txt-id">v₂</text>
          <rect x="0" y="35" width="40" height="20" fill="var(--color-accent)" rx="2" />
          <rect x="42" y="35" width="60" height="20" fill="var(--color-accent)" rx="2" opacity="0.8" />
          <rect x="104" y="35" width="20" height="20" fill="var(--color-accent)" rx="2" opacity="0.4" />
          <text x="-15" y="85" class="txt-id">v₃</text>
          <rect x="0" y="70" width="20" height="20" fill="var(--color-accent)" rx="2" opacity="0.2" />
          <rect x="22" y="70" width="30" height="20" fill="var(--color-accent)" rx="2" opacity="0.5" />
          <rect x="54" y="70" width="70" height="20" fill="var(--color-accent)" rx="2" />
        </g>
      </g>

      <!-- LOSS COMPONENT -->
      <g transform="translate(850, 160)">
        <rect x="0" y="0" width="130" height="100" rx="15" class="box-loss" />
        <text x="65" y="45" text-anchor="middle" class="label-loss-title text-danger">LOSS L</text>
        <text x="65" y="80" text-anchor="middle" class="label-sub bold" style="font-weight: 800">∑ CE(Yᵢ, Ŷᵢ)</text>
      </g>

      <!-- Connections to Loss -->
      <g stroke="var(--color-danger)" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.3">
        <path d="M800,60 C830,60 830,140 850,150" fill="none" />
        <path d="M800,280 C830,280 830,230 850,210" fill="none" />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.pretrain-container { width: 100%; padding: 0.5rem 2rem; background: white; border-radius: 1rem; }
.pretrain-svg { width: 100%; height: auto; font-family: 'Inter', sans-serif; }

.box-input { fill: var(--bg-neutral); stroke: var(--border-subtle); stroke-width: 2; }
.box-process-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 2; }
.box-process-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2; }
.box-loss { fill: var(--bg-danger); stroke: var(--color-danger); stroke-width: 2; }

.label-main { font-size: 14px; font-weight: 800; letter-spacing: 0.05em; }
.label-box { font-size: 16px; font-weight: 700; }
.label-sub { font-size: 11px; font-weight: 500; fill: var(--color-secondary); }
.label-loss-title { font-size: 16px; font-weight: 900; }

.txt-id { font-size: 11px; font-weight: 800; fill: var(--color-secondary); }
.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }

/* Oracle Animation Sequence: Edge then Node, 5s Cycle */
.grow-branch-seq1 { animation: grow-path1 5s infinite; }
.grow-branch-seq2 { animation: grow-path2 5s infinite; }
.grow-branch-seq3 { animation: grow-path3 5s infinite; }
.grow-branch-seq4 { animation: grow-path4 5s infinite; }
.grow-branch-seq5 { animation: grow-path5 5s infinite; }

.pop-node-seq1 { animation: pop-path1 5s infinite; }
.pop-node-seq2 { animation: pop-path2 5s infinite; }
.pop-node-seq3 { animation: pop-path3 5s infinite; }
.pop-node-seq4 { animation: pop-path4 5s infinite; }
.pop-node-seq5 { animation: pop-path5 5s infinite; }

@keyframes grow-path1 { 0% { stroke-dashoffset: 40; opacity: 0; } 15%, 100% { stroke-dashoffset: 0; opacity: 1; } }
@keyframes grow-path2 { 0%, 20% { stroke-dashoffset: 40; opacity: 0; } 35%, 100% { stroke-dashoffset: 0; opacity: 1; } }
@keyframes grow-path3 { 0%, 40% { stroke-dashoffset: 40; opacity: 0; } 55%, 100% { stroke-dashoffset: 0; opacity: 1; } }
@keyframes grow-path4 { 0%, 60% { stroke-dashoffset: 40; opacity: 0; } 75%, 100% { stroke-dashoffset: 0; opacity: 1; } }
@keyframes grow-path5 { 0%, 80% { stroke-dashoffset: 40; opacity: 0; } 95%, 100% { stroke-dashoffset: 0; opacity: 1; } }

@keyframes pop-path1 { 0%, 15% { transform: scale(0); opacity: 0; } 20%, 100% { transform: scale(1); opacity: 1; } }
@keyframes pop-path2 { 0%, 35% { transform: scale(0); opacity: 0; } 40%, 100% { transform: scale(1); opacity: 1; } }
@keyframes pop-path3 { 0%, 55% { transform: scale(0); opacity: 0; } 60%, 100% { transform: scale(1); opacity: 1; } }
@keyframes pop-path4 { 0%, 75% { transform: scale(0); opacity: 0; } 80%, 100% { transform: scale(1); opacity: 1; } }
@keyframes pop-path5 { 0%, 95% { transform: scale(0); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
</style>


