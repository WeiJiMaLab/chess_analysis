<template>
  <div class="pretrain-container">
    <svg viewBox="0 0 1000 440" xmlns="http://www.w3.org/2000/svg" class="pretrain-svg">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-secondary" />
        </marker>
        <marker id="arrowhead-success" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-success" />
        </marker>
        <marker id="arrowhead-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-accent" />
        </marker>
      </defs>

      <!-- STAGE 1: INPUT PARTIAL TREE -->
      <g transform="translate(40, 80)">
        <text x="100" y="-30" text-anchor="middle" class="label-main text-accent uppercase">1. Partial Tree (T)</text>
        <rect x="0" y="0" width="200" height="220" rx="12" class="box-input" stroke-width="2.5" />
        
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
        <text x="100" y="205" text-anchor="middle" class="label-sub bold uppercase letter-spacing">Initial State xᵢ</text>
      </g>

      <!-- STATIC DATA FLOW -->
      <g transform="translate(250, 180)">
        <path d="M0,-30 C40,-30 40,-100 75,-100" fill="none" stroke="var(--border-subtle)" stroke-width="2.5" marker-end="url(#arrowhead)" />
        <path d="M0,30 C40,30 40,100 75,100" fill="none" stroke="var(--border-subtle)" stroke-width="2.5" marker-end="url(#arrowhead)" />
      </g>

      <!-- STAGE 2: ENGINES -->
      <g transform="translate(325, 30)">
        <rect x="0" y="0" width="220" height="180" rx="12" class="box-process-success" stroke-width="2.5" />
        <text x="110" y="30" text-anchor="middle" class="label-box text-success uppercase">Oracle Search</text>
        
        <g transform="translate(110, 60) scale(0.48)">
           <path d="M0,0 L-50,70 M0,0 L50,70 M-50,70 L-80,140 M-50,70 L-20,140" stroke="var(--color-accent)" stroke-width="4" stroke-linecap="round" opacity="0.6" />
           <circle cx="0" cy="0" r="14" fill="white" stroke="var(--color-accent)" stroke-width="3" />
           <circle cx="-50" cy="70" r="14" fill="white" stroke="var(--color-accent)" stroke-width="3" />
           <circle cx="50" cy="70" r="10" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-width="2" />
           <circle cx="-80" cy="140" r="14" fill="white" stroke="var(--color-accent)" stroke-width="3" />
           <circle cx="-20" cy="140" r="10" fill="var(--bg-neutral)" stroke="var(--color-secondary)" stroke-width="2" />

           <g class="mcts-expansion">
              <path d="M-80,140 L-120,220" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" />
              <circle cx="-120" cy="220" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" />
              <path d="M50,70 L100,140" stroke="var(--color-success)" stroke-width="4" stroke-dasharray="4 4" />
              <circle cx="100" cy="140" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="2" />
           </g>
        </g>
      </g>

      <g transform="translate(325, 260)">
        <rect x="0" y="0" width="220" height="130" rx="12" class="box-process-accent" stroke-width="2.5" />
        <text x="110" y="35" text-anchor="middle" class="label-box text-accent uppercase">GNN Forward</text>
        <rect x="40" y="60" width="140" height="30" rx="6" fill="var(--color-accent)" opacity="0.1" />
        <text x="110" y="80" text-anchor="middle" style="font-size: 13px; font-weight: 800" fill="var(--color-accent)" class="font-diagram">GNN(T, x)</text>
        <text x="110" y="115" text-anchor="middle" class="label-sub bold uppercase">(Latent Summary)</text>
      </g>

      <!-- OUTPUT FLOW -->
      <g transform="translate(545, 115)">
        <path d="M0,0 L85,0" fill="none" stroke="var(--color-success)" stroke-width="3.5" marker-end="url(#arrowhead-success)" />
      </g>
      <g transform="translate(545, 325)">
        <path d="M0,0 L85,0" fill="none" stroke="var(--color-accent)" stroke-width="3.5" marker-end="url(#arrowhead-accent)" />
      </g>

      <!-- STAGE 3: WDL VECTORS -->
      <g transform="translate(640, 20)">
        <text x="85" y="-15" text-anchor="middle" class="label-main text-success uppercase">Actual WDL (Y)</text>
        <rect x="0" y="0" width="170" height="155" rx="12" fill="var(--bg-neutral)" stroke="var(--color-success)" stroke-dasharray="4 4" stroke-width="2.5" />
        <g transform="translate(30, 30)">
          <text x="-15" y="18" class="txt-id">v₁</text>
          <rect x="0" y="0" width="50" height="26" fill="var(--color-success)" rx="4" />
          <rect x="54" y="0" width="50" height="26" fill="var(--color-accent)" rx="4" opacity="0.2" /> 
          
          <text x="-15" y="62" class="txt-id">v₂</text>
          <rect x="0" y="44" width="25" height="26" fill="var(--color-success)" rx="4" />
          <rect x="29" y="44" width="80" height="26" fill="var(--color-accent)" rx="4" opacity="0.3" />

          <text x="-15" y="106" class="txt-id">v₃</text>
          <rect x="0" y="88" width="80" height="26" fill="var(--color-success)" rx="4" />
          <rect x="84" y="88" width="20" height="26" fill="var(--color-accent)" rx="4" opacity="0.1" />
        </g>
      </g>

      <g transform="translate(640, 250)">
        <text x="85" y="-15" text-anchor="middle" class="label-main text-accent uppercase">Predicted WDL (Ŷ)</text>
        <rect x="0" y="0" width="170" height="155" rx="12" fill="var(--bg-neutral)" stroke="var(--color-accent)" stroke-width="2.5" />
        <g transform="translate(30, 30)">
          <text x="-15" y="18" class="txt-id">v₁</text>
          <rect x="0" y="0" width="40" height="26" fill="var(--color-accent)" rx="4" />
          <rect x="44" y="0" width="60" height="26" fill="var(--color-accent)" rx="4" opacity="0.6" />
          
          <text x="-15" y="62" class="txt-id">v₂</text>
          <rect x="0" y="44" width="50" height="26" fill="var(--color-accent)" rx="4" />
          <rect x="54" y="44" width="60" height="26" fill="var(--color-accent)" rx="4" opacity="0.8" />

          <text x="-15" y="106" class="txt-id">v₃</text>
          <rect x="0" y="88" width="70" height="26" fill="var(--color-accent)" rx="4" />
          <rect x="74" y="88" width="30" height="26" fill="var(--color-accent)" rx="4" opacity="0.4" />
        </g>
      </g>

      <!-- LOSS COMPONENT -->
      <g transform="translate(860, 160)">
        <rect x="0" y="0" width="130" height="115" rx="16" class="box-loss" stroke-width="3" />
        <text x="65" y="48" text-anchor="middle" class="label-loss-title text-danger uppercase">Loss L</text>
        <text x="65" y="90" text-anchor="middle" class="label-formula bold salient">∑ CE(Yᵢ, Ŷᵢ)</text>
      </g>

      <!-- Connections to Loss -->
      <g stroke="var(--color-danger)" stroke-width="2.5" stroke-dasharray="4 4" opacity="0.4">
        <path d="M810,95 C830,95 840,190 860,190" fill="none" />
        <path d="M810,325 C830,325 840,240 860,240" fill="none" />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.pretrain-container { width: 100%; padding: 0.5rem 1rem; background: white; border-radius: 1rem; }
.pretrain-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.box-input { fill: var(--bg-neutral); stroke: var(--border-subtle); }
.box-process-success { fill: var(--bg-success); stroke: var(--color-success); }
.box-process-accent { fill: var(--bg-accent); stroke: var(--color-accent); }
.box-loss { fill: var(--bg-danger); stroke: var(--color-danger); }

.label-main { font-size: 18px; font-weight: 800; font-family: var(--font-diagram); letter-spacing: -0.01em; }
.label-box { font-size: 20px; font-weight: 800; font-family: var(--font-diagram); }
.label-sub { font-size: 13px; font-weight: 700; fill: var(--color-secondary); }
.label-loss-title { font-size: 22px; font-weight: 900; font-family: var(--font-diagram); }
.label-formula { font-size: 15px; font-family: var(--font-mono); fill: var(--color-danger); }

.txt-id { font-size: 13px; font-weight: 800; fill: var(--color-secondary); }
.font-diagram { font-family: var(--font-diagram); }

.uppercase { text-transform: uppercase; }
.letter-spacing { letter-spacing: 0.05em; }
.bold { font-weight: 800; }
.salient { opacity: 0.9; }
</style>
