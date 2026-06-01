<template>
  <div class="pretrain-container">
    <svg viewBox="0 0 1000 480" xmlns="http://www.w3.org/2000/svg" class="pretrain-svg">
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

      <g transform="translate(0, 35)">
        <!-- STAGE 1: INPUT SEARCH TREE -->
        <g transform="translate(40, 80)">
          <text x="100" y="-30" text-anchor="middle" class="label-main text-accent uppercase">Input Search Tree (T)</text>
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
          <text x="110" y="24" text-anchor="middle" class="label-box text-success uppercase">
            <tspan x="110" dy="0">Oracle</tspan>
            <tspan x="110" dy="20">Consolidation</tspan>
          </text>
          
          <g transform="translate(110, 75) scale(0.48)">
             <path d="M0,0 L-50,70 M0,0 L50,70 M-50,70 L-80,140 M-50,70 L-20,140" stroke="var(--color-success)" stroke-width="4" stroke-linecap="round" />
             <circle cx="0" cy="0" r="14" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="3" />
             <circle cx="-50" cy="70" r="14" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="3" />
             <circle cx="50" cy="70" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="3" />
             <circle cx="-80" cy="140" r="14" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="3" />
             <circle cx="-20" cy="140" r="10" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="3" />
          </g>
        </g>

        <g transform="translate(325, 260)">
          <rect x="0" y="0" width="220" height="130" rx="12" class="box-process-accent" stroke-width="2.5" />
          <text x="110" y="28" text-anchor="middle" class="label-box text-accent uppercase">
            <tspan x="110" dy="0">GNN</tspan>
            <tspan x="110" dy="20">Representation</tspan>
          </text>
          <rect x="40" y="65" width="140" height="30" rx="6" fill="var(--color-accent)" opacity="0.1" />
          <text x="110" y="85" text-anchor="middle" style="font-size: 13px; font-weight: 800" fill="var(--color-accent)" class="font-diagram">GNN(T, x)</text>
          <text x="110" y="115" text-anchor="middle" class="label-sub bold uppercase text-accent-dark salient">(Latent Summary)</text>
        </g>

        <!-- OUTPUT FLOW -->
        <g transform="translate(545, 115)">
          <path d="M0,0 L85,0" fill="none" stroke="var(--color-success)" stroke-width="3.5" marker-end="url(#arrowhead-success)" />
        </g>
        <g transform="translate(545, 325)">
          <path d="M0,0 L85,0" fill="none" stroke="var(--color-accent)" stroke-width="3.5" marker-end="url(#arrowhead-accent)" />
        </g>

        <!-- STAGE 3: WDL VECTORS -->
        <g transform="translate(650, 20)">
          <text x="75" y="-25" text-anchor="middle" class="label-main text-success uppercase">Actual WDL (Y)</text>
          <text x="75" y="-8" text-anchor="middle" class="label-tiny uppercase text-success opacity-70">Search-Consolidated</text>
          <rect x="0" y="0" width="150" height="155" rx="12" fill="var(--bg-neutral)" stroke="var(--color-success)" stroke-dasharray="4 4" stroke-width="2.5" />
          <g transform="translate(35, 30)">
            <text x="-18" y="18" class="txt-id">v₁</text>
            <rect x="0" y="0" width="70" height="26" fill="var(--color-success)" rx="4" />
            <rect x="72" y="0" width="20" height="26" fill="var(--color-success)" rx="4" opacity="0.2" />
            <rect x="94" y="0" width="6" height="26" fill="var(--color-success)" rx="4" opacity="0.1" /> 
            
            <text x="-18" y="62" class="txt-id">v₂</text>
            <rect x="0" y="44" width="10" height="26" fill="var(--color-success)" rx="4" />
            <rect x="12" y="44" width="60" height="26" fill="var(--color-success)" rx="4" opacity="0.4" />
            <rect x="74" y="44" width="26" height="26" fill="var(--color-success)" rx="4" opacity="0.1" />

            <text x="-18" y="106" class="txt-id">v₃</text>
            <rect x="0" y="88" width="50" height="26" fill="var(--color-success)" rx="4" />
            <rect x="52" y="88" width="30" height="26" fill="var(--color-success)" rx="4" opacity="0.2" />
            <rect x="84" y="88" width="16" height="26" fill="var(--color-success)" rx="4" opacity="0.1" />
          </g>
        </g>

        <g transform="translate(650, 250)">
          <text x="75" y="-25" text-anchor="middle" class="label-main text-accent uppercase">Predicted WDL (Ŷ)</text>
          <text x="75" y="-8" text-anchor="middle" class="label-tiny uppercase text-accent opacity-70">Tree-Learned</text>
          <rect x="0" y="0" width="150" height="155" rx="12" fill="var(--bg-neutral)" stroke="var(--color-accent)" stroke-width="2.5" />
          <g transform="translate(35, 30)">
            <text x="-18" y="18" class="txt-id">v₁</text>
            <rect x="0" y="0" width="50" height="26" fill="var(--color-accent)" rx="4" />
            <rect x="52" y="0" width="30" height="26" fill="var(--color-accent)" rx="4" opacity="0.6" />
            <rect x="84" y="0" width="16" height="26" fill="var(--color-accent)" rx="4" opacity="0.3" />
            
            <text x="-18" y="62" class="txt-id">v₂</text>
            <rect x="0" y="44" width="15" height="26" fill="var(--color-accent)" rx="4" />
            <rect x="17" y="44" width="50" height="26" fill="var(--color-accent)" rx="4" opacity="0.8" />
            <rect x="69" y="44" width="31" height="26" fill="var(--color-accent)" rx="4" opacity="0.4" />

            <text x="-18" y="106" class="txt-id">v₃</text>
            <rect x="0" y="88" width="40" height="26" fill="var(--color-accent)" rx="4" />
            <rect x="42" y="88" width="30" height="26" fill="var(--color-accent)" rx="4" opacity="0.4" />
            <rect x="74" y="88" width="26" height="26" fill="var(--color-accent)" rx="4" opacity="0.2" />
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

.label-main { font-size: 18px; font-weight: 800; letter-spacing: -0.01em; }
.label-box { font-size: 20px; font-weight: 800; }
.label-sub { font-size: 13px; font-weight: 700; fill: var(--color-secondary); }
.label-loss-title { font-size: 22px; font-weight: 900; }
.label-formula { font-size: 15px; font-family: var(--font-mono); fill: var(--color-danger); }
.label-tiny { font-size: 15px; font-weight: 700; letter-spacing: 0.02em; }
.txt-id { font-size: 13px; font-weight: 800; fill: var(--color-secondary); }
.font-diagram { font-family: var(--font-diagram); }

.uppercase { text-transform: uppercase; }
.letter-spacing { letter-spacing: 0.05em; }
.bold { font-weight: 800; }
.salient { opacity: 0.9; }
</style>
