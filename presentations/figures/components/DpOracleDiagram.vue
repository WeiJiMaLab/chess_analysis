<template>
  <div class="dp-tree-container">
    <svg viewBox="0 0 880 440" class="dp-svg">
      <!-- CENTRAL SPINE -->
      <line x1="160" y1="30" x2="160" y2="350" stroke="var(--border-subtle)" stroke-width="2" stroke-dasharray="4 4" />
      
      <!-- NODE 62 -->
      <g transform="translate(160, 50)">
        <!-- Decision Branch (Slid Up) -->
        <g transform="translate(0, 42)">
          <line x1="0" y1="-22" x2="0" y2="65" stroke="var(--color-primary)" stroke-width="3" />
          <circle cx="0" cy="0" r="4.5" fill="var(--color-primary)" />
          <path d="M 0 0 L 175 0" stroke="var(--color-accent)" stroke-width="3.5" marker-end="url(#arrow-accent)" />
          <rect x="20" y="-13" width="90" height="26" rx="4" fill="white" stroke="var(--color-accent)" stroke-width="1.5" />
          <text x="65" y="5" text-anchor="middle" class="label-tiny text-accent bold">π_halt(T₆₂)</text>
          <text x="145" y="-8" text-anchor="middle" class="label-tiny text-accent bold">act</text>
          
          <g transform="translate(180, -20)">
            <rect x="0" y="0" width="170" height="40" rx="6" class="bg-accent" stroke-width="2.5" />
            <text x="85" y="25" text-anchor="middle" class="label-box-text text-accent bold">V<tspan baseline-shift="sub" font-size="0.7em">oracle</tspan>(best₆₂)</text>
          </g>
          <!-- Continue (Left of Bar) -->
          <text x="-12" y="32" text-anchor="end" class="label-tiny text-success bold">continue ↓</text>
        </g>
        <!-- Node Circle -->
        <circle cx="0" cy="0" r="26" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text y="8" text-anchor="middle" class="label-large text-primary bold">62</text>
        <text x="-40" y="5" text-anchor="end" class="label-node-desc text-secondary bold">Depth 62</text>
      </g>

      <!-- NODE 63 -->
      <g transform="translate(160, 155)">
        <!-- Decision Branch (Slid Up) -->
        <g transform="translate(0, 42)">
          <line x1="0" y1="-22" x2="0" y2="65" stroke="var(--color-primary)" stroke-width="3" />
          <circle cx="0" cy="0" r="4.5" fill="var(--color-primary)" />
          <path d="M 0 0 L 175 0" stroke="var(--color-accent)" stroke-width="3.5" marker-end="url(#arrow-accent)" />
          <rect x="20" y="-13" width="90" height="26" rx="4" fill="white" stroke="var(--color-accent)" stroke-width="1.5" />
          <text x="65" y="5" text-anchor="middle" class="label-tiny text-accent bold">π_halt(T₆₃)</text>
          <text x="145" y="-8" text-anchor="middle" class="label-tiny text-accent bold">act</text>
          
          <g transform="translate(180, -20)">
            <rect x="0" y="0" width="170" height="40" rx="6" class="bg-accent" stroke-width="2.5" />
            <text x="85" y="25" text-anchor="middle" class="label-box-text text-accent bold">V<tspan baseline-shift="sub" font-size="0.7em">oracle</tspan>(best₆₃)</text>
          </g>
          <!-- Continue (Left of Bar) -->
          <text x="-12" y="32" text-anchor="end" class="label-tiny text-success bold">continue ↓</text>
        </g>
        <!-- Node Circle -->
        <circle cx="0" cy="0" r="26" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text y="8" text-anchor="middle" class="label-large text-primary bold">63</text>
        <text x="-40" y="5" text-anchor="end" class="label-node-desc text-secondary bold">Depth 63</text>
      </g>

      <!-- NODE 64 (HORIZON) -->
      <g transform="translate(160, 260)">
        <g transform="translate(0, 40)">
          <line x1="0" y1="-20" x2="0" y2="12" stroke="var(--color-danger)" stroke-width="3" />
          <circle cx="0" cy="12" r="4.5" fill="var(--color-danger)" />
          <path d="M 0 12 L 135 12" stroke="var(--color-danger)" stroke-width="3.5" marker-end="url(#arrow-danger)" />
          <text x="65" y="4" text-anchor="middle" class="label-tiny text-danger bold uppercase">Force Halt</text>
          
          <g transform="translate(140, -10)">
            <rect x="0" y="0" width="170" height="40" rx="6" class="bg-danger" stroke-width="2.5" />
            <text x="85" y="25" text-anchor="middle" class="label-box-text text-danger bold">V<tspan baseline-shift="sub" font-size="0.7em">oracle</tspan>(best₆₄)</text>
          </g>
        </g>

        <!-- Do Not Enter Sign -->
        <g transform="translate(0, 105)">
          <circle cx="0" cy="0" r="16" fill="var(--color-danger)" />
          <rect x="-10" y="-2" width="20" height="4" fill="white" rx="1" />
          <text y="32" text-anchor="middle" class="label-tiny text-danger bold uppercase">Horizon</text>
        </g>
        <line x1="0" y1="60" x2="0" y2="90" stroke="var(--color-danger)" stroke-width="3" stroke-dasharray="4 4" />

        <!-- Node Circle -->
        <circle cx="0" cy="0" r="26" fill="white" stroke="var(--color-primary)" stroke-width="4" />
        <text y="8" text-anchor="middle" class="label-large text-primary bold">64</text>
        <text x="-40" y="5" text-anchor="end" class="label-node-desc text-secondary bold">Depth 64</text>
      </g>

      <!-- RECURSIVE LOGIC OVERLAY -->
      <g transform="translate(540, 60)">
        <rect x="0" y="0" width="280" height="280" rx="16" fill="var(--bg-neutral)" stroke="var(--border-subtle)" stroke-width="2" />
        <text x="140" y="30" text-anchor="middle" class="formula-label text-primary uppercase bold">Search Consolidation</text>
        
        <g transform="translate(20, 65)">
          <text x="0" y="0" class="formula-label text-secondary salient bold">1. Ground Truth (V*)</text>
          <text x="10" y="22" class="formula-main text-danger">V*(T₆₄) = V<tspan baseline-shift="sub" font-size="0.7em">oracle</tspan>(best₆₄)</text>
          <text x="10" y="44" class="formula-main text-primary">V*(Tₖ) = max(V<tspan baseline-shift="sub" font-size="0.7em">act</tspan>, V*(Tₖ₊₁))</text>
        </g>

        <g transform="translate(20, 145)">
          <text x="0" y="0" class="formula-label text-secondary salient bold">2. Objective</text>
          <text x="10" y="25" class="formula-main bold text-accent">V_π(T) → V*(T)</text>
          <text x="10" y="48" class="label-tiny text-secondary" font-style="italic">Learn to predict the</text>
          <text x="10" y="62" class="label-tiny text-secondary" font-style="italic">consolidated value.</text>
        </g>

        <path d="M 140 120 L 140 140" fill="none" stroke="var(--color-accent)" stroke-width="2" marker-end="url(#arrow-accent)" />
      </g>

      <!-- GLOBAL FLOW ARROW (Backing Out) -->
      <g transform="translate(25, 360)">
        <path d="M 0 0 L 0 -310" fill="none" stroke="var(--color-secondary)" stroke-width="2.5" stroke-dasharray="6 6" class="anim-backout" marker-end="url(#arrow-secondary)" />
        <text x="-15" y="-155" transform="rotate(-90, -15, -155)" text-anchor="middle" class="label-tiny text-secondary bold uppercase opacity-50">Backing Out Logic</text>
      </g>

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
.dp-tree-container { width: 100%; padding: 0.25rem 0; background: white; }
.dp-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.label-large { font-weight: 800; }
.label-node-desc { font-weight: 700; }
.label-tiny { }
.label-box-text { font-weight: 800; font-family: var(--font-mono); }

.formula-label { letter-spacing: 0.05em; }
.formula-main { font-family: var(--font-mono); }

.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-danger { fill: var(--color-danger); }
.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }

.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); }
.bg-danger { fill: var(--bg-danger); stroke: var(--color-danger); }

.anim-backout { animation: flowUp 3s infinite linear; }
@keyframes flowUp {
  0% { stroke-dashoffset: 24; opacity: 0; }
  20%, 80% { opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0; }
}
.bold { font-weight: 700; }
.uppercase { text-transform: uppercase; }
</style>
