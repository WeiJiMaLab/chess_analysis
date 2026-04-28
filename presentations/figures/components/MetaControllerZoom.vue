<script setup lang="ts">
/**
 * Detailed Meta-Controller Architecture Zoom.
 * Layout:
 * Top: Feature Encoding Subplot (850px wide)
 * Bottom: Full Pipeline (Board -> MLP -> GNN -> Readout -> Decision)
 */
</script>

<template>
  <div class="zoom-container">
    <svg viewBox="0 0 1200 500" xmlns="http://www.w3.org/2000/svg" class="zoom-svg">
      <defs>
        <marker id="arr-m" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-primary" />
        </marker>
        <marker id="arr-i" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent" />
        </marker>
        <marker id="arr-accent" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-accent" />
        </marker>
        <marker id="arr-mini-p" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--color-secondary)" />
        </marker>
      </defs>

      <!-- SUBPLOT: Node Representation Detail (Upper Left - Major Focal Point) -->
      <g transform="translate(10, 10)">
        <rect x="0" y="0" width="850" height="120" rx="16" class="bg-neutral" stroke-width="3" />
        <text x="25" y="30" class="subplot-title uppercase text-accent">Feature Encoding Detail (Serialized State)</text>
        
        <!-- Board -->
        <g transform="translate(40, 60)">
          <rect x="0" y="0" width="180" height="50" rx="8" class="box hollow" stroke-width="2.5" />
          <text x="90" y="32" text-anchor="middle" class="subplot-label">Board State (s)</text>
          
          <line x1="180" y1="25" x2="255" y2="25" class="edge" stroke-width="2.5" marker-end="url(#arr-mini-p)" />
          <text x="217" y="8" text-anchor="middle" class="subplot-tiny bold salient">
            <tspan x="217" dy="0">Leela</tspan>
            <tspan x="217" dy="11">Features</tspan>
          </text>
        </g>

        <!-- MLP -->
        <g transform="translate(295, 60)">
          <rect x="0" y="0" width="220" height="50" rx="8" class="bg-primary" />
          <text x="110" y="32" text-anchor="middle" class="subplot-label text-primary">MLP State-Encoder</text>

          <line x1="220" y1="25" x2="305" y2="25" class="edge" stroke-width="2.5" marker-end="url(#arr-mini-p)" stroke="var(--color-accent)" />
          <text x="262" y="16" text-anchor="middle" class="subplot-tiny text-accent bold uppercase">Embedding</text>
        </g>
        
        <!-- Result x -->
        <g transform="translate(600, 60)">
          <rect x="0" y="0" width="220" height="50" rx="8" class="box hollow" stroke="var(--color-accent)" stroke-width="4" />
          <text x="110" y="32" text-anchor="middle" class="subplot-label text-accent bold">Input Vector (xᵢ)</text>
        </g>
      </g>

      <!-- MAIN ARCHITECTURE (Bottom) -->
      <g transform="translate(0, 160)">
        <!-- Vertical Axis Label -->
        <text x="10" y="100" transform="rotate(-90 10,100)" text-anchor="middle" class="label-ph uppercase opacity-50">Global Pipeline</text>

        <g transform="translate(30, 20)">
          <!-- Feature Node x -->
          <g transform="translate(0, 25)">
            <rect x="0" y="0" width="130" height="100" rx="14" class="box hollow" stroke-width="2.5" />
            <text x="65" y="35" text-anchor="middle" class="label-bold text-accent">Serialized</text>
            <text x="65" y="55" text-anchor="middle" class="label-bold text-accent">Input (x)</text>
            <text x="65" y="75" text-anchor="middle" class="tiny text-accent salient">Nodes of T</text>

            <line x1="130" y1="45" x2="210" y2="45" class="edge" marker-end="url(#arr-i)" />
          </g>

          <!-- GNN Backbone -->
          <g transform="translate(210, 40)">
            <rect x="0" y="0" width="160" height="100" rx="14" class="bg-accent" />
            <text x="80" y="40" text-anchor="middle" class="label-large text-accent">GNN</text>
            <text x="80" y="65" text-anchor="middle" class="label-large text-accent">Backbone</text>
            <text x="80" y="85" text-anchor="middle" class="tiny text-accent salient">(Bi-Sweep)</text>

            <!-- Recurrent state loops (stylized) -->



            <line x1="160" y1="50" x2="210" y2="50" class="edge" marker-end="url(#arr-i)" />
          </g>

          <!-- Readout Head -->
          <g transform="translate(420, 30)">
            <rect x="0" y="0" width="180" height="120" rx="14" class="bg-success" />
            <text x="90" y="45" text-anchor="middle" class="label-large text-success">Readout Head</text>
            <text x="90" y="75" text-anchor="middle" class="label-bold text-success">(Halt Controller)</text>
            <text x="90" y="100" text-anchor="middle" class="tiny text-success salient">Logistic/Policy Hub</text>
          </g>

          <!-- Readout OUT connection -->
          <line x1="600" y1="90" x2="640" y2="90" class="edge" stroke="var(--color-success)" marker-end="url(#arr-m)" stroke-width="2" />

          <!-- Branch Point Junction -->
          <g transform="translate(645, 70)">
            <circle cx="20" cy="20" r="20" class="box junction" />
            <text x="20" y="26" text-anchor="middle" class="label-bold">0/1</text>
            
            <!-- Path 1: ACT -->
            <line x1="40" y1="20" x2="110" y2="20" class="edge" marker-end="url(#arr-m)" />
            <text x="115" y="25" class="label-bold text-secondary uppercase">ACT</text>

            <!-- Path 0: CONTINUE (Loop Back) -->
            <path d="M 20 40 L 20 100 L -600 100 L -600 70" fill="none" class="edge-loop" marker-end="url(#arr-m)" />
            <text x="-300" y="95" text-anchor="middle" class="label-tiny bold text-accent">0: CONTINUE SEARCH</text>
          </g>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.zoom-container { width: 100%; padding: 1rem; background: white; border-radius: 2rem; }
.zoom-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.subplot-title { font-weight: 800; letter-spacing: 0.05em; }
.subplot-label { font-weight: 600; fill: var(--color-primary); }
.subplot-tiny { }

.label-large { font-weight: 800; }
.label-bold { font-weight: 700; }
.label-tiny { }
.label-ph { font-weight: 800; letter-spacing: 0.1em; }
.tiny { }

.edge { stroke: var(--color-primary); stroke-width: 2.5; fill: none; }
.edge-loop { stroke: var(--color-accent); stroke-width: 2.5; fill: none; stroke-dasharray: 6; }
.edge-accent { stroke: var(--color-accent); stroke-width: 1.5; fill: none; }

.junction { fill: #fff; stroke: var(--color-primary); stroke-width: 3; }
.box { fill: none; stroke: var(--color-primary); stroke-width: 2; }
.box.hollow { fill: var(--bg-neutral); stroke: var(--border-subtle); }

.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2; }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 2; }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); stroke-width: 2; }
.bg-neutral { fill: var(--bg-neutral); stroke: var(--border-subtle); }

.text-accent { fill: var(--color-accent); }
.text-info { fill: var(--color-info); }
.text-success { fill: var(--color-success); }
.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }

.uppercase { text-transform: uppercase; }
.salient { opacity: 0.8; }
.bold { font-weight: 700; }
</style>
