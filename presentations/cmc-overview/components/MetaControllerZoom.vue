<script setup lang="ts">
/**
 * Detailed "Under the Hood" View of the Meta-Controller.
 * High-fidelity schematic for Node Representation and Architecture.
 */
</script>

<template>
  <div class="zoom-container">
    <svg viewBox="0 0 1050 360" xmlns="http://www.w3.org/2000/svg" class="zoom-svg">
      <defs>
        <marker id="arr-m" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-secondary" />
        </marker>
        <marker id="arr-i" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent" />
        </marker>
        <marker id="arr-mini-p" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-secondary" />
        </marker>
      </defs>

      <!-- SUBPLOT: Node Representation Detail (Upper Left - Major Focal Point) -->
      <g transform="translate(10, 10)">
        <rect x="0" y="0" width="560" height="110" rx="16" class="bg-neutral" stroke-width="2" />
        <text x="15" y="24" class="subplot-title uppercase text-accent">Feature Encoding Detail (Serialized State)</text>
        
        <!-- Board -->
        <g transform="translate(20, 50)">
          <rect x="0" y="0" width="130" height="45" rx="8" class="box hollow" />
          <text x="65" y="28" text-anchor="middle" class="subplot-label">Board State (s)</text>
          
          <line x1="130" y1="22.5" x2="185" y2="22.5" class="edge" stroke-width="2" marker-end="url(#arr-mini-p)" />
          <text x="157" y="14" text-anchor="middle" class="subplot-tiny bold salient">Leela-MCTS</text>
        </g>

        <!-- MLP -->
        <g transform="translate(205, 50)">
          <rect x="0" y="0" width="160" height="45" rx="8" class="bg-primary" />
          <text x="80" y="28" text-anchor="middle" class="subplot-label">MLP State-Encoder</text>

          <line x1="160" y1="22.5" x2="215" y2="22.5" class="edge" stroke-width="2" marker-end="url(#arr-mini-p)" stroke="var(--color-accent)" />
          <text x="187" y="14" text-anchor="middle" class="subplot-tiny text-accent bold uppercase">Embedding</text>
        </g>
        
        <!-- Result x -->
        <g transform="translate(420, 50)">
          <rect x="0" y="0" width="120" height="45" rx="8" class="box hollow" stroke="var(--color-accent)" stroke-width="2.5" />
          <text x="60" y="28" text-anchor="middle" class="subplot-label text-accent bold">Input Vector (x)</text>
        </g>
      </g>

      <!-- MAIN ARCHITECTURE -->
      <g transform="translate(0, 140)">
        <!-- 1. External Board State -->
        <g transform="translate(10, 75)">
          <rect x="0" y="0" width="100" height="50" rx="10" class="box hollow" />
          <text x="50" y="32" text-anchor="middle" class="label-bold">Board (s)</text>
          <line x1="100" y1="25" x2="150" y2="25" class="edge" marker-end="url(#arr-m)" />
        </g>

        <!-- 2. Leela (Planner) creating the Tree -->
        <g transform="translate(150, 55)">
          <rect x="0" y="0" width="130" height="90" rx="12" class="bg-primary" />
          <text x="65" y="40" text-anchor="middle" class="label-large">Leela</text>
          <text x="65" y="65" text-anchor="middle" class="tiny text-secondary">(MCTS Planner)</text>
          
          <path d="M130 45 L190 45" class="edge" marker-end="url(#arr-m)" />
          <text x="160" y="35" text-anchor="middle" class="label italic text-secondary salient">Tree (T)</text>
        </g>

        <!-- 3. Meta-Controller Wrapper -->
        <g transform="translate(320, 10)">
          <rect x="0" y="0" width="620" height="180" rx="20" fill="none" stroke="var(--color-accent)" stroke-width="2.5" stroke-dasharray="8" />
          <text x="605" y="22" text-anchor="end" class="subplot-title uppercase salient">Meta-Controller Submodules</text>

          <!-- Input Node Feed -->
          <g transform="translate(30, 45)">
            <rect x="0" y="0" width="130" height="90" rx="12" class="box hollow-accent" />
            <text x="65" y="35" text-anchor="middle" class="label-bold text-accent">Serialized</text>
            <text x="65" y="55" text-anchor="middle" class="label-bold text-accent">Input (x)</text>
            <text x="65" y="75" text-anchor="middle" class="tiny text-accent salient">Nodes of T</text>

            <line x1="130" y1="45" x2="180" y2="45" class="edge" marker-end="url(#arr-i)" />
          </g>

          <!-- GNN Backbone -->
          <g transform="translate(210, 40)">
            <rect x="0" y="0" width="150" height="100" rx="14" class="bg-accent" />
            <text x="75" y="40" text-anchor="middle" class="label-large text-accent">GNN</text>
            <text x="75" y="65" text-anchor="middle" class="label-large text-accent">Backbone</text>
            <text x="75" y="85" text-anchor="middle" class="tiny text-accent salient">(Bi-Sweep)</text>

            <line x1="150" y1="50" x2="230" y2="50" class="edge" marker-end="url(#arr-i)" />
            <text x="250" y="42" text-anchor="middle" class="label-bold text-secondary">h_root</text>
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
            
            <!-- Arrow to ACT -->
            <line x1="40" y1="20" x2="90" y2="20" class="edge" marker-end="url(#arr-m)" stroke-width="2" />
            <text x="65" y="12" text-anchor="middle" class="tiny bold text-primary">1</text>
            <text x="95" y="26" class="label-large uppercase">ACT</text>
          </g> 
        </g>

        <!-- RECURSION Path -->
        <path d="M1000 100 L1000 260 L215 260 L215 147" fill="none" class="edge loop-path" marker-end="url(#arr-m)" stroke-width="2.5" />
        <text x="607" y="254" text-anchor="middle" class="label-large text-secondary salient uppercase letter-spacing">0: Continue Search (Expand More Nodes)</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.zoom-container { width: 100%; padding: 1.5rem 0; background: white; }
.zoom-svg { width: 100%; height: auto; font-family: var(--font-body); overflow: visible; }
.box { stroke-width: 1.5; }
.junction { fill: #fff; stroke: var(--color-primary); stroke-width: 2; }
.hollow { fill: #fff; stroke: var(--border-subtle); }
.hollow-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2; }

/* Readability Scaling */
.label { font-size: 14px; fill: var(--color-primary); }
.label-bold { font-size: 15px; font-weight: 700; fill: var(--color-primary); }
.label-large { font-size: 18px; font-weight: 800; fill: var(--color-primary); font-family: var(--font-header); }

.subplot-title { font-size: 16px; font-weight: 800; fill: var(--color-accent); letter-spacing: 0.05em; font-family: var(--font-header); }
.subplot-label { font-size: 14px; font-weight: 700; fill: var(--color-primary); }
.subplot-tiny { font-size: 11px; font-weight: 800; fill: var(--color-secondary); text-transform: uppercase; }

.tiny { font-size: 11px; }
.salient { opacity: 0.9 !important; }
.uppercase { text-transform: uppercase; }
.letter-spacing { letter-spacing: 0.05em; }
.italic { font-style: italic; }

.edge { stroke: var(--color-secondary); stroke-width: 1.5; fill: none; }
.loop-path { stroke: var(--color-secondary); stroke-dasharray: 6; }

.fill-secondary { fill: var(--color-secondary); }
.fill-accent { fill: var(--color-accent); }
.bg-neutral { fill: var(--bg-neutral); stroke: var(--border-subtle); }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); }
</style>
