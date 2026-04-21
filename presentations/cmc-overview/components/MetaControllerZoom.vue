<script setup lang="ts">
/**
 * Detailed "Under the Hood" View of the Meta-Controller.
 * Refined geometry: moved subplot to left, removed redundant arrows.
 */
</script>

<template>
  <div class="zoom-container">
    <svg viewBox="0 0 1020 340" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arr-m" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="text-secondary" />
        </marker>
        <marker id="arr-i" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="text-accent" />
        </marker>
        <marker id="arr-mini-p" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" class="text-secondary" />
        </marker>
      </defs>

      <!-- SUBPLOT: Node Encoding Detail (Upper Left) -->
      <g transform="translate(10, -15)">
        <rect x="0" y="0" width="310" height="60" rx="8" class="bg-neutral" stroke-width="1" />
        <text x="10" y="15" class="title-tiny uppercase bold opacity-60">Node Representation Detail</text>
        
        <!-- Board -->
        <rect x="10" y="25" width="70" height="25" rx="4" class="box hollow" />
        <text x="45" y="41" text-anchor="middle" class="label-tiny bold">Board (s)</text>

        <line x1="80" y1="37.5" x2="110" y2="37.5" class="edge" stroke-width="1" marker-end="url(#arr-mini-p)" />
        <text x="95" y="32" text-anchor="middle" class="tiny-label-mini salient">Leela</text>

        <!-- MLP -->
        <rect x="110" y="25" width="80" height="25" rx="4" class="box bg-primary" />
        <text x="150" y="41" text-anchor="middle" class="label-tiny bold">MLP Encoder</text>

        <line x1="190" y1="37.5" x2="220" y2="37.5" class="edge" stroke-width="1" marker-end="url(#arr-mini-p)" stroke="var(--color-accent)" />
        
        <!-- Result x -->
        <rect x="220" y="25" width="80" height="25" rx="4" class="box hollow" stroke="var(--color-accent)" />
        <text x="260" y="41" text-anchor="middle" class="label-tiny text-accent bold">Vector (x)</text>
      </g>

      <!-- MAIN ARCHITECTURE -->
      <g transform="translate(0, 70)">
        <!-- 1. External Board State -->
        <g transform="translate(10, 80)">
          <rect x="0" y="0" width="80" height="40" rx="6" class="box hollow" />
          <text x="40" y="25" text-anchor="middle" class="label-bold">Board (s)</text>
          <line x1="80" y1="20" x2="120" y2="20" class="edge" marker-end="url(#arr-m)" />
        </g>

        <!-- 2. Leela (Planner) creating the Tree -->
        <g transform="translate(110, 60)">
          <rect x="0" y="0" width="100" height="80" rx="8" class="box bg-primary" />
          <text x="50" y="30" text-anchor="middle" class="label bold">Leela</text>
          <text x="50" y="50" text-anchor="middle" class="tiny text-secondary opacity-70">(MCTS)</text>
          
          <path d="M100 40 L170 40" class="edge" marker-end="url(#arr-m)" />
          <text x="135" y="32" text-anchor="middle" class="tiny italic salient">Tree (T)</text>
        </g>

        <!-- 3. Meta-Controller Wrapper -->
        <g transform="translate(260, 25)">
          <rect x="0" y="0" width="535" height="150" rx="12" fill="none" stroke="var(--color-accent)" stroke-width="1.5" stroke-dasharray="6" />
          <text x="525" y="15" text-anchor="end" class="title-tiny-alt uppercase salient">Meta-Controller Submodules</text>

          <!-- Input Node Feed -->
          <g transform="translate(20, 35)">
            <rect x="0" y="0" width="100" height="80" rx="8" class="box hollow-accent" />
            <text x="50" y="35" text-anchor="middle" class="label text-accent bold">Serialized</text>
            <text x="50" y="50" text-anchor="middle" class="label text-accent bold">Input (x)</text>
            <text x="50" y="65" text-anchor="middle" class="tiny text-accent salient">Nodes of T</text>

            <line x1="100" y1="40" x2="140" y2="40" class="edge" marker-end="url(#arr-i)" />
          </g>

          <!-- GNN Backbone -->
          <g transform="translate(160, 30)">
            <rect x="0" y="0" width="120" height="90" rx="10" class="box bg-accent" />
            <text x="60" y="40" text-anchor="middle" class="label text-accent bold">GNN</text>
            <text x="60" y="60" text-anchor="middle" class="label text-accent bold">Backbone</text>
            <text x="60" y="75" text-anchor="middle" class="tiny text-accent salient">(Bi-Sweep)</text>

            <line x1="120" y1="45" x2="200" y2="45" class="edge" marker-end="url(#arr-i)" />
            <text x="215" y="38" text-anchor="middle" class="tiny-i bold text-secondary">h_root</text>
          </g>

          <!-- Readout Head -->
          <g transform="translate(360, 20)">
            <rect x="0" y="0" width="160" height="110" rx="10" class="box bg-success" />
            <text x="80" y="45" text-anchor="middle" class="label text-success bold">Readout Head</text>
            <text x="80" y="65" text-anchor="middle" class="label text-success bold">(Halt Controller)</text>
            <text x="80" y="85" text-anchor="middle" class="tiny text-success salient">Logistic/Policy Hub</text>
          </g>

          <!-- Readout OUT connection -->
          <line x1="520" y1="70" x2="550" y2="70" class="edge" stroke="var(--color-success)" marker-end="url(#arr-m)" />

          <!-- Branch Point Junction -->
          <g transform="translate(555, 55)">
            <rect x="0" y="0" width="30" height="30" rx="15" class="box junction" />
            <text x="15" y="20" text-anchor="middle" class="label bold">0/1</text>
            
            <!-- Arrow to ACT -->
            <line x1="30" y1="15" x2="80" y2="15" class="edge" marker-end="url(#arr-m)" />
            <text x="85" y="19" class="label bold-large">ACT</text>
          </g> 
        </g>

        <!-- RECURSION Path -->
        <path d="M830 110 L830 240 L160 240 L160 142" fill="none" class="edge loop-path" marker-end="url(#arr-m)" />
        <text x="492" y="234" text-anchor="middle" class="label bold text-secondary salient">CONTINUE: Expand More Nodes</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.zoom-container {
  width: 100%;
  padding: 1rem 0;
  background: white;
}
svg {
  width: 100%;
  height: auto;
  font-family: 'Inter', sans-serif;
  overflow: visible;
}
.box { stroke-width: 1.5; }
.junction { fill: #fff; stroke: var(--color-primary); stroke-width: 1.5; }
.hollow { fill: #fff; stroke: var(--border-subtle); }
.hollow-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2; }
.filled-primary { fill: var(--color-primary); stroke: none; }
.fill-accent { fill: var(--color-accent); stroke: var(--color-accent); }
.fill-success { fill: var(--color-success); stroke: var(--color-success); }
.white { fill: #fff; }
.bold { font-weight: 700; }
.bold-large { font-weight: 800; }
.label { font-size: 11px; fill: var(--color-primary); }
.label-bold { font-size: 11px; font-weight: 700; fill: var(--color-primary); }
.label-tiny { font-size: 8px; fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }
.tiny { font-size: 9px; }
.tiny-label-mini { font-size: 6px; font-weight: 800; fill: var(--color-secondary); text-transform: uppercase; }
.title-tiny { font-size: 7px; fill: var(--color-accent); }
.title-tiny-alt { font-size: 9px; font-weight: 800; fill: var(--color-accent); }
.text-accent { fill: var(--color-accent); }
.tiny-i { font-size: 9px; font-style: italic; }
.edge { stroke: var(--color-secondary); stroke-width: 1.5; fill: none; }
.loop-path { stroke: var(--color-secondary); stroke-width: 1.5; }
.salient { opacity: 0.8 !important; }
.opacity-60 { opacity: 0.6; }
.uppercase { text-transform: uppercase; }
.italic { font-style: italic; }
</style>
