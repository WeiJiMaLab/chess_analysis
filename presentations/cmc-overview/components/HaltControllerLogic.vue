<script setup lang="ts">
/**
 * Visualizing the Halt Controller logic:
 * Environment -> Tree -> GNN -> Halt Probability -> Decision.
 */
</script>

<template>
  <div class="logic-container">
    <svg viewBox="0 0 850 340" xmlns="http://www.w3.org/2000/svg" class="logic-svg">
      <defs>
        <marker id="arr-std" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-primary" />
        </marker>
        <marker id="arr-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent" />
        </marker>
        <marker id="arr-success" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-success" />
        </marker>
      </defs>

      <!-- 1. TOP AXIS: EXECUTION (THE AGENT) -->
      <g transform="translate(40, 100)">
        <text x="260" y="-50" text-anchor="middle" class="label-ph uppercase text-secondary">Inference Flow (Mechanistic)</text>
        
        <!-- Input State -->
        <g transform="translate(0, 0)">
          <rect x="0" y="0" width="80" height="80" rx="8" class="bg-neutral" />
          <text x="40" y="35" text-anchor="middle" class="label bold">s_t</text>
          <text x="40" y="55" text-anchor="middle" class="tiny opacity-70">Board</text>
          <line x1="80" y1="40" x2="120" y2="40" class="edge" marker-end="url(#arr-std)" />
        </g>

        <!-- GNN Summary -->
        <g transform="translate(120, 0)">
          <rect x="0" y="0" width="140" height="80" rx="12" class="bg-accent" />
          <text x="70" y="35" text-anchor="middle" class="label-bold text-accent">GNN(T, s)</text>
          <text x="70" y="55" text-anchor="middle" class="tiny text-accent opacity-70">Latent Summary</text>
        </g>

        <!-- Decision Junction -->
        <line x1="260" y1="40" x2="320" y2="40" class="edge" marker-end="url(#arr-std)" />
        
        <circle cx="340" cy="40" r="20" class="box junction" />
        <text x="340" y="46" text-anchor="middle" class="label-bold">0/1</text>

        <!-- BRANCH 1: CONTINUE (0) -->
        <path d="M340 20 L340 -20 L180 -20 L180 0" fill="none" class="edge loop-path" marker-end="url(#arr-accent)" />
        <text x="325" y="10" text-anchor="middle" class="tiny bold text-accent">0</text>
        <text x="260" y="-28" text-anchor="middle" class="tiny-i bold text-accent">Continue Search</text>

        <!-- BRANCH 2: HALT (1) -->
        <line x1="360" y1="40" x2="440" y2="40" class="edge" marker-end="url(#arr-std)" />
        <text x="400" y="32" text-anchor="middle" class="tiny bold text-primary">1</text>
        <rect x="440" y="20" width="80" height="40" rx="6" class="bg-primary" />
        <text x="480" y="45" text-anchor="middle" class="label-bold text-primary uppercase">Act</text>
      </g>

      <!-- 2. BOTTOM AXIS: LEARNING (THE ENVIRONMENT) -->
      <g transform="translate(40, 240)">
        <text x="260" y="75" text-anchor="middle" class="label-ph uppercase text-success">Learning Signal (Policy Gradient)</text>

        <!-- Oracle -->
        <g transform="translate(0, 0)">
          <rect x="0" y="0" width="100" height="50" rx="8" class="box hollow-success" />
          <text x="50" y="22" text-anchor="middle" class="label-tiny bold text-success">DP Oracle</text>
          <text x="50" y="38" text-anchor="middle" class="tiny text-success">(V*)</text>
          
          <line x1="100" y1="25" x2="140" y2="25" class="edge-success" marker-end="url(#arr-success)" />
        </g>

        <!-- Objective Function -->
        <g transform="translate(170, 0)">
          <text x="180" y="15" text-anchor="middle" class="formula bold">R(k) = Value(k) - Cost(k)</text>
          <text x="180" y="40" text-anchor="middle" class="tiny opacity-70">Economy of Thought Objective</text>
        </g>

        <!-- Policy Gradient Feedback Path -->
        <path d="M480 20 L480 -30 L40 -30 L40 -100" fill="none" class="edge feedback-path" marker-end="url(#arr-success)" />
        <text x="260" y="-40" text-anchor="middle" class="label-tiny bold text-success uppercase bg-success-label">Policy Gradient Update</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.logic-container { width: 100%; padding: 1rem; background: white; border-radius: 1rem; }
.logic-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.label-ph { font-size: 13px; font-weight: 800; font-family: var(--font-diagram); letter-spacing: 0.1em; }
.label { font-size: 14px; fill: var(--color-primary); }
.label-bold { font-size: 15px; font-weight: 700; font-family: var(--font-diagram); }
.label-tiny { font-size: 12px; }
.formula { font-size: 16px; font-family: var(--font-mono); fill: var(--color-primary); }
.tiny { font-size: 11px; }
.tiny-i { font-size: 10px; font-style: italic; }

.edge { stroke: var(--color-primary); stroke-width: 2; fill: none; }
.edge-success { stroke: var(--color-success); stroke-width: 2; fill: none; }
.feedback-path { stroke: var(--color-success); stroke-width: 2.5; stroke-dasharray: 6; opacity: 0.6; }
.loop-path { stroke: var(--color-accent); stroke-width: 2; stroke-dasharray: 4; }

.junction { fill: #fff; stroke: var(--color-primary); stroke-width: 2; }
.hollow-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 2; }

.bg-neutral { fill: var(--bg-neutral); stroke: var(--border-subtle); stroke-width: 1.5; }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); stroke-width: 1.5; }
.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 1.5; }
.bg-success-label { fill: white; font-size: 11px; }

.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.text-primary { fill: var(--color-primary); }
.text-secondary { fill: var(--color-secondary); }

.fill-primary { fill: var(--color-primary); }
.fill-accent { fill: var(--color-accent); }
.fill-success { fill: var(--color-success); }
.uppercase { text-transform: uppercase; }
.opacity-70 { opacity: 0.7; }
.bold { font-weight: 700; }
</style>
