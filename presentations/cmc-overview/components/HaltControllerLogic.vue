<script setup lang="ts">
/**
 * Overhauled Halt Controller Logic.
 * A dual-axis "Agent-in-a-Loop" narrative.
 * Top: Execution (h_root -> MLP -> Decision)
 * Bottom: Learning (Oracle -> Reward -> Policy Update)
 */
</script>

<template>
  <div class="halt-logic-container">
    <svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <!-- Standard Arrow -->
        <marker id="arr-std" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-secondary" />
        </marker>
        <!-- Accent Arrow (State) -->
        <marker id="arr-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent" />
        </marker>
        <!-- Success Arrow (Learning) -->
        <marker id="arr-success" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" class="fill-success" />
        </marker>
      </defs>

      <!-- 1. TOP AXIS: EXECUTION (THE AGENT) -->
      <g transform="translate(40, 60)">
        <!-- Input State -->
        <circle cx="20" cy="40" r="18" fill="var(--bg-accent)" stroke="var(--color-accent)" stroke-width="2" />
        <text x="20" y="44" text-anchor="middle" class="label-bold text-accent">h_r</text>
        <text x="20" y="72" text-anchor="middle" class="tiny opacity-60">GNN State</text>

        <line x1="40" y1="40" x2="100" y2="40" class="edge" marker-end="url(#arr-accent)" />

        <!-- Halt MLP Module -->
        <g transform="translate(100, 0)">
          <rect x="0" y="0" width="160" height="80" rx="10" class="box bg-accent" />
          <text x="80" y="35" text-anchor="middle" class="label text-accent bold uppercase letter-spacing">Halt MLP</text>
          <text x="80" y="55" text-anchor="middle" class="tiny text-accent opacity-80">(Policy Head)</text>
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
      <g transform="translate(140, 180)">
        <!-- Environment / Oracle Wrapper -->
        <rect x="0" y="0" width="560" height="120" rx="12" fill="var(--bg-success)" stroke="var(--color-success)" stroke-width="1.5" stroke-dasharray="4" />
        <text x="10" y="20" class="title-tiny text-success bold uppercase">Learning Environment</text>

        <!-- DP Oracle -->
        <g transform="translate(30, 40)">
          <rect x="0" y="0" width="100" height="50" rx="8" class="box hollow-success" />
          <text x="50" y="22" text-anchor="middle" class="label-tiny bold text-success">DP Oracle</text>
          <text x="50" y="38" text-anchor="middle" class="tiny text-success">(V*)</text>
          
          <line x1="100" y1="25" x2="140" y2="25" class="edge-success" marker-end="url(#arr-success)" />
        </g>

        <!-- Objective Function -->
        <g transform="translate(170, 30)">
          <text x="180" y="35" text-anchor="middle" class="formula bold">R(k) = Value(k) - Cost(k)</text>
          <text x="180" y="65" text-anchor="middle" class="tiny opacity-70">Economy of Thought Objective</text>
        </g>

        <!-- Policy Gradient Feedback Path -->
        <path d="M480 30 L480 -30 L40 -30 L40 -100" fill="none" class="edge feedback-path" marker-end="url(#arr-success)" />
        <text x="260" y="-40" text-anchor="middle" class="label-tiny bold text-success uppercase bg-success-label">Policy Gradient Update</text>
      </g>

      <!-- Connection from Result to Environment -->
      <path d="M520 100 L560 100 L560 180" fill="none" class="edge" stroke="var(--color-secondary)" />
    </svg>
  </div>
</template>

<style scoped>
.halt-logic-container {
  width: 100%;
  padding: 1.5rem;
  background: white;
  border-radius: 12px;
  border: 1px solid var(--border-subtle);
}
svg {
  width: 100%;
  height: auto;
  font-family: var(--font-body);
}
.box { stroke-width: 1.5; }
.junction { fill: #fff; stroke: var(--color-accent); stroke-width: 2; }
.hollow-success { fill: var(--bg-success); stroke: var(--color-success); }

.label { font-size: 13px; fill: var(--color-primary); }
.label-bold { font-size: 14px; font-weight: 700; fill: var(--color-primary); }
.label-large { font-size: 16px; font-weight: 800; fill: var(--color-primary); }
.label-tiny { font-size: 11px; }

.formula { font-size: 20px; font-weight: 700; fill: var(--color-primary); font-family: var(--font-header); letter-spacing: -0.01em; }
.tiny { font-size: 10px; }
.tiny-i { font-size: 10px; font-style: italic; }
.text-accent { fill: var(--color-accent); }
.text-success { fill: var(--color-success); }
.title-tiny { font-size: 10px; font-weight: 800; fill: var(--color-success); }

.edge { stroke: var(--color-secondary); stroke-width: 1.5; fill: none; }
.edge-success { stroke: var(--color-success); stroke-width: 1.5; fill: none; }
.loop-path { stroke: var(--color-accent); stroke-dasharray: 4; stroke-width: 1.25; }
.feedback-path { stroke: var(--color-success); stroke-dasharray: 6; stroke-width: 2; }

.bg-success-label { fill: var(--bg-success); }
.opacity-60 { opacity: 0.6; }
.opacity-70 { opacity: 0.7; }
.opacity-80 { opacity: 0.8; }
.uppercase { text-transform: uppercase; }
.letter-spacing { letter-spacing: 0.05em; }
</style>
