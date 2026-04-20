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
          <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8" />
        </marker>
        <!-- Indigo Arrow (State) -->
        <marker id="arr-indigo" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#6366f1" />
        </marker>
        <!-- Orange Arrow (Learning) -->
        <marker id="arr-orange" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 Z" fill="#f59e0b" />
        </marker>
      </defs>

      <!-- 1. TOP AXIS: EXECUTION (THE AGENT) -->
      <g transform="translate(40, 60)">
        <!-- Input State -->
        <circle cx="20" cy="40" r="18" fill="#f5f3ff" stroke="#6366f1" stroke-width="2" />
        <text x="20" y="44" text-anchor="middle" class="label-bold indigo-txt">h_r</text>
        <text x="20" y="72" text-anchor="middle" class="tiny opacity-60">GNN State</text>

        <line x1="40" y1="40" x2="100" y2="40" class="edge" marker-end="url(#arr-indigo)" />

        <!-- Halt MLP Module -->
        <g transform="translate(100, 0)">
          <rect x="0" y="0" width="160" height="80" rx="10" class="box fill-indigo" />
          <text x="80" y="35" text-anchor="middle" class="label white bold uppercase letter-spacing">Halt MLP</text>
          <text x="80" y="55" text-anchor="middle" class="tiny white opacity-80">(Policy Head)</text>
        </g>

        <!-- Decision Junction -->
        <line x1="260" y1="40" x2="320" y2="40" class="edge" marker-end="url(#arr-std)" />
        
        <circle cx="340" cy="40" r="20" class="box junction" />
        <text x="340" y="45" text-anchor="middle" class="label bold">?</text>

        <!-- BRANCH 1: CONTINUE -->
        <path d="M340 20 L340 -20 L180 -20 L180 0" fill="none" class="edge loop-path" marker-end="url(#arr-indigo)" />
        <text x="260" y="-28" text-anchor="middle" class="tiny-i bold indigo-txt">Continue Search</text>

        <!-- BRANCH 2: HALT -->
        <line x1="360" y1="40" x2="440" y2="40" class="edge" marker-end="url(#arr-std)" />
        <rect x="440" y="20" width="80" height="40" rx="6" class="box filled-slate" />
        <text x="480" y="45" text-anchor="middle" class="label white bold">ACT</text>
      </g>

      <!-- 2. BOTTOM AXIS: LEARNING (THE ENVIRONMENT) -->
      <g transform="translate(140, 180)">
        <!-- Environment / Oracle Wrapper -->
        <rect x="0" y="0" width="560" height="120" rx="12" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4" />
        <text x="10" y="20" class="title-tiny orange-txt bold uppercase">Learning Environment</text>

        <!-- DP Oracle -->
        <g transform="translate(30, 40)">
          <rect x="0" y="0" width="100" height="50" rx="8" class="box hollow-orange" />
          <text x="50" y="22" text-anchor="middle" class="label-tiny bold orange-txt">DP Oracle</text>
          <text x="50" y="38" text-anchor="middle" class="tiny orange-txt">(V*)</text>
          
          <line x1="100" y1="25" x2="140" y2="25" class="edge-orange" marker-end="url(#arr-orange)" />
        </g>

        <!-- Objective Function -->
        <g transform="translate(170, 30)">
          <text x="180" y="35" text-anchor="middle" class="formula bold">R(k) = Value(k) - Cost(k)</text>
          <text x="180" y="65" text-anchor="middle" class="tiny opacity-70">Economy of Thought Objective</text>
        </g>

        <!-- Policy Gradient Feedback Path -->
        <path d="M480 30 L480 -30 L40 -30 L40 -100" fill="none" class="edge feedback-path" marker-end="url(#arr-orange)" />
        <text x="260" y="-40" text-anchor="middle" class="label-tiny bold orange-txt uppercase bg-white">Policy Gradient Update</text>
      </g>

      <!-- Connection from Result to Environment -->
      <path d="M520 100 L560 100 L560 180" fill="none" class="edge" stroke="#94a3b8" />
    </svg>
  </div>
</template>

<style scoped>
.halt-logic-container {
  width: 100%;
  padding: 1.5rem;
  background: white;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}
svg {
  width: 100%;
  height: auto;
  font-family: 'Inter', sans-serif;
}
.box { stroke-width: 1.5; }
.junction { fill: #fff; stroke: #6366f1; stroke-width: 2; }
.hollow-orange { fill: #fff4ed; stroke: #f59e0b; }
.filled-slate { fill: #475569; stroke: none; }
.fill-indigo { fill: #6366f1; stroke: #4f46e5; }
.white { fill: #fff; }
.bold { font-weight: 700; }
.label { font-size: 11px; fill: #1e293b; }
.label-bold { font-size: 13px; font-weight: 700; fill: #1e293b; }
.label-tiny { font-size: 10px; }
.formula { font-size: 18px; fill: #1e293b; letter-spacing: 0.05em; }
.tiny { font-size: 9px; }
.tiny-i { font-size: 9px; font-style: italic; }
.indigo-txt { fill: #6366f1; }
.orange-txt { fill: #f59e0b; }
.tag { font-size: 8px; font-weight: 900; fill: #6366f1; }
.title-tiny { font-size: 8px; }
.edge { stroke: #94a3b8; stroke-width: 1.5; fill: none; }
.edge-orange { stroke: #f59e0b; stroke-width: 1.5; fill: none; }
.loop-path { stroke: #6366f1; stroke-dasharray: 4; stroke-width: 1.25; }
.feedback-path { stroke: #f59e0b; stroke-dasharray: 6; stroke-width: 2; }
.bg-white { fill: #fffbeb; }
.opacity-60 { opacity: 0.6; }
.opacity-70 { opacity: 0.7; }
.opacity-80 { opacity: 0.8; }
.uppercase { text-transform: uppercase; }
.letter-spacing { letter-spacing: 0.1em; }
</style>
