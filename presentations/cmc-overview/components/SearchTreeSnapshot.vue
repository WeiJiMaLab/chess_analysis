<script setup lang="ts">
/**
 * Representation & Architecture: Node Serialization -> GNN -> Readout Head.
 */
</script>

<template>
  <div class="arch-detailed">
    <!-- Part 1: Node Serialization -->
    <div class="section">
      <h3 class="step-title">1. Node Serialization</h3>
      <svg viewBox="0 0 450 80" xmlns="http://www.w3.org/2000/svg" class="flow-svg">
        <defs>
          <marker id="arr-m" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8" />
          </marker>
        </defs>

        <rect x="5" y="20" width="70" height="30" rx="4" class="box hollow-vsmall" />
        <text x="40" y="38" text-anchor="middle" class="txt-tiny bold">Board (s)</text>

        <line x1="75" y1="35" x2="105" y2="35" class="edge" marker-end="url(#arr-m)" />

        <rect x="105" y="15" width="100" height="40" rx="6" class="box filled-slate" />
        <text x="155" y="40" text-anchor="middle" class="txt-tiny white bold">Leela + MLP</text>

        <line x1="205" y1="35" x2="235" y2="35" class="edge" marker-end="url(#arr-m)" />

        <rect x="235" y="20" width="80" height="30" rx="4" class="box hollow-indigo-bold" />
        <text x="275" y="38" text-anchor="middle" class="txt-tiny bold indigo-txt">Vector (x)</text>

        <path d="M315 35 L340 35 L340 55" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="4" marker-end="url(#arr-m)" />
      </svg>
    </div>

    <!-- Part 2: Meta-Controller Wrapper -->
    <div class="section mt-2">
      <h3 class="step-title">2. Under the Hood</h3>
      <svg viewBox="0 0 450 160" xmlns="http://www.w3.org/2000/svg" class="arch-svg">
        <!-- Meta-Controller Body -->
        <rect x="5" y="10" width="440" height="140" rx="10" fill="none" stroke="#6366f1" stroke-width="1.25" stroke-dasharray="6" />
        <text x="435" y="25" text-anchor="end" class="tiny-label uppercase opacity-50">Meta-Controller Module</text>

        <!-- Input Tree -->
        <rect x="20" y="50" width="80" height="40" rx="4" class="box hollow" />
        <text x="60" y="70" text-anchor="middle" class="txt-tiny bold">Input Tree</text>
        <text x="60" y="82" text-anchor="middle" class="txt-vsmall opacity-60">(Nodes = x)</text>

        <line x1="100" y1="70" x2="135" y2="70" class="edge" marker-end="url(#arr-m)" />

        <!-- GNN Backbone -->
        <rect x="135" y="40" width="110" height="60" rx="8" class="box filled-indigo" />
        <text x="190" y="70" text-anchor="middle" class="txt bold white">GNN Backbone</text>
        <text x="190" y="85" text-anchor="middle" class="txt-vsmall white opacity-80">(Tree Processing)</text>

        <line x1="245" y1="70" x2="280" y2="70" class="edge" marker-end="url(#arr-m)" />
        <text x="262" y="62" class="tiny bold indigo-txt">h_root</text>

        <!-- Readout Head -->
        <rect x="280" y="45" width="140" height="50" rx="8" class="box filled-emerald" />
        <text x="350" y="70" text-anchor="middle" class="txt bold white">Readout Head</text>
        <text x="350" y="85" text-anchor="middle" class="txt-vsmall white opacity-80">(Halt Controller)</text>

        <!-- Final Decision -->
        <line x1="350" y1="95" x2="350" y2="120" class="edge" marker-end="url(#arr-m)" />
        <text x="360" y="112" class="tiny bold emerald-txt">Decision</text>

        <rect x="300" y="120" width="100" height="25" rx="4" class="box hollow success-border" />
        <text x="350" y="136" text-anchor="middle" class="txt-tiny bold">Halt / Continue</text>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.arch-detailed {
  width: 100%;
  padding: 0.5rem 1rem;
  background: white;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  font-family: 'Inter', sans-serif;
}
.section { padding: 0.25rem 0; }
.mt-2 { margin-top: 0.5rem; }
.step-title {
  font-size: 13px;
  font-weight: 800;
  margin: 0 0 0.25rem 0;
  color: #1e293b;
}
.box { stroke-width: 1.25; }
.hollow { fill: #fff; stroke: #94a3b8; }
.hollow-vsmall { fill: #fff; stroke: #64748b; stroke-width: 1; }
.hollow-indigo-bold { fill: #f5f3ff; stroke: #6366f1; stroke-width: 1.5; }
.filled-slate { fill: #475569; stroke: none; }
.filled-indigo { fill: #6366f1; stroke: none; }
.filled-emerald { fill: #10b981; stroke: none; }
.success-border { stroke: #10b981; stroke-width: 1.5; }
.white { fill: #fff; }
.bold { font-weight: 700; }
.indigo-txt { fill: #6366f1; }
.emerald-txt { fill: #059669; }
.txt { font-size: 10px; }
.txt-tiny { font-size: 9px; }
.txt-vsmall { font-size: 8px; }
.tiny { font-size: 9px; }
.tiny-label { font-size: 7px; font-weight: 800; }
.edge { stroke: #94a3b8; stroke-width: 1.25; fill: none; }
.opacity-80 { opacity: 0.8; }
.opacity-60 { opacity: 0.6; }
.opacity-50 { opacity: 0.5; }
svg { display: block; width: 100%; height: auto; }
.flow-svg { max-height: 80px; }
.arch-svg { max-height: 160px; }
</style>
