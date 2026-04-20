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
          <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8" />
        </marker>
        <marker id="arr-i" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#6366f1" />
        </marker>
        <marker id="arr-mini-p" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="#94a3b8" />
        </marker>
      </defs>

      <!-- SUBPLOT: Node Encoding Detail (Upper Left - MOVED) -->
      <g transform="translate(10, -15)">
        <rect x="0" y="0" width="310" height="60" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
        <text x="10" y="15" class="title-tiny uppercase bold opacity-60">Node Representation Detail</text>
        
        <!-- Board -->
        <rect x="10" y="25" width="70" height="25" rx="4" class="box hollow" />
        <text x="45" y="41" text-anchor="middle" class="label-tiny bold">Board (s)</text>

        <line x1="80" y1="37.5" x2="110" y2="37.5" class="edge" stroke-width="1" marker-end="url(#arr-mini-p)" />
        <text x="95" y="32" text-anchor="middle" class="tiny-label-mini salient">Leela</text>

        <!-- MLP -->
        <rect x="110" y="25" width="80" height="25" rx="4" class="box filled-slate" />
        <text x="150" y="41" text-anchor="middle" class="label-tiny white bold">MLP Encoder</text>

        <line x1="190" y1="37.5" x2="220" y2="37.5" class="edge" stroke-width="1" marker-end="url(#arr-mini-p)" stroke="#6366f1" />
        
        <!-- Result x -->
        <rect x="220" y="25" width="80" height="25" rx="4" class="box hollow" stroke="#6366f1" />
        <text x="260" y="41" text-anchor="middle" class="label-tiny indigo-txt bold">Vector (x)</text>
      </g>

      <!-- MAIN ARCHITECTURE (Shifted down slightly to clear subplot) -->
      <g transform="translate(0, 70)">
        <!-- 1. External Board State -->
        <g transform="translate(10, 80)">
          <rect x="0" y="0" width="80" height="40" rx="6" class="box hollow" />
          <text x="40" y="25" text-anchor="middle" class="label-bold">Board (s)</text>
          <line x1="80" y1="20" x2="120" y2="20" class="edge" marker-end="url(#arr-m)" />
        </g>

        <!-- 2. Leela (Planner) creating the Tree -->
        <g transform="translate(110, 60)">
          <rect x="0" y="0" width="100" height="80" rx="8" class="box filled-slate" />
          <text x="50" y="30" text-anchor="middle" class="label white bold">Leela</text>
          <text x="50" y="50" text-anchor="middle" class="tiny white opacity-70">(MCTS)</text>
          
          <path d="M100 40 L170 40" class="edge" marker-end="url(#arr-m)" />
          <text x="135" y="32" text-anchor="middle" class="tiny italic salient">Tree (T)</text>
        </g>

        <!-- 3. Meta-Controller Wrapper -->
        <g transform="translate(260, 25)">
          <rect x="0" y="0" width="535" height="150" rx="12" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-dasharray="6" />
          <text x="525" y="15" text-anchor="end" class="title-tiny-alt uppercase salient">Meta-Controller Submodules</text>

          <!-- Input Node Feed -->
          <g transform="translate(20, 35)">
            <rect x="0" y="0" width="100" height="80" rx="8" class="box hollow-indigo-alt" />
            <text x="50" y="35" text-anchor="middle" class="label indigo-txt bold">Serialized</text>
            <text x="50" y="50" text-anchor="middle" class="label indigo-txt bold">Input (x)</text>
            <text x="50" y="65" text-anchor="middle" class="tiny indigo-txt salient">Nodes of T</text>

            <line x1="100" y1="40" x2="140" y2="40" class="edge" marker-end="url(#arr-i)" />
          </g>

          <!-- GNN Backbone -->
          <g transform="translate(160, 30)">
            <rect x="0" y="0" width="120" height="90" rx="10" class="box fill-indigo" />
            <text x="60" y="40" text-anchor="middle" class="label white bold">GNN</text>
            <text x="60" y="60" text-anchor="middle" class="label white bold">Backbone</text>
            <text x="60" y="75" text-anchor="middle" class="tiny white salient">(Bi-Sweep)</text>

            <line x1="120" y1="45" x2="200" y2="45" class="edge" marker-end="url(#arr-i)" />
            <text x="215" y="38" text-anchor="middle" class="tiny-i bold orange-txt">h_root</text>
          </g>

          <!-- Readout Head -->
          <g transform="translate(360, 20)">
            <rect x="0" y="0" width="160" height="110" rx="10" class="box fill-emerald" />
            <text x="80" y="45" text-anchor="middle" class="label white bold">Readout Head</text>
            <text x="80" y="65" text-anchor="middle" class="label white bold">(Halt Controller)</text>
            <text x="80" y="85" text-anchor="middle" class="tiny white salient">Logistic/Policy Hub</text>
          </g>

          <!-- Readout OUT connection -->
          <line x1="520" y1="70" x2="550" y2="70" class="edge" stroke="#10b981" marker-end="url(#arr-m)" />

          <!-- Branch Point Junction -->
          <g transform="translate(555, 55)">
            <rect x="0" y="0" width="30" height="30" rx="15" class="box junction" />
            <text x="15" y="20" text-anchor="middle" class="label bold">0/1</text>
            
            <!-- Arrow to ACT -->
            <line x1="30" y1="15" x2="80" y2="15" class="edge" marker-end="url(#arr-m)" />
            <text x="85" y="19" class="label bold-large dark-txt">ACT</text>
          </g> 
        </g>

        <!-- RECURSION Path: (Spurious junction-arrow removed, path starts directly) -->
        <!-- Global Junction Bottom is (260+555+15, 25+55+30) = (830, 110) -->
        <path d="M830 110 L830 240 L160 240 L160 142" fill="none" class="edge loop-path" marker-end="url(#arr-m)" />
        <text x="492" y="234" text-anchor="middle" class="label bold gray-txt salient">CONTINUE: Expand More Nodes</text>
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
.junction { fill: #fff; stroke: #1e293b; stroke-width: 1.5; }
.hollow { fill: #fff; stroke: #cbd5e1; }
.hollow-indigo-alt { fill: #f5f3ff; stroke: #6366f1; stroke-width: 2; }
.filled-slate { fill: #475569; stroke: none; }
.fill-indigo { fill: #6366f1; stroke: #4f46e5; }
.fill-emerald { fill: #10b981; stroke: #059669; }
.white { fill: #fff; }
.bold { font-weight: 700; }
.bold-large { font-weight: 800; }
.label { font-size: 11px; fill: #1e293b; }
.label-bold { font-size: 11px; font-weight: 700; fill: #1e293b; }
.label-tiny { font-size: 8px; fill: #1e293b; }
.dark-txt { fill: #0f172a; }
.gray-txt { fill: #64748b; }
.tiny { font-size: 9px; }
.tiny-label-mini { font-size: 6px; font-weight: 800; fill: #94a3b8; text-transform: uppercase; }
.title-tiny { font-size: 7px; fill: #6366f1; }
.title-tiny-alt { font-size: 9px; font-weight: 800; fill: #6366f1; }
.indigo-txt { fill: #4f46e5; }
.orange-txt { fill: #f59e0b; }
.tiny-i { font-size: 9px; font-style: italic; }
.edge { stroke: #94a3b8; stroke-width: 1.5; fill: none; }
.loop-path { stroke: #94a3b8; stroke-width: 1.5; }
.salient { opacity: 0.8 !important; }
.opacity-60 { opacity: 0.6; }
.opacity-80 { opacity: 0.8; }
.uppercase { text-transform: uppercase; }
.italic { font-style: italic; }
</style>
