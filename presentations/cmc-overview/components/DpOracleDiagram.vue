<template>
  <div class="dp-tree-container bg-white p-6 rounded-3xl border border-slate-100 shadow-sm">
    <svg viewBox="0 0 900 360" class="w-full h-auto" font-family="Inter, sans-serif">
      <!-- Background / Horizon Area -->
      <rect x="680" y="20" width="200" height="320" fill="#f8fafc" rx="12" />
      <text x="780" y="45" text-anchor="middle" class="txt-horizon uppercase tracking-widest text-xs font-bold text-slate-400">Horizon Limit</text>
      
      <!-- Connection Edges -->
      <g fill="none" stroke-width="3">
        <!-- 63 -> Act -->
        <path d="M 180 180 L 180 100" stroke="#6366f1" marker-end="url(#arrow-indigo)" />
        <!-- 63 -> Continue -->
        <path d="M 180 180 L 480 180" stroke="#10b981" marker-end="url(#arrow-emerald)" />
        
        <!-- 64 -> Act -->
        <path d="M 480 180 L 480 100" stroke="#6366f1" marker-end="url(#arrow-indigo)" />
        <!-- 64 -> Continue (Horizon) -->
        <path d="M 480 180 L 730 180" stroke="#f43f5e" marker-end="url(#arrow-rose)" />
      </g>

      <!-- Edge Labels -->
      <text x="190" y="130" class="txt-edge font-bold fill-indigo-600">Halt (Act)</text>
      <text x="330" y="170" text-anchor="middle" class="txt-edge font-bold fill-emerald-600">Continue</text>
      
      <text x="490" y="130" class="txt-edge font-bold fill-indigo-600">Halt (Act)</text>
      <text x="605" y="170" text-anchor="middle" class="txt-edge font-bold fill-rose-600">Continue</text>

      <!-- Nodes -->
      <g class="nodes">
        <circle cx="180" cy="180" r="24" fill="white" stroke="#334155" stroke-width="4" />
        <text x="180" y="186" text-anchor="middle" class="text-lg font-black fill-slate-800">63</text>
        <text x="180" y="225" text-anchor="middle" class="text-xs font-bold fill-slate-500">Depth k=63</text>

        <circle cx="480" cy="180" r="24" fill="white" stroke="#334155" stroke-width="4" />
        <text x="480" y="186" text-anchor="middle" class="text-lg font-black fill-slate-800">64</text>
        <text x="480" y="225" text-anchor="middle" class="text-xs font-bold fill-slate-500">Depth k=64</text>
      </g>

      <!-- Terminal Value Boxes -->
      <!-- V(act, 63) -->
      <rect x="120" y="50" width="120" height="40" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" />
      <text x="180" y="75" text-anchor="middle" class="text-sm font-black fill-indigo-700">V(act, 63)</text>

      <!-- V(act, 64) -->
      <rect x="420" y="50" width="120" height="40" rx="6" fill="#e0e7ff" stroke="#6366f1" stroke-width="2" />
      <text x="480" y="75" text-anchor="middle" class="text-sm font-black fill-indigo-700">V(act, 64)</text>

      <!-- Horizon Penalty -->
      <rect x="710" y="160" width="140" height="40" rx="6" fill="#fff1f2" stroke="#f43f5e" stroke-width="2" />
      <text x="780" y="185" text-anchor="middle" class="text-sm font-black fill-rose-600">Reward = 0</text>

      <!-- DP Equations Overlay -->
      <!-- Step 1: k=64 -->
      <g transform="translate(480, 260)">
        <rect x="-130" y="0" width="260" height="70" rx="8" fill="white" stroke="#e2e8f0" stroke-width="2" />
        <text x="-115" y="22" class="text-xs font-bold fill-indigo-600">Step 1: Solve Horizon</text>
        <text x="-115" y="42" class="text-[10.5px] font-mono fill-slate-600">V(continue, 64) = 0</text>
        <text x="-115" y="60" class="text-[10.5px] font-mono font-bold fill-slate-900">V(64) = P(act)·V(act, 64) + (1-P)·0</text>
      </g>

      <!-- Step 2: k=63 -->
      <g transform="translate(180, 260)">
        <rect x="-130" y="0" width="260" height="70" rx="8" fill="white" stroke="#e2e8f0" stroke-width="2" />
        <text x="-115" y="22" class="text-xs font-bold fill-emerald-600">Step 2: Back Out</text>
        <text x="-115" y="42" class="text-[10.5px] font-mono fill-slate-600">V(continue, 63) = V(64)</text>
        <text x="-115" y="60" class="text-[10.5px] font-mono font-bold fill-slate-900">V(63) = P(act)·V(act, 63) + (1-P)·V(64)</text>
      </g>

      <!-- DP Backing Out Arrow -->
      <path d="M 340 295 L 320 295" fill="none" stroke="#94a3b8" stroke-width="3" stroke-dasharray="4 4" marker-end="url(#arrow-slate)" class="anim-backout" />

      <defs>
        <marker id="arrow-indigo" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="#6366f1" /></marker>
        <marker id="arrow-emerald" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="#10b981" /></marker>
        <marker id="arrow-rose" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="#f43f5e" /></marker>
        <marker id="arrow-slate" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><polygon points="0 0, 6 2, 0 4" fill="#94a3b8" /></marker>
      </defs>
    </svg>
  </div>
</template>

<style scoped>
.dp-tree-container { width: 100%; }
.txt-edge { font-size: 11px; }

.anim-backout {
  animation: flowBack 3s infinite linear;
}

@keyframes flowBack {
  0% { stroke-dashoffset: 16; opacity: 0; }
  20%, 80% { opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0; }
}
</style>
