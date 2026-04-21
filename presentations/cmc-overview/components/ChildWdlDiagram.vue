<template>
  <div class="child-wdl-diag">
    <div class="grid grid-cols-2 gap-12">
      <!-- PANEL A: SELF PREDICTION -->
      <div class="panel">
        <div class="panel-header text-slate-400">
          Naive: Self-Targeting
          <div class="panel-subtitle text-rose-400">"What is MY value?"</div>
        </div>
        <svg viewBox="0 0 400 300" class="w-full h-auto">
          <g transform="translate(200, 60)">
            <path d="M0,0 L-60,100 M0,0 L60,100" stroke="#e2e8f0" stroke-width="3" />
            <circle cx="0" cy="0" r="22" fill="white" stroke="#94a3b8" stroke-width="4" />
            <circle cx="-60" cy="100" r="14" fill="white" stroke="#cbd5e1" stroke-width="2.5" />
            <circle cx="60" cy="100" r="14" fill="white" stroke="#cbd5e1" stroke-width="2.5" />

            <!-- Pointing to self -->
            <path d="M0,-25 L0,-10" stroke="#f43f5e" stroke-width="4" marker-end="url(#arrow-rose)" class="signal-bounce" transform="translate(0, -30)" />
            
            <g class="cross">
              <line x1="-18" y1="-18" x2="18" y2="18" stroke="#f43f5e" stroke-width="5" stroke-linecap="round" />
              <line x1="18" y1="-18" x2="-18" y2="18" stroke="#f43f5e" stroke-width="5" stroke-linecap="round" />
            </g>
          </g>
        </svg>
      </div>

      <!-- PANEL B: CHILD WDL -->
      <div class="panel highlight">
        <div class="panel-header text-indigo-600">
          Alternative: Child-Query
          <div class="panel-subtitle text-indigo-400">"What are THEIR values?"</div>
        </div>
        <svg viewBox="0 0 400 300" class="w-full h-auto">
          <g transform="translate(200, 60)">
            <path d="M0,0 L-60,100 M0,0 L60,100" stroke="#818cf8" stroke-width="3" opacity="0.3" />
            
            <circle cx="0" cy="0" r="22" fill="white" stroke="#6366f1" stroke-width="4" />
            <circle cx="-60" cy="100" r="18" fill="white" stroke="#6366f1" stroke-width="3" />
            <circle cx="60" cy="100" r="18" fill="white" stroke="#6366f1" stroke-width="3" />

            <!-- Calibration Signals (Pointing to children) -->
            <path d="M-15,22 L-45,80" stroke="#f43f5e" stroke-width="3" marker-end="url(#arrow-rose)" class="signal-flow" />
            <path d="M15,22 L45,80" stroke="#f43f5e" stroke-width="3" marker-end="url(#arrow-rose)" class="signal-flow" />
          </g>
          <defs>
            <marker id="arrow-rose" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#f43f5e" />
            </marker>
          </defs>
        </svg>
      </div>
    </div>
  </div>
</template>

<style scoped>
.child-wdl-diag { width: 100%; font-family: 'Inter', sans-serif; }

.panel {
  background: #fdfdfd;
  padding: 2rem;
  border-radius: 1.25rem;
  border: 1px solid #f1f5f9;
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.panel.highlight {
  background: #f5f3ff;
  border-color: #e0e7ff;
  box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.05);
}

.panel-header {
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 1rem;
  text-align: center;
}

.panel-subtitle {
  font-size: 14px;
  font-weight: 600;
  text-transform: none;
  letter-spacing: normal;
  margin-top: 0.25rem;
  font-style: italic;
}

.footer-note { font-size: 11px; margin-top: 1rem; }

.label-loss { font-size: 10px; font-weight: 900; text-transform: uppercase; }
.indigo-text { fill: #4f46e5; }

.signal-bounce { animation: bounce 1.5s infinite; }
.signal-flow { animation: flow-line 2s infinite; }

@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(5px); }
}

@keyframes flow-line {
  0% { stroke-dasharray: 0, 100; opacity: 0; }
  20% { opacity: 1; }
  80% { opacity: 1; }
  100% { stroke-dasharray: 100, 0; opacity: 0; }
}

.cross { opacity: 0.6; }
</style>
