<template>
  <div class="child-wdl-diag">
    <div class="grid grid-cols-2 gap-12">
      <!-- PANEL A: SELF PREDICTION -->
      <div class="panel">
        <div class="panel-header">
          Naive: Self-Targeting
          <div class="panel-subtitle text-danger">"What is MY value?"</div>
        </div>
        <svg viewBox="0 0 400 300" class="w-full h-auto">
          <g transform="translate(200, 60)">
            <path d="M0,0 L-60,100 M0,0 L60,100" stroke="var(--border-subtle)" stroke-width="3" />
            <circle cx="0" cy="0" r="22" fill="white" stroke="var(--color-secondary)" stroke-width="4" />
            <circle cx="-60" cy="100" r="14" fill="white" stroke="var(--border-subtle)" stroke-width="2.5" />
            <circle cx="60" cy="100" r="14" fill="white" stroke="var(--border-subtle)" stroke-width="2.5" />

            <!-- Pointing to self -->
            <path d="M0,-25 L0,-10" stroke="var(--color-danger)" stroke-width="4" marker-end="url(#arrow-danger)" class="signal-bounce" transform="translate(0, -30)" />
            
            <g class="cross">
              <line x1="-18" y1="-18" x2="18" y2="18" stroke="var(--color-danger)" stroke-width="5" stroke-linecap="round" />
              <line x1="18" y1="-18" x2="-18" y2="18" stroke="var(--color-danger)" stroke-width="5" stroke-linecap="round" />
            </g>
          </g>
          <defs>
            <marker id="arrow-danger" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--color-danger)" />
            </marker>
          </defs>
        </svg>
      </div>

      <!-- PANEL B: CHILD WDL -->
      <div class="panel highlight">
        <div class="panel-header">
          Alternative: Child-Query
          <div class="panel-subtitle text-accent">"What are THEIR values?"</div>
        </div>
        <svg viewBox="0 0 400 300" class="w-full h-auto">
          <g transform="translate(200, 60)">
            <path d="M0,0 L-60,100 M0,0 L60,100" stroke="var(--color-accent)" stroke-width="3" opacity="0.3" />
            
            <circle cx="0" cy="0" r="22" fill="white" stroke="var(--color-accent)" stroke-width="4" />
            <circle cx="-60" cy="100" r="18" fill="white" stroke="var(--color-accent)" stroke-width="3" />
            <circle cx="60" cy="100" r="18" fill="white" stroke="var(--color-accent)" stroke-width="3" />

            <!-- Calibration Signals (Pointing to children) -->
            <path d="M-15,22 L-45,80" stroke="var(--color-success)" stroke-width="3" marker-end="url(#arrow-success)" class="signal-flow" />
            <path d="M15,22 L45,80" stroke="var(--color-success)" stroke-width="3" marker-end="url(#arrow-success)" class="signal-flow" />
          </g>
          <defs>
            <marker id="arrow-success" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="var(--color-success)" />
            </marker>
          </defs>
        </svg>
      </div>
    </div>
  </div>
</template>

<style scoped>
.child-wdl-diag { width: 100%; font-family: var(--font-body); }

.panel {
  background: var(--bg-neutral);
  padding: 2.5rem;
  border-radius: 1.5rem;
  border: 1px solid var(--border-subtle);
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  transition: all 0.3s ease;
}

.panel.highlight {
  background: white;
  border-color: var(--color-accent);
  border-width: 2px;
  box-shadow: 0 20px 40px -10px rgba(99, 102, 241, 0.08);
}

.panel-header {
  font-weight: 800;
  font-family: var(--font-header);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 1rem;
  text-align: center;
  color: var(--color-secondary);
}

.panel.highlight .panel-header {
  color: var(--color-accent);
}

.panel-subtitle {
  font-weight: 800;
  text-transform: none;
  letter-spacing: -0.01em;
  margin-top: 0.5rem;
}

.text-danger { color: var(--color-danger); fill: var(--color-danger); }
.text-accent { color: var(--color-accent); fill: var(--color-accent); }
.text-success { color: var(--color-success); fill: var(--color-success); }

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

.cross { opacity: 0.6; stroke: var(--color-danger); }
</style>
