<template>
  <div class="policy-diag">
    <svg viewBox="0 0 880 260" xmlns="http://www.w3.org/2000/svg" class="policy-svg">
      <!-- Axis Labels -->
      <text x="400" y="240" text-anchor="middle" class="axis-label uppercase opacity-40">Search Depth (k)</text>
      <line x1="50" y1="210" x2="750" y2="210" stroke="#e2e8f0" stroke-width="2" />

      <!-- 1. VALUE CURVE (Indigo) -->
      <path d="M 50 200 C 150 50, 350 50, 750 50" fill="none" stroke="var(--color-accent)" stroke-width="3" stroke-linecap="round" />
      <text x="758" y="55" class="legend-label text-accent uppercase">Reward</text>

      <!-- 2. COST LINE (Slate) -->
      <line x1="50" y1="200" x2="750" y2="120" stroke="var(--color-secondary)" stroke-width="2" stroke-dasharray="4" />
      <text x="758" y="125" class="legend-label text-secondary uppercase">Thinking Cost</text>

      <!-- 3. NET REWARD CURVE (Emerald) -->
      <path d="M 50 200 Q 300 80, 750 180" fill="none" stroke="var(--color-success)" stroke-width="4" />
      <text x="758" y="185" class="legend-label text-success bold uppercase">Net Reward $R_k$</text>

      <!-- 4. THE PEAK (Signal) -->
      <g class="peak-marker">
        <line x1="380" y1="40" x2="380" y2="210" stroke="var(--color-primary)" stroke-width="1.5" stroke-dasharray="2" />
        <circle cx="380" cy="134" r="6" fill="var(--color-success)" />
        
        <!-- Optimal Depth Tag -->
        <text x="380" y="30" text-anchor="middle" class="txt-small bold uppercase text-primary">Optimal Depth k*</text>
      </g>

      <!-- 5. TRAINING REGIONS -->
      <g transform="translate(50, 225)">
        <rect x="0" y="0" width="330" height="6" fill="var(--bg-neutral)" rx="3" stroke="var(--border-subtle)" />
        <text x="165" y="18" text-anchor="middle" class="micro-label opacity-70 uppercase">Target: CONTINUE</text>
        
        <rect x="330" y="0" width="370" height="6" fill="var(--color-accent-light)" rx="3" />
        <text x="515" y="18" text-anchor="middle" class="micro-label text-accent-dark bold uppercase">Target: HALT</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.policy-diag { width: 100%; padding: 1rem 0; background: white; }
.policy-svg { width: 100%; height: auto; font-family: var(--font-diagram); overflow: visible; }

.axis-label { font-weight: 700; fill: var(--color-secondary); }
.legend-label { font-weight: 800; }
.micro-label { letter-spacing: 0.05em; }
.txt-small { }
.bold { font-weight: 800; }
.uppercase { text-transform: uppercase; }
.opacity-40 { opacity: 0.4; }
.opacity-70 { opacity: 0.7; }

.text-accent { fill: var(--color-accent); }
.text-secondary { fill: var(--color-secondary); }
.text-success { fill: var(--color-success); }
.text-primary { fill: var(--color-primary); }
.text-accent-dark { fill: var(--color-accent-dark); }
</style>
