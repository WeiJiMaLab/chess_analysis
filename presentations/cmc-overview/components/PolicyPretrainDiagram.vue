<template>
  <div class="policy-diag">
    <svg viewBox="0 0 880 240" xmlns="http://www.w3.org/2000/svg">
      <!-- Axis Labels -->
      <text x="400" y="230" text-anchor="middle" class="axis-label uppercase opacity-40">Search Depth (k)</text>
      <line x1="50" y1="210" x2="750" y2="210" stroke="#e2e8f0" stroke-width="1.5" />

      <!-- 1. VALUE CURVE (Muted Indigo) -->
      <path d="M 50 200 C 150 50, 350 50, 750 50" fill="none" stroke="#6366f1" stroke-width="2.5" stroke-linecap="round" />
      <text x="755" y="55" class="legend-label indigo-txt">Reward</text>

      <!-- 2. COST LINE (Slate) -->
      <line x1="50" y1="200" x2="750" y2="120" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4" />
      <text x="755" y="125" class="legend-label slate-txt">Thinking Cost</text>

      <!-- 3. NET REWARD CURVE (Muted Emerald) -->
      <path d="M 50 200 Q 300 80, 750 180" fill="none" stroke="#10b981" stroke-width="3" />
      <text x="755" y="185" class="legend-label emerald-txt bold">Net Reward (R_k)</text>

      <!-- 4. THE PEAK (Signal) -->
      <g class="peak-marker">
        <line x1="380" y1="40" x2="380" y2="210" stroke="#1e293b" stroke-width="1" stroke-dasharray="2" />
        <circle cx="380" cy="134" r="5" fill="#10b981" />
        
        <!-- Optimal Depth Tag -->
        <text x="380" y="30" text-anchor="middle" class="txt-small bold uppercase slate-txt">Optimal Depth (k*)</text>
      </g>

      <!-- 5. TRAINING REGIONS -->
      <g transform="translate(50, 215)">
        <rect x="0" y="0" width="330" height="4" fill="#f1f5f9" rx="2" />
        <text x="165" y="14" text-anchor="middle" class="micro-label opacity-50 uppercase">Target: CONTINUE</text>
        
        <rect x="330" y="0" width="370" height="4" fill="#6366f1" rx="2" />
        <text x="515" y="14" text-anchor="middle" class="micro-label indigo-txt bold uppercase">Target: HALT</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.policy-diag { width: 100%; font-family: 'Inter', sans-serif; }
.axis-label { font-size: 10px; font-weight: 700; fill: #64748b; }
.legend-label { font-size: 10px; font-weight: 600; }
.indigo-txt { fill: #6366f1; }
.slate-txt { fill: #94a3b8; }
.emerald-txt { fill: #059669; }
.micro-label { font-size: 8px; letter-spacing: 0.05em; }
.txt-small { font-size: 11px; }
.bold { font-weight: 800; }
.uppercase { text-transform: uppercase; }
.opacity-40 { opacity: 0.4; }
.opacity-50 { opacity: 0.5; }
</style>
