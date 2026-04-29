<script setup lang="ts">
/**
 * Single-source CMC diagram:
 * global pipeline + feature encoding + GNN sweeps in one SVG coordinate system.
 */
</script>

<template>
  <div class="zoom-container unified-figure">
    <svg viewBox="0 0 1620 1000" xmlns="http://www.w3.org/2000/svg" class="zoom-svg">
      <defs>
        <marker id="arr-main" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-primary" />
        </marker>
        <marker id="arr-accent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" class="fill-accent" />
        </marker>
        <marker id="arr-up" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-accent-dark" />
        </marker>
        <marker id="arr-down" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-info-dark" />
        </marker>
        <marker id="arr-secondary" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" class="fill-secondary" />
        </marker>
      </defs>

      <!-- Global pipeline (left column) -->
      <g transform="translate(62, 42)">
        <text x="8" y="22" class="t-banner uppercase text-secondary">Global Pipeline</text>

        <g transform="translate(0, 36)">
          <rect x="30" y="0" width="300" height="120" rx="13" class="box hollow" />
          <text x="180" y="44" text-anchor="middle" class="t-node bold text-accent">Serialized</text>
          <text x="180" y="73" text-anchor="middle" class="t-node bold text-accent">Input (x)</text>
          <text x="180" y="102" text-anchor="middle" class="t-micro text-accent">Nodes of T</text>
        </g>
        <line x1="180" y1="162" x2="180" y2="188" class="edge" marker-end="url(#arr-accent)" />

        <g transform="translate(0, 192)">
          <rect x="30" y="0" width="300" height="104" rx="13" class="bg-accent" />
          <text x="180" y="42" text-anchor="middle" class="t-node bold text-accent">GNN</text>
          <text x="180" y="70" text-anchor="middle" class="t-node bold text-accent">Backbone</text>
          <text x="180" y="93" text-anchor="middle" class="t-micro text-accent">(Bi-Sweep)</text>
        </g>
        <line x1="180" y1="300" x2="180" y2="324" class="edge" marker-end="url(#arr-accent)" />

        <g transform="translate(0, 328)">
          <rect x="30" y="0" width="300" height="110" rx="13" class="bg-success" />
          <text x="180" y="44" text-anchor="middle" class="t-node bold text-success">
            <tspan x="180" dy="0">Halt</tspan>
            <tspan x="180" dy="1.06em">Controller</tspan>
          </text>
        </g>
        <line x1="180" y1="442" x2="180" y2="468" class="edge" stroke="var(--color-success)" marker-end="url(#arr-main)" />

        <g transform="translate(159, 472)">
          <circle cx="21" cy="21" r="21" class="junction" />
          <text x="21" y="27" text-anchor="middle" class="t-micro text-primary">0/1</text>
          <line x1="42" y1="21" x2="168" y2="21" class="edge" marker-end="url(#arr-main)" />
          <text x="180" y="27" class="t-node bold text-secondary uppercase">ACT</text>
        </g>

        <path d="M 180 514 L 180 596 L -34 596" fill="none" class="edge-loop" />
        <path d="M -34 596 L -34 102" fill="none" class="edge-loop" />
        <line x1="-34" y1="596" x2="0" y2="596" class="edge-loop" />
        <line x1="-34" y1="102" x2="0" y2="102" class="edge-loop" marker-end="url(#arr-main)" />
        <g transform="translate(-44, 330) rotate(-90)">
          <text x="0" y="0" text-anchor="middle" class="t-node bold text-accent">0: CONTINUE SEARCH</text>
        </g>
      </g>

      <!-- Encoding pipeline strip (wide rects to fit labels; tight vertical spacing) -->
      <g transform="translate(546, 20)">
        <g transform="translate(244, 4)">
          <rect x="0" y="0" width="328" height="34" rx="8" class="box hollow" />
          <text x="164" y="23" text-anchor="middle" class="t-node text-primary">Board State (s)</text>
        </g>
        <line x1="408" y1="38" x2="408" y2="56" class="edge" stroke-width="2" marker-end="url(#arr-secondary)" />
        <text x="442" y="52" class="t-micro text-secondary">Leela Features</text>

        <g transform="translate(244, 58)">
          <rect x="0" y="0" width="328" height="34" rx="8" class="bg-primary" />
          <text x="164" y="23" text-anchor="middle" class="t-node text-primary">MLP State-Encoder</text>
        </g>
        <line x1="408" y1="92" x2="408" y2="110" class="edge" stroke="var(--color-accent)" stroke-width="2" marker-end="url(#arr-secondary)" />
        <text x="442" y="106" class="t-micro text-accent">Embedding</text>

        <g transform="translate(244, 116)">
          <rect x="0" y="0" width="328" height="40" rx="8" class="box hollow" stroke="var(--color-accent)" stroke-width="2.5" />
          <text x="164" y="26" text-anchor="middle" class="t-node bold text-accent">Input Vector (xᵢ)</text>
        </g>
      </g>

      <!-- GNN two-sweep mechanism: taller internals, rects sized to labels -->
      <g transform="translate(544, 190)">
        <line x1="590" y1="38" x2="590" y2="718" stroke="var(--bg-neutral)" stroke-width="3" stroke-dasharray="7" />

        <g transform="translate(-6, 152)">
          <rect x="0" y="0" width="182" height="58" rx="10" class="bg-neutral" />
          <text x="91" y="22" text-anchor="middle" class="t-micro italic text-secondary">Initial h</text>
          <text x="91" y="43" text-anchor="middle" class="t-node text-primary">h(0) = x</text>
        </g>
        <line x1="176" y1="181" x2="266" y2="181" stroke="var(--color-secondary)" stroke-width="2.5" marker-end="url(#arr-secondary)" />

        <g transform="translate(188, 32)">
          <path d="M0 166 L74 166" fill="none" stroke="var(--color-info-dark)" stroke-dasharray="4" marker-end="url(#arr-down)" />
          <text x="46" y="154" text-anchor="middle" class="t-micro text-info-dark bold">h(t)</text>

          <!-- cx=126 spine -->
          <!-- Upward sweep -->
          <g transform="translate(70, 0)">
            <rect x="11" y="142" width="230" height="112" rx="11" class="bg-accent-dark" />
            <text x="126" y="182" text-anchor="middle" class="t-node bold text-accent-dark">
              <tspan x="126" dy="0">GRU</tspan>
              <tspan x="126" dy="1.06em">Upward</tspan>
            </text>

            <line x1="126" y1="142" x2="126" y2="86" class="edge-up" marker-end="url(#arr-up)" />
            <rect x="10" y="48" width="232" height="38" rx="8" class="bg-neutral" />
            <text x="126" y="72" text-anchor="middle" class="t-micro bold text-primary">Parent Node</text>

            <line x1="126" y1="254" x2="126" y2="266" class="edge-up" marker-end="url(#arr-up)" />
            <rect x="4" y="266" width="244" height="52" rx="9" class="bg-accent-dark" />
            <text x="126" y="297" text-anchor="middle" class="t-node bold text-accent-dark">Attention</text>

            <g transform="translate(126, 386)">
              <rect x="-191" y="0" width="96" height="38" rx="8" class="bg-neutral" />
              <text x="-143" y="25" text-anchor="middle" class="t-micro text-primary">Child 1</text>
              <rect x="-46" y="0" width="92" height="38" rx="8" class="bg-neutral" />
              <text x="0" y="25" text-anchor="middle" class="t-micro text-primary">Child 2</text>
              <rect x="95" y="0" width="96" height="38" rx="8" class="bg-neutral" />
              <text x="143" y="25" text-anchor="middle" class="t-micro text-primary">Child 3</text>

              <path d="M-143 0 L-62 -54" class="edge-up" marker-end="url(#arr-up)" />
              <path d="M0 0 L0 -54" class="edge-up" marker-end="url(#arr-up)" />
              <path d="M143 0 L62 -54" class="edge-up" marker-end="url(#arr-up)" />
            </g>

            <path d="M241 198 L382 198 L382 294 L428 294" fill="none" stroke="var(--color-accent-dark)" stroke-width="3" stroke-dasharray="6" marker-end="url(#arr-up)" />
            <text x="334" y="186" text-anchor="middle" class="t-micro bold text-accent-dark">h(t + 1/2)</text>
          </g>

          <!-- Downward sweep -->
          <g transform="translate(482, 0)">
            <rect x="11" y="218" width="230" height="112" rx="11" class="bg-info-dark" />
            <text x="126" y="258" text-anchor="middle" class="t-node bold text-info-dark">
              <tspan x="126" dy="0">GRU</tspan>
              <tspan x="126" dy="1.06em">Downward</tspan>
            </text>

            <rect x="10" y="48" width="232" height="38" rx="8" class="bg-neutral" />
            <text x="126" y="72" text-anchor="middle" class="t-micro bold text-primary">Parent Node</text>

            <line x1="126" y1="86" x2="126" y2="124" class="edge-down" marker-end="url(#arr-down)" />
            <rect x="4" y="124" width="244" height="46" rx="9" class="bg-info-dark" />
            <text x="126" y="152" text-anchor="middle" class="t-node bold text-info-dark">Linear</text>
            <line x1="126" y1="170" x2="126" y2="218" class="edge-down" marker-end="url(#arr-down)" />

            <g transform="translate(126, 386)">
              <rect x="-191" y="0" width="96" height="38" rx="8" class="bg-neutral" />
              <text x="-143" y="25" text-anchor="middle" class="t-micro text-primary">Child 1</text>
              <rect x="-46" y="0" width="92" height="38" rx="8" class="bg-neutral" />
              <text x="0" y="25" text-anchor="middle" class="t-micro text-primary">Child 2</text>
              <rect x="95" y="0" width="96" height="38" rx="8" class="bg-neutral" />
              <text x="143" y="25" text-anchor="middle" class="t-micro text-primary">Child 3</text>

              <path d="M0 -78 L0 -24" class="edge-down" marker-end="url(#arr-down)" />
              <path d="M0 -52 L-143 -52 L-143 -24" fill="none" class="edge-down" marker-end="url(#arr-down)" />
              <path d="M0 -52 L143 -52 L143 -24" fill="none" class="edge-down" marker-end="url(#arr-down)" />
            </g>

            <path d="M241 274 L382 274" stroke="var(--color-info-dark)" stroke-width="3" stroke-dasharray="6" marker-end="url(#arr-down)" />
            <text x="310" y="262" text-anchor="middle" class="t-micro bold text-info-dark">h(t + 1)</text>
          </g>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.zoom-container {
  box-sizing: border-box;
  width: 100%;
  min-height: 0;
  flex: 1 1 auto;
  padding: 0.1rem 0.2rem 0.1rem 0;
  background: transparent;
}

.zoom-svg {
  width: 100%;
  height: auto;
  max-width: 100%;
  max-height: min(88vh, calc(100dvh - 3rem));
  display: block;
  overflow: visible;
  font-family: var(--font-diagram);
}

/* Unified text system across all parts in one coordinate space */
.unified-figure {
  --txt-node: calc(var(--font-size-sub) * 1.36);
  --txt-micro: calc(var(--txt-node) * 0.82);
  --txt-phase: calc(var(--txt-node) * 0.98);
  --txt-banner: calc(var(--txt-node) * 0.9);
}

.t-node { font-size: var(--txt-node) !important; }
.t-micro { font-size: var(--txt-micro) !important; }
.t-phase { font-size: var(--txt-phase) !important; }
.t-banner { font-size: var(--txt-banner) !important; }

.italic { font-style: italic; }
.bold { font-weight: 700; }
.uppercase { text-transform: uppercase; }
.salient { opacity: 0.83; }

.edge { stroke: var(--color-primary); stroke-width: 3; fill: none; }
.edge-loop { stroke: var(--color-accent); stroke-width: 3; fill: none; stroke-dasharray: 7; }
.edge-up { stroke: var(--color-accent-dark); stroke-width: 2.7; fill: none; }
.edge-down { stroke: var(--color-info-dark); stroke-width: 2.7; fill: none; }

.junction { fill: #fff; stroke: var(--color-primary); stroke-width: 3; }
.box { fill: none; stroke: var(--color-primary); stroke-width: 2.3; }
.box.hollow { fill: var(--bg-neutral); stroke: var(--border-subtle); }

.bg-accent { fill: var(--bg-accent); stroke: var(--color-accent); stroke-width: 2.3; }
.bg-accent-dark { fill: var(--bg-accent); stroke: var(--color-accent-dark); stroke-width: 2.3; }
.bg-info-dark { fill: var(--bg-info); stroke: var(--color-info-dark); stroke-width: 2.3; }
.bg-success { fill: var(--bg-success); stroke: var(--color-success); stroke-width: 2.3; }
.bg-primary { fill: var(--bg-primary); stroke: var(--color-primary); stroke-width: 2.3; }
.bg-neutral { fill: var(--bg-neutral); stroke: var(--border-subtle); stroke-width: 1.8; }

.text-accent { fill: var(--color-accent); }
.text-accent-dark { fill: var(--color-accent-dark); }
.text-info-dark { fill: var(--color-info-dark); }
.text-secondary { fill: var(--color-secondary); }
.text-primary { fill: var(--color-primary); }
.text-success { fill: var(--color-success); }

.fill-primary { fill: var(--color-primary); }
.fill-accent { fill: var(--color-accent); }
.fill-accent-dark { fill: var(--color-accent-dark); }
.fill-info-dark { fill: var(--color-info-dark); }
.fill-secondary { fill: var(--color-secondary); }
</style>
