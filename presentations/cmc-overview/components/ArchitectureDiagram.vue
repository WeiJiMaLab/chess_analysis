<script setup>
import { computed } from 'vue'
import rawClassic from '../assets/diagram-01.svg?raw'

const svgHtml = computed(() =>
  rawClassic.replace(/^\s*<\?xml[^>]*>\s*/i, '').replace(/\s+id="Layer_1"/, ''),
)
</script>

<template>
  <div
    class="architecture-diagram-host w-full max-h-[min(78vh,620px)] flex items-center justify-center"
  >
    <div
      class="architecture-diagram-svg-wrap w-full [&_svg]:w-full [&_svg]:h-auto [&_svg]:max-h-[min(78vh,620px)] [&_svg]:object-contain"
      v-html="svgHtml"
    />
  </div>
</template>

<style scoped>
.architecture-diagram-host :deep(svg text:not(.cls-35)),
.architecture-diagram-host :deep(svg text:not(.cls-35) tspan) {
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-weight: 600 !important;
  font-variation-settings: normal !important;
}

.architecture-diagram-host :deep(svg text.cls-35),
.architecture-diagram-host :deep(svg text.cls-35 tspan) {
  font-family: HelveticaNeue, 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
  font-weight: 400 !important;
}

.architecture-diagram-host :deep(svg text.cls-20),
.architecture-diagram-host :deep(svg text.cls-23) {
  text-anchor: middle !important;
  dominant-baseline: central !important;
}
</style>
