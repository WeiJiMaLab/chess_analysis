<script setup>
import { computed, useId } from 'vue'
import {
  VIEWBOX,
  ZONES,
  MODULES,
  EDGES,
  ANNOTATIONS,
  FLOW_LABELS,
  ZONE_MARKS,
  THEMES,
  anchorPoint,
} from './architectureModel.js'

const props = defineProps({
  /**
   * Code-driven visual / typographic style. Pair with the classic Illustrator
   * slide (`ArchitectureDiagram`) for the frozen original art.
   */
  variant: {
    type: String,
    default: 'minimal',
    validator: (v) => ['minimal', 'blueprint', 'editorial', 'cards', 'swiss'].includes(v),
  },
})

const t = computed(() => THEMES[props.variant])
const headerPad = computed(() => (t.value.header ? 42 : 0))

const uid = useId().replace(/[^a-zA-Z0-9]/g, '')
const markerId = `arch-arw-${uid}`
const filterId = `arch-sh-${uid}`

const modMap = computed(() => Object.fromEntries(MODULES.map((m) => [m.id, m])))

function fillFor(kind) {
  const th = t.value
  if (kind === 'accent') return th.moduleFillAccent
  if (kind === 'teal') return th.moduleFillTeal
  if (kind === 'child') return th.childFill
  return th.moduleFill
}

function strokeFor(kind) {
  const th = t.value
  if (kind === 'teal') return th.moduleStrokeTeal
  if (kind === 'child') return th.childStroke
  return th.moduleStroke
}

function layoutLabels(mod) {
  const cy = mod.y + mod.h / 2
  if (mod.lines.length === 1) {
    return [{ text: mod.lines[0].text, sub: false, y: cy + 4 }]
  }
  return [
    { text: mod.lines[0].text, sub: false, y: cy - 11 },
    { text: mod.lines[1].text, sub: true, y: cy + 15 },
  ]
}

function orthoPath(x1, y1, x2, y2) {
  const dx = x2 - x1
  const dy = y2 - y1
  if (Math.abs(dx) < 1.5 || Math.abs(dy) < 1.5) {
    return `M ${x1} ${y1} L ${x2} ${y2}`
  }
  const mx = (x1 + x2) / 2
  return `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`
}

function edgePathD(edge) {
  const map = modMap.value

  let x1
  let y1
  let x2
  let y2

  if (edge.fromPoint) {
    ;[x1, y1] = edge.fromPoint
  } else if (edge.from?.id) {
    const a = map[edge.from.id]
    if (!a) return ''
    ;[x1, y1] = anchorPoint(a, edge.from.anchor)
  } else {
    return ''
  }

  if (edge.toPoint) {
    ;[x2, y2] = edge.toPoint
  } else if (edge.to?.id) {
    const b = map[edge.to.id]
    if (!b) return ''
    ;[x2, y2] = anchorPoint(b, edge.to.anchor)
  } else {
    return ''
  }

  if (edge.fromPoint || edge.toPoint) {
    return orthoPath(x1, y1, x2, y2)
  }

  const dx = x2 - x1
  const dy = y2 - y1

  if (Math.abs(dx) < 1.5 || Math.abs(dy) < 1.5) {
    return `M ${x1} ${y1} L ${x2} ${y2}`
  }

  if (edge.from.anchor === 'right' || edge.from.anchor === 'left') {
    const midX = x1 + dx * 0.55
    return `M ${x1} ${y1} L ${midX} ${y1} L ${midX} ${y2} L ${x2} ${y2}`
  }

  if (edge.from.anchor === 'bottom') {
    const stub = Math.max(22, Math.min(48, Math.abs(dy) * 0.45))
    let yLeg = y1 + (y2 > y1 ? stub : -stub)
    const tid = edge.to?.id
    if (tid === 'c1' || tid === 'c2' || tid === 'c3') {
      yLeg += (['c1', 'c2', 'c3'].indexOf(tid) - 1) * 42
    } else if (tid === 'r1' || tid === 'r2' || tid === 'r3') {
      yLeg += (['r1', 'r2', 'r3'].indexOf(tid) - 1) * 42
    }
    return `M ${x1} ${y1} L ${x1} ${yLeg} L ${x2} ${yLeg} L ${x2} ${y2}`
  }

  if (edge.from.anchor === 'top') {
    const stub = Math.max(22, Math.min(48, Math.abs(dy) * 0.45))
    const yLeg = y2 < y1 ? y1 - stub : y1 + stub
    return `M ${x1} ${y1} L ${x1} ${yLeg} L ${x2} ${yLeg} L ${x2} ${y2}`
  }

  const mx = (x1 + x2) / 2
  return `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`
}

const resolvedEdges = computed(() =>
  EDGES.map((e, i) => ({
    key: `e-${i}`,
    d: e.customD ?? edgePathD(e),
    dashed: !!e.dashed,
  })).filter((e) => e.d),
)

function rxFor(kind) {
  const r = t.value.rx
  return kind === 'child' ? r.child : r.module
}
</script>

<template>
  <div class="arch-schematic-host w-full max-h-[min(82vh,680px)] flex items-center justify-center">
    <svg
      class="arch-schematic-svg max-h-[min(82vh,680px)] w-full"
      :viewBox="`0 0 ${VIEWBOX.w} ${VIEWBOX.h}`"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <marker
          :id="markerId"
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" :fill="t.edgeStroke" />
        </marker>
        <filter
          v-if="t.dropShadow"
          :id="filterId"
          x="-15%"
          y="-15%"
          width="130%"
          height="130%"
        >
          <feDropShadow
            dx="0"
            dy="2"
            stdDeviation="3"
            flood-color="#0f172a"
            flood-opacity="0.12"
          />
        </filter>
      </defs>

      <rect width="100%" height="100%" :fill="t.canvasBg" />

      <!-- In-figure title block -->
      <g v-if="t.header">
        <text
          :x="VIEWBOX.w / 2"
          y="20"
          text-anchor="middle"
          :fill="t.header.fill"
          :style="{ font: t.header.titleFont }"
        >
          {{ t.header.title }}
        </text>
        <text
          :x="VIEWBOX.w / 2"
          y="36"
          text-anchor="middle"
          :fill="t.header.subfill"
          :style="{ font: t.header.subtitleFont }"
        >
          {{ t.header.subtitle }}
        </text>
      </g>

      <g :transform="`translate(0 ${headerPad})`">
        <!-- Zone panels -->
        <g v-for="z in ZONES" :key="z.id">
          <rect
            :x="z.x"
            :y="z.y"
            :width="z.w"
            :height="z.h"
            :rx="t.rx.zone"
            :ry="t.rx.zone"
            :fill="t.zoneFill"
            :stroke="t.zoneStroke"
            :stroke-width="t.sw.zone"
          />
          <text
            :x="z.x + 14"
            :y="z.y + z.h - 11"
            :fill="t.zoneTitleColor"
            :style="{ font: t.zoneTitleFont }"
          >
            {{ z.label.toUpperCase() }}
          </text>
        </g>

        <!-- Large zone letters -->
        <text
          v-for="(zm, i) in ZONE_MARKS"
          :key="i"
          :x="zm.x"
          :y="zm.y"
          :fill="t.zoneMarkFill"
          :style="{ font: t.zoneMarkFont }"
        >
          {{ zm.text }}
        </text>

        <g>
          <path
            v-for="e in resolvedEdges"
            :key="e.key"
            :d="e.d"
            fill="none"
            :stroke="e.dashed ? t.edgeDashed : t.edgeStroke"
            :stroke-width="t.sw.edge"
            :stroke-dasharray="e.dashed ? '6 5' : undefined"
            :marker-end="`url(#${markerId})`"
          />
        </g>

        <g v-for="m in MODULES" :key="m.id">
          <rect
            :x="m.x"
            :y="m.y"
            :width="m.w"
            :height="m.h"
            :rx="rxFor(m.kind)"
            :ry="rxFor(m.kind)"
            :fill="fillFor(m.kind)"
            :stroke="strokeFor(m.kind)"
            :stroke-width="m.kind === 'child' ? t.sw.child : t.sw.module"
            :filter="t.dropShadow ? `url(#${filterId})` : undefined"
          />
          <text
            v-for="(ln, j) in layoutLabels(m)"
            :key="j"
            :x="m.x + m.w / 2"
            :y="ln.y"
            text-anchor="middle"
            :fill="t.moduleTextFill"
            :style="{ font: ln.sub ? t.subtitleFont : t.titleFont }"
          >
            {{ ln.text }}
          </text>
        </g>

        <text
          v-for="a in ANNOTATIONS"
          :key="a.id"
          :x="a.x"
          :y="a.y"
          :fill="t.annoFill"
          :style="{ font: t.annoFont }"
        >
          {{ a.text }}
        </text>

        <text
          v-for="(f, i) in FLOW_LABELS"
          :key="i"
          :x="f.x"
          :y="f.y"
          :fill="f.role === 'hint' ? t.hintFill : t.flowPillFill"
          :style="{ font: f.role === 'hint' ? t.hintFont : t.flowFont }"
        >
          {{ f.text }}
        </text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.arch-schematic-host {
  color-scheme: light;
}
</style>
