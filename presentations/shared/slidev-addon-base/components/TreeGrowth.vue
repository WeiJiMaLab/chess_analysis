<script setup>
// Three side-by-side tree-search animations growing node-by-node, to contrast how
// the SELECTOR shapes the tree: BFS (level-order, wide) · MCTS/UCB (deepens the
// promising line + explores) · PUCT (prior-focused). Reveal is driven by one step
// counter on a loop; each algorithm has a fixed layout + reveal order so the
// resulting SHAPES differ visibly. Pure SVG; matches the deck's component style.
import { ref, computed, onMounted, onUnmounted } from 'vue'

// node: {id, x, y, p(parent id|null), o(reveal order), kind}
// Coordinates are within a 0..300 x, 0..230 y panel.
function tree(spec) { return spec }

const BFS = tree([
  { id: 0, x: 150, y: 24, p: null, o: 0 },
  { id: 1, x: 70, y: 96, p: 0, o: 1 }, { id: 2, x: 150, y: 96, p: 0, o: 2 }, { id: 3, x: 230, y: 96, p: 0, o: 3 },
  { id: 4, x: 40, y: 168, p: 1, o: 4 }, { id: 5, x: 100, y: 168, p: 1, o: 5 },
  { id: 6, x: 150, y: 168, p: 2, o: 6 }, { id: 7, x: 200, y: 168, p: 3, o: 7 }, { id: 8, x: 260, y: 168, p: 3, o: 8 },
])
// MCTS: descend the best line, expand, "rollout" (dashed leaf), backup. Deepens.
const MCTS = tree([
  { id: 0, x: 150, y: 24, p: null, o: 0 },
  { id: 1, x: 90, y: 84, p: 0, o: 1 }, { id: 2, x: 210, y: 84, p: 0, o: 2 },
  { id: 3, x: 70, y: 144, p: 1, o: 3 }, { id: 4, x: 130, y: 144, p: 1, o: 4 },
  { id: 5, x: 110, y: 200, p: 4, o: 5 }, { id: 6, x: 165, y: 200, p: 4, o: 6 },
  { id: 7, x: 235, y: 144, p: 2, o: 7 },
])
// PUCT: a prior biases exploration toward the policy-preferred branch -> focused.
const PUCT = tree([
  { id: 0, x: 150, y: 24, p: null, o: 0 },
  { id: 1, x: 105, y: 88, p: 0, o: 1, fav: true }, { id: 2, x: 215, y: 88, p: 0, o: 4 },
  { id: 3, x: 80, y: 152, p: 1, o: 2, fav: true }, { id: 4, x: 135, y: 152, p: 1, o: 3 },
  { id: 5, x: 80, y: 210, p: 3, o: 5, fav: true }, { id: 6, x: 130, y: 210, p: 3, o: 6 },
])

const panels = [
  { key: 'BFS', name: 'Breadth-First', sub: 'expand every node, level by level', nodes: BFS, color: 'var(--c-neutral, #7F8C8D)', note: 'uniform & wide — no value guidance' },
  { key: 'MCTS', name: 'MCTS / UCB', sub: 'select by Q + exploration, expand, roll out, back up', nodes: MCTS, color: 'var(--c-accent, #C0392B)', note: 'deepens the promising line' },
  { key: 'PUCT', name: 'PUCT (ours)', sub: 'UCB + a prior on which child to try', nodes: PUCT, color: 'var(--c-primary, #2E86C1)', note: 'prior-focused; uniform prior ⇒ breadth-leaning UCB' },
]

const step = ref(0)
const maxO = Math.max(...panels.flatMap(p => p.nodes.map(n => n.o)))
let timer = null
function tick() { step.value = step.value >= maxO ? 0 : step.value + 1 }
onMounted(() => { timer = setInterval(tick, 850) })
onUnmounted(() => clearInterval(timer))

function shown(p) { return p.nodes.filter(n => n.o <= step.value) }
function edges(p) {
  return shown(p).filter(n => n.p !== null).map(n => {
    const par = p.nodes.find(m => m.id === n.p)
    return { x1: par.x, y1: par.y, x2: n.x, y2: n.y, fav: n.fav }
  })
}
function isFrontier(p, n) { return n.o === step.value }
</script>

<template>
  <div class="tg-wrap">
    <div v-for="p in panels" :key="p.key" class="tg-panel">
      <div class="tg-head"><span class="tg-name" :style="{ color: p.color }">{{ p.name }}</span><span class="tg-sub">{{ p.sub }}</span></div>
      <svg viewBox="0 0 300 230" class="tg-svg">
        <line v-for="(e, i) in edges(p)" :key="'e'+i" :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
              class="tg-edge" :class="{ 'tg-edge--fav': e.fav }" />
        <g v-for="n in shown(p)" :key="n.id">
          <circle :cx="n.x" :cy="n.y" :r="n.p === null ? 13 : 10"
                  :fill="n.p === null ? p.color : '#fff'" :stroke="p.color"
                  :class="{ 'tg-pulse': isFrontier(p, n) }" stroke-width="2.5" />
        </g>
      </svg>
      <div class="tg-note">{{ p.note }}</div>
    </div>
  </div>
</template>

<style scoped>
.tg-wrap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; width: 100%; }
.tg-panel { display: flex; flex-direction: column; align-items: center; }
.tg-head { text-align: center; margin-bottom: .25rem; min-height: 2.6rem; }
.tg-name { display: block; font-weight: 800; font-size: 1.05rem; }
.tg-sub { display: block; font-size: .72rem; opacity: .6; line-height: 1.15; }
.tg-svg { width: 100%; height: auto; }
.tg-edge { stroke: #b0b7bc; stroke-width: 2; }
.tg-edge--fav { stroke: var(--c-primary, #2E86C1); stroke-width: 3.2; }
.tg-note { font-size: .76rem; opacity: .7; margin-top: .35rem; text-align: center; font-style: italic; }
.tg-pulse { animation: tgp 0.85s ease-out; }
@keyframes tgp { 0% { r: 16px; opacity: .35; } 100% { opacity: 1; } }
</style>
