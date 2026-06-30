<script setup>
/**
 * TreeGrowth — three tree-search algorithms grown side-by-side, stepped through
 * the SAME rollout cycle SELECT → EXPAND → EVALUATE → BACK-UP so you can see how
 * the *selector* and the *evaluator* differ.  The trees are MONOCHROME (one hue):
 * a node's VALUE is encoded by OPACITY (faint = low value · solid = high value),
 * never by colour.  Faithful per-algorithm semantics (researched):
 *
 *   Best-First (greedy)   SELECT = the single highest-value frontier node;
 *                         EXPAND it; EVALUATE = read the heuristic of the new
 *                         node; BACK-UP.  Dives narrow & deep down the best line
 *                         (exploitation).            — Russell & Norvig / GfG
 *
 *   PUCT (AlphaZero, ours) SELECT = descend from root by
 *                         argmax[ Q(a) + c_puct·P(a)·√(ΣN)/(1+N(a)) ] to a leaf;
 *                         EXPAND; EVALUATE = the leaf's value straight from the
 *                         evaluator (NO rollout); BACK-UP the path.  With uniform
 *                         priors this is just UCB ⇒ spreads visits across root
 *                         moves: broad.              — AlphaZero / lczero.org
 *
 *   MCTS / UCB (UCT)      SELECT = descend by UCB1 [ Q + c·√(ln N_parent /N) ];
 *                         EXPAND; EVALUATE = a Monte-Carlo ROLLOUT — simulate a
 *                         faint chain to a terminal & read the outcome; BACK-UP
 *                         the rollout result up the path.   — UCT / MCTS survey
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'

// One monochrome hue for every tree.  Value is shown by OPACITY only.
const HUE = '#475569'        // slate — reference panels
const HUE_OURS = '#4338ca'   // indigo — "ours" panel (still monochrome within the tree)
function hue(p) { return p.ours ? HUE_OURS : HUE }
// value (0..1) → fill opacity (clamped so even low-value nodes are faintly visible)
function op(v) { return (0.12 + 0.83 * Math.max(0, Math.min(1, v))).toFixed(3) }

/* nodes: {id,x,y,p,v,r}   (r = rollout in which the node first appears)
   rollouts[k] = { sel:[ids root→leaf], add:[new child ids],
                   roll?:[{x,y}], rv?, from? }  (roll = MCTS simulated chain)   */

// Best-First — greedy: each rollout dives one level deeper down the best line.
const BEFS = {
  name: 'Best-First Search', sub: 'greedy · expand the best frontier node', ours: false,
  nodes: [
    { id: 'R', x: 120, y: 22, p: null, v: 0.5, r: 0 },
    { id: 'a', x: 78, y: 86, p: 'R', v: 0.82, r: 1 }, { id: 'b', x: 168, y: 86, p: 'R', v: 0.34, r: 1 },
    { id: 'a1', x: 50, y: 150, p: 'a', v: 0.9, r: 2 }, { id: 'a2', x: 104, y: 150, p: 'a', v: 0.5, r: 2 },
    { id: 'a11', x: 38, y: 214, p: 'a1', v: 0.95, r: 3 }, { id: 'a12', x: 80, y: 214, p: 'a1', v: 0.58, r: 3 },
  ],
  rollouts: [null,
    { sel: ['R'], add: ['a', 'b'] },
    { sel: ['R', 'a'], add: ['a1', 'a2'] },
    { sel: ['R', 'a', 'a1'], add: ['a11', 'a12'] },
  ],
  stage: ['pick the highest-value frontier node', 'expand it', 'read its heuristic value', 'back the value up'],
}

// PUCT — broad: visits fan across all root moves before deepening.
const PUCT = {
  name: 'PUCT', sub: 'Q + prior + exploration · ours', ours: true,
  nodes: [
    { id: 'R', x: 120, y: 22, p: null, v: 0.5, r: 0 },
    { id: 'a', x: 56, y: 90, p: 'R', v: 0.7, r: 1 }, { id: 'b', x: 120, y: 90, p: 'R', v: 0.58, r: 2 }, { id: 'c', x: 184, y: 90, p: 'R', v: 0.44, r: 3 },
    { id: 'a1', x: 38, y: 160, p: 'a', v: 0.64, r: 4 }, { id: 'a2', x: 86, y: 160, p: 'a', v: 0.52, r: 4 },
  ],
  rollouts: [null,
    { sel: ['R'], add: ['a'] },
    { sel: ['R'], add: ['b'] },
    { sel: ['R'], add: ['c'] },
    { sel: ['R', 'a'], add: ['a1', 'a2'] },
  ],
  stage: ['descend by Q + prior + exploration', 'expand the leaf', 'value straight from the evaluator', 'back the value up'],
}

// MCTS / UCB — deepens one line via Monte-Carlo rollouts to a terminal.
const MCTS = {
  name: 'MCTS / UCB', sub: 'descend by UCB · evaluate by rollout', ours: false,
  nodes: [
    { id: 'R', x: 120, y: 22, p: null, v: 0.5, r: 0 },
    { id: 'a', x: 80, y: 86, p: 'R', v: 0.62, r: 1 }, { id: 'b', x: 170, y: 86, p: 'R', v: 0.5, r: 4 },
    { id: 'a1', x: 56, y: 150, p: 'a', v: 0.72, r: 2 }, { id: 'a2', x: 106, y: 150, p: 'a', v: 0.48, r: 2 },
    { id: 'a11', x: 48, y: 214, p: 'a1', v: 0.78, r: 3 },
  ],
  rollouts: [null,
    { sel: ['R', 'a'], add: ['a'], roll: [{ x: 66, y: 130 }, { x: 58, y: 170 }, { x: 62, y: 212 }], rv: 0.68, from: 'a' },
    { sel: ['R', 'a'], add: ['a1', 'a2'], roll: [{ x: 44, y: 192 }, { x: 38, y: 228 }], rv: 0.74, from: 'a1' },
    { sel: ['R', 'a', 'a1'], add: ['a11'], roll: [{ x: 46, y: 240 }], rv: 0.6, from: 'a11' },
    { sel: ['R', 'b'], add: ['b'], roll: [{ x: 158, y: 128 }, { x: 166, y: 168 }, { x: 162, y: 208 }], rv: 0.55, from: 'b' },
  ],
  stage: ['descend by UCB', 'expand a child', 'roll out to a terminal', 'back the rollout value up'],
}

const panels = [BEFS, PUCT, MCTS]
const nRoll = computed(() => Math.max(...panels.map(p => p.rollouts.length - 1)))

const PH = ['select', 'expand', 'eval', 'backup']
const roll = ref(1)
const phase = ref(0)
let timer = null
function tick() {
  phase.value += 1
  if (phase.value >= PH.length) {
    phase.value = 0
    roll.value = roll.value >= nRoll.value ? 1 : roll.value + 1
  }
}
onMounted(() => { timer = setInterval(tick, 1000) })
onUnmounted(() => { clearInterval(timer); timer = null })
const ph = computed(() => PH[phase.value])

// current rollout descriptor for a panel (clamp: shorter searches hold their last)
function R(p) { return p.rollouts[Math.min(roll.value, p.rollouts.length - 1)] }
function rk(p) { return Math.min(roll.value, p.rollouts.length - 1) }
function node(p, id) { return p.nodes.find(n => n.id === id) }

// nodes visible: anything from earlier rollouts, the root, plus this rollout's
// new children once EXPAND has happened.
function shown(p) {
  const k = rk(p)
  return p.nodes.filter(n => n.r === 0 || n.r < k || (n.r === k && phase.value >= 1 && R(p).add.includes(n.id)))
}
// a node's value is "known" (opacity applied) only after EVALUATE for new nodes.
function evaluated(p, n) {
  const k = rk(p)
  if (n.r === 0 || n.r < k) return true
  return n.r === k && phase.value >= 2 && R(p).add.includes(n.id)
}
function fillOpacity(p, n) { return evaluated(p, n) ? op(n.v) : 0 }

// SELECT path = the chain from root to leaf being descended this rollout.
function onSelectedPath(p, id) { return R(p).sel.includes(id) }
// BACK-UP also lights the newly added children hanging off the leaf.
function onBackupNode(p, id) { return R(p).sel.includes(id) || R(p).add.includes(id) }

// edges
function edges(p) {
  return shown(p).filter(n => n.p).map(n => {
    const par = node(p, n.p)
    return { id: n.id, parent: n.p, x1: par.x, y1: par.y, x2: n.x, y2: n.y }
  })
}
// is this edge lit?  blink the SELECT path during select; flow it during backup.
function edgeOnPath(p, e) {
  const sel = R(p).sel
  const inSel = sel.includes(e.parent) && sel.includes(e.id)
  const leaf = sel[sel.length - 1]
  const inAdd = e.parent === leaf && R(p).add.includes(e.id)
  return inSel || inAdd
}

// MCTS rollout chain, only during EVALUATE.
function rollNow(p) { return ph.value === 'eval' && R(p).roll ? R(p) : null }

function label(p) { return ['①', '②', '③', '④'][phase.value] + '  ' + p.stage[phase.value] }
</script>

<template>
  <div class="tg">
    <div class="tg-banner">
      <span v-for="(s, i) in ['SELECT', 'EXPAND', 'EVALUATE', 'BACK-UP']" :key="s"
            class="tg-chip" :class="{ on: i === phase }">{{ s }}</span>
      <span class="tg-roll">rollout {{ roll }} / {{ nRoll }}</span>
      <span class="tg-scale">
        <i class="lo"></i> faint = low value
        <i class="hi"></i> solid = high value
      </span>
    </div>

    <div class="tg-grid">
      <div v-for="p in panels" :key="p.name" class="tg-panel" :class="{ ours: p.ours }">
        <div class="tg-head">
          <span class="tg-name" :class="{ acc: p.ours }">{{ p.name }}</span>
          <span class="tg-sub">{{ p.sub }}</span>
        </div>

        <svg viewBox="0 0 240 252" class="tg-svg" preserveAspectRatio="xMidYMid meet">
          <!-- tree edges -->
          <line v-for="e in edges(p)" :key="'e' + e.id"
                :x1="e.x1" :y1="e.y1" :x2="e.x2" :y2="e.y2"
                class="tg-edge"
                :class="{
                  blink: ph === 'select' && edgeOnPath(p, e),
                  flow: ph === 'backup' && edgeOnPath(p, e),
                }"
                :style="(ph === 'select' || ph === 'backup') && edgeOnPath(p, e) ? { stroke: hue(p) } : {}" />

          <!-- MCTS: faint Monte-Carlo rollout chain down to a terminal -->
          <template v-if="rollNow(p)">
            <polyline
              :points="[node(p, rollNow(p).from), ...rollNow(p).roll].map(pt => `${pt.x},${pt.y}`).join(' ')"
              class="tg-rollline" :style="{ stroke: hue(p) }" />
            <circle v-for="(pt, i) in rollNow(p).roll" :key="'rc' + i"
                    :cx="pt.x" :cy="pt.y" :r="i === rollNow(p).roll.length - 1 ? 5.5 : 3.5"
                    class="tg-rollnode" :style="{ fill: hue(p) }"
                    :fill-opacity="i === rollNow(p).roll.length - 1 ? op(rollNow(p).rv) : 0.22" />
          </template>

          <!-- nodes -->
          <g v-for="n in shown(p)" :key="n.id">
            <!-- blink ring on the SELECT path; steady ring during BACK-UP -->
            <circle v-if="(ph === 'select' && onSelectedPath(p, n.id)) || (ph === 'backup' && onBackupNode(p, n.id))"
                    :cx="n.x" :cy="n.y" :r="(n.p === null ? 9 : 7) + 4"
                    class="tg-ring" :class="{ blink: ph === 'select' }" :style="{ stroke: hue(p) }" />
            <!-- node: monochrome stroke, value carried by fill opacity -->
            <circle :cx="n.x" :cy="n.y" :r="n.p === null ? 9 : 7" stroke-width="2"
                    :stroke="hue(p)" :fill="hue(p)" :fill-opacity="fillOpacity(p, n)"
                    class="tg-node" :class="{ pop: n.r === rk(p) && phase === 1 && R(p).add.includes(n.id) }" />
          </g>
        </svg>

        <div class="tg-stage" :class="{ acc: p.ours }">{{ label(p) }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tg { width: 100%; display: flex; flex-direction: column; font-family: var(--font-body, 'Inter', sans-serif); }

.tg-banner { display: flex; align-items: center; justify-content: center; gap: .45rem; margin-bottom: .4rem; flex-wrap: wrap; }
.tg-chip {
  font-size: .58rem; font-weight: 700; letter-spacing: .07em; padding: .18rem .5rem; border-radius: 99px;
  background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; transition: all .25s; font-family: var(--font-mono, monospace);
}
.tg-chip.on { background: #4338ca; color: #fff; border-color: #4338ca; }
.tg-roll { font-size: .6rem; color: #94a3b8; font-family: var(--font-mono, monospace); margin-left: .3rem; }
.tg-scale { display: inline-flex; align-items: center; gap: .28rem; font-size: .58rem; color: #94a3b8; margin-left: .5rem; }
.tg-scale i { width: .72rem; height: .72rem; border-radius: 50%; display: inline-block; background: #475569; }
.tg-scale .lo { opacity: .18; } .tg-scale .hi { opacity: .95; }

.tg-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .8rem; }
.tg-panel { display: flex; flex-direction: column; align-items: center; padding: .25rem; border-radius: 10px; border: 1.5px solid transparent; }
.tg-panel.ours { border-color: #c7d2fe; background: #f8f7ff; }
.tg-head { text-align: center; margin-bottom: .1rem; min-height: 2.1rem; }
.tg-name { display: block; font-weight: 800; font-size: .9rem; letter-spacing: -.01em; color: #334155; }
.tg-name.acc { color: #4338ca; }
.tg-sub { display: block; font-size: .58rem; opacity: .6; line-height: 1.1; }
.tg-svg { width: 100%; height: 25vh; }

.tg-edge { stroke: #d8dee6; stroke-width: 2; transition: stroke .3s, stroke-width .3s; }
.tg-edge.blink { stroke-width: 3.4; animation: tg-blink .55s ease-in-out infinite; }
.tg-edge.flow { stroke-width: 3.4; stroke-dasharray: 5 4; animation: tg-up .6s linear infinite; }

.tg-node { transition: fill-opacity .45s ease; }
.tg-node.pop { animation: tg-pop .45s ease-out; }

.tg-ring { fill: none; stroke-width: 2.2; opacity: .85; }
.tg-ring.blink { animation: tg-blink .55s ease-in-out infinite; }

.tg-rollline { fill: none; stroke-width: 1.5; stroke-dasharray: 3 3; opacity: .5; animation: tg-up .6s linear infinite; }
.tg-rollnode { animation: tg-fade .3s ease-out; }

@keyframes tg-blink { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
@keyframes tg-up { to { stroke-dashoffset: -12; } }
@keyframes tg-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes tg-pop { 0% { r: 2px; opacity: .3; } 100% { opacity: 1; } }

.tg-stage { font-size: .66rem; margin-top: .2rem; text-align: center; font-weight: 600; color: #64748b; min-height: 1rem; line-height: 1.15; }
.tg-stage.acc { color: #4338ca; }
</style>
