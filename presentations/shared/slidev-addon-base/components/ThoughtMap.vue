<script setup>
/**
 * ThoughtMap — the branching question tree, driven by Slidev CLICKS so a single
 * map element can PAN across the tree, REVEAL children, RESOLVE questions, and
 * BLINK the next one — all as on-click animations within one slide. Because the
 * element persists across clicks, the camera pan is a smooth CSS transition; only
 * map↔content slide changes use Slidev's zoom transition.
 *
 * Each slide passes `seq` = a named click sequence (defined below). State at
 * click k = sequence[k]: { f: focus node, v: visible ids, r: resolved ids,
 * blink?: pulse the focus }. Set `clicks: <len-1>` in the slide's frontmatter.
 */
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const props = defineProps({ seq: { type: String, default: 'build' } })
const { $clicks } = useSlideContext()

const W = 1500, H = 1200
const nodes = [
  { id: 'q1',     cx: 750,  cy: 70,  w: 380, h: 92, q: 'Do People Meta-Control Their Thinking in Chess?', a: 'Yes — paced by decision width, not value-of-computation.' },
  { id: 'q1a',    cx: 360,  cy: 250, w: 360, h: 92, q: 'What Is Thinking Worth? (The Normative Model)', a: 'A PUCT search with a budgeted stopping rule.' },
  { id: 'q1b',    cx: 1140, cy: 250, w: 360, h: 92, q: 'Do People Behave as the Model Predicts?', a: 'No — so we ask why, and reframe.' },
  { id: 'q2',     cx: 150,  cy: 446, w: 300, h: 96, q: 'How Do We Model Searching?', a: 'An AlphaZero-style PUCT tree (no rollouts).' },
  { id: 'q3',     cx: 470,  cy: 446, w: 300, h: 96, q: 'Which Engine Evaluates the Tree?', a: 'lc0 → Stockfish: uniform prior, WDL value.' },
  { id: 'q4',     cx: 790,  cy: 446, w: 300, h: 96, q: 'When Should the Search Stop?', a: 'A budgeted oracle + a tree-stats readout.' },
  { id: 'qmatch', cx: 1140, cy: 446, w: 360, h: 96, q: "Do the Model's Signals Match Human Think-Time?", a: 'No — the drivers are structural, not VOC.' },
  { id: 'qwhy',   cx: 1140, cy: 636, w: 360, h: 86, q: 'If Not — Why Not?', a: 'Every VOC signal is a legal-moves proxy.' },
  { id: 'h5',     cx: 150,  cy: 840, w: 280, h: 104, q: 'Is It the Wrong Cost Shape?', a: 'No — regret is flat.' },
  { id: 'h6',     cx: 470,  cy: 840, w: 280, h: 104, q: 'Is Uncertainty Missing?', a: 'It recovers, but never beats.' },
  { id: 'h7',     cx: 790,  cy: 840, w: 280, h: 104, q: 'Is It Hindsight Asymmetry?', a: 'No — the causal halter is ≈ 0.' },
  { id: 'h8',     cx: 1110, cy: 840, w: 280, h: 104, q: 'Is It Just a Legal-Moves Proxy?', a: 'Yes — RT is decision difficulty.', star: true },
  { id: 'h9',     cx: 1410, cy: 840, w: 280, h: 104, q: 'Is the Evaluator Too Strong?', a: 'N matters; UCI_Elo is a no-op.' },
  { id: 'finding', cx: 1110, cy: 1000, w: 360, h: 92, q: 'So What Is Think-Time, Really?', a: 'Satisficed decision difficulty: size − satisfaction + sharpness.' },
  { id: 'plan',   cx: 1110, cy: 1130, w: 380, h: 78, q: 'The Plan: A Meta-Rational Reward − Cost Model', a: '', plan: true },
]
const byId = Object.fromEntries(nodes.map(n => [n.id, n]))
const ALL = nodes.map(n => n.id)
const edges = [
  ['q1', 'q1a'], ['q1', 'q1b'], ['q1a', 'q2'], ['q1a', 'q3'], ['q1a', 'q4'],
  ['q1b', 'qmatch'], ['qmatch', 'qwhy'],
  ['qwhy', 'h5'], ['qwhy', 'h6'], ['qwhy', 'h7'], ['qwhy', 'h8'], ['qwhy', 'h9'],
  ['h8', 'finding'], ['finding', 'plan'],
]
const parentOf = {}; edges.forEach(([a, b]) => { parentOf[b] = a })

// ── click sequences ──────────────────────────────────────────────────────────
const V1 = ['q1', 'q1a', 'q1b', 'q2', 'q3', 'q4']
const V2 = [...V1, 'qmatch']
const V3 = [...V2, 'qwhy', 'h5', 'h6', 'h7', 'h8', 'h9']
const V4 = [...V3, 'finding']
const Rm = ['q2', 'q3', 'q4', 'q1a', 'qmatch']          // resolved through the pivot
const Rh = [...Rm, 'h5', 'h6', 'h7', 'h8', 'qwhy', 'h9'] // resolved through all hypotheses
const interlude = (prev, next, opts = {}) => ([
  { f: prev, v: opts.v0 || V1, r: opts.r0 },
  { f: prev, v: opts.v0 || V1, r: opts.r1 },
  { f: next, v: opts.v1 || opts.v0 || V1, r: opts.r1, blink: true },
])
const seqs = {
  build: [
    { f: 'all', v: ALL, r: [] },
    { f: 'q1',  v: ['q1'], r: [] },
    { f: 'q1',  v: ['q1', 'q1a', 'q1b'], r: [] },
    { f: 'q1a', v: ['q1', 'q1a', 'q1b'], r: [] },
    { f: 'q1a', v: V1, r: [] },
    { f: 'q2',  v: V1, r: [], blink: true },
  ],
  'after-q2': interlude('q2', 'q3', { r0: [], r1: ['q2'] }),
  'after-q3': interlude('q3', 'q4', { r0: ['q2'], r1: ['q2', 'q3'] }),
  // resolve q4 (+ q1a subtree), back up, reveal qmatch on the other branch
  'after-q4': [
    { f: 'q4', v: V1, r: ['q2', 'q3'] },
    { f: 'q1a', v: V1, r: ['q2', 'q3', 'q4', 'q1a'] },
    { f: 'qmatch', v: V2, r: ['q2', 'q3', 'q4', 'q1a'], blink: true },
  ],
  // resolve qmatch · select "why not?" · expand it (children emerge) · select the first suspect
  'after-qmatch': [
    { f: 'qmatch', v: V2, r: ['q2', 'q3', 'q4', 'q1a'] },
    { f: 'qmatch', v: V2, r: Rm },
    { f: 'qwhy', v: [...V2, 'qwhy'], r: Rm },
    { f: 'qwhy', v: V3, r: Rm },
    { f: 'h5', v: V3, r: Rm, blink: true },
  ],
  'after-h5': interlude('h5', 'h6', { v0: V3, r0: Rm, r1: [...Rm, 'h5'] }),
  'after-h6': interlude('h6', 'h7', { v0: V3, r0: [...Rm, 'h5'], r1: [...Rm, 'h5', 'h6'] }),
  'after-h7': interlude('h7', 'h8', { v0: V3, r0: [...Rm, 'h5', 'h6'], r1: [...Rm, 'h5', 'h6', 'h7'] }),
  // resolving h8 also resolves "why not?"
  'after-h8': interlude('h8', 'h9', { v0: V3, r0: [...Rm, 'h5', 'h6', 'h7'], r1: [...Rm, 'h5', 'h6', 'h7', 'h8', 'qwhy'] }),
  // resolve h9, reveal the finding
  'after-h9': [
    { f: 'h9', v: V3, r: [...Rm, 'h5', 'h6', 'h7', 'h8', 'qwhy'] },
    { f: 'h9', v: V3, r: Rh },
    { f: 'finding', v: V4, r: Rh, blink: true },
  ],
  // resolve the finding, reveal the plan
  'after-finding': [
    { f: 'finding', v: V4, r: Rh },
    { f: 'finding', v: V4, r: [...Rh, 'finding'] },
    { f: 'plan', v: [...V4, 'plan'], r: [...Rh, 'finding'], blink: true },
  ],
}

const state = computed(() => {
  const s = seqs[props.seq] || seqs.build
  return s[Math.max(0, Math.min($clicks.value, s.length - 1))]
})
const vis = computed(() => new Set(state.value.v))
const resolvedSet = computed(() => new Set(state.value.r))
const focus = computed(() => state.value.f)

// camera frames the focus node + its visible parent/children (local branch)
const cam = computed(() => {
  let ids, fill
  if (state.value.blink && byId[focus.value]) {            // selected → centre tightly on it (we're about to dive in)
    ids = [focus.value]; fill = 0.5
  } else if (focus.value === 'all' || !byId[focus.value]) { // whole tree
    ids = [...vis.value]; fill = 0.82
  } else {                                                  // a branch: focus + visible parent + children
    const set = new Set([focus.value])
    const p = parentOf[focus.value]; if (p && vis.value.has(p)) set.add(p)
    edges.forEach(([a, b]) => { if (a === focus.value && vis.value.has(b)) set.add(b) })
    ids = [...set]; fill = 0.9
  }
  const ns = ids.map(i => byId[i]), pad = 80
  const x0 = Math.min(...ns.map(n => n.cx - n.w / 2)), y0 = Math.min(...ns.map(n => n.cy - n.h / 2))
  const x1 = Math.max(...ns.map(n => n.cx + n.w / 2)), y1 = Math.max(...ns.map(n => n.cy + n.h / 2))
  const s = fill * Math.min(W / (x1 - x0 + pad * 2), H / (y1 - y0 + pad * 2))
  return `translate(${W / 2 - s * (x0 + x1) / 2}px, ${H / 2 - s * (y0 + y1) / 2}px) scale(${s})`
})

const visibleNodes = computed(() => nodes.filter(n => vis.value.has(n.id)))
function nodeState(n) {
  // resolved takes precedence over active, so a node turns green the moment its
  // answer appears (the resolve click) — not on the next click.
  if (resolvedSet.value.has(n.id)) return 'resolved'
  if (n.id === focus.value) return 'active'
  return 'pending'
}
// emerge: a newly revealed child animates out from behind its parent node
function onEnter(el, done) {
  const n = byId[el.getAttribute('data-id')], p = byId[parentOf[el.getAttribute('data-id')]]
  if (!p) { done(); return }
  el.animate(
    [{ transform: `translate(${p.cx - n.cx}px, ${p.cy - n.cy}px) scale(0.25)`, opacity: 0 },
     { transform: 'translate(0px, 0px) scale(1)', opacity: 1 }],
    { duration: 480, easing: 'cubic-bezier(0.34, 1.4, 0.64, 1)' }
  ).onfinish = done
}
function onLeave(el, done) {
  el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, easing: 'ease' }).onfinish = done
}
function showAnswer(n) { return resolvedSet.value.has(n.id) && n.a }
const drawn = computed(() => edges.filter(([a, b]) => vis.value.has(a) && vis.value.has(b)).map(([a, b]) => {
  const s = byId[a], d = byId[b]
  const my = (s.cy + d.cy) / 2
  return { path: `M ${s.cx} ${s.cy + s.h / 2} C ${s.cx} ${my}, ${d.cx} ${my}, ${d.cx} ${d.cy - d.h / 2}`,
           lit: nodeState(s) !== 'pending' && nodeState(d) !== 'pending', key: a + b }
}))
</script>

<template>
  <div class="tm-wrap">
    <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" class="tm-svg">
      <defs>
        <marker id="tm-a" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#cbd5e1" /></marker>
        <marker id="tm-al" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#6366f1" /></marker>
      </defs>
      <g class="tm-cam" :style="{ transform: cam }">
        <g>
          <path v-for="e in drawn" :key="e.key" :d="e.path" fill="none"
                class="tm-edge" :class="{ lit: e.lit }" :marker-end="e.lit ? 'url(#tm-al)' : 'url(#tm-a)'" />
        </g>
        <TransitionGroup tag="g" :css="false" @enter="onEnter" @leave="onLeave">
          <foreignObject v-for="n in visibleNodes" :key="n.id" :data-id="n.id" :x="n.cx - n.w / 2" :y="n.cy - n.h / 2" :width="n.w" :height="n.h">
            <div class="tm-node" :class="[`is-${nodeState(n)}`, { plan: n.plan, star: n.star, blink: state.blink && n.id === focus }]">
              <span v-if="n.star && nodeState(n) !== 'pending'" class="tm-badge">promising</span>
              <div class="tm-q">{{ n.q }}</div>
              <div v-if="showAnswer(n)" class="tm-a">{{ n.a }}</div>
            </div>
          </foreignObject>
        </TransitionGroup>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.tm-wrap { width: 100%; height: 100%; }
.tm-svg { width: 100%; height: 100%; display: block; overflow: hidden; font-family: var(--font-body, 'Inter', sans-serif); }
.tm-cam { transform-origin: 0 0; transition: transform 0.8s cubic-bezier(0.4, 0, 0.2, 1); }

.tm-edge { stroke: #e2e8f0; stroke-width: 2.5; transition: stroke .4s; }
.tm-edge.lit { stroke: #818cf8; stroke-width: 3.5; }

.tm-node {
  box-sizing: border-box; width: 100%; height: 100%; border-radius: 14px; padding: 10px 16px;
  display: flex; flex-direction: column; justify-content: center; border: 2.5px solid; background: #fff;
  position: relative; overflow: hidden; transition: background .5s, border-color .5s;
}
.tm-q { font-size: 19px; font-weight: 700; line-height: 1.18; letter-spacing: -0.02em; transition: color .5s; }
.tm-a { font-size: 15px; line-height: 1.25; margin-top: 6px; font-weight: 600; }

.tm-node.is-pending { border-color: #cbd5e1; }
.tm-node.is-pending .tm-q { color: #64748b; font-weight: 600; }
.tm-node.is-active { border-color: #6366f1; background: #f5f3ff; box-shadow: 0 8px 24px -8px rgba(99,102,241,.4); }
.tm-node.is-active .tm-q { color: #0f172a; }
.tm-node.is-active .tm-a { color: #0f766e; }
.tm-node.is-resolved { border-color: #5eead4; background: #f0fdfa; }
.tm-node.is-resolved .tm-q { color: #475569; font-weight: 600; font-size: 17px; }
.tm-node.is-resolved .tm-a { color: #0f766e; }
.tm-node.star.is-active, .tm-node.star.is-resolved { border-color: #0d9488; background: #f0fdfa; }
.tm-badge { position: absolute; top: 8px; right: 10px; font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #0f766e; background: #ccfbf1; padding: 2px 7px; border-radius: 99px; }
.tm-node.plan.is-active, .tm-node.plan.is-resolved { border-color: #4338ca; background: #4338ca; }
.tm-node.plan.is-active .tm-q, .tm-node.plan.is-resolved .tm-q { color: #fff; }

/* blink the just-selected (next) question — the whole node alternates opacity */
.tm-node.blink { animation: tm-blink 0.95s ease-in-out infinite; }
@keyframes tm-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
