<script setup>
/**
 * ThoughtMap — the branching question tree, driven by Slidev CLICKS.
 *
 * The map is a BeFS of sorts: SELECT a node (blink) → content slide EVALUATES it
 * → return and BACK-PROPAGATE (resolve / green) → EXPAND next subtree → repeat.
 *
 * Data: one nested TREE object. Everything else (edges, parentOf, childrenOf,
 * visible sets, resolved sets, per-slide step arrays) is derived from the tree
 * and a VISITS list that encodes the traversal order.
 *
 * seq prop: 'build' for the opening slide, or a node id (e.g. 'q2') for the
 * map slide that follows that node's content slide. 'after-X' is also accepted
 * for backwards compat.
 */
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const props = defineProps({ seq: { type: String, default: 'build' } })
const { $clicks } = useSlideContext()

const W = 1760, H = 1360

// ── Tree (single source of truth) ─────────────────────────────────────────────
const TREE = {
  id: 'q1', cx: 750, cy: 60, w: 380, h: 92,
  q: 'Do People Meta-Control Their Thinking in Chess?',
  a: 'Yes — paced by decision width, not value-of-computation.',
  children: [
    {
      id: 'q1a', cx: 360, cy: 250, w: 360, h: 92,
      q: 'What Is Thinking Worth? (The Normative Model)',
      a: 'A PUCT search with a budgeted, RL-trained stopping rule.',
      children: [
        {
          id: 'q2', cx: 150, cy: 446, w: 300, h: 96,
          q: 'How Do We Model Planning?', a: 'An AlphaZero-style PUCT tree (no rollouts).',
          children: [
            { id: 'q3', cx: 150, cy: 620, w: 300, h: 92, q: 'Which Engine Do We Use?', a: 'lc0 → Stockfish: uniform prior, WDL value.', children: [] },
          ],
        },
        {
          id: 'q4', cx: 560, cy: 446, w: 300, h: 96,
          q: 'When Should the Search Stop?', a: 'A budgeted oracle; an RL-trained readout on advantage.',
          children: [
            { id: 'h7', cx: 560, cy: 620, w: 300, h: 92, q: "Does step*'s Hindsight Inflate It?", a: 'No — causal halter ≈ hindsight oracle.', children: [] },
          ],
        },
      ],
    },
    {
      id: 'q1b', cx: 1140, cy: 250, w: 360, h: 92,
      q: 'How Do People Spend Their Time Thinking?',
      a: 'On decision width, not value-of-computation.',
      children: [
        {
          id: 'qmatch', cx: 1140, cy: 446, w: 360, h: 96,
          q: "Do the Model's Signals Match Human Think-Time?",
          a: 'No — the drivers are structural, not VOC.',
          children: [
            {
              id: 'qwhy', cx: 1140, cy: 636, w: 360, h: 86,
              q: 'If Not — Why Not?',
              a: 'Every VOC signal is a legal-moves proxy.',
              children: [
                { id: 'h5', cx:  870, cy: 840, w: 280, h: 104, q: 'Is It the Wrong Cost Shape?',  a: 'No — regret is flat.',           children: [] },
                { id: 'h6', cx: 1170, cy: 840, w: 280, h: 104, q: 'Is Uncertainty Missing?',      a: 'It recovers, but never beats.',  children: [] },
                { id: 'h9', cx: 1470, cy: 840, w: 280, h: 104, q: 'Is the Evaluator Too Strong?', a: 'No — only N matters, not Elo.',   children: [] },
                {
                  id: 'h8', cx: 1170, cy: 1000, w: 300, h: 104, star: true,
                  q: 'Is It Just a Legal-Moves Proxy?',
                  a: 'Yes — RT is decision difficulty.',
                  children: [
                    {
                      id: 'finding', cx: 1170, cy: 1150, w: 360, h: 88,
                      q: 'So What Is Think-Time, Really?',
                      a: 'Satisficed difficulty: size − satisfaction + sharpness.',
                      children: [
                        { id: 'plan', cx: 1170, cy: 1275, w: 380, h: 76, plan: true,
                          q: '★ The Umbrella: It Was the Cost of the Leaves', a: '', children: [] },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        },
      ],
    },
  ],
}

// ── Flatten tree → nodes, edges, lookup maps ──────────────────────────────────
const nodes = [], edges = [], parentOf = {}, childrenOf = {}
;(function walk(n, p) {
  nodes.push(n)
  childrenOf[n.id] = n.children.map(c => c.id)
  if (p) { edges.push([p.id, n.id]); parentOf[n.id] = p.id }
  n.children.forEach(c => walk(c, n))
})(TREE, null)
const byId = Object.fromEntries(nodes.map(n => [n.id, n]))
const ALL  = nodes.map(n => n.id)

// ── BeFS visit order ──────────────────────────────────────────────────────────
// Each entry is one content slide. Fields beyond `id` encode domain-specific beats:
//   also_resolve  nodes that resolve at the same time (e.g. parent whose subtree settled)
//   pan_to        where the camera rests on the resolve step (backs up visually)
//   expand        nodes revealed on the final blink step (combined with next selection)
//   expand2       a second expand beat before the blink (gets its own click)
const VISITS = [
  { id: 'q2',     expand: ['q3'] },
  { id: 'q3',     pan_to: 'q2' },
  { id: 'q4',     expand: ['h7'] },
  { id: 'h7',     also_resolve: ['q1a'], pan_to: 'q1a', expand: ['qmatch'] },
  { id: 'qmatch', expand: ['qwhy'], expand2: childrenOf['qwhy'] },
  { id: 'h5' },
  { id: 'h6' },
  { id: 'h9' },
  { id: 'h8',     also_resolve: ['qwhy'], expand: ['finding'] },
  { id: 'finding', expand: ['plan'] },
  { id: 'plan' },
]

// Base visible set — established by the 'build' sequence
const BASE_VIS = new Set(['q1', 'q1a', 'q1b', 'q2', 'q4'])

// Cumulative resolved set *before* visiting VISITS[idx]
function resolvedBefore(idx) {
  const r = new Set()
  for (let i = 0; i < idx; i++) {
    const v = VISITS[i]
    r.add(v.id);
    (v.also_resolve || []).forEach(x => r.add(x))
  }
  return r
}

// Cumulative visible set *before* visiting VISITS[idx]
function visibleBefore(idx) {
  const vis = new Set(BASE_VIS)
  for (let i = 0; i < idx; i++) {
    const v = VISITS[i];
    (v.expand  || []).forEach(x => vis.add(x));
    (v.expand2 || []).forEach(x => vis.add(x))
  }
  return vis
}

function arr(set) { return [...set] }

// Generate the click-step array for the map slide that follows node `id`'s content slide.
//   step 0  land: id is focus (was blinking at end of previous map)
//   step 1  resolve: id (+ also_resolve) turn green; camera pans to pan_to if set
//   step 2  (if expand2) expand first batch
//   step 3  (if expand2) expand second batch, then blink next
//           OR (if expand only) expand + blink next (combined)
//           OR blink next with no expansion
function makeSeq(id) {
  const idx = VISITS.findIndex(v => v.id === id)
  if (idx < 0) return []
  const v    = VISITS[idx]
  const next = VISITS[idx + 1]?.id

  const vis0   = visibleBefore(idx)
  const rBefore = resolvedBefore(idx)
  const rAfter  = new Set([...rBefore, id, ...(v.also_resolve || [])])

  const steps = [
    { f: id,              v: arr(vis0), r: arr(rBefore) },           // land
    { f: v.pan_to || id,  v: arr(vis0), r: arr(rAfter)  },           // resolve (+ optional pan)
  ]

  if (v.expand2 && next) {
    const vis1 = new Set([...vis0, ...v.expand])
    const vis2 = new Set([...vis1, ...v.expand2])
    const pivot = v.expand[0]  // intermediate node; shift focus here so camera sees its children
    steps.push({ f: id,    v: arr(vis1), r: arr(rAfter) })           // expand first batch
    steps.push({ f: pivot, v: arr(vis2), r: arr(rAfter) })           // expand second batch (pivot's children)
    steps.push({ f: next,  v: arr(vis2), r: arr(rAfter), blink: true })
  } else if (v.expand && next) {
    const vis1 = new Set([...vis0, ...v.expand])
    steps.push({ f: next, v: arr(vis1), r: arr(rAfter), blink: true }) // expand + blink (one beat)
  } else if (next) {
    steps.push({ f: next, v: arr(vis0), r: arr(rAfter), blink: true }) // just blink
  }

  return steps
}

// ── Sequences ─────────────────────────────────────────────────────────────────
const seqs = {
  build: [
    { f: 'all', v: ALL,         r: [] },
    { f: 'q1',  v: ['q1'],      r: [] },
    { f: 'q1',  v: ['q1','q1a','q1b'], r: [] },
    { f: 'q1a', v: ['q1','q1a','q1b'], r: [] },
    { f: 'q1a', v: arr(BASE_VIS), r: [] },
    { f: 'q2',  v: arr(BASE_VIS), r: [], blink: true },
  ],
}
VISITS.forEach(v => { seqs[v.id] = makeSeq(v.id) })

// ── Reactive state ─────────────────────────────────────────────────────────────
const state = computed(() => {
  // accept both 'q2' and legacy 'after-q2'
  const key = props.seq.replace(/^after-/, '')
  const s = seqs[key] || seqs.build
  return s[Math.max(0, Math.min($clicks.value, s.length - 1))]
})
const vis         = computed(() => new Set(state.value.v))
const resolvedSet = computed(() => new Set(state.value.r))
const focus       = computed(() => state.value.f)

// ── Camera ─────────────────────────────────────────────────────────────────────
const cam = computed(() => {
  let ids, fill
  if (state.value.blink && byId[focus.value]) {
    ids = [focus.value]; fill = 0.5                                   // tight: about to dive in
  } else if (focus.value === 'all' || !byId[focus.value]) {
    ids = [...vis.value]; fill = 0.82                                 // whole tree
  } else {
    const set = new Set([focus.value])
    const p = parentOf[focus.value]; if (p && vis.value.has(p)) set.add(p)
    edges.forEach(([a, b]) => { if (a === focus.value && vis.value.has(b)) set.add(b) })
    ids = [...set]; fill = 0.9                                        // local branch
  }
  const ns = ids.map(i => byId[i]), pad = 80
  const x0 = Math.min(...ns.map(n => n.cx - n.w / 2)), y0 = Math.min(...ns.map(n => n.cy - n.h / 2))
  const x1 = Math.max(...ns.map(n => n.cx + n.w / 2)), y1 = Math.max(...ns.map(n => n.cy + n.h / 2))
  const s = fill * Math.min(W / (x1 - x0 + pad * 2), H / (y1 - y0 + pad * 2))
  return `translate(${W / 2 - s * (x0 + x1) / 2}px, ${H / 2 - s * (y0 + y1) / 2}px) scale(${s})`
})

// ── Helpers ───────────────────────────────────────────────────────────────────
const visibleNodes = computed(() => nodes.filter(n => vis.value.has(n.id)))

function nodeState(n) {
  if (resolvedSet.value.has(n.id)) return 'resolved'  // resolved > active (turns green immediately)
  if (n.id === focus.value)        return 'active'
  return 'pending'
}

// emerge: newly revealed child animates out from behind its parent
function onEnter(el, done) {
  const n = byId[el.getAttribute('data-id')], p = byId[parentOf[el.getAttribute('data-id')]]
  if (!p) { done(); return }
  el.animate(
    [{ transform: `translate(${p.cx - n.cx}px, ${p.cy - n.cy}px) scale(0.25)`, opacity: 0 },
     { transform: 'translate(0px,0px) scale(1)', opacity: 1 }],
    { duration: 480, easing: 'cubic-bezier(0.34,1.4,0.64,1)' }
  ).onfinish = done
}
function onLeave(el, done) {
  el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, easing: 'ease' }).onfinish = done
}

function showAnswer(n) { return resolvedSet.value.has(n.id) && n.a }

const drawn = computed(() => edges
  .filter(([a, b]) => vis.value.has(a) && vis.value.has(b))
  .map(([a, b]) => {
    const s = byId[a], d = byId[b], my = (s.cy + d.cy) / 2
    return {
      path: `M ${s.cx} ${s.cy + s.h / 2} C ${s.cx} ${my}, ${d.cx} ${my}, ${d.cx} ${d.cy - d.h / 2}`,
      lit: nodeState(s) !== 'pending' && nodeState(d) !== 'pending',
      key: a + b,
    }
  })
)
</script>

<template>
  <div class="tm-wrap">
    <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" class="tm-svg">
      <defs>
        <marker id="tm-a"  markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#cbd5e1" /></marker>
        <marker id="tm-al" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#6366f1" /></marker>
      </defs>
      <g class="tm-cam" :style="{ transform: cam }">
        <g>
          <path v-for="e in drawn" :key="e.key" :d="e.path" fill="none"
                class="tm-edge" :class="{ lit: e.lit }" :marker-end="e.lit ? 'url(#tm-al)' : 'url(#tm-a)'" />
        </g>
        <TransitionGroup tag="g" :css="false" @enter="onEnter" @leave="onLeave">
          <foreignObject v-for="n in visibleNodes" :key="n.id" :data-id="n.id"
                         :x="n.cx - n.w / 2" :y="n.cy - n.h / 2" :width="n.w" :height="n.h">
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
.tm-svg  { width: 100%; height: 100%; display: block; overflow: hidden; font-family: var(--font-body, 'Inter', sans-serif); }
.tm-cam  { transform-origin: 0 0; transition: transform 0.8s cubic-bezier(0.4, 0, 0.2, 1); }

.tm-edge     { stroke: #e2e8f0; stroke-width: 2.5; transition: stroke .4s; }
.tm-edge.lit { stroke: #818cf8; stroke-width: 3.5; }

.tm-node {
  box-sizing: border-box; width: 100%; height: 100%; border-radius: 14px; padding: 10px 16px;
  display: flex; flex-direction: column; justify-content: center; border: 2.5px solid; background: #fff;
  position: relative; overflow: hidden; transition: background .5s, border-color .5s;
}
.tm-q { font-size: 19px; font-weight: 700; line-height: 1.18; letter-spacing: -0.02em; transition: color .5s; }
.tm-a { font-size: 15px; line-height: 1.25; margin-top: 6px; font-weight: 600; }

.tm-node.is-pending  { border-color: #cbd5e1; }
.tm-node.is-pending .tm-q  { color: #64748b; font-weight: 600; }
.tm-node.is-active   { border-color: #6366f1; background: #f5f3ff; box-shadow: 0 8px 24px -8px rgba(99,102,241,.4); }
.tm-node.is-active .tm-q   { color: #0f172a; }
.tm-node.is-active .tm-a   { color: #0f766e; }
.tm-node.is-resolved { border-color: #5eead4; background: #f0fdfa; }
.tm-node.is-resolved .tm-q { color: #475569; font-weight: 600; font-size: 17px; }
.tm-node.is-resolved .tm-a { color: #0f766e; }
.tm-node.star.is-active, .tm-node.star.is-resolved { border-color: #0d9488; background: #f0fdfa; }
.tm-badge { position: absolute; top: 8px; right: 10px; font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #0f766e; background: #ccfbf1; padding: 2px 7px; border-radius: 99px; }
.tm-node.plan.is-active .tm-q, .tm-node.plan.is-resolved .tm-q { color: #fff; }
.tm-node.plan.is-active, .tm-node.plan.is-resolved { border-color: #4338ca; background: #4338ca; }

.tm-node.blink { animation: tm-blink 0.95s ease-in-out infinite; }
@keyframes tm-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
