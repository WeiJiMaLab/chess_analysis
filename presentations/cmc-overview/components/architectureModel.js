/**
 * Semantic layout — viewBox 0 0 1120 528 (moderate zone B height; pitchfork children; header-safe).
 */

export const VIEWBOX = { w: 1120, h: 528 }

export const ZONES = [
  { id: 'zoneA', x: 18, y: 32, w: 296, h: 388, label: 'Control & halting' },
  { id: 'zoneB', x: 332, y: 32, w: 770, h: 388, label: 'Tree message passing' },
]

/** Tighter boxes: heights matched to 2-line / 1-line labels at scaled fonts */
export const MODULES = [
  {
    id: 'engine',
    x: 36,
    y: 50,
    w: 238,
    h: 52,
    lines: [{ text: 'Chess Engine' }, { text: 'Node expansion', sub: true }],
    kind: 'accent',
  },
  {
    id: 'gnn',
    x: 36,
    y: 110,
    w: 238,
    h: 44,
    lines: [{ text: 'GNN' }, { text: 'Propagation', sub: true }],
    kind: 'accent',
  },
  {
    id: 'halt',
    x: 36,
    y: 162,
    w: 238,
    h: 44,
    lines: [{ text: 'Halt' }, { text: 'Controller', sub: true }],
    kind: 'teal',
  },
  {
    id: 'parentC',
    x: 418,
    y: 76,
    w: 200,
    h: 32,
    lines: [{ text: 'Parent node' }],
    kind: 'neutral',
  },
  {
    id: 'gruUp',
    x: 418,
    y: 138,
    w: 200,
    h: 44,
    lines: [{ text: 'GRU' }, { text: 'Upward', sub: true }],
    kind: 'accent',
  },
  {
    id: 'attention',
    x: 418,
    y: 212,
    w: 200,
    h: 32,
    lines: [{ text: 'Attention' }],
    kind: 'neutral',
  },
  { id: 'c1', x: 386, y: 307, w: 76, h: 32, lines: [{ text: 'Child 1' }], kind: 'child' },
  { id: 'c2', x: 478, y: 307, w: 76, h: 32, lines: [{ text: 'Child 2' }], kind: 'child' },
  { id: 'c3', x: 570, y: 307, w: 76, h: 32, lines: [{ text: 'Child 3' }], kind: 'child' },
  {
    id: 'parentR',
    x: 724,
    y: 76,
    w: 218,
    h: 32,
    lines: [{ text: 'Parent node' }],
    kind: 'neutral',
  },
  {
    id: 'linear',
    x: 724,
    y: 138,
    w: 218,
    h: 30,
    lines: [{ text: 'Linear' }],
    kind: 'neutral',
  },
  {
    id: 'gruDown',
    x: 724,
    y: 200,
    w: 218,
    h: 44,
    lines: [{ text: 'GRU' }, { text: 'Downward', sub: true }],
    kind: 'accent',
  },
  { id: 'r1', x: 692, y: 307, w: 78, h: 32, lines: [{ text: 'Child 1' }], kind: 'child' },
  { id: 'r2', x: 786, y: 307, w: 78, h: 32, lines: [{ text: 'Child 2' }], kind: 'child' },
  { id: 'r3', x: 880, y: 307, w: 78, h: 32, lines: [{ text: 'Child 3' }], kind: 'child' },
]

/** Pitchfork: shared trunk to (516,278), center prong Child 2, outer prongs tertiary */
const PITCH_L = {
  attnCx: 518,
  attnB: 244,
  forkY: 278,
  mid: 516,
  c1: 424,
  c3: 608,
  childT: 307,
}
const PITCH_R = {
  gruCx: 833,
  gruB: 244,
  forkY: 278,
  mid: 825,
  r1: 731,
  r3: 919,
  childT: 307,
}

export const EDGES = [
  { from: { id: 'engine', anchor: 'bottom' }, to: { id: 'gnn', anchor: 'top' } },
  { from: { id: 'gnn', anchor: 'bottom' }, to: { id: 'halt', anchor: 'top' } },
  /** Halting controller → loop back vs enter tree (matches diagram-01.svg) */
  {
    customD: `M 145 206 L 145 244 L 52 244 L 52 108 L 36 108 L 36 76`,
    dashed: true,
  },
  {
    customD: `M 168 206 L 168 230 L 352 230 L 352 92 L 418 92`,
  },
  { from: { id: 'parentC', anchor: 'bottom' }, to: { id: 'gruUp', anchor: 'top' } },
  { from: { id: 'gruUp', anchor: 'bottom' }, to: { id: 'attention', anchor: 'top' } },
  {
    customD: `M ${PITCH_L.attnCx} ${PITCH_L.attnB} L ${PITCH_L.attnCx} 266 L ${PITCH_L.mid} 266 L ${PITCH_L.mid} ${PITCH_L.forkY} L ${PITCH_L.c1} ${PITCH_L.forkY} L ${PITCH_L.c1} ${PITCH_L.childT}`,
  },
  {
    customD: `M ${PITCH_L.attnCx} ${PITCH_L.attnB} L ${PITCH_L.attnCx} 266 L ${PITCH_L.mid} 266 L ${PITCH_L.mid} ${PITCH_L.forkY} L ${PITCH_L.mid} ${PITCH_L.childT}`,
  },
  {
    customD: `M ${PITCH_L.attnCx} ${PITCH_L.attnB} L ${PITCH_L.attnCx} 266 L ${PITCH_L.mid} 266 L ${PITCH_L.mid} ${PITCH_L.forkY} L ${PITCH_L.c3} ${PITCH_L.forkY} L ${PITCH_L.c3} ${PITCH_L.childT}`,
  },
  { from: { id: 'parentR', anchor: 'bottom' }, to: { id: 'linear', anchor: 'top' } },
  { from: { id: 'linear', anchor: 'bottom' }, to: { id: 'gruDown', anchor: 'top' } },
  {
    customD: `M ${PITCH_R.gruCx} ${PITCH_R.gruB} L ${PITCH_R.gruCx} 266 L ${PITCH_R.mid} 266 L ${PITCH_R.mid} ${PITCH_R.forkY} L ${PITCH_R.r1} ${PITCH_R.forkY} L ${PITCH_R.r1} ${PITCH_R.childT}`,
  },
  {
    customD: `M ${PITCH_R.gruCx} ${PITCH_R.gruB} L ${PITCH_R.gruCx} 266 L ${PITCH_R.mid} 266 L ${PITCH_R.mid} ${PITCH_R.forkY} L ${PITCH_R.mid} ${PITCH_R.childT}`,
  },
  {
    customD: `M ${PITCH_R.gruCx} ${PITCH_R.gruB} L ${PITCH_R.gruCx} 266 L ${PITCH_R.mid} 266 L ${PITCH_R.mid} ${PITCH_R.forkY} L ${PITCH_R.r3} ${PITCH_R.forkY} L ${PITCH_R.r3} ${PITCH_R.childT}`,
  },
  {
    from: { id: 'gruUp', anchor: 'right' },
    to: { id: 'parentR', anchor: 'left' },
    dashed: true,
  },
  /** State labels drawn as annotations; edges terminate beside glyphs */
  { fromPoint: [376, 160], to: { id: 'gruUp', anchor: 'left' } },
  { from: { id: 'gruDown', anchor: 'right' }, toPoint: [968, 222] },
]

export const ANNOTATIONS = [
  { id: 'h1', x: 328, y: 160, text: 'h(t)', role: 'state' },
  { id: 'h2', x: 648, y: 160, text: 'h(t+\u00bd)', role: 'state' },
  { id: 'h3', x: 978, y: 222, text: 'h(t+1)', role: 'state' },
]

export const FLOW_LABELS = [
  { x: 42, y: 228, text: 'CONTINUE', role: 'pill' },
  { x: 188, y: 228, text: 'ACT', role: 'pill' },
  { x: 418, y: 44, text: 'For each node …', role: 'hint' },
]

export const ZONE_MARKS = [
  { x: 28, y: 46, text: 'A' },
  { x: 340, y: 46, text: 'B' },
]

const BIG = {
  zoneTitle: "600 13px 'Inter', ui-sans-serif, system-ui, sans-serif",
  zoneMark: "700 48px 'Inter', ui-sans-serif, system-ui, sans-serif",
  title: "600 16px 'JetBrains Mono', ui-monospace, monospace",
  subtitle: "500 13px 'JetBrains Mono', ui-monospace, monospace",
  anno: "700 15px 'JetBrains Mono', ui-monospace, monospace",
  flow: "600 14px 'JetBrains Mono', ui-monospace, monospace",
  hint: "500 13px 'Inter', ui-sans-serif, sans-serif",
}

export const THEMES = {
  minimal: {
    canvasBg: '#f8fafc',
    zoneStroke: '#cbd5e1',
    zoneFill: 'rgba(148, 163, 184, 0.07)',
    zoneTitleColor: '#64748b',
    zoneTitleFont: BIG.zoneTitle,
    zoneMarkFont: BIG.zoneMark,
    zoneMarkFill: '#e2e8f0',
    moduleStroke: '#475569',
    moduleFill: '#ffffff',
    moduleFillAccent: '#eef2ff',
    moduleFillTeal: '#ecfdf5',
    moduleStrokeTeal: '#0d9488',
    childFill: '#e0e7ff',
    childStroke: '#4f46e5',
    edgeStroke: '#6366f1',
    edgeDashed: '#94a3b8',
    titleFont: BIG.title,
    subtitleFont: BIG.subtitle,
    annoFont: BIG.anno,
    flowFont: BIG.flow,
    hintFont: BIG.hint,
    hintFill: '#64748b',
    moduleTextFill: '#1e293b',
    annoFill: '#4338ca',
    flowPillFill: '#be123c',
    rx: { zone: 12, module: 10, child: 8 },
    sw: { zone: 1.35, module: 2, child: 1.75, edge: 2.5 },
    header: null,
    dropShadow: false,
  },
  blueprint: {
    canvasBg: '#0b1224',
    zoneStroke: '#334155',
    zoneFill: 'rgba(56, 189, 248, 0.06)',
    zoneTitleColor: '#94a3b8',
    zoneTitleFont: "600 12px 'IBM Plex Mono', ui-monospace, monospace",
    zoneMarkFont: "700 44px 'IBM Plex Mono', ui-monospace, monospace",
    zoneMarkFill: 'rgba(148, 163, 184, 0.12)',
    moduleStroke: '#38bdf8',
    moduleFill: 'rgba(56, 189, 248, 0.12)',
    moduleFillAccent: 'rgba(129, 140, 248, 0.15)',
    moduleFillTeal: 'rgba(45, 212, 191, 0.12)',
    moduleStrokeTeal: '#2dd4bf',
    childFill: 'rgba(56, 189, 248, 0.1)',
    childStroke: '#7dd3fc',
    edgeStroke: '#a5b4fc',
    edgeDashed: '#64748b',
    titleFont: "600 15px 'IBM Plex Mono', ui-monospace, monospace",
    subtitleFont: "500 12px 'IBM Plex Mono', ui-monospace, monospace",
    annoFont: "700 14px 'IBM Plex Mono', ui-monospace, monospace",
    flowFont: "600 13px 'IBM Plex Mono', ui-monospace, monospace",
    hintFont: "500 12px 'IBM Plex Mono', ui-monospace, monospace",
    hintFill: '#94a3b8',
    moduleTextFill: '#f1f5f9',
    annoFill: '#fde68a',
    flowPillFill: '#fca5a5',
    rx: { zone: 10, module: 8, child: 6 },
    sw: { zone: 1.1, module: 1.75, child: 1.5, edge: 2.25 },
    header: {
      title: 'Architecture schematic',
      subtitle: 'Meta-controller over tree-structured GNN',
      titleFont: "600 14px 'IBM Plex Mono', ui-monospace, monospace",
      subtitleFont: "400 12px 'IBM Plex Mono', ui-monospace, monospace",
      fill: '#e2e8f0',
      subfill: '#94a3b8',
    },
    dropShadow: false,
  },
  editorial: {
    canvasBg: '#fafaf9',
    zoneStroke: '#d6d3d1',
    zoneFill: 'rgba(168, 162, 158, 0.06)',
    zoneTitleColor: '#78716c',
    zoneTitleFont: "600 13px 'Source Serif 4', Georgia, serif",
    zoneMarkFont: "700 44px 'Source Serif 4', Georgia, serif",
    zoneMarkFill: '#e7e5e4',
    moduleStroke: '#44403c',
    moduleFill: '#ffffff',
    moduleFillAccent: '#f5f5f4',
    moduleFillTeal: '#f0fdfa',
    moduleStrokeTeal: '#0f766e',
    childFill: '#faf5ff',
    childStroke: '#6b21a8',
    edgeStroke: '#57534e',
    edgeDashed: '#a8a29e',
    titleFont: "600 16px 'JetBrains Mono', ui-monospace, monospace",
    subtitleFont: "500 13px 'JetBrains Mono', ui-monospace, monospace",
    annoFont: "700 15px 'Source Serif 4', Georgia, serif",
    flowFont: "700 14px 'Source Serif 4', Georgia, serif",
    hintFont: "italic 500 13px 'Source Serif 4', Georgia, serif",
    hintFill: '#57534e',
    moduleTextFill: '#1c1917',
    annoFill: '#6b21a8',
    flowPillFill: '#9f1239',
    rx: { zone: 4, module: 6, child: 5 },
    sw: { zone: 1.6, module: 2.1, child: 1.9, edge: 2.5 },
    header: {
      title: 'System layout',
      subtitle: 'Chess engine · propagation · halting',
      titleFont: "700 17px 'Source Serif 4', Georgia, serif",
      subtitleFont: "400 13px 'Source Serif 4', Georgia, serif",
      fill: '#1c1917',
      subfill: '#57534e',
    },
    dropShadow: false,
  },
  cards: {
    canvasBg: '#f1f5f9',
    zoneStroke: 'transparent',
    zoneFill: 'transparent',
    zoneTitleColor: '#64748b',
    zoneTitleFont: BIG.zoneTitle,
    zoneMarkFont: BIG.zoneMark,
    zoneMarkFill: '#cbd5e1',
    moduleStroke: '#e2e8f0',
    moduleFill: '#ffffff',
    moduleFillAccent: '#ffffff',
    moduleFillTeal: '#ffffff',
    moduleStrokeTeal: '#14b8a6',
    childFill: '#ffffff',
    childStroke: '#6366f1',
    edgeStroke: '#818cf8',
    edgeDashed: '#94a3b8',
    titleFont: "600 16px 'Space Grotesk', sans-serif",
    subtitleFont: "500 13px 'JetBrains Mono', monospace",
    annoFont: "700 14px 'JetBrains Mono', monospace",
    flowFont: "600 14px 'Space Grotesk', sans-serif",
    hintFont: "500 13px 'Inter', sans-serif",
    hintFill: '#475569',
    moduleTextFill: '#0f172a',
    annoFill: '#4f46e5',
    flowPillFill: '#be123c',
    rx: { zone: 16, module: 14, child: 10 },
    sw: { zone: 0, module: 1.35, child: 1.35, edge: 2.5 },
    header: null,
    dropShadow: true,
  },
  swiss: {
    canvasBg: '#ffffff',
    zoneStroke: '#0f172a',
    zoneFill: '#ffffff',
    zoneTitleColor: '#0f172a',
    zoneTitleFont: "800 11px 'Inter', sans-serif",
    zoneMarkFont: "900 40px 'Inter', sans-serif",
    zoneMarkFill: '#0f172a',
    moduleStroke: '#0f172a',
    moduleFill: '#ffffff',
    moduleFillAccent: '#ffffff',
    moduleFillTeal: '#ffffff',
    moduleStrokeTeal: '#0f172a',
    childFill: '#ffffff',
    childStroke: '#0f172a',
    edgeStroke: '#0f172a',
    edgeDashed: '#64748b',
    titleFont: "800 15px 'Inter', sans-serif",
    subtitleFont: "600 12px 'Inter', sans-serif",
    annoFont: "800 13px 'JetBrains Mono', monospace",
    flowFont: "800 12px 'Inter', sans-serif",
    hintFont: "700 12px 'Inter', sans-serif",
    hintFill: '#0f172a',
    moduleTextFill: '#0f172a',
    annoFill: '#0f172a',
    flowPillFill: '#0f172a',
    rx: { zone: 0, module: 0, child: 0 },
    sw: { zone: 2.75, module: 2.75, child: 2.75, edge: 2.75 },
    header: {
      title: 'HIGH-LEVEL ARCHITECTURE',
      subtitle: 'FIG. 1 — TREE SEARCH META-CONTROL',
      titleFont: "900 15px 'Inter', sans-serif",
      subtitleFont: "600 10px 'JetBrains Mono', monospace",
      fill: '#0f172a',
      subfill: '#475569',
    },
    dropShadow: false,
  },
}

export function anchorPoint(mod, anchor) {
  const cx = mod.x + mod.w / 2
  const cy = mod.y + mod.h / 2
  switch (anchor) {
    case 'top':
      return [cx, mod.y]
    case 'bottom':
      return [cx, mod.y + mod.h]
    case 'left':
      return [mod.x, cy]
    case 'right':
      return [mod.x + mod.w, cy]
    case 'c':
    default:
      return [cx, cy]
  }
}

export function moduleById(id) {
  return MODULES.find((m) => m.id === id)
}
