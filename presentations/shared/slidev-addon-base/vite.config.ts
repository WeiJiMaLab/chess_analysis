import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

// This addon is loaded by every deck (src/<deck>.md). The Slidev project root
// is `src/`, and figures are served from `src/public/figures`, a symlink chain
// that resolves to `chess_analysis/human_analytics/figures` (outside `src/`).
// Vite's dev server refuses to serve files outside the project root unless they
// are explicitly allowed, so widen `server.fs.allow` to the presentations root,
// the repo root, and the real figures directory.
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const addonRoot = __dirname
const presentationsRoot = path.resolve(addonRoot, '../..')
const repoRoot = path.resolve(presentationsRoot, '..')
const figuresRoot = path.resolve(repoRoot, 'human_analytics/figures')

export default defineConfig({
  server: {
    fs: {
      allow: [presentationsRoot, repoRoot, figuresRoot],
    },
  },
})
