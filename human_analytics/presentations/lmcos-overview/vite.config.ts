import { defineConfig } from 'vite'

// Allow loading analysis figures from `chess_analysis/human_analytics/figures/` (relative to this package).
export default defineConfig({
  server: {
    fs: {
      allow: ['.', '..', '../..', '../../..', '../../../..'],
    },
  },
})
