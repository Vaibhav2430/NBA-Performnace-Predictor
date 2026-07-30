import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict':          'http://localhost:8000',
      '/search':           'http://localhost:8000',
      '/games/today':      'http://localhost:8000',
      '/team_preview':     'http://localhost:8000',
      '/wnba':             'http://localhost:8000',
      '/accuracy':         'http://localhost:8000',
    },
  },
})
