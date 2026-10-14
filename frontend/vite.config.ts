import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The Python backend (FastAPI) serves /api; in dev Vite proxies to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8000' } },
  // bundle under /_app so it can never collide with the /assets/:id UI route when served by FastAPI
  build: { chunkSizeWarningLimit: 1200, assetsDir: '_app' },
})
