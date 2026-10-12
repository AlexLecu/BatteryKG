import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// In development /api goes to the FastAPI app (uvicorn api.main:app --port 8000)
export default defineConfig({
  plugins: [vue()],
  build: { chunkSizeWarningLimit: 600 }, // vis-network alone is ~530 kB, on the graph page only
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
})
