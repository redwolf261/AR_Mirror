import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    proxy: {
      '/api':    'http://localhost:5050',
      '/stream': 'http://localhost:5050',
    },
  },
})
