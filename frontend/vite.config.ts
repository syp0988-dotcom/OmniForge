import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  root: __dirname,
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    // Bind every interface (IPv4 + IPv6). Vite's default is `localhost`, which
    // on Windows resolves to ::1 only — so http://127.0.0.1:5173/ was refused
    // even though the server was running. Set `host: '127.0.0.1'` instead if
    // you want to keep the dev server off the local network.
    host: true,
    port: 5173,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vue: ['vue'],
          markdown: ['markdown-it', 'highlight.js'],
          icons: ['lucide-vue-next'],
        },
      },
    },
  }
})
