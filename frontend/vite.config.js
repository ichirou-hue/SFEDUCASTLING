import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8005',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
    css: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      // Порог считается по модулям, для которых уже есть тесты. Новые
      // тестируемые модули нужно добавлять сюда, чтобы покрытие не падало.
      include: [
        'src/apiErrors.js',
        'src/theme.js',
        'src/learningLinks.js',
        'src/components/AchievementsWidget.jsx',
        'src/components/UserRecordsWidget.jsx',
      ],
      exclude: ['src/main.jsx', 'src/test/**'],
      thresholds: {
        lines: 85,
        functions: 90,
        branches: 70,
        statements: 80,
      },
    },
  },
})
