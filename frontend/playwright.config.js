import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
 testDir: './e2e', retries: 1,
 use: { baseURL: 'http://127.0.0.1:4173', trace: 'retain-on-failure' },
 projects: [{name:'desktop',use:{...devices['Desktop Chrome']}},{name:'mobile',use:{...devices['Pixel 7']}}],
 webServer: {command:'npm run preview -- --host 127.0.0.1 --port 4173',port:4173},
 reporter: [['list'],['html',{open:'never'}]],
});
