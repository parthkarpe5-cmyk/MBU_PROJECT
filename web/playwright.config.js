import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';
const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
export default defineConfig({
  testDir:'./tests/browser',timeout:90000,workers:1,
  use:{baseURL:'http://127.0.0.1:4273',headless:true,launchOptions:{
    ...(process.platform==='win32' && existsSync(edge)?{executablePath:edge}:{}),
    args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']
  }},
  webServer:{command:'node ../node_modules/vite/bin/vite.js preview . --outDir ../dist --host 127.0.0.1 --port 4273 --strictPort',url:'http://127.0.0.1:4273',reuseExistingServer:false},
  reporter:'list'
});
