import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
execFileSync(process.execPath, ['web/scripts/prepare-assets.mjs'], { cwd: root, stdio: 'inherit' });
execFileSync(process.execPath, ['node_modules/vite/bin/vite.js', 'build', 'web', '--outDir', '../dist'], { cwd: root, stdio: 'inherit' });
