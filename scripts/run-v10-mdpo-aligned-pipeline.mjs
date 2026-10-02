import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo');
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const options = { baseUrl: 'http://127.0.0.1:5174', pollSeconds: 60 };
for (let i = 0; i < process.argv.length; i += 1) if (process.argv[i].startsWith('--')) { const key = process.argv[i].slice(2); options[key] = process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[++i] : true; }
const read = (file) => fs.readFile(file, 'utf8').then(JSON.parse).catch((error) => error?.code === 'ENOENT' ? null : Promise.reject(error));
async function run(script, args = []) { return new Promise((resolve, reject) => { const child = spawn(process.execPath, [script, ...args], { cwd: root, windowsHide: true, stdio: 'inherit' }); child.once('error', reject); child.once('exit', (code) => code === 0 ? resolve() : reject(new Error(script + ' exited ' + code))); }); }
async function waitForTraining() {
  while (true) {
    const ledger = await read(path.join(mdpo, 'aligned_sweep', 'aligned_sweep_ledger.json'));
    const trained = (ledger?.configurations || []).filter((row) => row.status === 'trained').length;
    const failed = (ledger?.configurations || []).filter((row) => row.status === 'failed').length;
    console.log(JSON.stringify({ stage: 'train_81_grid_aligned', trained, failed, total: 81, checked_at: new Date().toISOString() }));
    if (failed > 0) throw new Error('Aligned 81-grid training has failed configurations: ' + failed);
    if (trained === 81 && failed === 0) return;
    await sleep(Math.max(10, Number(options.pollSeconds)) * 1000);
  }
}
async function waitForVal11() {
  while (true) {
    const ledger = await read(path.join(mdpo, 'aligned_val11_sweep_ledger.json'));
    const evaluated = (ledger?.configurations || []).filter((row) => row.status === 'evaluated').length;
    const failed = (ledger?.configurations || []).filter((row) => row.status === 'failed').length;
    console.log(JSON.stringify({ stage: 'evaluate_81_grid_val11_aligned', evaluated, failed, total: 81, checked_at: new Date().toISOString() }));
    if (failed > 0) throw new Error('Aligned val11 has failed configurations: ' + failed);
    if (evaluated === 81 && failed === 0) return;
    await sleep(Math.max(10, Number(options.pollSeconds)) * 1000);
  }
}
await waitForTraining();
await run('scripts/run-v10-mdpo-aligned-val11-sweep.mjs', ['--baseUrl', String(options.baseUrl), '--timeoutMinutes', '240']);
await waitForVal11();
await run('scripts/select-v10-mdpo-aligned-hyperparameters.mjs');
await run('scripts/run-v10-mdpo-ablations.mjs', ['--aligned', 'true', '--selection', path.join(mdpo, 'aligned_hyperparameter_selection.json'), '--outputDir', path.join(mdpo, 'aligned_ablations')]);
await run('scripts/run-v10-mdpo-ablation-val11.mjs', ['--aligned', 'true', '--baseUrl', String(options.baseUrl), '--timeoutMinutes', '240']);
console.log(JSON.stringify({ status: 'aligned_core_and_ablation_pipeline_complete', evaluation_policy: 'safety_priority_v3_aligned' }));
