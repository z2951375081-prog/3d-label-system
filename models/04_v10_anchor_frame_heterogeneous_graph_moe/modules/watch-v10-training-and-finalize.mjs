import fs from 'node:fs/promises';
import path from 'node:path';
import { execFile, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const logFile = path.join(experiments, 'service-logs', 'v10-training-watch.log');
const execFileAsync = promisify(execFile);
const args = process.argv.slice(2);
const pidIndex = args.indexOf('--pid');
const pid = Number(pidIndex >= 0 ? args[pidIndex + 1] : 0);
const intervalMs = 10 * 60 * 1000;
const progressIntervalMs = intervalMs;
const startedAt = Date.now();

if (!Number.isInteger(pid) || pid <= 0) throw new Error('watcher requires --pid <training-pid>');
const stamp = async (message) => {
  const line = `[${new Date().toISOString()}] ${message}`;
  await fs.mkdir(path.dirname(logFile), { recursive: true });
  await fs.appendFile(logFile, `${line}\n`, 'utf8');
  console.log(line);
};
async function processInfo() {
  try {
    const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-Command', `(Get-CimInstance Win32_Process -Filter "ProcessId = ${pid}") | Select-Object ProcessId,Name,CommandLine,CreationDate | ConvertTo-Json -Compress`], { windowsHide: true });
    if (!stdout.trim()) return null;
    return JSON.parse(stdout);
  } catch { return null; }
}
async function waitForTraining() {
  let lastProgress = 0;
  await stamp(`Watching v10 training PID ${pid}; process checks and progress logs are every ten minutes.`);
  while (true) {
    const info = await processInfo();
    if (!info) return;
    if (Date.now() - lastProgress >= progressIntervalMs) {
      lastProgress = Date.now();
      await stamp(`Training still running: PID ${pid}, elapsed_hours=${((Date.now() - startedAt) / 3600000).toFixed(2)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
try {
  await waitForTraining();
  await stamp(`Training PID ${pid} ended; starting v10 completion audit.`);
  const child = spawn(process.execPath, ['scripts/finalize-v10-training.mjs'], { cwd: root, windowsHide: true, stdio: 'inherit' });
  const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('close', resolve); });
  await stamp(`v10 completion audit exited with code ${code}.`);
  if (code !== 0) { process.exitCode = code || 2; }
  else {
    await stamp('v10 layout passed finalization; starting the complete local Qwen and visualization workflow.');
    const completion = spawn(process.execPath, ['scripts/complete-v10-layout-and-qwen.mjs'], { cwd: root, windowsHide: true, stdio: 'inherit' });
    const completionCode = await new Promise((resolve, reject) => { completion.on('error', reject); completion.on('close', resolve); });
    await stamp(`v10 Qwen and visualization workflow exited with code ${completionCode}.`);
    process.exitCode = completionCode || 0;
  }
} catch (error) {
  await stamp(`Watcher failed: ${error.stack || error.message}`);
  process.exitCode = 2;
}
