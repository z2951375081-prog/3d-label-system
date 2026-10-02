import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { buildMdpoLiveResults } from '../lib/mdpo-live-results.mjs';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const mdpo = path.join(root, 'experiments', 'mdpo');
const outputFile = path.join(root, 'public', 'mdpo-live-results.json');
const stateFile = path.join(mdpo, 'live_results_watcher.json');
const once = process.argv.includes('--once');

async function update() {
  const payload = await buildMdpoLiveResults({ root });
  const temporary = outputFile + '.' + randomUUID() + '.tmp';
  await fs.writeFile(temporary, JSON.stringify(payload, null, 2) + '\n', { flag: 'wx' });
  try {
    await fs.rename(temporary, outputFile);
  } catch (error) {
    if (process.platform !== 'win32' || !['EPERM', 'EACCES', 'EBUSY'].includes(error?.code)) throw error;
    await fs.copyFile(temporary, outputFile);
    await fs.rm(temporary, { force: true });
  }
  const watcherState = {
    version: 'v10_mdpo_live_results_watcher_v2', pid: process.pid, updated_at: payload.generated_at,
    output: path.relative(root, outputFile).split(path.sep).join('/'),
    latest_candidate_id: payload.latest?.candidate_id || null, evaluated: payload.progress.evaluated,
    running_candidate_id: payload.progress.running?.id || null,
    aligned_candidate_id: payload.aligned?.current?.id || null,
    aligned_epoch: payload.aligned?.current_metrics?.epoch || null
  };
  await fs.writeFile(stateFile, JSON.stringify(watcherState, null, 2) + '\n');
  process.stdout.write(JSON.stringify(watcherState) + '\n');
}

await update();
if (!once) setInterval(() => update().catch((error) => process.stderr.write(error.stack + '\n')), 30_000);
