import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const parseArgs = (argv) => Object.fromEntries(argv.reduce((rows, item, index) => {
  if (!item.startsWith('--')) return rows;
  rows.push([item.slice(2), argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[index + 1] : true]);
  return rows;
}, []));
const options = parseArgs(process.argv.slice(2));
const collectorPid = Number(options.collectorPid);
const intervalMinutes = Number(options.intervalMinutes || 60);
const firstCheckAt = options.firstCheckAt ? new Date(String(options.firstCheckAt)) : new Date(Date.now() + intervalMinutes * 60_000);
const output = path.resolve(root, String(options.output || 'experiments/mdpo/collection_runs/train33_hourly_monitor.jsonl'));
const stdoutFile = path.resolve(root, String(options.stdout || 'experiments/mdpo/collection_runs/train33_strict_resume2_20260922.stdout.log'));
const stderrFile = path.resolve(root, String(options.stderr || 'experiments/mdpo/collection_runs/train33_strict_resume2_20260922.stderr.log'));
const datasetFile = path.join(root, 'experiments', 'mdpo', 'train_pairs.json');

if (!Number.isInteger(collectorPid) || collectorPid <= 0) throw new Error('A positive --collectorPid is required');
if (!Number.isFinite(intervalMinutes) || intervalMinutes < 1) throw new Error('--intervalMinutes must be at least 1');
if (!Number.isFinite(firstCheckAt.getTime())) throw new Error('--firstCheckAt must be an ISO date');
if (!output.startsWith(path.join(root, 'experiments', 'mdpo', 'collection_runs') + path.sep)) throw new Error('Monitor output must stay inside experiments/mdpo/collection_runs');

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const processAlive = (pid) => {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error?.code === 'ESRCH') return false; throw error; }
};
const statSummary = async (file) => fs.stat(file).then((stat) => ({ bytes: stat.size, modified_at: stat.mtime.toISOString() }), (error) => {
  if (error?.code === 'ENOENT') return null;
  throw error;
});

async function snapshot() {
  const bytes = await fs.readFile(datasetFile);
  const dataset = JSON.parse(bytes.toString('utf8'));
  const sampleCount = new Set(dataset.pairs.map((pair) => `${pair.category}/${pair.sample_id}`)).size;
  const alive = processAlive(collectorPid);
  const record = {
    checked_at: new Date().toISOString(), collector_pid: collectorPid, collector_alive: alive,
    sample_count: sampleCount, pair_count: dataset.pairs.length,
    dataset_sha256: createHash('sha256').update(bytes).digest('hex'),
    stdout: await statSummary(stdoutFile), stderr: await statSummary(stderrFile)
  };
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.appendFile(output, JSON.stringify(record) + '\n', 'utf8');
  process.stdout.write(`${JSON.stringify(record)}\n`);
  return alive;
}

const firstDelay = Math.max(0, firstCheckAt.getTime() - Date.now());
if (firstDelay) await sleep(firstDelay);
while (await snapshot()) await sleep(intervalMinutes * 60_000);
