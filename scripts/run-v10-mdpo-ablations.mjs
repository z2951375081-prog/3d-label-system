import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { latestMdpoCheckpoint, validateCompletedMdpoTrainingRun } from '../lib/mdpo-training-resume.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const [referenceHash, datasetHash] = await Promise.all([
  fs.readFile(path.join(root, 'experiments', 'layout_model.json')).then(digest),
  fs.readFile(path.join(root, 'experiments', 'mdpo', 'train_pairs.json')).then(digest)
]);
const args = Object.fromEntries(process.argv.slice(2).reduce((rows, item, index, all) => item.startsWith('--') ? [...rows, [item.slice(2), all[index + 1] && !all[index + 1].startsWith('--') ? all[index + 1] : true]] : rows, []));
const aligned = args.aligned === true || String(args.aligned).toLowerCase() === 'true';
const selectionFile = path.resolve(String(args.selection || path.join(root, 'experiments', 'mdpo', aligned ? 'aligned_hyperparameter_selection.json' : 'hyperparameter_selection.json')));
const outputRoot = path.resolve(String(args.outputDir || path.join(root, 'experiments', 'mdpo', aligned ? 'aligned_ablations' : 'ablations')));
const selection = await fs.readFile(selectionFile, 'utf8').then(JSON.parse).catch((error) => {
  if (error?.code === 'ENOENT') throw new Error(`Missing authoritative val11 hyperparameter selection: ${path.relative(root, selectionFile).split(path.sep).join('/')}`);
  throw error;
});
if ((!aligned && (selection.version !== 'v10_mdpo_hyperparameter_selection_v1' || selection.status !== 'selected_by_complete_authoritative_val11')) || (aligned && (selection.version !== 'v10_mdpo_aligned_hyperparameter_selection_v1' || selection.status !== 'selected_by_complete_aligned_val11'))
    || selection.test_used_for_selection !== false || selection.val11_gate_complete !== true) throw new Error('Ablations require a complete authoritative val11-selected base configuration; training proxies cannot select it');
const base = selection.selected?.hyperparameters;
for (const key of ['learningRate', 'beta', 'lambdaMulti', 'rank']) if (!Number.isFinite(Number(base?.[key]))) throw new Error(`Selected val11 configuration missing ${key}`);
const seed = Number.isFinite(Number(base.seed)) ? Number(base.seed) : 17017;
const common = {
  learningRate: Number(base.learningRate), beta: Number(base.beta), lambdaMulti: Number(base.lambdaMulti), rank: Number(base.rank),
  lambdaDpo: Number.isFinite(Number(base.lambdaDpo)) ? Number(base.lambdaDpo) : 1,
  lambdaKl: Number.isFinite(Number(base.lambdaKl)) ? Number(base.lambdaKl) : 0.01,
  epochs: Number.isFinite(Number(base.epochs)) ? Number(base.epochs) : 30, seed,
  ...(aligned ? { safetyAlignment: true, overlapPairWeight: 24, worstOverflowWeight: 20, cvarOverflowWeight: 12, preferenceSafetyWeight: 0.5 } : {})
};
const configurations = [
  { id: 'full_fixed_variance', overrides: {} },
  { id: 'no_multi_only_overall_dpo', overrides: { lambdaMulti: 0 } },
  { id: 'no_overall_only_multidimensional', overrides: { lambdaDpo: 0 } },
  { id: 'no_reference_kl', overrides: { lambdaKl: 0, noReference: true } },
  { id: 'no_text_clarity', overrides: { textWeight: 0 } },
  { id: 'no_leader_line_clarity', overrides: { leaderWeight: 0 } },
  { id: 'equal_multidimensional_weights', overrides: { compositionWeight: 1 / 6, hierarchyWeight: 1 / 6, balanceWeight: 1 / 6, manualStyleWeight: 1 / 6, textWeight: 1 / 6, leaderWeight: 1 / 6 } },
  { id: 'clarity_emphasis_weights', overrides: { compositionWeight: 0.15, hierarchyWeight: 0.15, balanceWeight: 0.10, manualStyleWeight: 0.10, textWeight: 0.25, leaderWeight: 0.25 } },
  { id: 'learnable_variance', overrides: { learnableVariance: true, varianceLearningRate: Number(base.varianceLearningRate || 1e-4) } }
];
await fs.mkdir(outputRoot, { recursive: true });
const ledgerFile = path.join(outputRoot, 'ablation_ledger.json');
const ledger = await fs.readFile(ledgerFile, 'utf8').then(JSON.parse).catch(() => ({ version: aligned ? 'v10_mdpo_aligned_ablation_ledger_v1' : 'v10_mdpo_ablation_ledger_v1', evaluation_policy: aligned ? 'safety_priority_v3_aligned' : 'safety_priority_v2', configurations: [] }));

async function run(configuration) {
  const directory = path.join(outputRoot, configuration.id), stdoutFile = path.join(directory, 'stdout.log'), stderrFile = path.join(directory, 'stderr.log');
  const hyperparameters = { ...common, ...configuration.overrides };
  await fs.mkdir(directory, { recursive: true });
  const artifactState = await Promise.all(['layout_model_v10_mdpo_candidate.json', 'checkpoint_ledger.json'].map((name) => fs.access(path.join(directory, name)).then(() => true, (error) => {
    if (error?.code === 'ENOENT') return false;
    throw error;
  })));
  if (artifactState.some(Boolean)) {
    if (!artifactState.every(Boolean)) throw new Error(`${configuration.id} has partial immutable final training artifacts; preserve for manual audit`);
    await validateCompletedMdpoTrainingRun({ outputDir: directory, root, referenceHash, datasetHash, options: hyperparameters });
    return { directory, hyperparameters, reused_existing_candidate: true };
  }
  const resume = await latestMdpoCheckpoint(path.join(directory, 'checkpoints'));
  const argv = ['scripts/train-v10-mdpo.mjs', '--outputDir', directory, ...Object.entries(hyperparameters).flatMap(([key, value]) => [`--${key}`, String(value)]), ...(resume ? ['--resume', resume] : [])];
  const out = await fs.open(stdoutFile, 'a');
  let err;
  try {
    err = await fs.open(stderrFile, 'a');
    const child = spawn(process.execPath, argv, { cwd: root, windowsHide: true, stdio: ['ignore', out.fd, err.fd] });
    const code = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', resolve); });
    if (code !== 0) throw new Error(`${configuration.id} exited ${code}`);
    return { directory, hyperparameters };
  } finally { await out.close(); if (err) await err.close(); }
}

for (const configuration of configurations) {
  if (ledger.configurations.some((row) => row.id === configuration.id && row.status === 'trained')) {
    await validateCompletedMdpoTrainingRun({ outputDir: path.join(outputRoot, configuration.id), root, referenceHash, datasetHash,
      options: { ...common, ...configuration.overrides } });
    continue;
  }
  const startedAt = new Date().toISOString();
  ledger.current = { id: configuration.id, started_at: startedAt };
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  try {
    const result = await run(configuration);
    const candidate = await validateCompletedMdpoTrainingRun({ outputDir: result.directory, root, referenceHash, datasetHash,
      options: result.hyperparameters });
    ledger.configurations = ledger.configurations.filter((row) => row.id !== configuration.id);
    ledger.configurations.push({ id: configuration.id, status: 'trained', started_at: startedAt, completed_at: new Date().toISOString(), hyperparameters: result.hyperparameters, best_epoch: candidate.mdpo.best_epoch, proxy_val_diagnostic_only: candidate.mdpo.best_val, trainable_parameters: candidate.mdpo.trainable_parameters, candidate_file: path.relative(root, path.join(result.directory, 'layout_model_v10_mdpo_candidate.json')).split(path.sep).join('/'), authoritative_val11_pending: true });
  } catch (error) {
    ledger.configurations = ledger.configurations.filter((row) => row.id !== configuration.id);
    ledger.configurations.push({ id: configuration.id, status: 'failed', started_at: startedAt, completed_at: new Date().toISOString(), error: error.message });
  }
  ledger.current = null;
  ledger.base_selection = path.relative(root, selectionFile).split(path.sep).join('/');
  ledger.fixed_seed = seed;
  ledger.test_used_for_selection = false;
  ledger.required = configurations.map((row) => row.id);
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
}
const trainedCount = ledger.configurations.filter((row) => row.status === 'trained').length;
console.log(JSON.stringify({ total: configurations.length, trained: trainedCount, ledger: path.relative(root, ledgerFile).split(path.sep).join('/') }, null, 2));
if (trainedCount !== configurations.length) {
  console.error(`v10-MDPO ablation training incomplete: ${trainedCount}/${configurations.length}; rerun resumes validated checkpoints and retries failures`);
  process.exitCode = 2;
}
