import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { latestMdpoCheckpoint, validateCompletedMdpoTrainingRun } from '../lib/mdpo-training-resume.mjs';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';
import { validateMdpoFinalDatasetAudit } from '../lib/mdpo-final-dataset-audit.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sweepDir = path.join(root, 'experiments', 'mdpo', 'sweep');
const fixedSeed = 17017;
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const [referenceHash, datasetHash] = await Promise.all([
  fs.readFile(path.join(root, 'experiments', 'layout_model.json')).then(digest),
  fs.readFile(path.join(root, 'experiments', 'mdpo', 'train_pairs.json')).then(digest)
]);
const [dataset, manifest] = await Promise.all([
  fs.readFile(path.join(root, 'experiments', 'mdpo', 'train_pairs.json'), 'utf8').then(JSON.parse),
  fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8').then(JSON.parse)
]);
if (dataset.reference_model_sha256 !== referenceHash) throw new Error('81-grid MDPO training requires the frozen original v10 reference');
validateMdpoDataset(dataset, manifest, { requireComplete: true });
validateMdpoFinalDatasetAudit({ datasetBytes: await fs.readFile(path.join(root, 'experiments', 'mdpo', 'train_pairs.json')),
  audit: await fs.readFile(path.join(root, 'experiments', 'mdpo', 'dataset_audit_final.json'), 'utf8').then(JSON.parse), referenceSha256: referenceHash });
const learningRates = [5e-5, 1e-4, 2e-4], betas = [0.05, 0.10, 0.20], lambdaMultis = [0.1, 0.3, 0.5], ranks = [2, 4, 8];
const configurations = learningRates.flatMap((learningRate) =>
  betas.flatMap((beta) =>
    lambdaMultis.flatMap((lambdaMulti) =>
      ranks.map((rank) => ({ learningRate, beta, lambdaMulti, rank })))));
const existing = await fs.readFile(path.join(sweepDir, 'sweep_ledger.json'), 'utf8').then(JSON.parse).catch(() => ({ version: 'v10_mdpo_hyperparameter_sweep_v1', configurations: [] }));
await fs.mkdir(sweepDir, { recursive: true });
async function run(configuration) {
  const id = `lr_${configuration.learningRate}_beta_${configuration.beta}_multi_${configuration.lambdaMulti}_rank_${configuration.rank}`.replaceAll('.', 'p');
  const outputDir = path.join(sweepDir, id), stdout = path.join(outputDir, 'stdout.log'), stderr = path.join(outputDir, 'stderr.log');
  await fs.mkdir(outputDir, { recursive: true });
  const candidateFile = path.join(outputDir, 'layout_model_v10_mdpo_candidate.json');
  const ledgerFile = path.join(outputDir, 'checkpoint_ledger.json');
  const exists = await Promise.all([candidateFile, ledgerFile].map((file) => fs.access(file).then(() => true, (error) => {
    if (error?.code === 'ENOENT') return false;
    throw error;
  })));
  if (exists.some(Boolean)) {
    if (!exists.every(Boolean)) throw new Error(`${id} has partial immutable final training artifacts; preserve for manual audit`);
    await validateCompletedMdpoTrainingRun({ outputDir, root, referenceHash, datasetHash, options: { ...configuration, seed: fixedSeed, learnableVariance: false, noReference: false } });
    return { id, outputDir, reused_existing_candidate: true };
  }
  const resume = await latestMdpoCheckpoint(path.join(outputDir, 'checkpoints'));
  const out = await fs.open(stdout, 'a');
  let err;
  try {
    err = await fs.open(stderr, 'a');
    const child = spawn(process.execPath, ['scripts/train-v10-mdpo.mjs', '--outputDir', outputDir, '--learningRate', String(configuration.learningRate), '--beta', String(configuration.beta), '--lambdaMulti', String(configuration.lambdaMulti), '--rank', String(configuration.rank), '--seed', String(fixedSeed), ...(resume ? ['--resume', resume] : [])], { cwd: root, windowsHide: true, stdio: ['ignore', out.fd, err.fd] });
    const code = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', resolve); });
    if (code !== 0) throw new Error(`${id} exited ${code}`);
    return { id, outputDir };
  } finally { await out.close(); if (err) await err.close(); }
}
for (let index = 0; index < configurations.length; index++) {
  const configuration = configurations[index], id = `lr_${configuration.learningRate}_beta_${configuration.beta}_multi_${configuration.lambdaMulti}_rank_${configuration.rank}`.replaceAll('.', 'p');
  if (existing.configurations.some((item) => item.id === id && item.status === 'trained')) {
    await validateCompletedMdpoTrainingRun({ outputDir: path.join(sweepDir, id), root, referenceHash, datasetHash,
      options: { ...configuration, seed: fixedSeed, learnableVariance: false, noReference: false } });
    continue;
  }
  const startedAt = new Date().toISOString();
  try {
    const result = await run(configuration);
    const candidate = await validateCompletedMdpoTrainingRun({ outputDir: result.outputDir, root, referenceHash, datasetHash,
      options: { ...configuration, seed: fixedSeed, learnableVariance: false, noReference: false } });
    existing.configurations = existing.configurations.filter((item) => item.id !== id);
    existing.configurations.push({ id, status: 'trained', started_at: startedAt, completed_at: new Date().toISOString(), hyperparameters: { ...configuration, seed: fixedSeed }, proxy_val: candidate.mdpo.best_val, best_epoch: candidate.mdpo.best_epoch, deployment_status: 'diagnostic_only_pending_authoritative_val11_six_view_gate' });
  } catch (error) {
    existing.configurations = existing.configurations.filter((item) => item.id !== id);
    existing.configurations.push({ id, status: 'failed', started_at: startedAt, completed_at: new Date().toISOString(), hyperparameters: configuration, error: error.message });
  }
  existing.required_grid = { learning_rates: learningRates, betas, lambda_multis: lambdaMultis, ranks, total: configurations.length };
  existing.fixed_seed = fixedSeed;
  existing.test_used_for_selection = false;
  await fs.writeFile(path.join(sweepDir, 'sweep_ledger.json'), JSON.stringify(existing, null, 2) + '\n');
}
const trainedCount = existing.configurations.filter((item) => item.status === 'trained').length;
console.log(JSON.stringify({ total: configurations.length, trained: trainedCount, ledger: path.relative(root, path.join(sweepDir, 'sweep_ledger.json')).split(path.sep).join('/') }, null, 2));
if (trainedCount !== configurations.length) {
  console.error(`v10-MDPO 81-grid incomplete: ${trainedCount}/${configurations.length}; rerun resumes validated checkpoints and retries failures`);
  process.exitCode = 2;
}
