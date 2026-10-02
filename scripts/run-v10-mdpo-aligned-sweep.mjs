import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { latestMdpoCheckpoint, validateCompletedMdpoTrainingRun } from '../lib/mdpo-training-resume.mjs';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';
import { validateMdpoFinalDatasetAudit } from '../lib/mdpo-final-dataset-audit.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo');
const sweepDir = path.join(mdpo, 'aligned_sweep');
const ledgerFile = path.join(sweepDir, 'aligned_sweep_ledger.json');
const fixedSeed = 17017;
const safetyAlignment = { protocol: 'safety_priority_v3_aligned', overlapPairWeight: 24, worstOverflowWeight: 20, cvarOverflowWeight: 12, preferenceSafetyWeight: 0.5 };
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const idFor = ({ learningRate, beta, lambdaMulti, rank }) => ('lr_' + learningRate + '_beta_' + beta + '_multi_' + lambdaMulti + '_rank_' + rank).replaceAll('.', 'p');
const learningRates = [5e-5, 1e-4, 2e-4], betas = [0.05, 0.10, 0.20], lambdaMultis = [0.1, 0.3, 0.5], ranks = [2, 4, 8];
const configurations = learningRates.flatMap((learningRate) => betas.flatMap((beta) => lambdaMultis.flatMap((lambdaMulti) => ranks.map((rank) => ({ learningRate, beta, lambdaMulti, rank })))));
const referenceHash = digest(await fs.readFile(path.join(root, 'experiments', 'layout_model.json')));
const datasetBytes = await fs.readFile(path.join(mdpo, 'train_pairs.json'));
const datasetHash = digest(datasetBytes);
const [dataset, manifest] = await Promise.all([JSON.parse(datasetBytes.toString('utf8')), fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8').then(JSON.parse)]);
if (dataset.reference_model_sha256 !== referenceHash) throw new Error('Aligned MDPO sweep requires the frozen original v10 reference');
validateMdpoDataset(dataset, manifest, { requireComplete: true });
validateMdpoFinalDatasetAudit({ datasetBytes, audit: await fs.readFile(path.join(mdpo, 'dataset_audit_final.json'), 'utf8').then(JSON.parse), referenceSha256: referenceHash });
await fs.mkdir(sweepDir, { recursive: true });
let ledger = await fs.readFile(ledgerFile, 'utf8').then(JSON.parse).catch(() => ({ version: 'v10_mdpo_aligned_sweep_v1', policy: safetyAlignment, configurations: [] }));
ledger.required_grid = { learning_rates: learningRates, betas, lambda_multis: lambdaMultis, ranks, total: configurations.length };
ledger.fixed_seed = fixedSeed; ledger.test_used_for_selection = false; ledger.updated_at = new Date().toISOString();
await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');

async function validateAligned(outputDir, configuration) {
  const candidate = await validateCompletedMdpoTrainingRun({ outputDir, root, referenceHash, datasetHash, options: { ...configuration, seed: fixedSeed, learnableVariance: false, noReference: false, safetyAlignment: true, overlapPairWeight: safetyAlignment.overlapPairWeight, worstOverflowWeight: safetyAlignment.worstOverflowWeight, cvarOverflowWeight: safetyAlignment.cvarOverflowWeight, preferenceSafetyWeight: safetyAlignment.preferenceSafetyWeight } });
  const alignment = candidate.mdpo?.safety_alignment;
  if (!alignment || alignment.protocol !== safetyAlignment.protocol || alignment.overlap_pair_weight !== safetyAlignment.overlapPairWeight || alignment.worst_view_overflow_weight !== safetyAlignment.worstOverflowWeight || alignment.cvar_overflow_weight !== safetyAlignment.cvarOverflowWeight || !alignment.final_gate_metrics.includes('overlap_pairs') || !alignment.final_gate_metrics.includes('worst_view_overflow')) throw new Error('Aligned candidate is missing v3 safety objective provenance');
  return candidate;
}

async function run(configuration) {
  const id = idFor(configuration), outputDir = path.join(sweepDir, id);
  await fs.mkdir(outputDir, { recursive: true });
  const candidateFile = path.join(outputDir, 'layout_model_v10_mdpo_candidate.json');
  const checkpointFile = path.join(outputDir, 'checkpoint_ledger.json');
  const both = await Promise.all([candidateFile, checkpointFile].map((file) => fs.access(file).then(() => true, (error) => error?.code === 'ENOENT' ? false : Promise.reject(error))));
  if (both.every(Boolean)) return { id, outputDir, reused_existing_candidate: true };
  if (both.some(Boolean)) throw new Error(id + ' has partial aligned artifacts');
  const resume = await latestMdpoCheckpoint(path.join(outputDir, 'checkpoints'));
  const stdout = await fs.open(path.join(outputDir, 'stdout.log'), 'a'), stderr = await fs.open(path.join(outputDir, 'stderr.log'), 'a');
  try {
    const args = ['scripts/train-v10-mdpo.mjs', '--outputDir', outputDir, '--learningRate', String(configuration.learningRate), '--beta', String(configuration.beta), '--lambdaMulti', String(configuration.lambdaMulti), '--rank', String(configuration.rank), '--seed', String(fixedSeed), '--safetyAlignment', 'true', '--overlapPairWeight', String(safetyAlignment.overlapPairWeight), '--worstOverflowWeight', String(safetyAlignment.worstOverflowWeight), '--cvarOverflowWeight', String(safetyAlignment.cvarOverflowWeight), '--preferenceSafetyWeight', String(safetyAlignment.preferenceSafetyWeight)];
    if (resume) args.push('--resume', resume);
    const code = await new Promise((resolve, reject) => { const child = spawn(process.execPath, args, { cwd: root, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd] }); child.once('error', reject); child.once('exit', resolve); });
    if (code !== 0) throw new Error(id + ' aligned training exited ' + code);
    return { id, outputDir, reused_existing_candidate: false };
  } finally { await stdout.close(); await stderr.close(); }
}

for (const configuration of configurations) {
  const id = idFor(configuration), startedAt = new Date().toISOString();
  const prior = ledger.configurations.find((item) => item.id === id && item.status === 'trained');
  ledger.current = prior ? null : { id, hyperparameters: configuration, started_at: startedAt, status: 'training' };
  ledger.updated_at = new Date().toISOString();
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  try {
    const result = prior ? { id, outputDir: path.join(sweepDir, id), reused_existing_candidate: true } : await run(configuration);
    const candidate = await validateAligned(result.outputDir, configuration);
    ledger.configurations = ledger.configurations.filter((item) => item.id !== id).concat({ id, status: 'trained', started_at: prior?.started_at || startedAt, completed_at: new Date().toISOString(), hyperparameters: { ...configuration, seed: fixedSeed, safetyAlignment: true, ...safetyAlignment }, policy: safetyAlignment, proxy_val: candidate.mdpo.best_val, best_epoch: candidate.mdpo.best_epoch, deployment_status: 'diagnostic_only_pending_authoritative_val11_v3_aligned_gate' });
  } catch (error) {
    ledger.configurations = ledger.configurations.filter((item) => item.id !== id).concat({ id, status: 'failed', started_at: startedAt, completed_at: new Date().toISOString(), hyperparameters: configuration, policy: safetyAlignment, error: error.message });
  }
  ledger.current = null;
  ledger.required_grid = { learning_rates: learningRates, betas, lambda_multis: lambdaMultis, ranks, total: configurations.length };
  ledger.fixed_seed = fixedSeed; ledger.test_used_for_selection = false; ledger.updated_at = new Date().toISOString();
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  console.log(JSON.stringify({ id, status: ledger.configurations.find((item) => item.id === id)?.status, trained: ledger.configurations.filter((item) => item.status === 'trained').length, total: configurations.length }));
}
const trained = ledger.configurations.filter((item) => item.status === 'trained').length;
console.log(JSON.stringify({ policy: safetyAlignment, trained, total: configurations.length, ledger: path.relative(root, ledgerFile).split(path.sep).join('/') }, null, 2));
if (trained !== configurations.length) process.exitCode = 2;
