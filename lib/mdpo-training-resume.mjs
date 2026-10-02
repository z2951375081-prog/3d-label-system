import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { MDPO_UNIFIED_NINE_METRICS } from './mdpo-unified-training-metrics.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
export const MDPO_RESUME_HYPERPARAMETERS = Object.freeze(['epochs', 'learningRate', 'beta', 'lambdaMulti', 'lambdaDpo', 'lambdaKl',
  'lambdaSup', 'lambdaSafe', 'lambdaCenterTail', 'rank', 'alpha', 'dropout', 'clipNorm', 'compositionWeight', 'hierarchyWeight',
  'balanceWeight', 'manualStyleWeight', 'textWeight', 'leaderWeight', 'learnableVariance', 'noReference', 'varianceLearningRate',
  'varianceMinimum', 'varianceMaximum', 'patience', 'minDelta', 'seed', 'styleGridSize']);

export function mdpoRng(seed, restoredState = null) {
  let state = restoredState === null ? (Number(seed) >>> 0) || 17 : Number(restoredState);
  if (!Number.isInteger(state) || state < 0 || state > 0xffffffff) throw new Error('Invalid MDPO resumed random state');
  const next = () => { state = (1664525 * state + 1013904223) >>> 0; return state / 2 ** 32; };
  next.state = () => state;
  return next;
}

export async function latestMdpoCheckpoint(checkpointDir) {
  const directory = path.resolve(checkpointDir);
  const names = await fs.readdir(directory).catch((error) => { if (error?.code === 'ENOENT') return []; throw error; });
  const checkpoints = names.filter((name) => /^epoch_\d{3}\.json$/.test(name)).sort();
  return checkpoints.length ? path.join(directory, checkpoints.at(-1)) : null;
}

export async function restoreMdpoTrainingState({ checkpointFile, checkpointDir, root, referenceHash, datasetHash, options }) {
  const file = path.resolve(checkpointFile), directory = path.resolve(checkpointDir);
  if (path.dirname(file) !== directory || !/^epoch_\d{3}\.json$/.test(path.basename(file))) throw new Error('Resume checkpoint is outside this training run');
  const bytes = await fs.readFile(file), restored = JSON.parse(bytes.toString('utf8'));
  if (restored.version !== 'v10_mdpo_checkpoint_v1' || restored.reference_model_sha256 !== referenceHash
      || restored.dataset_sha256 !== datasetHash || !Number.isInteger(restored.epoch) || restored.epoch < 1
      || path.basename(file) !== `epoch_${String(restored.epoch).padStart(3, '0')}.json`)
    throw new Error('Resume checkpoint provenance mismatch');
  for (const name of MDPO_RESUME_HYPERPARAMETERS) {
    if (restored.hyperparameters?.[name] !== options[name]) throw new Error(`Resume checkpoint hyperparameter mismatch: ${name}`);
  }
  if (!Number.isInteger(restored.random_state) || !restored.best_state?.epoch
      || !Number.isFinite(restored.best_state?.val?.total) || !Number.isInteger(restored.stale_epochs)
      || restored.stale_epochs < 0 || restored.stale_epochs > options.patience || restored.epoch !== restored.history?.length
      || !Array.isArray(restored.ledger) || restored.ledger.length !== restored.epoch - 1)
    throw new Error('Resume checkpoint lacks complete deterministic training state');
  const previousLedger = restored.ledger;
  for (const entry of previousLedger) {
    const checkedFile = path.resolve(root, String(entry.file || ''));
    if (path.dirname(checkedFile) !== directory || path.basename(checkedFile) !== `epoch_${String(entry.epoch).padStart(3, '0')}.json`
        || digest(await fs.readFile(checkedFile)) !== entry.sha256)
      throw new Error(`Prior immutable checkpoint changed during resume: ${entry.epoch}`);
  }
  const entry = { epoch: restored.epoch, file: path.relative(root, file).split(path.sep).join('/'),
    sha256: digest(bytes), val_total: restored.history.at(-1).val.total };
  const ledger = [...previousLedger, entry];
  if (ledger.some((row, index) => row.epoch !== index + 1 || !Number.isFinite(row.val_total))
      || restored.best_state.epoch > restored.epoch || restored.best_state.epoch < 1)
    throw new Error('Resume checkpoint ledger is incomplete or inconsistent');
  const bestCheckpoint = ledger[restored.best_state.epoch - 1];
  if (restored.best_state.val.total !== bestCheckpoint.val_total) throw new Error('Resume best checkpoint and ledger mismatch');
  return { adapters: restored.adapters, biases: restored.biases, variance: restored.variance,
    startEpoch: restored.epoch + 1, history: restored.history, ledger, best: { ...restored.best_state, checkpoint: bestCheckpoint },
    bestVal: restored.best_state.val.total, stale: restored.stale_epochs, randomState: restored.random_state };
}

export async function validateCompletedMdpoTrainingRun({ outputDir, root, referenceHash, datasetHash, options }) {
  const directory = path.resolve(outputDir);
  const [candidateBytes, ledgerBytes] = await Promise.all([
    fs.readFile(path.join(directory, 'layout_model_v10_mdpo_candidate.json')),
    fs.readFile(path.join(directory, 'checkpoint_ledger.json'))
  ]);
  const candidate = JSON.parse(candidateBytes.toString('utf8'));
  const ledger = JSON.parse(ledgerBytes.toString('utf8'));
  if (candidate.version !== 'layout_model_v10_mdpo_candidate' || candidate.status !== 'diagnostic_only_requires_full_val11_gate'
      || candidate.reference?.sha256 !== referenceHash || candidate.mdpo?.dataset_sha256 !== datasetHash
      || ledger.version !== 'v10_mdpo_checkpoint_ledger_v1' || ledger.reference_model_sha256 !== referenceHash
      || ledger.dataset_sha256 !== datasetHash || ledger.selected_without_test !== true
      || !Array.isArray(ledger.checkpoints) || !ledger.checkpoints.length
      || JSON.stringify(candidate.mdpo.checkpoint_ledger) !== JSON.stringify(ledger.checkpoints)
      || candidate.mdpo.best_epoch !== ledger.best?.epoch || candidate.architecture?.qwen_inference_input !== false
      || candidate.mdpo.preference_holdout?.heldout_pair_count !== 33
      || candidate.mdpo.preference_holdout?.gradient_pair_count < 165
      || !/^[a-f0-9]{64}$/.test(candidate.mdpo.preference_holdout?.holdout_sha256 || '')
      || candidate.mdpo.weight_update_evidence?.matrix_count !== 19
      || candidate.mdpo.weight_update_evidence?.trainable_parameters !== candidate.mdpo.trainable_parameters
      || Object.keys(candidate.mdpo.weight_update_evidence?.modules || {}).length !== 7
      || Object.values(candidate.mdpo.weight_update_evidence?.modules || {}).some((module) => module.changed_matrices < 1)
      || candidate.mdpo.weight_update_evidence?.ablation?.effective !== true
      || candidate.mdpo.weight_update_evidence?.matrices?.some((row) => !Number.isFinite(row.maximum_absolute_update))
      || !Number.isFinite(candidate.mdpo.hyperparameters?.lambdaSafe) || candidate.mdpo.hyperparameters.lambdaSafe < 0
      || !String(candidate.mdpo.objective?.geometry_safe || '').includes('lambdaSafe')
      || !Array.isArray(candidate.mdpo.history) || !candidate.mdpo.history.length
      || candidate.mdpo.history.some((row) => !Number.isFinite(row.train?.geometry_safe) || row.train.geometry_safe < 0)
      || candidate.mdpo.train_unified_metrics?.baseline?.sample_count !== 33
      || candidate.mdpo.train_unified_metrics?.candidate?.sample_count !== 33
      || Object.keys(candidate.mdpo.train_unified_metrics?.delta || {}).length !== MDPO_UNIFIED_NINE_METRICS.length
      || MDPO_UNIFIED_NINE_METRICS.some((name) => !Number.isFinite(candidate.mdpo.train_unified_metrics?.baseline?.metrics?.[name])
        || !Number.isFinite(candidate.mdpo.train_unified_metrics?.candidate?.metrics?.[name])
        || !Number.isFinite(candidate.mdpo.train_unified_metrics?.delta?.[name]))
      || candidate.split_policy?.train33_preference_holdout_no_gradient !== true)
    throw new Error('Completed MDPO training candidate/ledger provenance invalid');
  for (const name of MDPO_RESUME_HYPERPARAMETERS) {
    if (Object.hasOwn(options, name) && candidate.mdpo.hyperparameters?.[name] !== options[name])
      throw new Error(`Completed MDPO training hyperparameter mismatch: ${name}`);
  }
  for (const [index, entry] of ledger.checkpoints.entries()) {
    const checkedFile = path.resolve(root, String(entry.file || ''));
    if (path.dirname(checkedFile) !== path.join(directory, 'checkpoints') || entry.epoch !== index + 1
        || path.basename(checkedFile) !== `epoch_${String(entry.epoch).padStart(3, '0')}.json`
        || digest(await fs.readFile(checkedFile)) !== entry.sha256) throw new Error(`Completed MDPO training checkpoint corrupted: ${entry.epoch}`);
  }
  if (JSON.stringify(ledger.best?.checkpoint) !== JSON.stringify(ledger.checkpoints[ledger.best.epoch - 1]))
    throw new Error('Completed MDPO training best checkpoint changed');
  return candidate;
}
