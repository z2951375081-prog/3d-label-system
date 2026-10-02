import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { MDPO_RESUME_HYPERPARAMETERS, latestMdpoCheckpoint, mdpoRng, restoreMdpoTrainingState, validateCompletedMdpoTrainingRun } from '../lib/mdpo-training-resume.mjs';

const random = mdpoRng(17018);
random(); random();
const restoredRandom = mdpoRng(17018, random.state());
assert.equal(restoredRandom(), random());
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'v10-mdpo-resume-test-'));
const checkpointDir = path.join(root, 'checkpoints');
await fs.mkdir(checkpointDir);
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const referenceHash = 'a'.repeat(64), datasetHash = 'b'.repeat(64);
const options = Object.fromEntries(MDPO_RESUME_HYPERPARAMETERS.map((name) => [name, ['learnableVariance', 'noReference'].includes(name) ? false : 1]));
const best = { epoch: 1, val: { total: 0.4 }, adapters: { best: true }, biases: {}, variance: {} };
const base = { version: 'v10_mdpo_checkpoint_v1', reference_model_sha256: referenceHash, dataset_sha256: datasetHash,
  hyperparameters: options, adapters: { current: true }, biases: {}, variance: {}, random_state: random.state() };
try {
  const file1 = path.join(checkpointDir, 'epoch_001.json');
  const checkpoint1 = { ...base, epoch: 1, history: [{ val: { total: 0.4 } }], ledger: [], best_state: best, stale_epochs: 0 };
  const bytes1 = Buffer.from(JSON.stringify(checkpoint1));
  await fs.writeFile(file1, bytes1, { flag: 'wx' });
  const file2 = path.join(checkpointDir, 'epoch_002.json');
  const checkpoint2 = { ...base, epoch: 2, history: [...checkpoint1.history, { val: { total: 0.6 } }],
    ledger: [{ epoch: 1, file: 'checkpoints/epoch_001.json', sha256: sha(bytes1), val_total: 0.4 }], best_state: best, stale_epochs: 1 };
  await fs.writeFile(file2, JSON.stringify(checkpoint2), { flag: 'wx' });
  assert.equal(await latestMdpoCheckpoint(checkpointDir), file2);
  const input = { checkpointFile: file2, checkpointDir, root, referenceHash, datasetHash, options };
  const state = await restoreMdpoTrainingState(input);
  assert.equal(state.startEpoch, 3);
  assert.equal(state.best.epoch, 1);
  assert.equal(state.best.checkpoint.sha256, sha(bytes1));
  assert.equal(state.bestVal, 0.4);
  assert.equal(state.stale, 1);
  assert.equal(state.ledger.length, 2);
  assert.equal(mdpoRng(1, state.randomState)(), random());
  const candidate = { version: 'layout_model_v10_mdpo_candidate', status: 'diagnostic_only_requires_full_val11_gate',
    reference: { sha256: referenceHash }, architecture: { qwen_inference_input: false },
    split_policy: { train33_preference_holdout_no_gradient: true },
    mdpo: { dataset_sha256: datasetHash, hyperparameters: options, objective: { geometry_safe: 'lambdaSafe * geometry' },
      history: [{ train: { geometry_safe: 0.2 } }], checkpoint_ledger: state.ledger, best_epoch: 1,
      trainable_parameters: 7676,
      preference_holdout: { heldout_pair_count: 33, gradient_pair_count: 165, holdout_sha256: 'f'.repeat(64) },
      weight_update_evidence: { matrix_count: 19, trainable_parameters: 7676, ablation: { effective: true },
        modules: Object.fromEntries(Array.from({ length: 7 }, (_, index) => [`module_${index}`, { changed_matrices: 1 }])),
        matrices: [{ maximum_absolute_update: 0.1 }] },
      train_unified_metrics: { baseline: { sample_count: 33, metrics: Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'].map((name) => [name, 0])) },
        candidate: { sample_count: 33, metrics: Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'].map((name) => [name, 0])) },
        delta: Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'].map((name) => [name, 0])) } } };
  await fs.writeFile(path.join(root, 'layout_model_v10_mdpo_candidate.json'), JSON.stringify(candidate), { flag: 'wx' });
  await fs.writeFile(path.join(root, 'checkpoint_ledger.json'), JSON.stringify({ version: 'v10_mdpo_checkpoint_ledger_v1',
    reference_model_sha256: referenceHash, dataset_sha256: datasetHash, selected_without_test: true,
    checkpoints: state.ledger, best: { ...best, checkpoint: state.ledger[0] } }), { flag: 'wx' });
  assert.equal((await validateCompletedMdpoTrainingRun({ outputDir: root, root, referenceHash, datasetHash, options })).mdpo.best_epoch, 1);
  await assert.rejects(() => validateCompletedMdpoTrainingRun({ outputDir: root, root, referenceHash, datasetHash,
    options: { beta: 2 } }), /hyperparameter mismatch/);
  await assert.rejects(() => restoreMdpoTrainingState({ ...input, options: { ...options, beta: 2 } }), /hyperparameter mismatch/);
  await assert.rejects(() => restoreMdpoTrainingState({ ...input, options: { ...options, lambdaSafe: 2 } }), /lambdaSafe/);
  await assert.rejects(() => restoreMdpoTrainingState({ ...input, referenceHash: 'c'.repeat(64) }), /provenance mismatch/);
  await fs.writeFile(file1, '{"changed":true}');
  await assert.rejects(() => restoreMdpoTrainingState(input), /Prior immutable checkpoint changed/);
  await assert.rejects(() => validateCompletedMdpoTrainingRun({ outputDir: root, root, referenceHash, datasetHash, options }), /checkpoint corrupted/);
  const [sweepSource, ablationSource] = await Promise.all([
    fs.readFile(new URL('./run-v10-mdpo-sweep.mjs', import.meta.url), 'utf8'),
    fs.readFile(new URL('./run-v10-mdpo-ablations.mjs', import.meta.url), 'utf8')
  ]);
  for (const [name, source] of [['81-grid', sweepSource], ['ablation', ablationSource]]) {
    assert.match(source, /status === 'trained'\)\) \{[\s\S]*?await validateCompletedMdpoTrainingRun\([\s\S]*?continue;/,
      `${name} resume must revalidate ledger-marked trained artifacts before skipping them`);
  }
  console.log('v10-MDPO exact random/best/ledger training resume and tamper refusal tests passed.');
} finally {
  const resolved = path.resolve(root), prefix = path.resolve(os.tmpdir()) + path.sep;
  if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('v10-mdpo-resume-test-')) throw new Error('Unsafe MDPO resume test cleanup target');
  await fs.rm(resolved, { force: true, recursive: true });
}
