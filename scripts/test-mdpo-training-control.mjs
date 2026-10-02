import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { MDPO_UI_TRAINING_DEFAULTS, normalizeMdpoUiTrainingRequest, mdpoUiTrainingArgs } from '../lib/mdpo-training-control.mjs';
import { MDPO_WEIGHTS } from '../lib/mdpo-continuous-policy.mjs';

const normalized = normalizeMdpoUiTrainingRequest({ model: 'v10_mdpo', epochs: 42, learningRate: 5e-5,
  rank: 8, textWeight: 0.3, leaderWeight: 0.3, lambdaSafe: 2 });
assert.equal(normalized.epochs, 42);
assert.equal(normalized.rank, 8);
assert.equal(normalized.lambdaSafe, 2);
assert.equal(normalized.compositionWeight, MDPO_WEIGHTS.composition_harmony);
assert.equal(MDPO_UI_TRAINING_DEFAULTS.textWeight, MDPO_WEIGHTS.text_clarity);
const argv = mdpoUiTrainingArgs(normalized, 'independent-candidate');
assert.equal(argv[0], 'scripts/train-v10-mdpo.mjs');
assert.deepEqual(argv.slice(1, 3), ['--outputDir', 'independent-candidate']);
assert.deepEqual(argv.slice(argv.indexOf('--learningRate'), argv.indexOf('--learningRate') + 2), ['--learningRate', '0.00005']);
assert(!argv.includes('--allowIncomplete'));
for (const bad of [
  { model: 'v10' }, { rank: -1 }, { rank: 2.5 }, { epochs: 0 }, { beta: NaN }, { lambdaSafe: 0 },
  { dataset: 'test11' }, { reference: 'other-model.json' }, { allowIncomplete: true },
  { outputDir: '.' }, { lambdaKl: 0.01, noReference: true },
  { lambdaMulti: 1, compositionWeight: 0, hierarchyWeight: 0, balanceWeight: 0,
    manualStyleWeight: 0, textWeight: 0, leaderWeight: 0 }
]) assert.throws(() => normalizeMdpoUiTrainingRequest(bad));
const html = await fs.readFile(new URL('../public/index.html', import.meta.url), 'utf8');
const app = await fs.readFile(new URL('../public/app.js', import.meta.url), 'utf8');
const server = await fs.readFile(new URL('../server.mjs', import.meta.url), 'utf8');
assert.match(html, /id="mdpoStartTrainingBtn"/);
assert.match(html, /id="mdpoTrainModel"/);
assert.match(app, /bindMdpoUiTrainingControls\(\);/);
assert.match(server, /mdpoTrainingDatasetReadiness\(\)/);
assert.match(server, /validateMdpoDataset\(dataset, manifest, \{ requireComplete: true \}\)/);
assert.match(server, /blocks_manual_training: Boolean\(ledger && !terminal\)/);
assert.match(server, /status: 'queued'/);
assert.match(server, /dispatchQueuedMdpoUiTraining\(\)/);
assert.match(app, /81-grid 训练已完成/);
assert.match(app, /训练完成 · epoch/);
assert.match(server, /validateCompletedMdpoTrainingRun\(/);
assert.match(server, /activation_policy: 'diagnostic_only_requires_full_val11_gate'/);
assert.match(server, /url\.pathname === '\/api\/mdpo-training-control' && req\.method === 'POST'/);
console.log('MDPO UI training control parameters, isolation, safety preflight and wiring tests passed.');
