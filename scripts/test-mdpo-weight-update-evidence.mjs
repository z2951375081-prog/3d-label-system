import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { initializeV10BiasTuning, initializeV10Lora, materializeV10BiasTuning, materializeV10Lora } from '../lib/mdpo-lora.mjs';
import { evaluateMdpoWeightUpdateEvidence } from '../lib/mdpo-weight-update-evidence.mjs';
import { buildV10TrainingGraphs, forwardV10TrainingGraph } from './train-3d-human-style-layout-model.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const model = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'layout_model.json'), 'utf8'));
const completeManifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const manifest = { samples: [completeManifest.samples.find((sample) => sample.split === 'train')] };
const graph = (await buildV10TrainingGraphs(manifest, 'train', 20, 'v10'))[0];
const adapters = initializeV10Lora(model.network, { rank: 4, alpha: 8, seed: 17 });
const biases = initializeV10BiasTuning(model.network);
adapters.matrices[0].B[0][0] = 0.01;
biases.biases[0].delta[0] = 0.02;
const trained = materializeV10BiasTuning(materializeV10Lora(model.network, adapters), biases);
const evidence = evaluateMdpoWeightUpdateEvidence({ referenceNetwork: model.network, trainedNetwork: trained,
  adapters, biases, graph, forward: forwardV10TrainingGraph });
assert.equal(evidence.matrix_count, 19);
assert.equal(evidence.trainable_parameters, 7676);
assert.equal(evidence.bias_deltas[0].changed_biases, 1);
assert.equal(evidence.matrices[0].changed_weights > 0, true);
assert.equal(evidence.ablation.effective, true);
assert.ok(evidence.trainable_fraction > 0 && evidence.trainable_fraction < 1);
const corrupt = structuredClone(trained);
corrupt.moe.router.bias[0] += 1;
assert.throws(() => evaluateMdpoWeightUpdateEvidence({ referenceNetwork: model.network, trainedNetwork: corrupt,
  adapters, biases, graph, forward: forwardV10TrainingGraph }), /tuned bias/);
console.log('v10-MDPO per-module update, parameter count and LoRA ablation evidence tests passed.');
