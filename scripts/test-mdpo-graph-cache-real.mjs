import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadMdpoTrainingGraphs } from '../lib/mdpo-graph-cache.mjs';
import { buildV10TrainingGraphs, forwardV10TrainingGraph, v10GeometrySafetyLoss, v10SupervisedObjectiveLoss, v10SupervisedOutputGradient } from './train-3d-human-style-layout-model.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fullManifest = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'dataset_manifest.json'), 'utf8'));
const manifest = { samples: ['train', 'val'].map((split) => fullManifest.samples.find((sample) => sample.split === split)) };
const manifestBytes = Buffer.from(JSON.stringify(manifest));
const model = JSON.parse(await fs.readFile(path.join(root, 'experiments', 'layout_model.json'), 'utf8'));
const scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'v10-mdpo-real-graph-cache-test-'));
try {
  const params = { root, manifest, manifestBytes, gridSize: 20, directory: scratch, buildGraphs: buildV10TrainingGraphs };
  const first = await loadMdpoTrainingGraphs(params);
  const second = await loadMdpoTrainingGraphs(params);
  assert.equal(first.cache.status, 'built');
  assert.equal(second.cache.status, 'hit');
  for (const split of ['trainGraphs', 'valGraphs']) {
    assert.equal(first[split].length, 1);
    assert.equal(second[split].length, 1);
    const before = forwardV10TrainingGraph(model.network, first[split][0]).output;
    const after = forwardV10TrainingGraph(model.network, second[split][0]).output;
    assert.deepEqual(after, before, `${split} cached graph must preserve frozen v10 output`);
    const raw = first[split][0].depthGrids.main.depth;
    const restored = second[split][0].depthGrids.main.depth;
    assert.ok(raw instanceof Float32Array && restored instanceof Float32Array);
    assert.deepEqual(restored, raw, `${split} depth grid must preserve all infinity sentinels`);
  }
  const graph = first.trainGraphs[0], run = forwardV10TrainingGraph(model.network, graph);
  const safetyOptions = { baseWeight: 0, styleWeight: 0, directionWeight: 0, viewWeight: 0.25,
    worstViewWeight: 2, cvarViewWeight: 1, stereoWeight: 1, textClarityWeight: 3, leaderCrossingWeight: 3.5 };
  const disabled = v10SupervisedOutputGradient(run, graph, { ...safetyOptions, viewWeight: 0 });
  assert.ok(disabled.flat().every((value) => value === 0), 'base/style/direction/view disabled must yield zero output gradient');
  const safetyGradient = v10SupervisedOutputGradient(run, graph, safetyOptions);
  assert.ok(safetyGradient.flat().every(Number.isFinite));
  assert.ok(safetyGradient.flat().some((value) => Math.abs(value) > 1e-12), 'independent geometry-safe objective must reach v10 outputs');
  assert.ok(Number.isFinite(v10GeometrySafetyLoss(run.output, graph, safetyOptions)));
  const safetyAxis = safetyGradient.flat().findIndex((value) => Math.abs(value) > 1e-12);
  const safetyNode = Math.floor(safetyAxis / 6), safetyCoordinate = safetyAxis % 6;
  const safetyEpsilon = 1e-4;
  const safetyAbove = run.output.map((row) => [...row]), safetyBelow = run.output.map((row) => [...row]);
  safetyAbove[safetyNode][safetyCoordinate] += safetyEpsilon;
  safetyBelow[safetyNode][safetyCoordinate] -= safetyEpsilon;
  const numericSafetyGradient = (v10GeometrySafetyLoss(safetyAbove, graph, safetyOptions)
    - v10GeometrySafetyLoss(safetyBelow, graph, safetyOptions)) / (2 * safetyEpsilon);
  assert.ok(Math.abs(numericSafetyGradient - safetyGradient[safetyNode][safetyCoordinate]) < 1e-7,
    'independent geometry-safe loss curve must match its actual output gradient');
  const supervisedOptions = { baseWeight: 1, styleWeight: 0.2, directionWeight: 1.25, viewWeight: 0 };
  const supervisedLoss = v10SupervisedObjectiveLoss(run.output, graph, supervisedOptions);
  assert.ok(Number.isFinite(supervisedLoss.total) && supervisedLoss.total >= 0);
  assert.ok(['regression', 'style', 'leader_direction'].every((name) => Number.isFinite(supervisedLoss[name])));
  const supervisedGradient = v10SupervisedOutputGradient(run, graph, supervisedOptions);
  for (const axis of [0, 3]) {
    const epsilon = 1e-4;
    const above = run.output.map((row) => [...row]);
    const below = run.output.map((row) => [...row]);
    above[0][axis] += epsilon;
    below[0][axis] -= epsilon;
    const numeric = (v10SupervisedObjectiveLoss(above, graph, supervisedOptions).total
      - v10SupervisedObjectiveLoss(below, graph, supervisedOptions).total) / (2 * epsilon);
    assert.ok(Math.abs(supervisedGradient[0][axis] - numeric) < 1e-5,
      `supervised loss curve and actual gradient must agree on local axis ${axis}`);
  }
  console.log('v10-MDPO real train/val graph cache frozen-policy forward-equivalence passed.');
} finally {
  if (!path.resolve(scratch).startsWith(path.resolve(os.tmpdir()) + path.sep) || !path.basename(scratch).startsWith('v10-mdpo-real-graph-cache-test-')) throw new Error('Unsafe real graph cache test cleanup target');
  await fs.rm(scratch, { recursive: true, force: true });
}
