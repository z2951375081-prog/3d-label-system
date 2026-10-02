import assert from 'node:assert/strict';
import { LOCAL_LAYOUT_OUTPUT_DIM, STYLE_ADAPTER_KEYS, runStyleAdapterMoE, styleAdapterMoEContract } from '../lib/style-adapter-moe.mjs';
import { annealStylePreservingLayout, fiveViewSafetyGate, stylePreservationLoss } from '../lib/style-preserving-annealer.mjs';

const hiddenDim = 8;
const styleDim = 4;
const adapterDim = 3;
const matrix = (rows, columns, scale = 0.01) => Array.from({ length: rows }, (_, row) => Array.from({ length: columns }, (_, column) => scale * ((row + 1) * (column + 2))));
const vector = (length, value = 0) => Array.from({ length }, () => value);
const moe = {
  style_adapter: { enabled: true, version: 'style_adapter_moe_v1', style_embedding_dim: styleDim, adapter_dim: adapterDim },
  style_embeddings: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
  router: { weights: matrix(STYLE_ADAPTER_KEYS.length, hiddenDim, 0.02), bias: [0, 0.1, -0.1] },
  experts: STYLE_ADAPTER_KEYS.map((styleKey, expertIndex) => ({
    name: `${styleKey}_expert`,
    style_key: styleKey,
    style_adapter: { down_weights: matrix(adapterDim, hiddenDim + styleDim, 0.01), down_bias: vector(adapterDim), up_weights: matrix(hiddenDim, adapterDim, 0.01), up_bias: vector(hiddenDim) },
    hidden_weights: matrix(hiddenDim, hiddenDim, 0.01),
    hidden_bias: vector(hiddenDim, expertIndex * 0.01),
    weights: matrix(LOCAL_LAYOUT_OUTPUT_DIM, hiddenDim, 0.01),
    bias: vector(LOCAL_LAYOUT_OUTPUT_DIM, expertIndex * 0.02)
  }))
};
const hidden = Array.from({ length: hiddenDim }, (_, index) => (index + 1) / hiddenDim);
const blended = runStyleAdapterMoE(hidden, moe);
assert.equal(blended.output.length, LOCAL_LAYOUT_OUTPUT_DIM);
assert.equal(blended.expertOutputs.length, 3);
assert.ok(Math.abs(blended.routerWeights.reduce((sum, value) => sum + value, 0) - 1) < 1e-8);
assert.equal(blended.fusionMode, 'router_blend');
const forced = runStyleAdapterMoE(hidden, moe, { styleExpert: 'box_sides' });
assert.deepEqual(forced.mixtureWeights, [0, 1, 0]);
assert.equal(forced.selectedStyle, 'box_sides');
const explicit = runStyleAdapterMoE(hidden, moe, { styleWeights: { radial_ring: 2, box_sides: 1, anchor_adaptive: 1 } });
assert.deepEqual(explicit.mixtureWeights, [0.5, 0.25, 0.25]);
assert.equal(styleAdapterMoEContract(moe).output_dim, 6);

const bounds = { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], size: [2, 2, 2], radius: Math.sqrt(3) };
const labels = [
  { id: 'a', text: 'a', anchor: [0.4, 0, 0], center: [0.75, 0.2, 0.2], boxSize: [0.22, 0.09, 0.02] },
  { id: 'b', text: 'b', anchor: [-0.35, 0.1, 0], center: [-0.7, 0.25, 0.22], boxSize: [0.20, 0.08, 0.02] },
  { id: 'c', text: 'c', anchor: [0, -0.4, 0], center: [0.15, -0.75, 0.18], boxSize: [0.18, 0.08, 0.02] }
];
const emptyDepth = { gridSize: 12, depth: new Float32Array(144), farDepth: new Float32Array(144) };
emptyDepth.depth.fill(Infinity); emptyDepth.farDepth.fill(-Infinity);
const depthGrids = Object.fromEntries(['main', 'right', 'left', 'up', 'down'].map((view) => [view, emptyDepth]));
const moved = structuredClone(labels);
moved[0].center[0] += 0.1;
assert.ok(stylePreservationLoss(moved, labels, bounds) > 0);
const gateOptions = { maxWorstViewTextClarityLoss: 0.8 };
assert.equal(fiveViewSafetyGate(labels, bounds, depthGrids, gateOptions).safe, true);
const annealed = annealStylePreservingLayout(labels, bounds, { depthGrids, styleReference: labels, iterations: 12, seed: 7, hardSafetyGate: true, ...gateOptions });
assert.equal(annealed.labels.length, labels.length);
assert.equal(annealed.safety.views.length, 5);
assert.equal(annealed.accepted, true);

console.log(JSON.stringify({ ok: true, architecture: { shared_hidden_dim: hiddenDim, style_embedding_dim: styleDim, expert_count: 3, expert_output_dim: 6, fusion_modes: ['router_blend', 'explicit_blend', 'forced'] }, five_view_gate: annealed.safety.safe }, null, 2));

