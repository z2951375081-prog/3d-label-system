import assert from 'node:assert/strict';
import { cameraBasisForView, evaluateLayout } from '../lib/layout-optimizer.mjs';

const bounds = { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], size: [2, 2, 2], radius: 1 };
const basis = cameraBasisForView(bounds, 'main');
const depth = new Float32Array(64 * 64); depth.fill(Infinity);
const farDepth = new Float32Array(64 * 64); farDepth.fill(-Infinity);
const depthGrid = { depth, farDepth, gridSize: 64 };
const point = (angle, radius = 1.2) => bounds.center.map((value, axis) => value + basis.right[axis] * Math.cos(angle) * radius + basis.up[axis] * Math.sin(angle) * radius);
const labels = (angles) => angles.map((angle, index) => ({ id: `label-${index}`, text: `L${index}`, anchor: [...bounds.center], center: point(angle), boxSize: [0.035, 0.018, 0.005], bendPoints: [] }));
const options = { viewPolicy: 'single', depthGrids: { main: depthGrid } };
const distributed = evaluateLayout(labels(Array.from({ length: 8 }, (_, index) => index * Math.PI / 4)), bounds, options);
const concentrated = evaluateLayout(labels(Array.from({ length: 8 }, (_, index) => -0.18 + index * 0.05)), bounds, options);

assert.ok(distributed.directional_uniformity > 0.99, 'eight directions should be near-uniform');
assert.ok(distributed.directional_allocation_mismatch < 0.15, 'uniform labels should approximately match rasterized uniform free space');
assert.ok(concentrated.directional_allocation_mismatch > distributed.directional_allocation_mismatch + 0.5, 'one-direction cluster must have larger allocation mismatch');
assert.ok(concentrated.directional_concentration_excess > 0, 'one-direction cluster must trigger concentration excess');
assert.ok(concentrated.objective_score > distributed.objective_score, 'directional density energy must penalize clustered labels');
console.log(JSON.stringify({ status: 'passed', distributed: { mismatch: distributed.directional_allocation_mismatch, concentration: distributed.directional_concentration_excess, uniformity: distributed.directional_uniformity, objective: distributed.objective_score }, concentrated: { mismatch: concentrated.directional_allocation_mismatch, concentration: concentrated.directional_concentration_excess, uniformity: concentrated.directional_uniformity, objective: concentrated.objective_score } }, null, 2));
