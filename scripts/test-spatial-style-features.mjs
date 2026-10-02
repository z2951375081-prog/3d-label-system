import assert from 'node:assert/strict';
import { buildSpatialContext, computeSpatialStyleMetrics, pure3DLabelFeatures, PURE_3D_LABEL_FEATURE_NAMES } from '../lib/spatial-style-features.mjs';

const bounds = { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], size: [2, 2, 2], radius: Math.sqrt(3) };
const geometry = { triangles: [[[-0.4, -0.4, -0.4], [0.4, -0.4, -0.4], [0, 0.4, 0.4]]] };
const context = buildSpatialContext(geometry, bounds, { gridSize: 12 });
const labels = [
  { id: 'a', text: 'A', anchor: [0, 0, 0], center: [0.8, 0.8, 0.8], boxSize: [0.2, 0.1, 0.05], sourceObjs: ['part-a'], targetGroups: ['part-a'] },
  { id: 'b', text: 'B', anchor: [0, 0, 0], center: [-0.8, -0.8, 0.8], boxSize: [0.25, 0.1, 0.05], sourceObjs: ['part-b'], targetGroups: ['part-b'] }
];
const features = pure3DLabelFeatures(labels[0], bounds, 0, labels.length, context, labels);
const metrics = computeSpatialStyleMetrics(labels, bounds, context);
assert.equal(PURE_3D_LABEL_FEATURE_NAMES.length, 51);
assert.equal(features.length, 51);
assert.ok(features.every(Number.isFinite));
for (const key of ['air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization', 'mean_3d_clearance', 'min_3d_spacing', 'directional_uniformity']) assert.ok(Number.isFinite(metrics[key]), key);
console.log(JSON.stringify({ ok: true, feature_dim: features.length, grid_size: context.gridSize, metrics }, null, 2));
