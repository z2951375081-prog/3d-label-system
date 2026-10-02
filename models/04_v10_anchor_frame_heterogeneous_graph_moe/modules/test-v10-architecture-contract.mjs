import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildAnchorLocalFrame } from '../lib/anchor-frame-features.mjs';
import { buildHeterogeneousLayoutGraph, decodeLocalLayoutOutput } from '../lib/heterogeneous-layout-graph.mjs';
import { buildSpatialContext } from '../lib/spatial-style-features.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidateFiles = [
  path.join(root, 'experiments', 'layout_model_v10_anchor_frame_candidate.json'),
  path.join(root, 'experiments', '_v10_relation_smoke.json'),
  path.join(root, 'experiments', '_v10_direction_clarity_probe.json'),
  path.join(root, 'experiments', '_v10_probe.json')
];

async function firstJson(files) {
  for (const file of files) {
    try { return { file, value: JSON.parse(await fs.readFile(file, 'utf8')) }; } catch {}
  }
  throw new Error('没有可用于 v10 架构契约测试的候选模型');
}

const rotateZ = (point) => [-point[1], point[0], point[2]];
const close = (left, right, tolerance = 1e-6) => Math.abs(left - right) <= tolerance;
const bounds = { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], size: [2, 2, 2], radius: Math.sqrt(3) };
const geometry = { triangles: [
  [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0.8, 0.8, 0]],
  [[-0.8, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0]],
  [[0.8, -0.8, 0], [0.8, -0.8, 0.35], [0.8, 0.8, 0.35]]
] };
const labels = [
  { id: 'a', text: 'seat', anchor: [0.45, 0.1, 0], center: [0.75, 0.25, 0.35], boxSize: [0.25, 0.12, 0.03], sourceObjs: ['seat'], targetGroups: ['body'] },
  { id: 'b', text: 'back', anchor: [-0.3, 0.2, 0], center: [-0.7, 0.35, 0.4], boxSize: [0.22, 0.1, 0.03], sourceObjs: ['back'], targetGroups: ['body'] },
  { id: 'c', text: 'leg', anchor: [0.1, -0.45, 0], center: [0.35, -0.75, 0.25], boxSize: [0.18, 0.09, 0.03], sourceObjs: ['leg'], targetGroups: ['support'] }
];

const frame = buildAnchorLocalFrame(labels[0].anchor, geometry, bounds);
const local = frame.localCoordinates(labels[0].center);
const rotatedGeometry = { triangles: geometry.triangles.map((triangle) => triangle.map(rotateZ)) };
const rotatedFrame = buildAnchorLocalFrame(rotateZ(labels[0].anchor), rotatedGeometry, bounds);
const rotatedLocal = rotatedFrame.localCoordinates(rotateZ(labels[0].center));
assert.ok(close(Math.abs(local.u), Math.abs(rotatedLocal.u), 1e-5));
assert.ok(close(Math.abs(local.v), Math.abs(rotatedLocal.v), 1e-5));
assert.ok(close(local.normal, rotatedLocal.normal, 1e-5));
assert.ok(frame.normal_stability >= 0 && frame.normal_stability <= 1);
assert.ok([1, 2].includes(frame.symmetry_equivalent_frames));

const context = buildSpatialContext(geometry, bounds, { gridSize: 12 });
const graph = buildHeterogeneousLayoutGraph(labels, geometry, bounds, context);
assert.deepEqual(graph.nodeTypes, { anchor: labels.length, label: labels.length });
assert.equal(graph.node_feature_dim, 51);
assert.equal(graph.anchor_edge_dim, 18);
assert.equal(graph.relation_edge_dim, 10);
assert.equal(graph.edge_feature_dim, 28);
assert.equal(graph.relationTypes.anchor_to_label.length, labels.length ** 2);
assert.equal(graph.relationTypes.label_to_label.length, labels.length * (labels.length - 1));
assert.equal(graph.anchorEdgeFeatures[0][1].at(-2), 0);
assert.equal(graph.anchorEdgeFeatures[0][1].at(-1), 1);

const decoded = decodeLocalLayoutOutput([0.2, -0.1, 0.3, 0.05, -0.1, 0.2], labels[0], graph.frames[0], bounds);
const decodedLocal = graph.frames[0].localCoordinates(decoded.center);
assert.ok(close(decodedLocal.u / bounds.radius, 0.2));
assert.ok(close(decodedLocal.v / bounds.radius, -0.1));
assert.ok(close(decodedLocal.normal / bounds.radius, 0.3));

const emptyDepth = { gridSize: 12, depth: new Float32Array(144), farDepth: new Float32Array(144) };
emptyDepth.depth.fill(Infinity); emptyDepth.farDepth.fill(-Infinity);
const solidDepth = { gridSize: 12, depth: new Float32Array(144), farDepth: new Float32Array(144) };
solidDepth.depth.fill(-10); solidDepth.farDepth.fill(10);
const emptyViews = Object.fromEntries(['main', 'right', 'left', 'up', 'down'].map((name) => [name, emptyDepth]));
const solidViews = Object.fromEntries(['main', 'right', 'left', 'up', 'down'].map((name) => [name, solidDepth]));
const safe = evaluateViewConditionedLayout(labels, bounds, emptyViews);
const obstructed = evaluateViewConditionedLayout(labels, bounds, solidViews);
assert.equal(safe.main_view_weight, 0.4);
assert.equal(safe.view_weights.right, 0.15);
assert.ok(Number.isFinite(safe.cvar_view_occlusion));
assert.ok(Number.isFinite(safe.worst_view_text_clarity_loss));
assert.ok(obstructed.weighted_label_object_overlap > safe.weighted_label_object_overlap);
assert.ok(obstructed.worst_view_penetration > safe.worst_view_penetration);
assert.ok(obstructed.objective > safe.objective);

const { file, value: model } = await firstJson(candidateFiles);
const architecture = model.architecture || {};
assert.equal(model.version, 'layout_model_v10_anchor_frame_heterogeneous_graph_moe');
assert.equal(architecture.type, 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe');
assert.equal(architecture.generation_input, 'pure_3d');
assert.equal(architecture.visual_generation_input, false);
assert.equal(architecture.node_input_dim, 51);
assert.equal(architecture.geometry_feature_dim, 64);
assert.equal(architecture.dgcnn_edgeconv_layers, 2);
assert.equal(architecture.anchor_label_edge_dim, 18);
assert.equal(architecture.label_label_edge_dim, 10);
assert.equal(architecture.message_passing_layers, 2);
assert.equal(architecture.transformer_layers, 1);
assert.equal(architecture.output_semantics, 'local_u_local_v_local_normal_distance_and_log_size_xyz');
assert.equal(architecture.relation_parameterization, 'separate_anchor_to_label_and_label_to_label_message_weights');
assert.equal(model.network.fusion.visual_dim, 0);
assert.equal(model.network.fusion.weights[0].length, 128);
assert.equal(model.network.message_layers.length, 2);
for (const layer of model.network.message_layers) {
  assert.equal(layer.anchor_edge_weights[0].length, 18);
  assert.equal(layer.relation_edge_weights[0].length, 10);
}
assert.equal(model.network.transformer_layers.length, 1);
assert.ok([4, 5].includes(model.network.moe.experts.length));
assert.ok(model.training.leader_direction_loss_used);
assert.ok(model.training.per_view_font_clarity_loss_used);
assert.ok(model.training.leader_crossing_loss_used);
assert.ok(model.architecture.view_loss.terms.includes('leader_crossing'));
assert.ok(model.architecture.view_loss.leader_crossing_weight > 0);
assert.ok(Number.isFinite(model.training.metrics.val.leader_crossing));
assert.equal(model.training.visual_branch_used_in_generation, false);

console.log(JSON.stringify({
  ok: true,
  model_file: path.relative(root, file).replaceAll('\\', '/'),
  architecture: {
    node_dim: architecture.node_input_dim,
    geometry_dim: architecture.geometry_feature_dim,
    anchor_edge_dim: architecture.anchor_label_edge_dim,
    label_edge_dim: architecture.label_label_edge_dim,
    message_layers: architecture.message_passing_layers,
    transformer_layers: architecture.transformer_layers,
    experts: model.network.moe.experts.length,
    output: architecture.output_semantics
  },
  rotation_local_coordinates: { original: local, rotated: rotatedLocal },
  view_objectives: { safe: safe.objective, obstructed: obstructed.objective }
}, null, 2));
