import assert from 'node:assert/strict';
import { buildAnchorLocalFrame, anchorLabelEdgeFeatures, labelRelationEdgeFeatures, ANCHOR_LABEL_EDGE_FEATURE_NAMES, LABEL_RELATION_EDGE_FEATURE_NAMES } from '../lib/anchor-frame-features.mjs';
import { ANCHOR_LOCAL_LABEL_FEATURE_NAMES, anchorLocal3DLabelFeatures, buildSpatialContext } from '../lib/spatial-style-features.mjs';
import { evaluateViewConditionedLayout } from '../lib/view-conditioned-evaluator.mjs';
import { cameraBasisForView, repairLeaderCrossings } from '../lib/layout-optimizer.mjs';
import { buildHeterogeneousLayoutGraph } from '../lib/heterogeneous-layout-graph.mjs';

const bounds = { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], size: [2, 2, 2], radius: Math.sqrt(3) };
const geometry = { triangles: [
  [[-0.7, -0.7, 0], [0.7, -0.7, 0], [0.7, 0.7, 0]],
  [[-0.7, -0.7, 0], [0.7, 0.7, 0], [-0.7, 0.7, 0]]
] };
const labels = [
  { id: 'a', text: 'A', anchor: [0, 0, 0], center: [0.5, 0.2, 0.5], boxSize: [0.2, 0.1, 0.03], sourceObjs: ['part-a'], targetGroups: ['group-a'] },
  { id: 'b', text: 'B', anchor: [0.2, 0.1, 0], center: [-0.5, -0.2, 0.6], boxSize: [0.2, 0.1, 0.03], sourceObjs: ['part-b'], targetGroups: ['group-b'] }
];
const frame = buildAnchorLocalFrame(labels[0].anchor, geometry, bounds);
const context = buildSpatialContext(geometry, bounds, { gridSize: 12 });
const heterogeneous = buildHeterogeneousLayoutGraph(labels, geometry, bounds, context);
const features = anchorLocal3DLabelFeatures(labels[0], frame, bounds, 0, labels.length, context, labels);
const anchorEdge = anchorLabelEdgeFeatures(labels[0], frame, bounds);
const labelEdge = labelRelationEdgeFeatures(labels[0], labels[1], bounds);
const depth = { gridSize: 8, depth: new Float32Array(64), farDepth: new Float32Array(64) };
depth.depth.fill(Infinity); depth.farDepth.fill(-Infinity);
const viewMetrics = evaluateViewConditionedLayout(labels, bounds, { main: depth, right: depth, left: depth, up: depth, down: depth });
const solidDepth = { gridSize: 8, depth: new Float32Array(64), farDepth: new Float32Array(64) };
solidDepth.depth.fill(-10); solidDepth.farDepth.fill(10);
const obstructedViewMetrics = evaluateViewConditionedLayout(labels, bounds, { main: solidDepth, right: solidDepth, left: solidDepth, up: solidDepth, down: solidDepth });
assert.equal(ANCHOR_LOCAL_LABEL_FEATURE_NAMES.length, 51);
assert.equal(features.length, 51);
assert.equal(anchorEdge.length, ANCHOR_LABEL_EDGE_FEATURE_NAMES.length);
assert.equal(labelEdge.length, LABEL_RELATION_EDGE_FEATURE_NAMES.length);
assert.ok(features.every(Number.isFinite));
assert.equal(heterogeneous.nodeTypes.anchor, labels.length);
assert.equal(heterogeneous.nodeTypes.label, labels.length);
assert.equal(heterogeneous.relationTypes.anchor_to_label.length, labels.length ** 2);
assert.equal(heterogeneous.relationTypes.label_to_label.length, labels.length * (labels.length - 1));
assert.equal(heterogeneous.anchorEdgeFeatures[0][1].length, ANCHOR_LABEL_EDGE_FEATURE_NAMES.length);
assert.ok(frame.normal_stability >= 0 && frame.normal_stability <= 1);
assert.equal(frame.symmetry_equivalent_frames, 2);
assert.ok(viewMetrics.per_view.main);
assert.equal(viewMetrics.main_view_weight, 0.4);
assert.ok(Number.isFinite(viewMetrics.worst_view_free_space_mismatch));
assert.ok(obstructedViewMetrics.weighted_label_object_overlap > 0);
assert.ok(obstructedViewMetrics.worst_view_penetration > 0);
assert.ok(obstructedViewMetrics.objective > viewMetrics.objective);
const camera = cameraBasisForView(bounds, 'main');
const screenPoint = (x, y) => {
  const depth = bounds.radius * 3;
  const xScale = depth * (camera.sensorWidthMm / 2) / camera.focalLengthMm;
  const yScale = depth * (camera.sensorHeightMm / 2) / camera.focalLengthMm;
  return camera.eye.map((value, axis) => value + camera.forward[axis] * depth + camera.right[axis] * x * xScale + camera.up[axis] * y * yScale);
};
const crossingLabels = [
  { ...labels[0], anchor: screenPoint(-0.6, -0.6), center: screenPoint(0.6, 0.6) },
  { ...labels[1], anchor: screenPoint(0.6, -0.6), center: screenPoint(-0.6, 0.6) }
];
const uncrossedLabels = crossingLabels.map((label, index) => ({ ...label, center: screenPoint(index ? 0.6 : -0.6, 0.6) }));
const crossed = evaluateViewConditionedLayout(crossingLabels, bounds, { main: depth }, { views: ['main'], viewWeights: { main: 1 }, worstViewWeight: 2, cvarViewWeight: 1, leaderCrossingWeight: 3.5 });
const uncrossed = evaluateViewConditionedLayout(uncrossedLabels, bounds, { main: depth }, { views: ['main'], viewWeights: { main: 1 }, worstViewWeight: 2, cvarViewWeight: 1, leaderCrossingWeight: 3.5 });
assert.equal(crossed.per_view.main.leader_crossings, 1);
assert.equal(uncrossed.per_view.main.leader_crossings, 0);
const repaired = repairLeaderCrossings(crossingLabels, bounds, { seed: 23, viewPolicy: 'binocular' });
assert.equal(repaired.final.total, 0, 'deterministic repair must remove crossings in every view');
assert.equal(evaluateViewConditionedLayout(repaired.labels, bounds).worst_view_leader_crossing_count, 0);
assert.deepEqual(repaired.labels.map((label) => label.anchor), crossingLabels.map((label) => label.anchor), 'repair cannot move fixed anchors');
assert.ok(crossed.leader_crossing_objective > uncrossed.leader_crossing_objective + 0.5);
assert.ok(crossed.worst_view_leader_crossing_risk > uncrossed.worst_view_leader_crossing_risk);
assert.ok(crossed.cvar_view_leader_crossing_risk > uncrossed.cvar_view_leader_crossing_risk);
assert.ok(crossed.objective > uncrossed.objective);
console.log(JSON.stringify({ ok: true, feature_dim: features.length, anchor_edge_dim: anchorEdge.length, label_edge_dim: labelEdge.length, heterogeneous_relations: { anchor_to_label: heterogeneous.relationTypes.anchor_to_label.length, label_to_label: heterogeneous.relationTypes.label_to_label.length }, frame, view_metrics: { main_weight: viewMetrics.main_view_weight, objective: viewMetrics.objective, obstructed_objective: obstructedViewMetrics.objective, weighted_label_object_overlap: obstructedViewMetrics.weighted_label_object_overlap, worst_view_penetration: obstructedViewMetrics.worst_view_penetration } }, null, 2));
