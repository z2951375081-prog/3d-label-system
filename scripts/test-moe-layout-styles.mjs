import assert from 'node:assert/strict';
import { applyMoEStyleLayout, extractObjectContourPointCloud, scoreLayoutStyleExperts } from '../lib/moe-layout-styles.mjs';
import { optimizeLabels } from '../lib/layout-optimizer.mjs';

const roundBounds = { min: [-1, -1, -1], max: [1, 1, 1], center: [0, 0, 0], size: [2, 2, 2], radius: Math.sqrt(3) };
const roundGeometry = { triangles: [
  [[-1, -1, 0], [1, -1, 0], [1, 1, 0]],
  [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]
] };
function radialLabels(count = 8, sx = 1, sy = 1) {
  return Array.from({ length: count }, (_, index) => {
    const angle = index / count * Math.PI * 2;
    const anchor = [Math.cos(angle) * 0.6 * sx, Math.sin(angle) * 0.6 * sy, 0];
    return { id: `l${index}`, text: `label ${index}`, anchor, center: [anchor[0] * 1.2, anchor[1] * 1.2, 0], boxSize: [0.22, 0.09, 0.02], sourceObjs: [`part${index}`], targetGroups: [`part${index}`] };
  });
}

const contour = extractObjectContourPointCloud(roundGeometry, roundBounds, { view: 'main' });
assert.ok(contour.points.length >= 6);
assert.ok(contour.hull.length >= 3);
assert.ok(Number.isFinite(contour.projected_aspect));

const sphericalRouting = scoreLayoutStyleExperts(radialLabels(), roundBounds, roundGeometry, { view: 'main' });
assert.equal(sphericalRouting.selected, 'spherical');
assert.ok(Math.abs(Object.values(sphericalRouting.weights).reduce((sum, value) => sum + value, 0) - 1) < 1e-5);

const longBounds = { min: [-3, -0.4, -0.3], max: [3, 0.4, 0.3], center: [0, 0, 0], size: [6, 0.8, 0.6], radius: 3.1 };
const longGeometry = { triangles: [
  [[-3, -0.4, 0], [3, -0.4, 0], [3, 0.4, 0]],
  [[-3, -0.4, 0], [3, 0.4, 0], [-3, 0.4, 0]]
] };
const rectangularRouting = scoreLayoutStyleExperts(radialLabels(8, 3, 0.4), longBounds, longGeometry, { view: 'main' });
assert.equal(rectangularRouting.selected, 'rectangular');

const clusteredLabels = Array.from({ length: 6 }, (_, index) => {
  const anchor = [0.2 + index * 0.05, 0.05 + index * 0.03, 0];
  return { id: `c${index}`, text: `cluster ${index}`, anchor, center: [anchor[0] + 0.3, anchor[1] + 0.2, 0], boxSize: [0.2, 0.08, 0.02], sourceObjs: [], targetGroups: [] };
});
const irregularGeometry = { triangles: [
  [[0, 0, 0], [1, 0.1, 0], [0.2, 0.9, 0]],
  [[0, 0, 0], [-0.2, 0.2, 0], [-0.9, -0.3, 0]],
  [[0, 0, 0], [0.2, -0.2, 0], [0.1, -1, 0]]
] };
assert.equal(scoreLayoutStyleExperts(clusteredLabels, roundBounds, irregularGeometry, { view: 'main' }).selected, 'surround');

const spherical = applyMoEStyleLayout(radialLabels(), roundBounds, roundGeometry, { style: 'spherical', view: 'main' });
assert.equal(spherical.length, 8);
assert.ok(spherical.every((label) => label.layout_style_expert === 'spherical'));
assert.ok(spherical.every((label) => label.moe_style_routing && Number.isFinite(label.spherical_arc_angle)));
const uniqueAngles = new Set(spherical.map((label) => label.spherical_arc_angle.toFixed(2)));
assert.ok(uniqueAngles.size >= 6, 'spherical style should distribute labels around an arc/ring');

const rectangular = applyMoEStyleLayout(radialLabels(8, 3, 0.4), longBounds, longGeometry, { style: 'rectangular', view: 'main' });
assert.ok(rectangular.every((label) => label.layout_style_expert === 'rectangular'));
assert.ok(new Set(rectangular.map((label) => label.rectangular_side)).size >= 2);

const optimized = optimizeLabels(radialLabels(), roundBounds, { layoutStyle: 'auto', optimizer: 'rules', geometry: roundGeometry, viewPolicy: 'single' });
assert.ok(optimized.every((label) => label.objective.terms.includes('moe_style_initializer')));
assert.ok(optimized.some((label) => label.moe_style_selected === 'spherical'));

console.log(JSON.stringify({ ok: true, selected: { spherical: sphericalRouting.weights, rectangular: rectangularRouting.weights }, contour_points: contour.points.length }, null, 2));
