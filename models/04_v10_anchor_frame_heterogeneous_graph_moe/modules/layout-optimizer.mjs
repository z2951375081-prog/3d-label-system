// Shared, deterministic layout optimizer used by the browser and batch exporter.
// It is intentionally dependency-free so a research model can replace this module.

import { DATASET_CAMERA_PROTOCOL, DATASET_VIEW_NAMES, datasetCameraForBounds } from '../public/dataset-camera.js';
import { meshLabelSurfaceIntersection } from './mesh-label-intersection.mjs';
import { clampLeaderLength, normalizedLeaderLength, resolveLeaderLengthRange } from './leader-length-prior.mjs';
import { applyMoEStyleLayout, scoreLayoutStyleExperts } from './moe-layout-styles.mjs';

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a, value) { return [a[0] * value, a[1] * value, a[2] * value]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize(a) { const length = Math.max(Math.hypot(...a), 1e-8); return scale(a, 1 / length); }
function cloneLabels(labels) { return labels.map((label) => ({ ...label, center: [...label.center], anchor: [...label.anchor], boxSize: [...label.boxSize], bendPoints: (label.bendPoints || []).map((point) => [...point]) })); }
function seededRandom(seed = 17) { let state = (Number(seed) >>> 0) || 17; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }

function semanticKey(label) { return String(label.text || label.id).toLowerCase().replace(/[_\-\d]+/g, ' ').trim(); }

export function groupLabels(labels, policy = 'all', radius = 1) {
  if (policy === 'all') return labels;
  const groups = new Map();
  labels.forEach((label) => { const key = semanticKey(label); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(label); });
  const output = [];
  for (const candidates of groups.values()) {
    if (policy === 'symmetric' && candidates.length > 1) {
      const sorted = [...candidates].sort((a, b) => a.center[0] - b.center[0]);
      const spread = Math.abs(sorted.at(-1).center[0] - sorted[0].center[0]);
      if (spread > radius * 0.25) { output.push(sorted[0], sorted.at(-1)); continue; }
    }
    output.push([...candidates].sort((a, b) => (b.sourceObjs?.length || 0) - (a.sourceObjs?.length || 0))[0]);
  }
  return output;
}

function applySizePolicy(labels, bounds, policy) {
  const radius = bounds.radius;
  return labels.map((label) => {
    const output = { ...label, center: [...label.center], anchor: [...label.anchor], boxSize: [...label.boxSize], bendPoints: (label.bendPoints || []).map((point) => [...point]) };
    if (policy === 'fixed') output.boxSize = [radius * 0.32, radius * 0.07, radius * 0.012];
    if (policy === 'distance-aware') {
      const d = Math.max(0.2, Math.hypot(...output.center));
      const factor = Math.min(2.2, Math.max(0.55, d / radius));
      output.boxSize = output.boxSize.map((value) => value * factor);
    }
    return output;
  });
}

function cameraBasis(bounds, side) {
  const base = datasetCameraForBounds(bounds, 'main');
  const target = bounds.center;
  const eye = add(base.eye, scale(base.right, side * 0.032));
  const forward = normalize(sub(target, eye));
  let right = normalize(cross(forward, base.up));
  if (!Number.isFinite(right[0])) right = [1, 0, 0];
  const up = normalize(cross(right, forward));
  return { ...base, eye, forward, right, up, view: side < 0 ? 'left_eye' : 'right_eye' };
}

export const MULTI_VIEW_NAMES = [...DATASET_VIEW_NAMES];

export function cameraBasisForView(bounds, view = 'main') {
  return datasetCameraForBounds(bounds, view);
}

function project(point, basis, bounds) {
  const relative = sub(point, basis.eye);
  const depth = Math.max(1e-6, dot(relative, basis.forward));
  const focalLength = basis.focalLengthMm || DATASET_CAMERA_PROTOCOL.focalLengthMm;
  const halfSensorWidth = (basis.sensorWidthMm || DATASET_CAMERA_PROTOCOL.sensorWidthMm) / 2;
  const halfSensorHeight = (basis.sensorHeightMm || DATASET_CAMERA_PROTOCOL.sensorHeightMm) / 2;
  return { x: dot(relative, basis.right) / depth * focalLength / halfSensorWidth, y: dot(relative, basis.up) / depth * focalLength / halfSensorHeight, depth };
}

export function projectPointToView(point, bounds, view = 'main') { return project(point, cameraBasisForView(bounds, view), bounds); }

function projectedLabel(label, basis, bounds) {
  const center = project(label.center, basis, bounds);
  const anchor = project(label.anchor, basis, bounds);
  const horizontal = project(add(label.center, scale(basis.right, Math.abs(label.boxSize[0]) / 2)), basis, bounds);
  const vertical = project(add(label.center, scale(basis.up, Math.abs(label.boxSize[1]) / 2)), basis, bounds);
  const width = Math.max(0.002, Math.abs(horizontal.x - center.x));
  const height = Math.max(0.002, Math.abs(vertical.y - center.y));
  return { center, anchor, left: center.x - width, right: center.x + width, top: center.y + height, bottom: center.y - height, width, height };
}

export function projectLabelToView(label,bounds,view='main') { return projectedLabel(label,cameraBasisForView(bounds,view),bounds); }

export function parseObjTriangles(text) {
  const vertices = [];
  const triangles = [];
  for (const line of text.split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) vertices.push(parts.slice(1, 4).map(Number));
    if (parts[0] === 'f' && parts.length >= 4) {
      const indices = parts.slice(1).map((token) => { const raw = Number(token.split('/')[0]); return raw < 0 ? vertices.length + raw : raw - 1; }).filter((index) => vertices[index]);
      for (let index = 1; index < indices.length - 1; index += 1) triangles.push([vertices[indices[0]], vertices[indices[index]], vertices[indices[index + 1]]]);
    }
  }
  return { triangles };
}

export function buildDepthGrid(geometry, bounds, side = 0, gridSize = 64) {
  const depth = new Float32Array(gridSize * gridSize);
  const farDepth = new Float32Array(gridSize * gridSize);
  depth.fill(Infinity);
  farDepth.fill(-Infinity);
  const basis = typeof side === 'string' ? cameraBasisForView(bounds, side) : cameraBasis(bounds, side);
  for (const triangle of geometry?.triangles || []) {
    const projected = triangle.map((point) => project(point, basis, bounds));
    const minX = Math.max(0, Math.floor((Math.min(...projected.map((point) => point.x)) + 1.15) / 2.3 * gridSize));
    const maxX = Math.min(gridSize - 1, Math.ceil((Math.max(...projected.map((point) => point.x)) + 1.15) / 2.3 * gridSize));
    const minY = Math.max(0, Math.floor((Math.min(...projected.map((point) => point.y)) + 1.15) / 2.3 * gridSize));
    const maxY = Math.min(gridSize - 1, Math.ceil((Math.max(...projected.map((point) => point.y)) + 1.15) / 2.3 * gridSize));
    const area = orientation(projected[0], projected[1], projected[2]);
    if (Math.abs(area) < 1e-8) continue;
    for (let y = minY; y <= maxY; y += 1) for (let x = minX; x <= maxX; x += 1) {
      const point = { x: (x + 0.5) / gridSize * 2.3 - 1.15, y: (y + 0.5) / gridSize * 2.3 - 1.15 };
      const w0 = orientation(projected[1], projected[2], point) / area;
      const w1 = orientation(projected[2], projected[0], point) / area;
      const w2 = orientation(projected[0], projected[1], point) / area;
      if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        const sampleDepth = 1 / (w0 / projected[0].depth + w1 / projected[1].depth + w2 / projected[2].depth);
        depth[y * gridSize + x] = Math.min(depth[y * gridSize + x], sampleDepth);
        farDepth[y * gridSize + x] = Math.max(farDepth[y * gridSize + x], sampleDepth);
      }
    }
  }
  return { depth, farDepth, gridSize };
}

function labelDepthRelations(label, basis, bounds, depthGrid) {
  if (!depthGrid) return { object_over_label: 0, label_over_object: 0, penetration: 0, object_overlap: 0 };
  const projected = projectedLabel(label, basis, bounds);
  let samples = 0, objectOverlap = 0, objectOverLabel = 0, labelOverObject = 0, penetration = 0;
  const depthTolerance = Math.max(bounds.radius * 0.015, 0.01);
  for (let row = 0; row < 4; row += 1) for (let column = 0; column < 6; column += 1) {
    const x = projected.left + (column + 0.5) / 6 * (projected.right - projected.left);
    const y = projected.bottom + (row + 0.5) / 4 * (projected.top - projected.bottom);
    const gx = Math.floor((x + 1.15) / 2.3 * depthGrid.gridSize);
    const gy = Math.floor((y + 1.15) / 2.3 * depthGrid.gridSize);
    if (gx < 0 || gy < 0 || gx >= depthGrid.gridSize || gy >= depthGrid.gridSize) continue;
    const objectDepth = depthGrid.depth[gy * depthGrid.gridSize + gx];
    samples += 1;
    if (!Number.isFinite(objectDepth)) continue;
    objectOverlap += 1;
    const objectFarDepth = Number.isFinite(depthGrid.farDepth?.[gy * depthGrid.gridSize + gx]) ? depthGrid.farDepth[gy * depthGrid.gridSize + gx] : objectDepth;
    if (projected.center.depth >= objectDepth - depthTolerance && projected.center.depth <= objectFarDepth + depthTolerance) penetration += 1;
    else if (projected.center.depth > objectFarDepth) objectOverLabel += 1;
    else labelOverObject += 1;
  }
  const denominator = Math.max(1, samples);
  return {
    object_over_label: objectOverLabel / denominator,
    label_over_object: labelOverObject / denominator,
    penetration: penetration / denominator,
    object_overlap: objectOverlap / denominator
  };
}

function overlap(a, b) { return Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) * Math.max(0, Math.min(a.top, b.top) - Math.max(a.bottom, b.bottom)); }
function orientation(a, b, c) { return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x); }
function segmentsCross(a, b, c, d) { const ab = orientation(a, b, c) * orientation(a, b, d); const cd = orientation(c, d, a) * orientation(c, d, b); return ab < 0 && cd < 0; }
function pointSegmentDistance(point, start, end) {
  const dx = end.x - start.x, dy = end.y - start.y;
  const length2 = dx * dx + dy * dy;
  if (length2 <= 1e-12) return Math.hypot(point.x - start.x, point.y - start.y);
  const ratio = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / length2));
  return Math.hypot(point.x - (start.x + ratio * dx), point.y - (start.y + ratio * dy));
}
function segmentDistance(a, b, c, d) {
  if (segmentsCross(a, b, c, d)) return 0;
  return Math.min(pointSegmentDistance(a, c, d), pointSegmentDistance(b, c, d), pointSegmentDistance(c, a, b), pointSegmentDistance(d, a, b));
}
function projectionOverlapRatio(a, b, c, d) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const length = Math.max(Math.hypot(dx, dy), 1e-8);
  const ux = dx / length, uy = dy / length;
  const project = (point) => point.x * ux + point.y * uy;
  const first = [project(a), project(b)].sort((left, right) => left - right);
  const second = [project(c), project(d)].sort((left, right) => left - right);
  return Math.max(0, Math.min(first[1], second[1]) - Math.max(first[0], second[0])) / length;
}
function visibleLeaderOverlap(a, b, c, d, tolerance = 0.018) {
  if (segmentsCross(a, b, c, d)) return false;
  const firstLength = Math.hypot(b.x - a.x, b.y - a.y);
  const secondLength = Math.hypot(d.x - c.x, d.y - c.y);
  const threshold = Math.max(tolerance, Math.min(firstLength, secondLength) * 0.035);
  return segmentDistance(a, b, c, d) <= threshold && projectionOverlapRatio(a, b, c, d) >= 0.22;
}

const DIRECTION_NAMES = ['right', 'upper_right', 'up', 'upper_left', 'left', 'lower_left', 'down', 'lower_right'];
function directionIndex(x, y) {
  const angle = Math.atan2(y, x);
  return (Math.round(angle / (Math.PI / 4)) + 8) % 8;
}

function directionalSpaceStats(items, depthGrid) {
  const counts = Array(8).fill(0);
  for (const item of items) counts[directionIndex(item.center.x, item.center.y)] += 1;
  const free = Array(8).fill(0);
  if (depthGrid?.depth?.length && depthGrid.gridSize) {
    for (let y = 0; y < depthGrid.gridSize; y += 1) for (let x = 0; x < depthGrid.gridSize; x += 1) {
      const screenX = (x + 0.5) / depthGrid.gridSize * 2.3 - 1.15;
      const screenY = (y + 0.5) / depthGrid.gridSize * 2.3 - 1.15;
      if (Math.abs(screenX) > 1 || Math.abs(screenY) > 1 || Number.isFinite(depthGrid.depth[y * depthGrid.gridSize + x])) continue;
      free[directionIndex(screenX, screenY)] += 1;
    }
  } else free.fill(1);
  const totalLabels = Math.max(1, counts.reduce((sum, value) => sum + value, 0));
  const totalFree = Math.max(1, free.reduce((sum, value) => sum + value, 0));
  const labelShares = counts.map((value) => value / totalLabels);
  const freeShares = free.map((value) => value / totalFree);
  const allocationMismatch = 0.5 * labelShares.reduce((sum, value, index) => sum + Math.abs(value - freeShares[index]), 0);
  const concentrationExcess = counts.reduce((maximum, value, index) => Math.max(maximum, Math.max(0, value - Math.ceil(totalLabels * freeShares[index]) - 1) / totalLabels), 0);
  const entropyDenominator = Math.log(Math.max(2, Math.min(8, totalLabels)));
  const entropy = -labelShares.filter((value) => value > 0).reduce((sum, value) => sum + value * Math.log(value), 0);
  return {
    directional_allocation_mismatch: allocationMismatch,
    directional_concentration_excess: concentrationExcess,
    directional_uniformity: clamp(entropy / entropyDenominator, 0, 1),
    directional_label_counts: Object.fromEntries(DIRECTION_NAMES.map((name, index) => [name, counts[index]])),
    directional_label_shares: Object.fromEntries(DIRECTION_NAMES.map((name, index) => [name, Number(labelShares[index].toFixed(6))])),
    directional_free_space_shares: Object.fromEntries(DIRECTION_NAMES.map((name, index) => [name, Number(freeShares[index].toFixed(6))]))
  };
}

function viewEnergyStats(labels, bounds, basis, depthGrid) {
  const items = labels.map((label) => projectedLabel(label, basis, bounds));
  const directional = directionalSpaceStats(items, depthGrid);
  const pairCount = Math.max(1, items.length * Math.max(1, items.length - 1) / 2);
  let overlapArea = 0, overlapPairs = 0, crossings = 0, visibleLeaderOverlaps = 0, overflow = 0, labelArea = 0;
  let objectOverLabel = 0, labelOverObject = 0, penetration = 0, objectOverlap = 0;
  let textPixelHeight = 0, textFit = 0, textClipping = 0;
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    labelArea += item.width * 2 * item.height * 2;
    const overflowX = Math.max(0, Math.abs(item.center.x) + item.width - 1);
    const overflowY = Math.max(0, Math.abs(item.center.y) + item.height - 1);
    overflow += overflowX + overflowY;
    textClipping += clamp((overflowX / Math.max(item.width * 2, 1e-8)) + (overflowY / Math.max(item.height * 2, 1e-8)), 0, 1);
    const relation = labelDepthRelations(labels[index], basis, bounds, depthGrid);
    objectOverLabel += relation.object_over_label;
    labelOverObject += relation.label_over_object;
    penetration += relation.penetration;
    objectOverlap += relation.object_overlap;
    const pixelHeight = item.height * DATASET_CAMERA_PROTOCOL.imageHeight;
    const pixelWidth = item.width * DATASET_CAMERA_PROTOCOL.imageWidth;
    const fontPixels = Math.max(1, Math.min(pixelHeight * 0.68, 28));
    const estimatedTextWidth = Math.max(1, String(labels[index].text || '').length) * fontPixels * 0.58;
    textPixelHeight += pixelHeight;
    textFit += Math.min(1, pixelWidth * 0.92 / estimatedTextWidth);
    for (let other = index + 1; other < items.length; other += 1) {
      const area = overlap(item, items[other]);
      overlapArea += area;
      if (area > 0) overlapPairs += 1;
      if (segmentsCross(item.anchor, item.center, items[other].anchor, items[other].center)) crossings += 1;
      else if (visibleLeaderOverlap(item.anchor, item.center, items[other].anchor, items[other].center)) visibleLeaderOverlaps += 1;
    }
  }
  const count = Math.max(1, items.length);
  const meanPixelHeight = textPixelHeight / count;
  return {
    items,
    overlap_area: overlapArea,
    overlap_pairs: overlapPairs,
    overlap_ratio: overlapPairs / pairCount,
    olr: labelArea > 1e-8 ? Math.min(1, overlapArea / labelArea) : 0,
    lcd: crossings / pairCount,
    crossings,
    visible_leader_overlaps: visibleLeaderOverlaps,
    visible_leader_overlap_ratio: visibleLeaderOverlaps / pairCount,
    overflow_ratio: Math.min(1, overflow / Math.max(1, items.length * 0.4)),
    object_occlusion: objectOverLabel / count,
    label_object_occlusion: labelOverObject / count,
    object_penetration: penetration / count,
    object_projection_overlap: objectOverlap / count,
    text_pixel_height: meanPixelHeight,
    text_fit_ratio: textFit / count,
    text_clipping_ratio: textClipping / count,
    text_clarity: clamp((meanPixelHeight - 5) / 13, 0, 1) * (textFit / count) * (1 - textClipping / count),
    ...directional
  };
}

function scoreLayout(labels, bounds, options) {
  const weights = { overlap: 3.2, overflow: 4.5, crossing: 1.8, leader_overlap: 3.6, object_occlusion: 3.5, label_object_occlusion: 3.5, penetration: 5.0, text_clarity: 2.0, directional_density: 1.6, directional_concentration: 1.2, anchor_distance: 0.12, leader_shortfall: 4.0, leader_excess: 0.35, leader_target: 0.04, view_variance: 2.4, binocular: 2.8, ...(options.energyWeights || {}) };
  const viewNames = options.viewPolicy === 'single' ? ['main'] : MULTI_VIEW_NAMES;
  const stats = viewNames.map((name) => viewEnergyStats(labels, bounds, cameraBasisForView(bounds, name), options.depthGrids?.[name]));
  const viewWeights = viewNames.map((name) => name === 'main' ? 0.4 : 0.15);
  const viewWeightTotal = viewWeights.reduce((sum, value) => sum + value, 0) || 1;
  let score = stats.reduce((sum, item, index) => sum + viewWeights[index] * (item.olr * weights.overlap + item.overflow_ratio * weights.overflow + item.lcd * weights.crossing + item.visible_leader_overlap_ratio * weights.leader_overlap + item.object_occlusion * weights.object_occlusion + item.label_object_occlusion * weights.label_object_occlusion + item.object_penetration * weights.penetration + (1 - item.text_clarity) * weights.text_clarity + item.directional_allocation_mismatch * weights.directional_density + item.directional_concentration_excess * weights.directional_concentration), 0) / viewWeightTotal;
  const leaderRange = options.leaderLengthRange || resolveLeaderLengthRange(options.leaderLengthPrior, options.category);
  if (leaderRange) {
    const lengths = labels.map((label) => normalizedLeaderLength(label, bounds));
    score += mean(lengths.map((length) => Math.max(0, leaderRange.preferred_min - length) ** 2)) * weights.leader_shortfall;
    score += mean(lengths.map((length) => Math.max(0, length - leaderRange.preferred_max) ** 2)) * weights.leader_excess;
    score += mean(lengths.map((length) => Math.abs(length - leaderRange.target))) * weights.leader_target;
  } else {
    score += labels.reduce((sum, label) => sum + Math.min(3, distance(label.anchor, label.center) / bounds.radius), 0) / Math.max(1, labels.length) * weights.anchor_distance;
  }
  if (stats.length > 1) {
    const meanOlr = stats.reduce((sum, item) => sum + item.olr, 0) / stats.length;
    score += stats.reduce((sum, item) => sum + Math.abs(item.olr - meanOlr), 0) / stats.length * weights.view_variance;
  }
  if (options.viewPolicy !== 'single') {
    const stereo = [-1, 1].map((side) => labels.map((label) => projectedLabel(label, cameraBasis(bounds, side), bounds)));
    for (let index = 0; index < labels.length; index += 1) score += Math.max(0, Math.abs(stereo[0][index].center.x - stereo[1][index].center.x) - 0.12) * weights.binocular / Math.max(1, labels.length);
  }
  return score;
}

function insideBounds(point, bounds) { return point.every((value, index) => value >= bounds.min[index] && value <= bounds.max[index]); }

function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }
function standardDeviation(values) { const average = mean(values); return Math.sqrt(mean(values.map((value) => (value - average) ** 2))); }

export function compareLayoutToManual(labels, manualLabels, bounds) {
  if (!Array.isArray(labels) || !Array.isArray(manualLabels) || !labels.length || labels.length !== manualLabels.length) return null;
  const manualById = new Map(manualLabels.map((label) => [String(label.id), label]));
  const pairs = labels.map((label) => [label, manualById.get(String(label.id))]);
  if (pairs.some(([label, manual]) => !manual || label.text !== manual.text)) return null;
  const radius = Math.max(bounds.radius, 1e-6);
  const centerDistances = pairs.map(([label, manual]) => distance(label.center, manual.center) / radius);
  const sizeDistances = pairs.map(([label, manual]) => mean(label.boxSize.slice(0, 2).map((value, axis) => Math.abs(Math.log(Math.max(Math.abs(value), 1e-6) / Math.max(Math.abs(manual.boxSize[axis]), 1e-6))))));
  const leaderLengths = pairs.map(([label]) => distance(label.anchor, label.center) / radius);
  const manualLeaderLengths = pairs.map(([, manual]) => distance(manual.anchor, manual.center) / radius);
  const radial = pairs.map(([label]) => distance(label.center, bounds.center) / radius);
  const manualRadial = pairs.map(([, manual]) => distance(manual.center, bounds.center) / radius);
  const coordinateMeans = [0, 1, 2].map((axis) => mean(pairs.map(([label]) => (label.center[axis] - bounds.center[axis]) / radius)));
  const manualCoordinateMeans = [0, 1, 2].map((axis) => mean(pairs.map(([, manual]) => (manual.center[axis] - bounds.center[axis]) / radius)));
  const meanLeaderDifference = Math.abs(mean(leaderLengths) - mean(manualLeaderLengths));
  const leaderSpreadDifference = Math.abs(standardDeviation(leaderLengths) - standardDeviation(manualLeaderLengths));
  const radialMeanDifference = Math.abs(mean(radial) - mean(manualRadial));
  const radialSpreadDifference = Math.abs(standardDeviation(radial) - standardDeviation(manualRadial));
  const balanceDifference = mean(coordinateMeans.map((value, axis) => Math.abs(value - manualCoordinateMeans[axis])));
  const positionDistance = mean(centerDistances);
  const sizeDistance = mean(sizeDistances);
  const styleDistance = clamp(mean([meanLeaderDifference, leaderSpreadDifference, radialMeanDifference, radialSpreadDifference, balanceDifference, Math.min(1, sizeDistance)]), 0, 1);
  return {
    manual_center_distance_norm: Number(positionDistance.toFixed(6)),
    manual_size_distance_log: Number(sizeDistance.toFixed(6)),
    manual_style_distance: Number(styleDistance.toFixed(6)),
    manual_position_similarity: Number((1 - clamp(positionDistance / 2, 0, 1)).toFixed(6)),
    manual_style_similarity: Number((1 - styleDistance).toFixed(6)),
    style_components: {
      mean_leader_length_difference: Number(meanLeaderDifference.toFixed(6)),
      leader_length_spread_difference: Number(leaderSpreadDifference.toFixed(6)),
      radial_mean_difference: Number(radialMeanDifference.toFixed(6)),
      radial_spread_difference: Number(radialSpreadDifference.toFixed(6)),
      spatial_balance_difference: Number(balanceDifference.toFixed(6))
    },
    evaluation_only: true
  };
}

export function evaluateLayout(labels, bounds, options = {}) {
  const safeLabels = Array.isArray(labels) ? labels : [];
  const mainBasis = cameraBasisForView(bounds, 'main');
  const main = safeLabels.map((label) => projectedLabel(label, mainBasis, bounds));
  let overlapArea = 0;
  let overlapPairs = 0;
  let leaderCrossings = 0;
  let overflowAmount = 0;
  let anchorDistance = 0;
  let labelArea = 0;
  for (let index = 0; index < main.length; index += 1) {
    const item = main[index];
    labelArea += item.width * 2 * item.height * 2;
    overflowAmount += Math.max(0, Math.abs(item.center.x) + item.width - 1.12) + Math.max(0, Math.abs(item.center.y) + item.height - 1.12);
    anchorDistance += distance(safeLabels[index].anchor, safeLabels[index].center) / Math.max(bounds.radius, 1e-6);
    for (let other = index + 1; other < main.length; other += 1) {
      const area = overlap(item, main[other]);
      if (area > 0) { overlapArea += area; overlapPairs += 1; }
      if (segmentsCross(item.anchor, item.center, main[other].anchor, main[other].center)) leaderCrossings += 1;
    }
  }
  const pairCount = Math.max(1, (main.length * (main.length - 1)) / 2);
  const overlapRatio = Math.min(1, overlapPairs / pairCount);
  const olr = labelArea > 1e-8 ? Math.min(1, overlapArea / labelArea) : 0;
  const overflowRatio = Math.min(1, overflowAmount / Math.max(1, main.length * 0.4));
  const anchorCoverage = safeLabels.length ? safeLabels.filter((label) => insideBounds(label.anchor, bounds)).length / safeLabels.length : 0;
  const meanAnchorDistance = safeLabels.length ? anchorDistance / safeLabels.length : 0;
  const normalizedLeaderLengths = safeLabels.map((label) => normalizedLeaderLength(label, bounds));
  const leaderLengthRange = options.leaderLengthRange || resolveLeaderLengthRange(options.leaderLengthPrior, options.category);
  const leaderLengthCompliance = leaderLengthRange && normalizedLeaderLengths.length
    ? normalizedLeaderLengths.filter((length) => length >= leaderLengthRange.preferred_min && length <= leaderLengthRange.preferred_max).length / normalizedLeaderLengths.length
    : null;
  const leaderLengthShortfall = leaderLengthRange && normalizedLeaderLengths.length
    ? mean(normalizedLeaderLengths.map((length) => Math.max(0, leaderLengthRange.preferred_min - length)))
    : null;
  const readability = safeLabels.length ? clamp(5 - overlapRatio * 2.2 - overflowRatio * 1.5 - Math.min(1, leaderCrossings / pairCount) * 1.4 - Math.min(1, meanAnchorDistance / 4) * 0.8, 1, 5) : 0;
  const coverage = safeLabels.length ? anchorCoverage * 5 : 0;
  const occlusion = safeLabels.length ? clamp(5 - overlapRatio * 4 - overflowRatio * 2, 1, 5) : 0;
  const left = safeLabels.map((label) => projectedLabel(label, cameraBasis(bounds, -1), bounds));
  const right = safeLabels.map((label) => projectedLabel(label, cameraBasis(bounds, 1), bounds));
  const binocularDisparity = options.viewPolicy === 'binocular' && safeLabels.length ? right.reduce((sum, item, index) => sum + Math.abs(item.center.x - left[index].center.x), 0) / safeLabels.length : 0;
  let leftOverlapArea = 0;
  let rightOverlapArea = 0;
  let leftCrossings = 0;
  let rightCrossings = 0;
  for (let index = 0; index < safeLabels.length; index += 1) {
    for (let other = index + 1; other < safeLabels.length; other += 1) {
      leftOverlapArea += overlap(left[index], left[other]);
      rightOverlapArea += overlap(right[index], right[other]);
      if (segmentsCross(left[index].anchor, left[index].center, left[other].anchor, left[other].center)) leftCrossings += 1;
      if (segmentsCross(right[index].anchor, right[index].center, right[other].anchor, right[other].center)) rightCrossings += 1;
    }
  }
  const binocularOlr = Math.min(1, Math.max(leftOverlapArea, rightOverlapArea) / Math.max(labelArea, 1e-8));
  const binocularLcd = safeLabels.length > 1 ? Math.min(1, Math.max(leftCrossings, rightCrossings) / pairCount) : 0;
  let doubleVisionArea = 0;
  let doubleVisionDenominator = 0;
  if (options.viewPolicy === 'binocular') {
    for (let index = 0; index < safeLabels.length; index += 1) for (let other = index + 1; other < safeLabels.length; other += 1) {
      doubleVisionArea += Math.abs(overlap(left[index], left[other]) - overlap(right[index], right[other]));
      doubleVisionDenominator += Math.max(left[index].width * 2 * left[index].height * 2, right[index].width * 2 * right[index].height * 2);
    }
  }
  const dbv = doubleVisionDenominator > 1e-8 ? Math.min(1, doubleVisionArea / doubleVisionDenominator) : 0;
  const multiViewNames = options.viewPolicy === 'single' ? ['main'] : MULTI_VIEW_NAMES;
  const multiViewStats = Object.fromEntries(multiViewNames.map((name) => [name, viewEnergyStats(safeLabels, bounds, cameraBasisForView(bounds, name), options.depthGrids?.[name])]));
  const averageViewMetric = (key) => multiViewNames.reduce((sum, name) => sum + multiViewStats[name][key], 0) / Math.max(1, multiViewNames.length);
  const multiViewOlr = averageViewMetric('olr');
  const multiViewLcd = averageViewMetric('lcd');
  const multiViewVisibleLeaderOverlap = averageViewMetric('visible_leader_overlap_ratio');
  const multiViewOverflow = averageViewMetric('overflow_ratio');
  const multiViewObjectOcclusion = averageViewMetric('object_occlusion');
  const multiViewLabelObjectOcclusion = averageViewMetric('label_object_occlusion');
  const multiViewPenetration = averageViewMetric('object_penetration');
  const multiViewOverlapRatio = averageViewMetric('overlap_ratio');
  const multiViewTextClarity = averageViewMetric('text_clarity');
  const multiViewTextPixelHeight = averageViewMetric('text_pixel_height');
  const multiViewTextFit = averageViewMetric('text_fit_ratio');
  const multiViewTextClipping = averageViewMetric('text_clipping_ratio');
  const multiViewDirectionalMismatch = averageViewMetric('directional_allocation_mismatch');
  const multiViewDirectionalConcentration = averageViewMetric('directional_concentration_excess');
  const multiViewDirectionalUniformity = averageViewMetric('directional_uniformity');
  const multiViewReadability = safeLabels.length ? clamp(1 + 4 * multiViewTextClarity - multiViewOverlapRatio * 1.5 - multiViewLcd - Math.min(1, meanAnchorDistance / 4) * 0.5, 1, 5) : 0;
  const multiViewOcclusionScore = safeLabels.length ? clamp(5 - multiViewOverlapRatio * 3 - multiViewObjectOcclusion * 2 - multiViewLabelObjectOcclusion * 2 - multiViewPenetration * 3, 1, 5) : 0;
  const lcd = options.viewPolicy === 'single' ? Math.min(1, leaderCrossings / pairCount) : multiViewLcd;
  const finalOlr = options.viewPolicy === 'single' ? olr : multiViewOlr;
  const objectiveScore = scoreLayout(safeLabels, bounds, options);
  const viewMetrics = Object.fromEntries(Object.entries(multiViewStats).map(([name, item]) => [name, {
    olr: Number(item.olr.toFixed(4)), lcd: Number(item.lcd.toFixed(4)), overflow_ratio: Number(item.overflow_ratio.toFixed(4)),
    object_occlusion_ratio: Number(item.label_object_occlusion.toFixed(4)),
    object_label_occlusion_ratio: Number(item.object_occlusion.toFixed(4)), label_object_occlusion_ratio: Number(item.label_object_occlusion.toFixed(4)),
    object_penetration_ratio: Number(item.object_penetration.toFixed(4)), overlap_ratio: Number(item.overlap_ratio.toFixed(4)),
    text_pixel_height: Number(item.text_pixel_height.toFixed(3)), text_fit_ratio: Number(item.text_fit_ratio.toFixed(4)),
    text_clipping_ratio: Number(item.text_clipping_ratio.toFixed(4)), text_clarity: Number(item.text_clarity.toFixed(4)),
    directional_allocation_mismatch: Number(item.directional_allocation_mismatch.toFixed(4)),
    directional_concentration_excess: Number(item.directional_concentration_excess.toFixed(4)),
    directional_uniformity: Number(item.directional_uniformity.toFixed(4)),
    directional_label_counts: item.directional_label_counts,
    directional_label_shares: item.directional_label_shares,
    directional_free_space_shares: item.directional_free_space_shares
  }]));
  const manualSimilarity = compareLayoutToManual(safeLabels, options.manualReference, bounds);
  const meshIntersections = options.geometry ? Object.fromEntries(multiViewNames.map(view=>[view,meshLabelSurfaceIntersection(safeLabels,options.geometry,cameraBasisForView(bounds,view))])) : null;
  const intrinsicStyleBalance = safeLabels.length ? clamp(1 - Math.hypot(...[0, 1, 2].map(axis => mean(safeLabels.map(label => (label.center[axis] - bounds.center[axis]) / Math.max(bounds.radius, 1e-6))))), 0, 1) : 0;
  const widths = safeLabels.map(label => Math.max(Math.abs(label.boxSize[0]), 1e-6));
  const meanWidth = mean(widths);
  const intrinsicSizeConsistency = widths.length ? clamp(1 - mean(widths.map(width => Math.abs(Math.log(width / meanWidth)))), 0, 1) : 0;
  const normalizedRadius = Math.max(bounds.radius, 1e-6);
  const radialDistances = safeLabels.map(label => distance(label.center, bounds.center) / normalizedRadius);
  const leaderLengths = safeLabels.map(label => distance(label.anchor, label.center) / normalizedRadius);
  const pairDistances = [];
  for (let index = 0; index < safeLabels.length; index += 1) for (let other = index + 1; other < safeLabels.length; other += 1) pairDistances.push(distance(safeLabels[index].center, safeLabels[other].center) / normalizedRadius);
  const spacingMean = mean(pairDistances);
  const intrinsicSpacingConsistency = pairDistances.length ? clamp(1 - standardDeviation(pairDistances) / Math.max(spacingMean, 1e-6), 0, 1) : 1;
  const leaderMean = mean(leaderLengths);
  const intrinsicLeaderRhythm = leaderLengths.length ? clamp(1 - standardDeviation(leaderLengths) / Math.max(leaderMean, 1e-6), 0, 1) : 1;
  const widthPerCharacter = safeLabels.map((label, index) => widths[index] / Math.max(1, String(label.text || '').length) / normalizedRadius);
  const meanWidthPerCharacter = mean(widthPerCharacter);
  const intrinsicTextSizeFit = widthPerCharacter.length ? clamp(1 - mean(widthPerCharacter.map(value => Math.abs(Math.log(Math.max(value, 1e-9) / Math.max(meanWidthPerCharacter, 1e-9))))), 0, 1) : 0;
  const intrinsicCompositionHarmony = mean([intrinsicStyleBalance, intrinsicSizeConsistency, intrinsicSpacingConsistency, intrinsicLeaderRhythm, intrinsicTextSizeFit]);
  const intrinsicTerms = [
    [0.15, multiViewTextClarity * 5],
    [0.10, coverage],
    [0.08, (1 - multiViewOlr) * 5],
    [0.08, (1 - multiViewLabelObjectOcclusion) * 5],
    [0.04, (1 - multiViewObjectOcclusion) * 5],
    [0.10, (1 - multiViewPenetration) * 5],
    ...(meshIntersections ? [[0.10, (1 - mean(Object.values(meshIntersections).map(item=>item.surface_intersection_ratio))) * 5]] : []),
    [0.10, (1 - multiViewLcd) * 5],
    [0.05, (1 - Math.max(...multiViewNames.map((name) => Math.abs(multiViewStats[name].olr - multiViewOlr)), 0)) * 5]
  ];
  const intrinsicWeight = intrinsicTerms.reduce((sum, [weight]) => sum + weight, 0);
  const intrinsicQuality = intrinsicTerms.reduce((sum, [weight, value]) => sum + weight * value, 0) / intrinsicWeight;
  const qualityTerms = [...intrinsicTerms, ...(manualSimilarity ? [[0.10, manualSimilarity.manual_position_similarity * 5], [0.10, manualSimilarity.manual_style_similarity * 5]] : [])];
  const qualityWeight = qualityTerms.reduce((sum, [weight]) => sum + weight, 0);
  const multidimensionalQuality = qualityTerms.reduce((sum, [weight, value]) => sum + weight * value, 0) / qualityWeight;
  return {
    label_count: safeLabels.length,
    intrinsic_style_balance: Number(intrinsicStyleBalance.toFixed(6)),
    intrinsic_size_consistency: Number(intrinsicSizeConsistency.toFixed(6)),
    aesthetic_radial_mean: Number(mean(radialDistances).toFixed(6)),
    aesthetic_radial_spread: Number(standardDeviation(radialDistances).toFixed(6)),
    aesthetic_leader_length_mean: Number(leaderMean.toFixed(6)),
    aesthetic_leader_length_spread: Number(standardDeviation(leaderLengths).toFixed(6)),
    aesthetic_spacing_consistency: Number(intrinsicSpacingConsistency.toFixed(6)),
    aesthetic_text_size_fit: Number(intrinsicTextSizeFit.toFixed(6)),
    aesthetic_composition_harmony: Number(intrinsicCompositionHarmony.toFixed(6)),
    directional_allocation_mismatch: Number(multiViewDirectionalMismatch.toFixed(6)),
    directional_concentration_excess: Number(multiViewDirectionalConcentration.toFixed(6)),
    directional_uniformity: Number(multiViewDirectionalUniformity.toFixed(6)),
    intrinsic_quality_score: Number(intrinsicQuality.toFixed(6)),
    semantic_count: new Set(safeLabels.map(semanticKey)).size,
    readability: Number(multiViewReadability.toFixed(3)),
    text_clarity: Number((multiViewTextClarity * 5).toFixed(3)),
    text_pixel_height: Number(multiViewTextPixelHeight.toFixed(3)),
    text_fit_ratio: Number(multiViewTextFit.toFixed(3)),
    text_clipping_ratio: Number(multiViewTextClipping.toFixed(3)),
    coverage: Number(coverage.toFixed(3)),
    occlusion: Number(multiViewOcclusionScore.toFixed(3)),
    object_occlusion_ratio: Number(multiViewLabelObjectOcclusion.toFixed(3)),
    object_label_occlusion_ratio: Number(multiViewObjectOcclusion.toFixed(3)),
    label_object_occlusion_ratio: Number(multiViewLabelObjectOcclusion.toFixed(3)),
    object_penetration_ratio: Number(multiViewPenetration.toFixed(3)),
    penetration_method: 'five_view_front_back_depth_interval_proxy',
    mesh_surface_intersection: meshIntersections,
    mesh_surface_intersection_ratio: meshIntersections ? mean(Object.values(meshIntersections).map(item=>item.surface_intersection_ratio)) : null,
    text_measurement_method: 'projected_panel_height_and_character_width_proxy_not_canvas_font_measurement',
    overlap_pairs: overlapPairs,
    overlap_ratio: Number(multiViewOverlapRatio.toFixed(3)),
    label_label_occlusion_ratio: Number(multiViewOlr.toFixed(3)),
    olr: Number(finalOlr.toFixed(3)),
    leader_crossings: leaderCrossings,
    lcd: Number(lcd.toFixed(3)),
    visible_leader_overlaps: Number(multiViewStats.main?.visible_leader_overlaps || 0),
    visible_leader_overlap_ratio: Number(multiViewVisibleLeaderOverlap.toFixed(3)),
    dbv: Number(dbv.toFixed(3)),
    viewport_overflow_ratio: Number(multiViewOverflow.toFixed(3)),
    anchor_coverage: Number(anchorCoverage.toFixed(3)),
    mean_anchor_distance: Number(meanAnchorDistance.toFixed(3)),
    leader_length_compliance_ratio: leaderLengthCompliance === null ? null : Number(leaderLengthCompliance.toFixed(3)),
    leader_length_shortfall: leaderLengthShortfall === null ? null : Number(leaderLengthShortfall.toFixed(3)),
    leader_length_range: leaderLengthRange ? {
      source: leaderLengthRange.source,
      preferred_min: Number(leaderLengthRange.preferred_min.toFixed(6)),
      target: Number(leaderLengthRange.target.toFixed(6)),
      preferred_max: Number(leaderLengthRange.preferred_max.toFixed(6)),
      hard_min: Number(leaderLengthRange.hard_min.toFixed(6)),
      hard_max: Number(leaderLengthRange.hard_max.toFixed(6)),
      units: leaderLengthRange.units
    } : null,
    binocular_disparity: Number(binocularDisparity.toFixed(3)),
    objective_score: Number(objectiveScore.toFixed(3)),
    multidimensional_quality_score: Number(multidimensionalQuality.toFixed(3)),
    view_metrics: viewMetrics,
    multi_view_worst_olr: Number(Math.max(...multiViewNames.map((name) => multiViewStats[name].olr), 0).toFixed(3)),
    multi_view_worst_overflow: Number(Math.max(...multiViewNames.map((name) => multiViewStats[name].overflow_ratio), 0).toFixed(3)),
    multi_view_worst_visible_leader_overlap: Number(Math.max(...multiViewNames.map((name) => multiViewStats[name].visible_leader_overlap_ratio), 0).toFixed(3)),
    multi_view_worst_penetration: Number(Math.max(...multiViewNames.map((name) => multiViewStats[name].object_penetration), 0).toFixed(3)),
    manual_similarity: manualSimilarity,
    camera_protocol: DATASET_CAMERA_PROTOCOL.id,
    energy_weights: leaderLengthRange
      ? { overlap: 3.2, overflow: 4.5, crossing: 1.8, leader_overlap: 3.6, object_occlusion: 3.5, label_object_occlusion: 3.5, penetration: 5, text_clarity: 2, directional_density: 1.6, directional_concentration: 1.2, leader_shortfall: 4, leader_excess: 0.35, leader_target: 0.04, view_variance: 2.4, binocular: 2.8 }
      : { overlap: 3.2, overflow: 4.5, crossing: 1.8, leader_overlap: 3.6, object_occlusion: 3.5, label_object_occlusion: 3.5, penetration: 5, text_clarity: 2, directional_density: 1.6, directional_concentration: 1.2, anchor_distance: 0.12, view_variance: 2.4, binocular: 2.8 }
  };
}

export function annealLabels(labels, bounds, options = {}) {
  if (labels.length < 2) return { labels: cloneLabels(labels), initialScore: scoreLayout(labels, bounds, options), finalScore: scoreLayout(labels, bounds, options), iterations: 0 };
  const random = seededRandom(options.seed);
  const iterations = Math.max(20, Number(options.iterations || 180));
  const radius = bounds.radius;
  let current = cloneLabels(labels);
  let currentScore = scoreLayout(current, bounds, options);
  const initialScore = currentScore;
  let best = cloneLabels(current);
  let bestScore = currentScore;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const temperature = Math.max(0.035, 1 - iteration / iterations);
    const index = Math.floor(random() * current.length);
    const candidate = cloneLabels(current);
    const magnitude = radius * (0.2 * temperature + 0.008);
    candidate[index].center = add(candidate[index].center, [(random() * 2 - 1) * magnitude, (random() * 2 - 1) * magnitude, (random() * 2 - 1) * magnitude * 0.35]);
    if (options.leaderLengthRange) candidate[index] = clampLeaderLength(candidate[index], bounds, options.leaderLengthRange, 'hard');
    const candidateScore = scoreLayout(candidate, bounds, options);
    const accept = candidateScore <= currentScore || random() < Math.exp((currentScore - candidateScore) / Math.max(temperature, 0.05));
    if (accept) { current = candidate; currentScore = candidateScore; }
    if (currentScore < bestScore) { best = cloneLabels(current); bestScore = currentScore; }
  }
  return { labels: best, initialScore, finalScore: bestScore, iterations };
}

function projectedLeaderCrossings(labels, bounds, names = MULTI_VIEW_NAMES) {
  const pairs = [];
  const perView = {};
  for (const view of names) {
    const basis = cameraBasisForView(bounds, view);
    const projected = labels.map((label) => projectedLabel(label, basis, bounds));
    let count = 0;
    for (let left = 0; left < projected.length; left += 1) for (let right = left + 1; right < projected.length; right += 1) {
      if (!segmentsCross(projected[left].anchor, projected[left].center, projected[right].anchor, projected[right].center)) continue;
      count += 1;
      pairs.push([left, right]);
    }
    perView[view] = count;
  }
  return { pairs, per_view: perView, total: pairs.length };
}

// A deterministic geometric repair after annealing. Qwen never supplies these
// positions; if a collision-free arrangement cannot be found it stays unsafe
// and the downstream hard gate must refuse it rather than claim success.
export function repairLeaderCrossings(labels, bounds, options = {}) {
  const names = options.viewPolicy === 'single' ? ['main'] : MULTI_VIEW_NAMES;
  const initial = projectedLeaderCrossings(labels, bounds, names);
  if (!initial.total || labels.length < 2) return { labels, initial, final: initial, repaired: false };
  const rng = seededRandom(Number(options.seed ?? 17) + 830071);
  const maxSteps = Math.max(100, Number(options.crossingRepairIterations ?? 1200));
  let current = cloneLabels(labels), currentState = initial;
  let best = cloneLabels(current), bestState = initial;
  for (let step = 0; step < maxSteps && bestState.total; step += 1) {
    const pair = currentState.pairs[Math.floor(rng() * currentState.pairs.length)];
    const index = pair ? pair[Math.floor(rng() * 2)] : Math.floor(rng() * labels.length);
    const candidate = cloneLabels(current);
    const label = candidate[index];
    const original = label.center.map((value, axis) => value - label.anchor[axis]);
    const length = Math.max(bounds.radius * 0.05, Math.hypot(...original));
    const scatter = step % 5 === 0 ? 1 : 0.20 + 0.45 * (1 - step / maxSteps);
    const direction = original.map((value) => value / length + (rng() * 2 - 1) * scatter);
    const directionLength = Math.max(1e-8, Math.hypot(...direction));
    label.center = label.anchor.map((value, axis) => value + direction[axis] / directionLength * length);
    if (options.leaderLengthRange) candidate[index] = clampLeaderLength(label, bounds, options.leaderLengthRange, 'hard');
    const next = projectedLeaderCrossings(candidate, bounds, names);
    const temperature = Math.max(0.08, 0.65 * (1 - step / maxSteps));
    if (next.total <= currentState.total || rng() < Math.exp((currentState.total - next.total) / temperature)) {
      current = candidate;
      currentState = next;
    }
    if (next.total < bestState.total) {
      best = cloneLabels(candidate);
      bestState = next;
    }
  }
  return { labels: bestState.total === 0 ? best : labels, initial, final: bestState, repaired: bestState.total === 0 };
}

export function optimizeLabels(labels, bounds, options = {}) {
  const leaderLengthRange = options.leaderLengthRange || resolveLeaderLengthRange(options.leaderLengthPrior, options.category);
  const styleRequested = options.layoutStyle || options.styleExpert || null;
  const normalized = { viewPolicy: options.fixedLabels ? 'binocular' : options.viewPolicy || 'binocular', groupPolicy: options.groupPolicy || 'all', sizePolicy: options.sizePolicy || 'relative', optimizer: options.optimizer || 'rules', seed: options.seed ?? 17, iterations: options.iterations || 180, depthGrids: options.depthGrids, fixedLabels: Boolean(options.fixedLabels), energyWeights: options.energyWeights, category: options.category || null, leaderLengthPrior: options.leaderLengthPrior || null, leaderLengthRange, layoutStyle: styleRequested, styleView: options.styleView || 'main', geometry: options.geometry || null };
  const grouped = normalized.fixedLabels ? labels : groupLabels(labels, normalized.groupPolicy, bounds.radius);
  let output = applySizePolicy(grouped, bounds, normalized.sizePolicy);
  if (styleRequested) output = applyMoEStyleLayout(output, bounds, normalized.geometry, { style: styleRequested === true ? 'auto' : styleRequested, view: normalized.styleView, seed: normalized.seed });
  if (leaderLengthRange) output = output.map((label) => clampLeaderLength(label, bounds, leaderLengthRange, 'preferred'));
  if (normalized.viewPolicy !== 'single') output = output.map((label) => ({ ...label, view_constraints: { views: [...MULTI_VIEW_NAMES], stereo: ['left_eye', 'right_eye'], objective: 'joint_five_view_energy' } }));
  if (normalized.optimizer === 'annealing') {
    const result = annealLabels(output, bounds, normalized);
    const repaired = normalized.fixedLabels && normalized.viewPolicy !== 'single' ? repairLeaderCrossings(result.labels, bounds, normalized) : { labels: result.labels, repaired: false };
    output = repaired.labels.map((label) => ({ ...label, optimizer: 'simulated_annealing', objective: { initial_score: result.initialScore, final_score: scoreLayout(repaired.labels, bounds, normalized), iterations: result.iterations, crossing_repair_succeeded: repaired.repaired, views: normalized.viewPolicy === 'single' ? ['main'] : [...MULTI_VIEW_NAMES], terms: ['multi_view_overlap', 'multi_view_overflow', 'multi_view_leader_crossing', 'visible_leader_overlap', 'multi_view_object_occlusion', 'directional_label_count_vs_free_space', ...(styleRequested ? ['moe_style_initializer'] : []), ...(leaderLengthRange ? ['train_manual_leader_length_range'] : ['anchor_distance']), 'view_variance', 'binocular_disparity'] }, object_center: bounds.center }));
  } else {
    output = output.map((label) => ({ ...label, optimizer: 'rules_baseline', objective: { terms: ['grouping', 'size_policy', ...(styleRequested ? ['moe_style_initializer'] : [])], score: scoreLayout(output, bounds, normalized) }, object_center: bounds.center }));
  }
  return output;
}

export { scoreLayoutStyleExperts };

