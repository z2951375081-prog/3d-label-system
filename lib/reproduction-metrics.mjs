import { DATASET_CAMERA_PROTOCOL, datasetCameraForBounds } from '../public/dataset-camera.js';

const WIDTH = DATASET_CAMERA_PROTOCOL.imageWidth;
const HEIGHT = DATASET_CAMERA_PROTOCOL.imageHeight;
const IMAGE_DIAGONAL = Math.hypot(WIDTH, HEIGHT);
const IPD_METERS = 0.064;
const objectRectCache = new WeakMap();

function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale(a, value) { return [a[0] * value, a[1] * value, a[2] * value]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm(a, fallback = [1, 0, 0]) { const length = Math.hypot(...a); return length > 1e-12 ? scale(a, 1 / length) : [...fallback]; }
function distance2(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }
function round(value, digits = 9) { return Number(Number(value).toFixed(digits)); }
function clamp01(value) { return Math.max(0, Math.min(1, Number(value) || 0)); }

export const REPRODUCTION_QUALITY_WEIGHTS = Object.freeze({
  PCK_005: 0.10,
  PCK_010: 0.15,
  overlap_ratio: 0.20,
  crossing_degree: 0.15,
  overlap_pairs: 0.15,
  occluded_points: 0.10,
  intersections: 0.10,
  leader_length: 0.05
});

export const REPRODUCTION_QUALITY_WEIGHTS_V4 = Object.freeze({
  PCK_005: 0.15,
  PCK_010: 0.20,
  overlap_ratio: 0.15,
  crossing_degree: 0.20,
  overlap_pairs: 0.10,
  occluded_points: 0.08,
  intersections: 0.07,
  leader_length: 0.05
});

function reproductionQualityScoreWithWeights(metrics, labelCount, weights) {
  const labels = Math.max(1, Number(labelCount) || 1);
  const pairCapacity = Math.max(1, labels * (labels - 1) * 0.5);
  const components = {
    PCK_005: clamp01(metrics?.PCK_005),
    PCK_010: clamp01(metrics?.PCK_010),
    overlap_ratio: 1 - clamp01(Number(metrics?.OLR) / 0.25),
    crossing_degree: 1 - clamp01(Number(metrics?.LCD) / 0.10),
    overlap_pairs: 1 - clamp01(Number(metrics?.overlap_pairs) / pairCapacity),
    occluded_points: 1 - clamp01(Number(metrics?.occluded_points) / labels),
    intersections: 1 - clamp01(Number(metrics?.intersections) / pairCapacity),
    leader_length: 1 - clamp01(Number(metrics?.avg_leader_length) / 0.25)
  };
  return round(100 * Object.entries(weights)
    .reduce((sum, [name, weight]) => sum + weight * components[name], 0));
}

export function reproductionQualityScore(metrics, labelCount) {
  return reproductionQualityScoreWithWeights(metrics, labelCount, REPRODUCTION_QUALITY_WEIGHTS);
}

export function reproductionQualityScoreV4(metrics, labelCount) {
  return reproductionQualityScoreWithWeights(metrics, labelCount, REPRODUCTION_QUALITY_WEIGHTS_V4);
}

function eyeCamera(bounds, view, eyeOffset = 0) {
  const camera = datasetCameraForBounds(bounds, view);
  const baseEye = scale(camera.outward, DATASET_CAMERA_PROTOCOL.cameraDistance);
  return { ...camera, target: [0, 0, 0], eye: eyeOffset ? add(baseEye, scale(camera.right, eyeOffset)) : baseEye, eyeOffset };
}

function projectPoint(point, camera) {
  const relative = sub(point, camera.eye);
  const depth = Math.max(1e-6, dot(relative, camera.forward));
  const x = dot(relative, camera.right) / depth * camera.focalLengthMm / (camera.sensorWidthMm * 0.5);
  const y = dot(relative, camera.up) / depth * camera.focalLengthMm / (camera.sensorHeightMm * 0.5);
  return [(x + 1) * 0.5 * (WIDTH - 1), (1 - (y + 1) * 0.5) * (HEIGHT - 1), depth];
}

function projectedLabelRect(label, camera) {
  const center = label.center.map(Number);
  const half = label.boxSize.map((value) => Math.abs(Number(value)) * 0.5);
  const axisX = norm(camera.right), axisY = norm(camera.up), axisZ = norm(scale(camera.forward, -1));
  const points = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    points.push(projectPoint(add(add(add(center, scale(axisX, sx * half[0])), scale(axisY, sy * half[1])), scale(axisZ, sz * half[2])), camera));
  }
  return [Math.min(...points.map((point) => point[0])), Math.min(...points.map((point) => point[1])), Math.max(...points.map((point) => point[0])), Math.max(...points.map((point) => point[1]))];
}

function projectedSimpleLabelRect(label, camera) {
  const center = projectPoint(label.center, camera);
  const depth = Math.max(center[2], 1e-6);
  const width = Math.abs(label.boxSize[0]) * camera.focalLengthMm / (depth * camera.sensorWidthMm * 0.5) * 0.5 * (WIDTH - 1);
  const height = Math.abs(label.boxSize[1]) * camera.focalLengthMm / (depth * camera.sensorHeightMm * 0.5) * 0.5 * (HEIGHT - 1);
  return [center[0] - width * 0.5, center[1] - height * 0.5, center[0] + width * 0.5, center[1] + height * 0.5];
}

function objectRect(geometry, camera) {
  if (!geometry) return null;
  let cached = objectRectCache.get(geometry);
  if (!cached) { cached = new Map(); objectRectCache.set(geometry, cached); }
  const key = `${camera.view}:${camera.eyeOffset || 0}`;
  if (cached.has(key)) return cached.get(key);
  const vertices = geometry?.triangles?.flat() || geometry?.vertices || [];
  if (!vertices.length) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const vertex of vertices) {
    const point = projectPoint(vertex, camera);
    minX = Math.min(minX, point[0]); minY = Math.min(minY, point[1]);
    maxX = Math.max(maxX, point[0]); maxY = Math.max(maxY, point[1]);
  }
  const rect = [minX, minY, maxX, maxY];
  cached.set(key, rect);
  return rect;
}

function rectArea(rect) { return Math.max(1e-9, Math.max(0, rect[2] - rect[0]) * Math.max(0, rect[3] - rect[1])); }
function overlapArea(a, b) { return Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])); }
function pointInsideRect(point, rect) { return point[0] >= rect[0] && point[0] <= rect[2] && point[1] >= rect[1] && point[1] <= rect[3]; }
function orientation(a, b, c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); }
function onSegment(a, b, point) { return point[0] >= Math.min(a[0], b[0]) - 1e-9 && point[0] <= Math.max(a[0], b[0]) + 1e-9 && point[1] >= Math.min(a[1], b[1]) - 1e-9 && point[1] <= Math.max(a[1], b[1]) + 1e-9; }
function segmentsIntersect(a, b, c, d) {
  const o1 = orientation(a, b, c), o2 = orientation(a, b, d), o3 = orientation(c, d, a), o4 = orientation(c, d, b);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) return true;
  return (Math.abs(o1) <= 1e-9 && onSegment(a, b, c)) || (Math.abs(o2) <= 1e-9 && onSegment(a, b, d)) || (Math.abs(o3) <= 1e-9 && onSegment(c, d, a)) || (Math.abs(o4) <= 1e-9 && onSegment(c, d, b));
}
function segmentIntersection(a, b, c, d) {
  const denominator = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]);
  if (Math.abs(denominator) <= 1e-12) return [(a[0] + b[0] + c[0] + d[0]) * 0.25, (a[1] + b[1] + c[1] + d[1]) * 0.25];
  const determinant1 = a[0] * b[1] - a[1] * b[0];
  const determinant2 = c[0] * d[1] - c[1] * d[0];
  return [(determinant1 * (c[0] - d[0]) - (a[0] - b[0]) * determinant2) / denominator, (determinant1 * (c[1] - d[1]) - (a[1] - b[1]) * determinant2) / denominator];
}
function triangleArea(a, b, c) { return Math.abs(orientation(a, b, c)) * 0.5; }

function overlapAreas(labels, geometry, bounds, view, eyeOffset) {
  const camera = eyeCamera(bounds, view, eyeOffset);
  const rects = labels.map((label) => projectedSimpleLabelRect(label, camera));
  const object = objectRect(geometry, camera);
  return rects.map((rect, index) => {
    let total = object ? overlapArea(rect, object) : 0;
    for (let other = 0; other < rects.length; other += 1) if (other !== index) total += overlapArea(rect, rects[other]);
    return total;
  });
}

export function evaluateReproductionMetrics({ labels, manualLabels, geometry, bounds, view = 'main', dbvAvailable = true }) {
  if (labels.length !== manualLabels.length) throw new Error(`Metric label count mismatch: ${labels.length} != ${manualLabels.length}`);
  const camera = eyeCamera(bounds, view, 0);
  const anchors = labels.map((label) => projectPoint(label.anchor, camera));
  const centers = labels.map((label) => projectPoint(label.center, camera));
  const manualCenters = manualLabels.map((label) => projectPoint(label.center, camera));
  const rects = labels.map((label) => projectedLabelRect(label, camera));
  const object = objectRect(geometry, camera);
  const perLabelOverlap = Array(labels.length).fill(0);
  let overlapPairs = 0;
  for (let index = 0; index < labels.length; index += 1) {
    for (let other = index + 1; other < labels.length; other += 1) {
      const area = overlapArea(rects[index], rects[other]);
      if (area <= 1e-9) continue;
      overlapPairs += 1;
      perLabelOverlap[index] += area;
      perLabelOverlap[other] += area;
    }
    if (object) perLabelOverlap[index] += overlapArea(rects[index], object);
  }
  let occludedPoints = 0;
  for (let index = 0; index < labels.length; index += 1) for (let other = 0; other < anchors.length; other += 1) if (index !== other && pointInsideRect(anchors[other], rects[index])) occludedPoints += 1;
  let intersections = 0;
  let lcdTotal = 0;
  for (let index = 0; index < labels.length; index += 1) for (let other = index + 1; other < labels.length; other += 1) {
    if (!segmentsIntersect(anchors[index], centers[index], anchors[other], centers[other])) continue;
    intersections += 1;
    const point = segmentIntersection(anchors[index], centers[index], anchors[other], centers[other]);
    const area1 = triangleArea(point, centers[index], centers[other]);
    const area2 = triangleArea(point, anchors[index], anchors[other]);
    lcdTotal += Math.min(area1, area2) / Math.max(area1 + area2, 1e-12);
  }
  const centerErrors = centers.map((center, index) => distance2(center, manualCenters[index]) / IMAGE_DIAGONAL);
  const pck005 = mean(centerErrors.map((error) => error <= 0.05 ? 1 : 0));
  const pck010 = mean(centerErrors.map((error) => error <= 0.10 ? 1 : 0));
  const olr = mean(rects.map((rect, index) => perLabelOverlap[index] / rectArea(rect)));
  const lcd = labels.length ? lcdTotal / labels.length : 0;
  const avgLeaderLength = mean(centers.map((center, index) => distance2(center, anchors[index]))) / IMAGE_DIAGONAL;
  let dbv = null;
  if (dbvAvailable && labels.length) {
    const centerAreas = labels.map((label) => rectArea(projectedSimpleLabelRect(label, camera)));
    const left = overlapAreas(labels, geometry, bounds, view, -IPD_METERS * 0.5);
    const right = overlapAreas(labels, geometry, bounds, view, IPD_METERS * 0.5);
    dbv = mean(labels.map((_, index) => Math.abs(left[index] - right[index]) / Math.max(centerAreas[index], 1e-9)));
  }
  const qualityScore = reproductionQualityScore({ PCK_005: pck005, PCK_010: pck010, OLR: olr, LCD: lcd,
    avg_leader_length: avgLeaderLength, overlap_pairs: overlapPairs, occluded_points: occludedPoints, intersections }, labels.length);
  return { PCK_005: round(pck005), PCK_010: round(pck010), OLR: round(olr), LCD: round(lcd), DBV: dbv === null ? null : round(dbv), avg_leader_length: round(avgLeaderLength), overlap_pairs: overlapPairs, occluded_points: occludedPoints, intersections, quality_score: round(qualityScore) };
}

export const REPRODUCTION_METRIC_DIRECTIONS = Object.freeze({ PCK_005: 'higher', PCK_010: 'higher', OLR: 'lower', LCD: 'lower', DBV: 'lower', avg_leader_length: 'lower', overlap_pairs: 'lower', occluded_points: 'lower', intersections: 'lower', quality_score: 'higher' });
export const REPRODUCTION_METRIC_PROTOCOL = Object.freeze({ id: 'unified_reproduction_metrics_v3_safety_weighted_low_pck', camera: 'Hedgehog/BinoForce fixed-origin 50mm perspective cameras', image_size: [WIDTH, HEIGHT], camera_target: [0, 0, 0], camera_distance: DATASET_CAMERA_PROTOCOL.cameraDistance, manual_reference: 'data/Layout after_manual_adjust label centers', dbv_ipd_meters: IPD_METERS, quality_score: { scale: 100, pck_total_weight: 0.25, safety_and_legibility_weight: 0.75, weights: REPRODUCTION_QUALITY_WEIGHTS, normalization: { OLR: 0.25, LCD: 0.10, avg_leader_length: 0.25, overlap_pairs: 'label_pair_capacity', occluded_points: 'label_count', intersections: 'label_pair_capacity' } } });
export const REPRODUCTION_METRIC_PROTOCOL_V4 = Object.freeze({ id: 'unified_reproduction_metrics_v4_pck_lcd_emphasis', camera: 'Hedgehog/BinoForce fixed-origin 50mm perspective cameras', image_size: [WIDTH, HEIGHT], camera_target: [0, 0, 0], camera_distance: DATASET_CAMERA_PROTOCOL.cameraDistance, manual_reference: 'data/Layout after_manual_adjust label centers', dbv_ipd_meters: IPD_METERS, quality_score: { scale: 100, pck_total_weight: 0.35, lcd_weight: 0.20, safety_and_legibility_weight: 0.65, weights: REPRODUCTION_QUALITY_WEIGHTS_V4, normalization: { OLR: 0.25, LCD: 0.10, avg_leader_length: 0.25, overlap_pairs: 'label_pair_capacity', occluded_points: 'label_count', intersections: 'label_pair_capacity' } } });
