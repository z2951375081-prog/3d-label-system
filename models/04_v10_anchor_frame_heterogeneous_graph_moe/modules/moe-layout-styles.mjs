import { DATASET_CAMERA_PROTOCOL, datasetCameraForBounds } from '../public/dataset-camera.js';

const EPS = 1e-8;
const STYLE_NAMES = ['spherical', 'rectangular', 'surround'];

function clamp(value, min, max) { return Math.max(min, Math.min(max, Number(value))); }
function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }
function std(values) { const avg = mean(values); return Math.sqrt(mean(values.map((value) => (value - avg) ** 2))); }
function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale(a, value) { return [a[0] * value, a[1] * value, a[2] * value]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function finite(value, fallback = 0) { return Number.isFinite(Number(value)) ? Number(value) : fallback; }
function normalizeWeights(raw) {
  const clipped = Object.fromEntries(STYLE_NAMES.map((name) => [name, Math.max(0.001, finite(raw[name], 0.001))]));
  const total = Object.values(clipped).reduce((sum, value) => sum + value, 0) || 1;
  return Object.fromEntries(Object.entries(clipped).map(([key, value]) => [key, Number((value / total).toFixed(6))]));
}
function circularDistance(left, right) {
  const value = Math.abs(left - right) % (Math.PI * 2);
  return Math.min(value, Math.PI * 2 - value);
}
function viewBasis(bounds, view) { return datasetCameraForBounds(bounds, view || 'main'); }
function project(point, basis) {
  const relative = [point[0] - basis.eye[0], point[1] - basis.eye[1], point[2] - basis.eye[2]];
  const depth = Math.max(1e-6, dot(relative, basis.forward));
  const focalLength = basis.focalLengthMm || DATASET_CAMERA_PROTOCOL.focalLengthMm;
  const halfSensorWidth = (basis.sensorWidthMm || DATASET_CAMERA_PROTOCOL.sensorWidthMm) / 2;
  const halfSensorHeight = (basis.sensorHeightMm || DATASET_CAMERA_PROTOCOL.sensorHeightMm) / 2;
  return { x: dot(relative, basis.right) / depth * focalLength / halfSensorWidth, y: dot(relative, basis.up) / depth * focalLength / halfSensorHeight, depth };
}
function unproject(point, basis) {
  const focalLength = basis.focalLengthMm || DATASET_CAMERA_PROTOCOL.focalLengthMm;
  const halfSensorWidth = (basis.sensorWidthMm || DATASET_CAMERA_PROTOCOL.sensorWidthMm) / 2;
  const halfSensorHeight = (basis.sensorHeightMm || DATASET_CAMERA_PROTOCOL.sensorHeightMm) / 2;
  return add(add(add(basis.eye, scale(basis.forward, point.depth)), scale(basis.right, point.x * point.depth * halfSensorWidth / focalLength)), scale(basis.up, point.y * point.depth * halfSensorHeight / focalLength));
}
function bboxFromPoints(points) {
  if (!points.length) return { minX: -0.35, maxX: 0.35, minY: -0.35, maxY: 0.35, width: 0.7, height: 0.7, cx: 0, cy: 0 };
  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  return { minX, maxX, minY, maxY, width: Math.max(EPS, maxX - minX), height: Math.max(EPS, maxY - minY), cx: (minX + maxX) / 2, cy: (minY + maxY) / 2 };
}
function hull(points) {
  const sorted = [...points].sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
  if (sorted.length <= 3) return sorted;
  const cross = (origin, left, right) => (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x);
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper = [];
  for (let index = sorted.length - 1; index >= 0; index -= 1) {
    const point = sorted[index];
    while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
    upper.push(point);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}
function geometryPoints(geometry, bounds) {
  const vertices = [];
  for (const triangle of geometry?.triangles || []) for (const point of triangle || []) if (Array.isArray(point) && point.length >= 3) vertices.push(point);
  if (vertices.length) return vertices;
  const [minX, minY, minZ] = bounds.min;
  const [maxX, maxY, maxZ] = bounds.max;
  return [
    [minX, minY, minZ], [minX, minY, maxZ], [minX, maxY, minZ], [minX, maxY, maxZ],
    [maxX, minY, minZ], [maxX, minY, maxZ], [maxX, maxY, minZ], [maxX, maxY, maxZ]
  ];
}
function projectedLabelHalfSize(label, basis) {
  const center = project(label.center || label.anchor, basis);
  const horizontal = project(add(label.center || label.anchor, scale(basis.right, Math.abs(label.boxSize?.[0] || 0.1) / 2)), basis);
  const vertical = project(add(label.center || label.anchor, scale(basis.up, Math.abs(label.boxSize?.[1] || 0.04) / 2)), basis);
  return { width: Math.max(0.006, Math.abs(horizontal.x - center.x)), height: Math.max(0.006, Math.abs(vertical.y - center.y)) };
}
function angularEntropy(angles, bins = 8) {
  if (!angles.length) return 0;
  const counts = Array.from({ length: bins }, () => 0);
  for (const angle of angles) counts[Math.max(0, Math.min(bins - 1, Math.floor(((angle + Math.PI) / (Math.PI * 2)) * bins)))] += 1;
  const shares = counts.map((count) => count / angles.length).filter((share) => share > 0);
  return clamp(-shares.reduce((sum, share) => sum + share * Math.log(share), 0) / Math.log(Math.max(2, Math.min(bins, angles.length))), 0, 1);
}
function styleFromScores(scores) { return STYLE_NAMES.reduce((best, name) => scores[name] > scores[best] ? name : best, STYLE_NAMES[0]); }

export function extractObjectContourPointCloud(geometry, bounds, options = {}) {
  const basis = viewBasis(bounds, options.view || 'main');
  const projected = geometryPoints(geometry, bounds).map((point) => ({ ...project(point, basis), source: point }));
  const projectedHull = hull(projected);
  const bbox = bboxFromPoints(projectedHull.length ? projectedHull : projected);
  const cx = bbox.cx;
  const cy = bbox.cy;
  const radial = (projectedHull.length ? projectedHull : projected).map((point) => Math.hypot(point.x - cx, point.y - cy));
  const radialMean = Math.max(EPS, mean(radial));
  return {
    view: options.view || 'main',
    points: projected,
    hull: projectedHull,
    bbox,
    radial_mean: Number(radialMean.toFixed(6)),
    radial_std: Number(std(radial).toFixed(6)),
    radial_cv: Number(clamp(std(radial) / radialMean, 0, 2).toFixed(6)),
    projected_aspect: Number((Math.max(bbox.width, bbox.height) / Math.max(EPS, Math.min(bbox.width, bbox.height))).toFixed(6))
  };
}

export function computeLayoutStyleFeatures(labels = [], bounds, geometry = null, options = {}) {
  const view = options.view || 'main';
  const basis = viewBasis(bounds, view);
  const contour = extractObjectContourPointCloud(geometry, bounds, { view });
  const shape = (bounds.size || [1, 1, 1]).map((value) => Math.abs(value) / Math.max(bounds.radius || 1, EPS));
  const aspect3d = Math.max(...shape) / Math.max(EPS, Math.min(...shape));
  const anchorProjection = labels.map((label) => ({ label, point: project(label.anchor, basis) }));
  const angles = anchorProjection.map(({ point }) => Math.atan2(point.y - contour.bbox.cy, point.x - contour.bbox.cx));
  const distances = anchorProjection.map(({ point }) => Math.hypot(point.x - contour.bbox.cx, point.y - contour.bbox.cy));
  const anchorEntropy = angularEntropy(angles);
  const radialUniformity = 1 - clamp(std(distances) / Math.max(EPS, mean(distances)), 0, 1);
  const compactness = 1 - clamp((contour.projected_aspect - 1) / 2.2, 0, 1);
  const elongated2d = clamp((contour.projected_aspect - 1.35) / 2.7, 0, 1);
  const elongated3d = clamp((aspect3d - 1.6) / 3.4, 0, 1);
  const contourRegularity = 1 - clamp(contour.radial_cv / 0.42, 0, 1);
  return {
    view,
    label_count: labels.length,
    projected_aspect: contour.projected_aspect,
    aspect_3d: Number(aspect3d.toFixed(6)),
    contour_radial_cv: contour.radial_cv,
    contour_regularity: Number(contourRegularity.toFixed(6)),
    anchor_angular_entropy: Number(anchorEntropy.toFixed(6)),
    anchor_radial_uniformity: Number(radialUniformity.toFixed(6)),
    compactness: Number(compactness.toFixed(6)),
    elongated_2d: Number(elongated2d.toFixed(6)),
    elongated_3d: Number(elongated3d.toFixed(6)),
    contour
  };
}

export function scoreLayoutStyleExperts(labels = [], bounds, geometry = null, options = {}) {
  const features = computeLayoutStyleFeatures(labels, bounds, geometry, options);
  const elongation = Math.max(features.elongated_3d, features.elongated_2d * 0.45);
  const spherePenalty = clamp(features.elongated_3d * 0.55 + features.elongated_2d * 0.12, 0, 0.7);
  const raw = {
    spherical: (0.18 + features.compactness * 0.32 + features.contour_regularity * 0.22 + features.anchor_angular_entropy * 0.18 + features.anchor_radial_uniformity * 0.10) * (1 - spherePenalty),
    rectangular: 0.14 + elongation * 0.72 + (1 - features.compactness) * 0.14 + features.anchor_angular_entropy * 0.06,
    surround: 0.16 + (1 - features.contour_regularity) * 0.27 + (1 - features.anchor_radial_uniformity) * 0.20 + (1 - features.anchor_angular_entropy) * 0.12 + (1 - elongation) * 0.10
  };
  const weights = normalizeWeights(raw);
  return { selected: styleFromScores(weights), weights, raw_scores: raw, features: { ...features, contour: undefined }, contour: features.contour };
}

function sortedByAnchorAngle(labels, bounds, basis, contour) {
  return labels.map((label, index) => {
    const point = project(label.anchor, basis);
    return { label, index, point, angle: Math.atan2(point.y - contour.bbox.cy, point.x - contour.bbox.cx) };
  }).sort((a, b) => a.angle - b.angle || a.index - b.index);
}
function viewportClamp(point, half) {
  return { ...point, x: clamp(point.x, -0.98 + half.width, 0.98 - half.width), y: clamp(point.y, -0.98 + half.height, 0.98 - half.height) };
}
function annotate(label, center, style, routing, extra = {}) {
  return {
    ...label,
    center,
    bendPoints: [],
    layout_style_expert: style,
    moe_style_routing: routing.weights,
    moe_style_selected: routing.selected,
    moe_style_features: routing.features,
    initialization_source: `moe_${style}_style_initializer`,
    ...extra
  };
}

function sphericalLayout(labels, bounds, geometry, options, routing) {
  const basis = viewBasis(bounds, options.view || 'main');
  const contour = routing.contour || extractObjectContourPointCloud(geometry, bounds, options);
  const ordered = sortedByAnchorAngle(labels, bounds, basis, contour);
  const count = Math.max(1, ordered.length);
  const angleByIndex = new Map();
  const minGap = (Math.PI * 2) / count * 0.72;
  let previous = -Infinity;
  for (let order = 0; order < ordered.length; order += 1) {
    let angle = ordered[order].angle;
    if (order && circularDistance(angle, previous) < minGap && angle <= previous + minGap) angle = previous + minGap;
    previous = angle;
    angleByIndex.set(ordered[order].index, angle);
  }
  return labels.map((label, index) => {
    const anchor = project(label.anchor, basis);
    const half = projectedLabelHalfSize(label, basis);
    const baseAngle = angleByIndex.get(index) ?? Math.atan2(anchor.y - contour.bbox.cy, anchor.x - contour.bbox.cx);
    const margin = Number(options.margin ?? 0.08) + Math.max(half.width, half.height) * 1.15;
    const rx = Math.min(0.92 - half.width, contour.bbox.width / 2 + margin + half.width * 0.7);
    const ry = Math.min(0.92 - half.height, contour.bbox.height / 2 + margin + half.height * 0.7);
    const target = viewportClamp({ x: contour.bbox.cx + Math.cos(baseAngle) * rx, y: contour.bbox.cy + Math.sin(baseAngle) * ry, depth: anchor.depth }, half);
    return annotate(label, unproject(target, basis), 'spherical', routing, { spherical_arc_angle: Number(baseAngle.toFixed(6)) });
  });
}

function sideForPoint(point, bbox) {
  const dxLeft = Math.abs(point.x - bbox.minX);
  const dxRight = Math.abs(point.x - bbox.maxX);
  const dyTop = Math.abs(point.y - bbox.maxY);
  const dyBottom = Math.abs(point.y - bbox.minY);
  const min = Math.min(dxLeft, dxRight, dyTop, dyBottom);
  if (min === dxLeft) return 'left';
  if (min === dxRight) return 'right';
  if (min === dyTop) return 'top';
  return 'bottom';
}
function rectangularLayout(labels, bounds, geometry, options, routing) {
  const basis = viewBasis(bounds, options.view || 'main');
  const contour = routing.contour || extractObjectContourPointCloud(geometry, bounds, options);
  const projected = labels.map((label, index) => ({ label, index, anchor: project(label.anchor, basis), half: projectedLabelHalfSize(label, basis) }));
  const margin = Number(options.margin ?? 0.08);
  const sides = { left: [], right: [], top: [], bottom: [] };
  for (const item of projected) sides[sideForPoint(item.anchor, contour.bbox)].push(item);
  sides.left.sort((a, b) => b.anchor.y - a.anchor.y);
  sides.right.sort((a, b) => b.anchor.y - a.anchor.y);
  sides.top.sort((a, b) => a.anchor.x - b.anchor.x);
  sides.bottom.sort((a, b) => a.anchor.x - b.anchor.x);
  const targetByIndex = new Map();
  for (const [side, items] of Object.entries(sides)) {
    items.forEach((item, order) => {
      const t = (order + 1) / (items.length + 1);
      let point;
      if (side === 'left' || side === 'right') point = { x: side === 'left' ? contour.bbox.minX - margin - item.half.width : contour.bbox.maxX + margin + item.half.width, y: contour.bbox.maxY - t * contour.bbox.height, depth: item.anchor.depth };
      else point = { x: contour.bbox.minX + t * contour.bbox.width, y: side === 'top' ? contour.bbox.maxY + margin + item.half.height : contour.bbox.minY - margin - item.half.height, depth: item.anchor.depth };
      targetByIndex.set(item.index, { ...viewportClamp(point, item.half), side });
    });
  }
  return labels.map((label, index) => {
    const target = targetByIndex.get(index);
    return annotate(label, unproject(target, basis), 'rectangular', routing, { rectangular_side: target.side });
  });
}

function surroundLayout(labels, bounds, geometry, options, routing) {
  const basis = viewBasis(bounds, options.view || 'main');
  const contour = routing.contour || extractObjectContourPointCloud(geometry, bounds, options);
  return labels.map((label) => {
    const anchor = project(label.anchor, basis);
    const half = projectedLabelHalfSize(label, basis);
    const dx = anchor.x - contour.bbox.cx;
    const dy = anchor.y - contour.bbox.cy;
    const angle = Math.atan2(dy, dx);
    const margin = Number(options.margin ?? 0.075) + Math.max(half.width, half.height) * 0.8;
    const rx = contour.bbox.width / 2 + margin;
    const ry = contour.bbox.height / 2 + margin;
    const target = viewportClamp({ x: contour.bbox.cx + Math.cos(angle) * rx, y: contour.bbox.cy + Math.sin(angle) * ry, depth: anchor.depth }, half);
    return annotate(label, unproject(target, basis), 'surround', routing, { surround_angle: Number(angle.toFixed(6)) });
  });
}

export function applyMoEStyleLayout(labels = [], bounds, geometry = null, options = {}) {
  if (!Array.isArray(labels) || !labels.length) return [];
  const autoRouting = scoreLayoutStyleExperts(labels, bounds, geometry, options);
  const requested = String(options.style || options.layoutStyle || 'auto').toLowerCase();
  const selected = STYLE_NAMES.includes(requested) ? requested : autoRouting.selected;
  const routing = { ...autoRouting, selected };
  if (selected === 'rectangular') return rectangularLayout(labels, bounds, geometry, options, routing);
  if (selected === 'surround') return surroundLayout(labels, bounds, geometry, options, routing);
  return sphericalLayout(labels, bounds, geometry, options, routing);
}

export const MOE_STYLE_EXPERTS = STYLE_NAMES;


