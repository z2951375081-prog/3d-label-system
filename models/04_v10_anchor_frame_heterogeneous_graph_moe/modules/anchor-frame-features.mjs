// Anchor-conditioned local surface frames and heterogeneous graph edge
// contracts for the next pure-3D layout candidate.  This module is camera
// independent: view-space terms stay in the evaluator/safety branch.

const EPS = 1e-8;
const add = (a, b) => a.map((value, axis) => value + b[axis]);
const sub = (a, b) => a.map((value, axis) => value - b[axis]);
const scale = (a, value) => a.map((entry) => entry * value);
const dot = (a, b) => a.reduce((sum, value, axis) => sum + value * b[axis], 0);
const norm = (a) => Math.max(Math.hypot(...a), EPS);
const normalize = (a, fallback = [1, 0, 0]) => { const length = norm(a); return length > EPS ? scale(a, 1 / length) : [...fallback]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;

export const ANCHOR_LABEL_EDGE_FEATURE_NAMES = [
  'local_u', 'local_v', 'local_normal_distance',
  'anchor_normal_x', 'anchor_normal_y', 'anchor_normal_z',
  'leader_direction_x', 'leader_direction_y', 'leader_direction_z',
  'leader_length', 'normal_alignment', 'surface_curvature',
  'patch_extent', 'box_x', 'box_y', 'box_z',
  'same_source_part', 'same_target_group'
];

export const LABEL_RELATION_EDGE_FEATURE_NAMES = [
  'relative_center_x', 'relative_center_y', 'relative_center_z', 'center_distance',
  'size_ratio_x', 'size_ratio_y', 'size_ratio_z',
  'direction_cosine', 'same_source_part', 'same_target_group'
];

function triangleInfo(triangle) {
  const [a, b, c] = triangle || [];
  if (!a || !b || !c) return null;
  const ab = sub(b, a), ac = sub(c, a);
  const normalRaw = cross(ab, ac);
  const doubleArea = Math.hypot(...normalRaw);
  if (doubleArea <= EPS) return null;
  const area = doubleArea * 0.5;
  return {
    centroid: [0, 1, 2].map((axis) => (a[axis] + b[axis] + c[axis]) / 3),
    normal: normalize(normalRaw, [0, 1, 0]),
    area,
    doubleArea
  };
}

function projectToPlane(vector, normal) { return sub(vector, scale(normal, dot(vector, normal))); }

function powerIteration(matrix, seed, iterations = 48) {
  let vector = normalize(seed);
  for (let iteration = 0; iteration < iterations; iteration += 1) vector = normalize(matrix.map((row) => dot(row, vector)));
  return vector;
}

function weightedCovariance(points, weights, center) {
  const covariance = Array.from({ length: 3 }, () => Array(3).fill(0));
  for (let index = 0; index < points.length; index += 1) {
    const delta = sub(points[index], center);
    for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) covariance[row][column] += weights[index] * delta[row] * delta[column];
  }
  return covariance;
}

function nearestTriangleInfos(anchor, geometry, bounds, options = {}) {
  const triangles = (geometry?.triangles || []).map(triangleInfo).filter(Boolean);
  const radius = Math.max(Number(options.patchRadius) || bounds.radius * 0.22, bounds.radius * 0.015);
  const ranked = triangles.map((info) => ({ info, distance: norm(sub(info.centroid, anchor)) })).sort((a, b) => a.distance - b.distance);
  const selected = ranked.filter((item) => item.distance <= radius);
  const fallbackCount = Math.max(6, Number(options.fallbackTriangles || 12));
  const patch = (selected.length ? selected : ranked.slice(0, fallbackCount)).slice(0, Math.max(fallbackCount, selected.length));
  return { patch, radius };
}

function patchStatistics(patch, anchor, radius, bounds) {
  if (!patch.length) {
    const fallbackNormal = normalize(sub(anchor, bounds.center), [0, 1, 0]);
    return { weights: [], normal: fallbackNormal, curvature: 1, normalStability: 0, extent: 0 };
  }
  const weightsRaw = patch.map(({ info, distance }) => info.area / Math.max(distance + radius * 0.02, EPS));
  const weightTotal = Math.max(weightsRaw.reduce((sum, value) => sum + value, 0), EPS);
  const weights = weightsRaw.map((value) => value / weightTotal);
  const reference = patch[0].info.normal;
  const alignedNormals = patch.map(({ info }) => dot(info.normal, reference) < 0 ? scale(info.normal, -1) : info.normal);
  let normalSum = [0, 0, 0];
  for (let index = 0; index < patch.length; index += 1) normalSum = add(normalSum, scale(alignedNormals[index], weights[index]));
  let normal = normalize(normalSum, normalize(sub(anchor, bounds.center), [0, 1, 0]));
  const radial = sub(anchor, bounds.center);
  if (dot(normal, radial) < 0) normal = scale(normal, -1);
  const normalStability = clamp(alignedNormals.reduce((sum, candidate, index) => sum + weights[index] * Math.abs(dot(candidate, normal)), 0), 0, 1);
  const curvature = clamp(alignedNormals.reduce((sum, candidate, index) => sum + weights[index] * (1 - Math.abs(dot(candidate, normal))), 0), 0, 1);
  const extent = Math.sqrt(Math.max(0, patch.reduce((sum, item, index) => {
    const delta = sub(item.info.centroid, anchor);
    return sum + weights[index] * dot(delta, delta);
  }, 0)));
  return { weights, normal, curvature, normalStability, extent };
}

export function buildAnchorLocalFrame(anchor, geometry, bounds, options = {}) {
  const point = (anchor || bounds.center || [0, 0, 0]).map(Number);
  let { patch, radius } = nearestTriangleInfos(point, geometry, bounds, options);
  let statistics = patchStatistics(patch, point, radius, bounds);
  const curvatureThreshold = Number(options.curvatureExpansionThreshold ?? 0.18);
  const stabilityThreshold = Number(options.normalStabilityThreshold ?? 0.72);
  const maxPatchRadius = Math.max(radius, bounds.radius * Number(options.maxPatchRadiusRatio ?? 0.55));
  let expansionSteps = 0;
  while (expansionSteps < 2 && radius < maxPatchRadius && (statistics.curvature > curvatureThreshold || statistics.normalStability < stabilityThreshold || patch.length < 6)) {
    const expandedRadius = Math.min(maxPatchRadius, radius * Number(options.patchExpansionFactor ?? 1.65));
    if (expandedRadius <= radius + EPS) break;
    ({ patch, radius } = nearestTriangleInfos(point, geometry, bounds, { ...options, patchRadius: expandedRadius }));
    statistics = patchStatistics(patch, point, radius, bounds);
    expansionSteps += 1;
  }
  const { weights, normal, curvature, normalStability, extent } = statistics;
  const radial = sub(point, bounds.center);
  const centroids = patch.map(({ info }) => info.centroid);
  const covariance = weightedCovariance(centroids, weights, point);
  let tangent1 = projectToPlane(powerIteration(covariance, [1, 0.37, 0.19]), normal);
  if (norm(tangent1) < EPS) tangent1 = projectToPlane(Math.abs(normal[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], normal);
  tangent1 = normalize(tangent1, [1, 0, 0]);
  const radialTangent = projectToPlane(radial, normal);
  if (norm(radialTangent) > bounds.radius * 0.01 && dot(tangent1, radialTangent) < 0) tangent1 = scale(tangent1, -1);
  let tangent2 = normalize(cross(normal, tangent1), [0, 1, 0]);
  tangent1 = normalize(cross(tangent2, normal), tangent1);
  return {
    origin: point,
    tangent1,
    tangent2,
    normal,
    patch_radius: radius,
    patch_triangle_count: patch.length,
    curvature: clamp(curvature, 0, 1),
    normal_stability: normalStability,
    extent: finite(extent),
    degenerate_patch: patch.length < 2,
    adaptive_patch_expanded: expansionSteps > 0,
    patch_expansion_steps: expansionSteps,
    symmetry_equivalent_frames: options.allowSymmetricFrame === false ? 1 : 2,
    equivalentTangentDirections: options.allowSymmetricFrame === false ? [[...tangent1, ...tangent2]] : [[...tangent1, ...tangent2], [...scale(tangent1, -1), ...scale(tangent2, -1)]],
    localCoordinates(position) {
      const delta = sub(position, point);
      return { u: dot(delta, tangent1), v: dot(delta, tangent2), normal: dot(delta, normal) };
    },
    worldFromLocal(local) {
      return add(point, add(scale(tangent1, finite(local?.u)), add(scale(tangent2, finite(local?.v)), scale(normal, finite(local?.normal)))));
    }
  };
}

export function buildAnchorFrames(labels, geometry, bounds, options = {}) {
  return labels.map((label) => buildAnchorLocalFrame(label.anchor, geometry, bounds, options));
}

export function anchorLabelEdgeFeatures(label, frame, bounds, source = label) {
  const local = frame.localCoordinates(label.center);
  const delta = sub(label.center, frame.origin);
  const leaderLength = norm(delta) / Math.max(bounds.radius, EPS);
  const leaderDirection = scale(delta, 1 / Math.max(norm(delta), EPS));
  const box = (label.boxSize || [0, 0, 0]).map((value) => Math.abs(Number(value)) / Math.max(bounds.radius, EPS));
  const sameSourcePart = Number((source.sourceObjs || []).some((item) => (label.sourceObjs || []).includes(item)));
  const sameTargetGroup = Number((source.targetGroups || []).some((item) => (label.targetGroups || []).includes(item)));
  return [
    local.u / Math.max(bounds.radius, EPS), local.v / Math.max(bounds.radius, EPS), local.normal / Math.max(bounds.radius, EPS),
    ...frame.normal, ...leaderDirection, leaderLength,
    dot(leaderDirection, frame.normal), frame.curvature, frame.extent / Math.max(bounds.radius, EPS), ...box,
    sameSourcePart,
    sameTargetGroup
  ].map((value) => finite(value));
}

export function labelRelationEdgeFeatures(source, target, bounds) {
  const radius = Math.max(bounds.radius, EPS);
  const delta = sub(target.center, source.center);
  const distanceValue = norm(delta) / radius;
  const sourceDirection = normalize(sub(source.center, bounds.center));
  const targetDirection = normalize(sub(target.center, bounds.center));
  return [
    ...delta.map((value) => value / radius), distanceValue,
    ...(target.boxSize || [0, 0, 0]).map((value, axis) => Math.log(Math.max(Math.abs(Number(value)), EPS) / Math.max(Math.abs(Number(source.boxSize?.[axis] || 0)), EPS))),
    dot(sourceDirection, targetDirection),
    Number(source.sourceObjs?.some((item) => (target.sourceObjs || []).includes(item)) || false),
    Number(source.targetGroups?.some((item) => (target.targetGroups || []).includes(item)) || false)
  ].map((value) => finite(value));
}

export function rotateFrameSanity(frame, rotation) {
  const rotate = (vector) => rotation.map((row) => dot(row, vector));
  return { origin: rotate(frame.origin), tangent1: rotate(frame.tangent1), tangent2: rotate(frame.tangent2), normal: rotate(frame.normal) };
}
