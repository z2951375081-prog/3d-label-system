// Pure 3D spatial features used by the v9 generator and by the human-style
// auxiliary objective.  This module deliberately has no camera/projective
// dependencies: view-space measurements belong to the evaluator only.

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const EPS = 1e-8;
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const sub = (a, b) => a.map((value, axis) => value - b[axis]);
const dot = (a, b) => a.reduce((sum, value, axis) => sum + value * b[axis], 0);
const vectorNorm = (a) => Math.max(Math.hypot(...a), EPS);
const norm = (a) => vectorNorm(a);
const scale = (a, value) => a.map((entry) => entry * value);
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const standardDeviation = (values) => {
  if (!values.length) return 0;
  const average = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
};

import { anchorLabelEdgeFeatures, buildAnchorFrames, labelRelationEdgeFeatures } from './anchor-frame-features.mjs';

export { anchorLabelEdgeFeatures, buildAnchorFrames, labelRelationEdgeFeatures };

export const ANCHOR_LOCAL_LABEL_FEATURE_NAMES = [
  'local_u', 'local_v', 'local_normal_distance',
  'anchor_normal_x', 'anchor_normal_y', 'anchor_normal_z',
  'tangent1_x', 'tangent1_y', 'tangent1_z',
  'tangent2_x', 'tangent2_y', 'tangent2_z',
  'surface_curvature', 'patch_point_density', 'patch_extent',
  'radial_x', 'radial_y', 'radial_z',
  'box_x', 'box_y', 'box_z', 'box_ratio_xy', 'box_ratio_xz', 'box_ratio_yz',
  'source_count', 'target_count', 'text_length', 'candidate_index',
  'leader_length', 'leader_length_squared', 'leader_normal_component',
  'surface_clearance', 'nearest_label_spacing', 'mean_label_spacing', 'local_label_density',
  'air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization',
  'direction_octant_0', 'direction_octant_1', 'direction_octant_2', 'direction_octant_3',
  'direction_octant_4', 'direction_octant_5', 'direction_octant_6', 'direction_octant_7',
  'object_center_distance', 'box_volume_ratio', 'normal_tangent_angle', 'tangent_balance', 'out_of_sight_prior'
];

if (ANCHOR_LOCAL_LABEL_FEATURE_NAMES.length !== 51) throw new Error('锚点局部三维标签特征必须固定为 51D');

export const PURE_3D_LABEL_FEATURE_NAMES = [
  'anchor_x', 'anchor_y', 'anchor_z', 'center_x', 'center_y', 'center_z',
  'box_x', 'box_y', 'box_z', 'source_count', 'target_count', 'distance',
  'shape_x', 'shape_y', 'shape_z', 'candidate_index',
  'anchor_to_center_x', 'anchor_to_center_y', 'anchor_to_center_z',
  'box_ratio_xy', 'box_ratio_xz', 'box_ratio_yz',
  'center_direction_x', 'center_direction_y', 'center_direction_z',
  'surface_clearance', 'leader_length', 'leader_length_squared',
  'air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization',
  'local_label_density', 'nearest_label_spacing', 'mean_label_spacing',
  'direction_octant_0', 'direction_octant_1', 'direction_octant_2', 'direction_octant_3',
  'direction_octant_4', 'direction_octant_5', 'direction_octant_6', 'direction_octant_7',
  'text_length', 'surface_distance_mean', 'object_center_distance', 'label_box_volume_ratio',
  'box_aspect_x', 'box_aspect_y', 'box_aspect_z', 'direction_xy', 'direction_z'
];

if (PURE_3D_LABEL_FEATURE_NAMES.length !== 51) throw new Error('纯三维标签特征必须固定为 51D');

function allGeometryPoints(geometry) {
  const points = [];
  for (const triangle of geometry?.triangles || []) {
    if (!Array.isArray(triangle) || triangle.length < 3) continue;
    points.push(...triangle.map((point) => point.map(Number)));
    points.push([0, 1, 2].map((axis) => triangle.reduce((sum, point) => sum + Number(point[axis] || 0), 0) / 3));
    for (let index = 0; index < 3; index += 1) {
      const left = triangle[index], right = triangle[(index + 1) % 3];
      points.push([0, 1, 2].map((axis) => (Number(left[axis]) + Number(right[axis])) / 2));
    }
  }
  return points.filter((point) => point.every(Number.isFinite));
}

function voxelCoordinate(point, bounds, gridSize) {
  return [0, 1, 2].map((axis) => clamp(Math.floor(((point[axis] - bounds.min[axis]) / Math.max(bounds.size[axis], 1e-8)) * gridSize), 0, gridSize - 1));
}

function voxelIndex(x, y, z, gridSize) { return z * gridSize * gridSize + y * gridSize + x; }

function markDilated(mask, x, y, z, gridSize, radius = 1) {
  for (let dz = -radius; dz <= radius; dz += 1) for (let dy = -radius; dy <= radius; dy += 1) for (let dx = -radius; dx <= radius; dx += 1) {
    const nx = x + dx, ny = y + dy, nz = z + dz;
    if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize && nz >= 0 && nz < gridSize) mask[voxelIndex(nx, ny, nz, gridSize)] = 1;
  }
}

export function buildSpatialContext(geometry, bounds, options = {}) {
  const gridSize = Math.max(8, Math.min(32, Number(options.gridSize || 20)));
  const objectMask = new Uint8Array(gridSize ** 3);
  const allPoints = allGeometryPoints(geometry);
  for (const point of allPoints) {
    const [x, y, z] = voxelCoordinate(point, bounds, gridSize);
    markDilated(objectMask, x, y, z, gridSize, Number(options.surfaceDilation ?? 1));
  }
  const pointLimit = Math.max(0, Number(options.pointLimit || 0));
  const points = pointLimit > 0 && allPoints.length > pointLimit
    ? allPoints.filter((_, index) => index % Math.ceil(allPoints.length / pointLimit) === 0).slice(0, pointLimit)
    : allPoints;
  const objectVoxelCount = objectMask.reduce((sum, value) => sum + value, 0);
  return {
    gridSize,
    objectMask,
    objectVoxelCount,
    objectVoxelRatio: objectVoxelCount / Math.max(1, objectMask.length),
    points,
    bounds: { min: [...bounds.min], max: [...bounds.max], center: [...bounds.center], size: [...bounds.size], radius: bounds.radius }
  };
}

function boxVoxelMask(label, context) {
  const { gridSize } = context;
  const mask = new Uint8Array(gridSize ** 3);
  const half = (label.boxSize || [0, 0, 0]).map((value) => Math.abs(Number(value)) / 2);
  const min = label.center.map((value, axis) => value - half[axis]);
  const max = label.center.map((value, axis) => value + half[axis]);
  const from = [0, 1, 2].map((axis) => clamp(Math.floor(((min[axis] - context.bounds.min[axis]) / Math.max(context.bounds.size[axis], 1e-8)) * gridSize), 0, gridSize - 1));
  const to = [0, 1, 2].map((axis) => clamp(Math.ceil(((max[axis] - context.bounds.min[axis]) / Math.max(context.bounds.size[axis], 1e-8)) * gridSize), 0, gridSize - 1));
  for (let z = from[2]; z <= to[2]; z += 1) for (let y = from[1]; y <= to[1]; y += 1) for (let x = from[0]; x <= to[0]; x += 1) mask[voxelIndex(x, y, z, gridSize)] = 1;
  return mask;
}

function directionOctant(vector) {
  return (vector[0] >= 0 ? 1 : 0) + (vector[1] >= 0 ? 2 : 0) + (vector[2] >= 0 ? 4 : 0);
}

function directionStats(labels, bounds) {
  const counts = Array(8).fill(0);
  for (const label of labels) counts[directionOctant(label.center.map((value, axis) => value - bounds.center[axis]))] += 1;
  const total = Math.max(1, labels.length);
  const probabilities = counts.map((count) => count / total);
  const entropy = -probabilities.reduce((sum, probability) => sum + (probability > 0 ? probability * Math.log(probability) : 0), 0) / Math.log(8);
  return { counts, probabilities, entropy: clamp(entropy, 0, 1) };
}

export function computeSpatialStyleMetrics(labels = [], bounds, context = buildSpatialContext(null, bounds)) {
  const safeLabels = Array.isArray(labels) ? labels : [];
  const totalVoxels = Math.max(1, context.objectMask?.length || 1);
  const labelMask = new Uint8Array(totalVoxels);
  let objectLabelOverlap = 0;
  for (const label of safeLabels) {
    const mask = boxVoxelMask(label, context);
    for (let index = 0; index < totalVoxels; index += 1) {
      if (mask[index]) {
        labelMask[index] = 1;
        if (context.objectMask?.[index]) objectLabelOverlap += 1;
      }
    }
  }
  const objectVoxelRatio = finite(context.objectVoxelRatio);
  const airVoxelAvailableRatio = clamp(1 - objectVoxelRatio, 0, 1);
  const labelOccupiedVoxelRatio = labelMask.reduce((sum, value) => sum + value, 0) / totalVoxels;
  const airSpaceUtilization = labelOccupiedVoxelRatio / Math.max(airVoxelAvailableRatio, 1e-6);
  const spacing = [];
  const nearestSpacing = [];
  const densities = [];
  for (let index = 0; index < safeLabels.length; index += 1) {
    const distances = safeLabels.map((label, other) => other === index ? Infinity : distance(safeLabels[index].center, label.center) / Math.max(bounds.radius, 1e-8)).sort((a, b) => a - b);
    if (Number.isFinite(distances[0])) nearestSpacing.push(distances[0]);
    densities.push(distances.filter((value) => value < 0.75).length / Math.max(1, safeLabels.length - 1));
    for (let other = index + 1; other < safeLabels.length; other += 1) spacing.push(distance(safeLabels[index].center, safeLabels[other].center) / Math.max(bounds.radius, 1e-8));
  }
  const leaderLengths = safeLabels.map((label) => distance(label.anchor, label.center) / Math.max(bounds.radius, 1e-8));
  const clearances = safeLabels.map((label) => {
    if (!context.points?.length) return 1;
    let nearestSurface = Infinity;
    for (const point of context.points) nearestSurface = Math.min(nearestSurface, distance(label.center, point));
    const halfExtent = (label.boxSize || [0, 0, 0]).reduce((maximum, value) => Math.max(maximum, Math.abs(Number(value))), 0) / 2;
    return (nearestSurface - halfExtent) / Math.max(bounds.radius, 1e-8);
  });
  const direction = directionStats(safeLabels, bounds);
  return {
    air_voxel_available_ratio: airVoxelAvailableRatio,
    label_occupied_voxel_ratio: labelOccupiedVoxelRatio,
    air_space_utilization: clamp(airSpaceUtilization, 0, 10),
    object_label_voxel_overlap_ratio: objectLabelOverlap / totalVoxels,
    mean_3d_clearance: mean(clearances),
    min_3d_clearance: clearances.length ? Math.min(...clearances) : 0,
    mean_3d_spacing: mean(spacing),
    min_3d_spacing: nearestSpacing.length ? Math.min(...nearestSpacing) : 0,
    spacing_std: standardDeviation(spacing),
    local_label_density: mean(densities),
    directional_uniformity: direction.entropy,
    directional_counts: direction.counts,
    directional_shares: direction.probabilities,
    leader_length_mean: mean(leaderLengths),
    leader_length_std: standardDeviation(leaderLengths),
    leader_length_max: leaderLengths.length ? Math.max(...leaderLengths) : 0,
    out_of_sight_ratio: leaderLengths.length ? leaderLengths.filter((value) => value > 3).length / leaderLengths.length : 0,
    label_count: safeLabels.length
  };
}

function safeContextStats(labels, bounds, context) {
  return computeSpatialStyleMetrics(labels, bounds, context);
}

export function pure3DLabelFeatures(label, bounds, index = 0, total = 1, context, labels = []) {
  const radius = Math.max(bounds.radius, 1e-8);
  const allLabels = labels.length ? labels : [label];
  const stats = safeContextStats(allLabels, bounds, context || buildSpatialContext(null, bounds));
  const relative = label.center.map((value, axis) => value - label.anchor[axis]);
  const radial = label.center.map((value, axis) => value - bounds.center[axis]);
  const radialLength = Math.max(Math.hypot(...radial), 1e-8);
  const box = (label.boxSize || [0, 0, 0]).map((value) => Math.abs(Number(value)) / radius);
  const boxVolumeRatio = (box[0] * box[1] * box[2]) / Math.max(1, bounds.size.map((value) => Math.abs(value) / radius).reduce((a, b) => a * b, 1));
  const direction = radial.map((value) => value / radialLength);
  const octant = directionOctant(radial);
  const spacing = allLabels.filter((candidate) => candidate !== label).map((candidate) => distance(label.center, candidate.center) / radius);
  const nearest = spacing.length ? Math.min(...spacing) : 0;
  const average = mean(spacing);
  const textLength = [...String(label.text || '')].length;
  let surfaceDistance = 0;
  if (context?.points?.length) {
    surfaceDistance = Infinity;
    for (const point of context.points) surfaceDistance = Math.min(surfaceDistance, distance(label.center, point));
    surfaceDistance /= radius;
  }
  const output = [
    ...label.anchor.map((value, axis) => (value - bounds.center[axis]) / radius),
    ...label.center.map((value, axis) => (value - bounds.center[axis]) / radius),
    ...box,
    Math.min(1, (label.sourceObjs?.length || 0) / 4), Math.min(1, (label.targetGroups?.length || 0) / 4),
    Math.min(1, Math.hypot(...radial) / (radius * 4)),
    bounds.size[0] / radius, bounds.size[1] / radius, bounds.size[2] / radius,
    total > 1 ? index / (total - 1) : 0.5,
    ...relative.map((value) => value / radius),
    box[0] / Math.max(box[1], 1e-6), box[0] / Math.max(box[2], 1e-6), box[1] / Math.max(box[2], 1e-6),
    ...direction,
    (surfaceDistance - Math.max(...box) / 2),
    distance(label.anchor, label.center) / radius,
    (distance(label.anchor, label.center) / radius) ** 2,
    stats.air_voxel_available_ratio,
    stats.label_occupied_voxel_ratio,
    clamp(stats.air_space_utilization / 10, 0, 1),
    stats.local_label_density,
    nearest,
    average,
    ...Array.from({ length: 8 }, (_, directionIndex) => directionIndex === octant ? 1 : 0),
    Math.min(1, textLength / 64),
    surfaceDistance,
    Math.min(1, Math.hypot(...radial) / radius),
    clamp(boxVolumeRatio, 0, 1),
    box[0] / Math.max(box[1] + box[2], 1e-6),
    box[1] / Math.max(box[0] + box[2], 1e-6),
    box[2] / Math.max(box[0] + box[1], 1e-6),
    Math.hypot(direction[0], direction[1]),
    direction[2]
  ];
  return output.map((value) => finite(value));
}

export function anchorLocal3DLabelFeatures(label, frame, bounds, index = 0, total = 1, context, labels = []) {
  if (!frame?.localCoordinates) throw new Error('anchorLocal3DLabelFeatures 需要锚点局部坐标系');
  const radius = Math.max(bounds.radius, EPS);
  const allLabels = labels.length ? labels : [label];
  const stats = computeSpatialStyleMetrics(allLabels, bounds, context || buildSpatialContext(null, bounds));
  const local = frame.localCoordinates(label.center);
  const radial = label.center.map((value, axis) => value - bounds.center[axis]);
  const radialLength = Math.max(Math.hypot(...radial), EPS);
  const radialDirection = radial.map((value) => value / radialLength);
  const box = (label.boxSize || [0, 0, 0]).map((value) => Math.abs(Number(value)) / radius);
  const spacing = allLabels.filter((candidate) => candidate !== label).map((candidate) => distance(label.center, candidate.center) / radius);
  const nearest = spacing.length ? Math.min(...spacing) : 0;
  const average = mean(spacing);
  const leader = sub(label.center, label.anchor);
  const leaderLength = norm(leader) / radius;
  const leaderDirection = scale(leader, 1 / Math.max(norm(leader), EPS));
  let surfaceDistance = 0;
  if (context?.points?.length) {
    surfaceDistance = Infinity;
    for (const point of context.points) surfaceDistance = Math.min(surfaceDistance, distance(label.center, point));
    surfaceDistance /= radius;
  }
  const octant = directionOctant(radial);
  const boxVolumeRatio = (box[0] * box[1] * box[2]) / Math.max(1, bounds.size.map((value) => Math.abs(value) / radius).reduce((a, b) => a * b, 1));
  const output = [
    local.u / radius, local.v / radius, local.normal / radius,
    ...frame.normal, ...frame.tangent1, ...frame.tangent2,
    finite(frame.curvature), finite((frame.patch_triangle_count || 0) / 32), finite(frame.extent / radius),
    ...radialDirection,
    ...box,
    box[0] / Math.max(box[1], EPS), box[0] / Math.max(box[2], EPS), box[1] / Math.max(box[2], EPS),
    Math.min(1, (label.sourceObjs?.length || 0) / 4), Math.min(1, (label.targetGroups?.length || 0) / 4), Math.min(1, [...String(label.text || '')].length / 64),
    total > 1 ? index / (total - 1) : 0.5,
    leaderLength, leaderLength ** 2, dot(leaderDirection, frame.normal),
    surfaceDistance - Math.max(...box) / 2, nearest, average, stats.local_label_density,
    stats.air_voxel_available_ratio, stats.label_occupied_voxel_ratio, clamp(stats.air_space_utilization / 10, 0, 1),
    ...Array.from({ length: 8 }, (_, directionIndex) => directionIndex === octant ? 1 : 0),
    Math.min(1, radialLength / radius), clamp(boxVolumeRatio, 0, 1), Math.abs(dot(leaderDirection, frame.tangent1)), Math.abs(dot(leaderDirection, frame.tangent2)), stats.out_of_sight_ratio
  ];
  if (output.length !== ANCHOR_LOCAL_LABEL_FEATURE_NAMES.length) throw new Error(`局部三维特征维度错误：${output.length}`);
  return output.map((value) => finite(value));
}

export function spatialStyleLoss(predicted, target, options = {}) {
  const keys = ['air_voxel_available_ratio', 'label_occupied_voxel_ratio', 'air_space_utilization', 'mean_3d_clearance', 'min_3d_clearance', 'mean_3d_spacing', 'min_3d_spacing', 'spacing_std', 'local_label_density', ...(options.includeDirectionalUniformity === false ? [] : ['directional_uniformity']), 'leader_length_mean', 'leader_length_std', 'leader_length_max', 'out_of_sight_ratio'];
  return mean(keys.map((key) => {
    const scale = key === 'air_space_utilization' ? 0.1 : key.includes('clearance') || key.includes('spacing') || key.includes('leader') ? 0.5 : 1;
    return ((finite(predicted[key]) - finite(target[key])) / scale) ** 2;
  }));
}
