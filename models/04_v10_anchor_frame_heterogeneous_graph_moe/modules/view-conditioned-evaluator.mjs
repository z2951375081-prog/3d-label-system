// Per-view safety and free-space evaluator.  It consumes generated 3D labels
// and clean-OBJ depth grids; it never feeds pixels back into the 3D generator.

import { MULTI_VIEW_NAMES, projectLabelToView } from './layout-optimizer.mjs';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const DIRECTION_NAMES = ['right', 'upper_right', 'up', 'upper_left', 'left', 'lower_left', 'down', 'lower_right'];
const VIEW_WEIGHTS = { main: 0.4, right: 0.15, left: 0.15, up: 0.15, down: 0.15 };

function directionIndex(x, y) { return (Math.round(Math.atan2(y, x) / (Math.PI / 4)) + 8) % 8; }
function rectOverlap(a, b) { return Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) * Math.max(0, Math.min(a.top, b.top) - Math.max(a.bottom, b.bottom)); }
function cross2(a, b) { return a.x * b.y - a.y * b.x; }
function subtract2(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function sigmoid(value) { return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value)); }
function segmentsCross(a, b, c, d) {
  const r = subtract2(b, a), s = subtract2(d, c), ca = subtract2(c, a);
  const denominator = cross2(r, s);
  if (Math.abs(denominator) < 1e-10) return false;
  const t = cross2(ca, s) / denominator;
  const u = cross2(ca, r) / denominator;
  return t > 0 && t < 1 && u > 0 && u < 1;
}
function smoothLeaderCrossingRisk(first, second) {
  const a = first.anchor, b = first.center, c = second.anchor, d = second.center;
  if (Math.hypot(a.x - c.x, a.y - c.y) < 1e-4) return 0;
  const r = subtract2(b, a), s = subtract2(d, c), ca = subtract2(c, a);
  const lengthProduct = Math.max(1e-8, Math.hypot(r.x, r.y) * Math.hypot(s.x, s.y));
  const denominator = cross2(r, s);
  if (Math.abs(denominator) < 1e-8) return 0;
  const t = cross2(ca, s) / denominator;
  const u = cross2(ca, r) / denominator;
  const sharpness = 12;
  const inside = sigmoid(sharpness * t) * sigmoid(sharpness * (1 - t)) * sigmoid(sharpness * u) * sigmoid(sharpness * (1 - u));
  const angle = clamp(Math.abs(denominator) / lengthProduct, 0, 1);
  return clamp(inside * (0.25 + 0.75 * Math.sqrt(angle)), 0, 1);
}
function pixelDepth(grid, x, y) {
  const gx = Math.floor((x + 1.15) / 2.3 * grid.gridSize);
  const gy = Math.floor((y + 1.15) / 2.3 * grid.gridSize);
  if (gx < 0 || gy < 0 || gx >= grid.gridSize || gy >= grid.gridSize) return { object: false, near: Infinity, far: -Infinity };
  const index = gy * grid.gridSize + gx;
  return { object: Number.isFinite(grid.depth[index]), near: grid.depth[index], far: Number.isFinite(grid.farDepth?.[index]) ? grid.farDepth[index] : grid.depth[index] };
}

function perView(labels, bounds, view, depthGrid) {
  const items = labels.map((label) => projectLabelToView(label, bounds, view));
  const objectDepthTolerance = Math.max(bounds.radius * 0.015, 0.01);
  let labelObjectOverlap = 0, objectOcclusion = 0, penetration = 0, overflow = 0, labelArea = 0, textFit = 0, textClip = 0;
  let overlapArea = 0, overlapPairs = 0, overlapPairRisk = 0, leaderCrossings = 0, leaderCrossingRisk = 0;
  const counts = Array(8).fill(0), free = Array(8).fill(0);
  for (const item of items) {
    const area = Math.max(1e-8, item.width * 2 * item.height * 2);
    labelArea += area;
    counts[directionIndex(item.center.x, item.center.y)] += 1;
    const overflowX = Math.max(0, Math.abs(item.center.x) + item.width - 1);
    const overflowY = Math.max(0, Math.abs(item.center.y) + item.height - 1);
    overflow += overflowX + overflowY;
    textClip += clamp(overflowX / Math.max(2 * item.width, 1e-8) + overflowY / Math.max(2 * item.height, 1e-8), 0, 1);
    const pixelHeight = item.height * 500;
    const pixelWidth = item.width * 750;
    const fontPixels = Math.max(1, Math.min(pixelHeight * 0.68, 28));
    const estimatedTextWidth = Math.max(1, String(labels[items.indexOf(item)]?.text || '').length) * fontPixels * 0.58;
    textFit += Math.min(1, pixelWidth * 0.92 / Math.max(estimatedTextWidth, 1));
    const rows = 4, columns = 6;
    for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
      const x = item.left + (column + 0.5) / columns * (item.right - item.left);
      const y = item.bottom + (row + 0.5) / rows * (item.top - item.bottom);
      const depth = pixelDepth(depthGrid, x, y);
      if (!depth.object) continue;
      labelObjectOverlap += 1 / (rows * columns);
      if (item.center.depth >= depth.near - objectDepthTolerance && item.center.depth <= depth.far + objectDepthTolerance) penetration += 1 / (rows * columns);
      else if (item.center.depth > depth.far) objectOcclusion += 1 / (rows * columns);
    }
  }
  for (let y = 0; y < depthGrid.gridSize; y += 1) for (let x = 0; x < depthGrid.gridSize; x += 1) {
    const sx = (x + 0.5) / depthGrid.gridSize * 2.3 - 1.15;
    const sy = (y + 0.5) / depthGrid.gridSize * 2.3 - 1.15;
    if (Math.abs(sx) <= 1 && Math.abs(sy) <= 1 && !Number.isFinite(depthGrid.depth[y * depthGrid.gridSize + x])) free[directionIndex(sx, sy)] += 1;
  }
  for (let i = 0; i < items.length; i += 1) for (let j = i + 1; j < items.length; j += 1) {
    const area = rectOverlap(items[i], items[j]);
    overlapArea += area;
    if (area > 0) overlapPairs += 1;
    // Smooth proxy for the discrete overlap-pair count.  The final gate uses
    // overlap_pairs; training uses this normalized soft risk so small overlaps
    // across many pairs cannot hide behind a lower total overlap area.
    const firstArea = Math.max(1e-8, (items[i].right - items[i].left) * (items[i].top - items[i].bottom));
    const secondArea = Math.max(1e-8, (items[j].right - items[j].left) * (items[j].top - items[j].bottom));
    const normalizedOverlap = area / Math.min(firstArea, secondArea);
    overlapPairRisk += normalizedOverlap / (normalizedOverlap + 0.05);
    if (segmentsCross(items[i].anchor, items[i].center, items[j].anchor, items[j].center)) leaderCrossings += 1;
    leaderCrossingRisk += smoothLeaderCrossingRisk(items[i], items[j]);
  }
  const pairCount = Math.max(1, labels.length * Math.max(1, labels.length - 1) / 2);
  const totalLabels = Math.max(1, labels.length), totalFree = Math.max(1, free.reduce((sum, value) => sum + value, 0));
  const labelShares = counts.map((value) => value / totalLabels), freeShares = free.map((value) => value / totalFree);
  const freeSpaceMismatch = 0.5 * labelShares.reduce((sum, value, index) => sum + Math.abs(value - freeShares[index]), 0);
  const entropy = -labelShares.filter((value) => value > 0).reduce((sum, value) => sum + value * Math.log(value), 0) / Math.log(8);
  return {
    object_occlusion_ratio: objectOcclusion / totalLabels,
    label_object_overlap_ratio: labelObjectOverlap / totalLabels,
    penetration_ratio: penetration / totalLabels,
    label_overlap_ratio: labelArea > 0 ? overlapArea / labelArea : 0,
    overlap_pairs: overlapPairs,
    overlap_pair_risk: overlapPairRisk / pairCount,
    leader_crossings: leaderCrossings,
    leader_crossing_ratio: leaderCrossings / pairCount,
    leader_crossing_risk: leaderCrossingRisk / pairCount,
    overflow_ratio: Math.min(1, overflow / Math.max(1, labels.length * 0.4)),
    text_clarity: clamp((mean(items.map((item) => item.height * 500)) - 5) / 13, 0, 1) * (textFit / totalLabels) * (1 - textClip / totalLabels),
    free_space_distribution_mismatch: freeSpaceMismatch,
    free_space_uniformity: clamp(entropy, 0, 1),
    label_shares: Object.fromEntries(DIRECTION_NAMES.map((name, index) => [name, Number(labelShares[index].toFixed(6))])),
    free_space_shares: Object.fromEntries(DIRECTION_NAMES.map((name, index) => [name, Number(freeShares[index].toFixed(6))]))
  };
}

function cvar(values, fraction = 0.25) {
  const sorted = [...values].sort((a, b) => b - a);
  return mean(sorted.slice(0, Math.max(1, Math.ceil(sorted.length * fraction))));
}

function stereoConsistency(per_view, leftName = 'left', rightName = 'right') {
  const left = per_view[leftName];
  const right = per_view[rightName];
  if (!left || !right) return 0;
  const names = DIRECTION_NAMES;
  const shareDelta = names.reduce((sum, name) => sum + Math.abs(Number(left.label_shares?.[name] || 0) - Number(right.label_shares?.[name] || 0)), 0) * 0.5;
  const freeDelta = names.reduce((sum, name) => sum + Math.abs(Number(left.free_space_shares?.[name] || 0) - Number(right.free_space_shares?.[name] || 0)), 0) * 0.5;
  return clamp((shareDelta + freeDelta) * 0.5, 0, 1);
}

export function evaluateViewConditionedLayout(labels, bounds, depthGrids = {}, options = {}) {
  const names = options.views || MULTI_VIEW_NAMES;
  const weights = { ...VIEW_WEIGHTS, ...(options.viewWeights || {}) };
  const per_view = Object.fromEntries(names.map((view) => [view, perView(labels, bounds, view, depthGrids[view] || { gridSize: 1, depth: new Float32Array([Infinity]), farDepth: new Float32Array([-Infinity]) })]));
  const weighted = (key) => names.reduce((sum, view) => sum + (weights[view] || 0) * Number(per_view[view][key] || 0), 0);
  const silhouetteOverlapLosses = names.map((view) => per_view[view].label_object_overlap_ratio);
  const depthOcclusionLosses = names.map((view) => per_view[view].object_occlusion_ratio);
  const penetrationLosses = names.map((view) => per_view[view].penetration_ratio);
  const occlusionLosses = names.map((view) => per_view[view].object_occlusion_ratio + per_view[view].penetration_ratio);
  const balanceLosses = names.map((view) => per_view[view].free_space_distribution_mismatch);
  const overflowLosses = names.map((view) => per_view[view].overflow_ratio);
  const crossingRisks = names.map((view) => per_view[view].leader_crossing_risk);
  const crossingCounts = names.map((view) => per_view[view].leader_crossings);
  const worstOcclusion = Math.max(...occlusionLosses, 0);
  const worstSilhouetteOverlap = Math.max(...silhouetteOverlapLosses, 0);
  const worstDepthOcclusion = Math.max(...depthOcclusionLosses, 0);
  const worstPenetration = Math.max(...penetrationLosses, 0);
  const worstBalance = Math.max(...balanceLosses, 0);
  const worstOverflow = Math.max(...overflowLosses, 0);
  const worstCrossingRisk = Math.max(...crossingRisks, 0);
  const worstCrossingCount = Math.max(...crossingCounts, 0);
  const cvarOcclusion = cvar(occlusionLosses);
  const cvarSilhouetteOverlap = cvar(silhouetteOverlapLosses);
  const cvarDepthOcclusion = cvar(depthOcclusionLosses);
  const cvarBalance = cvar(balanceLosses);
  const cvarOverflow = cvar(overflowLosses);
  const cvarCrossingRisk = cvar(crossingRisks);
  const textClarityLosses = names.map((view) => 1 - Number(per_view[view].text_clarity || 0));
  const weightedTextClarityLoss = names.reduce((sum, view) => sum + (weights[view] || 0) * (1 - Number(per_view[view].text_clarity || 0)), 0);
  const worstTextClarityLoss = Math.max(...textClarityLosses, 0);
  const cvarTextClarityLoss = cvar(textClarityLosses);
  const stereo = stereoConsistency(per_view);
  const cvarViewWeight = Number(options.cvarViewWeight ?? 1);
  const stereoWeight = Number(options.stereoWeight ?? 1);
  const textClarityWeight = Number(options.textClarityWeight ?? 3);
  const leaderCrossingWeight = Number(options.leaderCrossingWeight ?? 3.5);
  const overlapPairWeight = Number(options.overlapPairWeight ?? 0);
  const worstOverflowWeight = Number(options.worstOverflowWeight ?? 0);
  const cvarOverflowWeight = Number(options.cvarOverflowWeight ?? 0);
  const leaderCrossingObjective = weighted('leader_crossing_risk') * leaderCrossingWeight + worstCrossingRisk * Number(options.worstViewWeight ?? 2) + cvarCrossingRisk * cvarViewWeight;
  return {
    per_view,
    view_weights: weights,
    main_view_weight: weights.main || 0,
    weighted_object_occlusion: weighted('object_occlusion_ratio'),
    weighted_label_object_overlap: weighted('label_object_overlap_ratio'),
    weighted_penetration: weighted('penetration_ratio'),
    weighted_label_overlap: weighted('label_overlap_ratio'),
    weighted_leader_crossing_ratio: weighted('leader_crossing_ratio'),
    weighted_leader_crossing_risk: weighted('leader_crossing_risk'),
    weighted_overflow: weighted('overflow_ratio'),
    weighted_overlap_pair_risk: weighted('overlap_pair_risk'),
    weighted_text_clarity: weighted('text_clarity'),
    weighted_text_clarity_loss: weightedTextClarityLoss,
    weighted_free_space_mismatch: weighted('free_space_distribution_mismatch'),
    weighted_free_space_uniformity: weighted('free_space_uniformity'),
    worst_view_occlusion: worstOcclusion,
    worst_view_label_object_overlap: worstSilhouetteOverlap,
    worst_view_depth_occlusion: worstDepthOcclusion,
    worst_view_penetration: worstPenetration,
    worst_view_free_space_mismatch: worstBalance,
    worst_view_overflow: worstOverflow,
    worst_view_leader_crossing_risk: worstCrossingRisk,
    worst_view_leader_crossing_count: worstCrossingCount,
    cvar_view_occlusion: cvarOcclusion,
    cvar_view_label_object_overlap: cvarSilhouetteOverlap,
    cvar_view_depth_occlusion: cvarDepthOcclusion,
    cvar_view_free_space_mismatch: cvarBalance,
    cvar_view_overflow: cvarOverflow,
    cvar_view_leader_crossing_risk: cvarCrossingRisk,
    worst_view_text_clarity_loss: worstTextClarityLoss,
    cvar_view_text_clarity_loss: cvarTextClarityLoss,
    stereo_consistency: stereo,
    leader_crossing_objective: leaderCrossingObjective,
    objective: weighted('label_object_overlap_ratio') * 4 + weighted('object_occlusion_ratio') * 4 + weighted('penetration_ratio') * 6 + weighted('label_overlap_ratio') * 3 + weighted('overflow_ratio') * 4 + weighted('free_space_distribution_mismatch') * 2 + weighted('overlap_pair_risk') * overlapPairWeight + worstOverflow * worstOverflowWeight + cvarOverflow * cvarOverflowWeight + weightedTextClarityLoss * textClarityWeight + (worstDepthOcclusion + worstPenetration) * (options.worstViewWeight ?? 2) + (cvarSilhouetteOverlap + cvarDepthOcclusion + cvarOcclusion + cvarBalance + cvarTextClarityLoss) * cvarViewWeight + worstTextClarityLoss * textClarityWeight + stereo * stereoWeight + leaderCrossingObjective
  };
}
