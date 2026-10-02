// Post-generation optimizer for the V10 pipeline.
// It preserves the selected style while allowing small geometric repairs, then
// applies a hard five-view safety gate before a candidate can be accepted.

import { evaluateViewConditionedLayout } from './view-conditioned-evaluator.mjs';

export const FIVE_VIEW_NAMES = Object.freeze(['main', 'right', 'left', 'up', 'down']);

const clamp = (value, min, max) => Math.max(min, Math.min(max, Number(value)));
const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
const clone = (value) => structuredClone(value);
const distance = (a, b) => Math.hypot(...a.map((value, axis) => value - b[axis]));

function seededRandom(seed = 17) {
  let state = Number(seed) >>> 0 || 17;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normalizedBoxDistance(candidate, reference) {
  return mean(candidate.boxSize.map((value, axis) => Math.abs(Math.log(Math.max(Math.abs(value), 1e-8) / Math.max(Math.abs(reference.boxSize[axis]), 1e-8)))));
}

/**
 * Style preservation is intentionally geometric and label-order invariant at
 * the aggregate level: leader lengths, radial distances, spacing, box aspect,
 * and per-label local displacement all contribute to the energy.
 */
export function stylePreservationLoss(candidate, reference, bounds, options = {}) {
  if (!Array.isArray(candidate) || !Array.isArray(reference) || candidate.length !== reference.length || !candidate.length) return 0;
  const radius = Math.max(Number(bounds?.radius || 0), 1e-8);
  const perLabel = candidate.map((label, index) => {
    const target = reference[index];
    const center = distance(label.center, target.center) / radius;
    const leader = Math.abs(distance(label.anchor, label.center) - distance(target.anchor, target.center)) / radius;
    const radial = Math.abs(distance(label.center, bounds.center) - distance(target.center, bounds.center)) / radius;
    const size = normalizedBoxDistance(label, target);
    const candidateAspect = Math.max(Math.abs(label.boxSize[0]) / Math.max(Math.abs(label.boxSize[1]), 1e-8), 1e-8);
    const targetAspect = Math.max(Math.abs(target.boxSize[0]) / Math.max(Math.abs(target.boxSize[1]), 1e-8), 1e-8);
    const aspect = Math.abs(Math.log(candidateAspect / targetAspect));
    return (center * Number(options.positionWeight ?? 0.35)) + (leader * Number(options.leaderWeight ?? 0.25)) + (radial * Number(options.radialWeight ?? 0.15)) + (size * Number(options.sizeWeight ?? 0.15)) + (aspect * Number(options.aspectWeight ?? 0.10));
  });
  const pairwise = [];
  for (let left = 0; left < candidate.length; left += 1) for (let right = left + 1; right < candidate.length; right += 1) {
    const candidateDistance = distance(candidate[left].center, candidate[right].center) / radius;
    const referenceDistance = distance(reference[left].center, reference[right].center) / radius;
    pairwise.push(Math.abs(candidateDistance - referenceDistance));
  }
  return mean(perLabel) + mean(pairwise) * Number(options.spacingWeight ?? 0.10);
}

function safetyViolations(diagnostics, thresholds) {
  const checks = [
    ['worst_view_occlusion', diagnostics.worst_view_occlusion, thresholds.maxWorstViewOcclusion],
    ['worst_view_penetration', diagnostics.worst_view_penetration, thresholds.maxWorstViewPenetration],
    ['worst_view_label_object_overlap', diagnostics.worst_view_label_object_overlap, thresholds.maxWorstViewLabelObjectOverlap],
    ['worst_view_overflow', diagnostics.worst_view_overflow, thresholds.maxWorstViewOverflow],
    ['worst_view_leader_crossing_risk', diagnostics.worst_view_leader_crossing_risk, thresholds.maxWorstViewLeaderCrossingRisk],
    ['worst_view_text_clarity_loss', diagnostics.worst_view_text_clarity_loss, thresholds.maxWorstViewTextClarityLoss]
  ];
  return checks.filter(([, value, limit]) => Number.isFinite(Number(limit)) && Number(value) > Number(limit)).map(([term, value, limit]) => ({ term, value: Number(value), limit: Number(limit) }));
}

export function fiveViewSafetyGate(labels, bounds, depthGrids, options = {}) {
  const diagnostics = evaluateViewConditionedLayout(labels, bounds, depthGrids, {
    worstViewWeight: Number(options.worstViewWeight ?? 2),
    cvarViewWeight: Number(options.cvarViewWeight ?? 1),
    stereoWeight: Number(options.stereoWeight ?? 1),
    textClarityWeight: Number(options.textClarityWeight ?? 3),
    leaderCrossingWeight: Number(options.leaderCrossingWeight ?? 3.5),
    overlapPairWeight: Number(options.overlapPairWeight ?? 0),
    worstOverflowWeight: Number(options.worstOverflowWeight ?? 0),
    cvarOverflowWeight: Number(options.cvarOverflowWeight ?? 0)
  });
  const thresholds = {
    maxWorstViewOcclusion: options.maxWorstViewOcclusion ?? 0.55,
    maxWorstViewPenetration: options.maxWorstViewPenetration ?? 0.35,
    maxWorstViewLabelObjectOverlap: options.maxWorstViewLabelObjectOverlap ?? 0.55,
    maxWorstViewOverflow: options.maxWorstViewOverflow ?? 0.25,
    maxWorstViewLeaderCrossingRisk: options.maxWorstViewLeaderCrossingRisk ?? 0.45,
    maxWorstViewTextClarityLoss: options.maxWorstViewTextClarityLoss ?? 0.65
  };
  const violations = safetyViolations(diagnostics, thresholds);
  return { safe: violations.length === 0, views: [...FIVE_VIEW_NAMES], thresholds, violations, objective: Number(diagnostics.objective || 0), diagnostics };
}

function proposeLayout(current, bounds, random, temperature) {
  const candidate = clone(current);
  const index = Math.floor(random() * candidate.length);
  const label = candidate[index];
  const radius = Math.max(Number(bounds.radius || 0), 1e-8);
  const magnitude = radius * (0.025 + 0.14 * temperature);
  label.center = label.center.map((value, axis) => value + (random() * 2 - 1) * magnitude * (axis === 2 ? 0.45 : 1));
  if (random() < 0.25) {
    const scale = Math.exp((random() * 2 - 1) * 0.08 * (0.3 + temperature));
    label.boxSize = label.boxSize.map((value) => Math.max(radius * 0.005, Math.abs(value) * scale));
  }
  return candidate;
}

/**
 * Simulated annealing after MoE decoding. The reference is the selected
 * expert output, so style preservation is explicit rather than accidental.
 */
export function annealStylePreservingLayout(initialLabels, bounds, options = {}) {
  if (!Array.isArray(initialLabels) || !initialLabels.length) return { labels: [], accepted: false, iterations: 0, safety: null };
  const missingViews = FIVE_VIEW_NAMES.filter((view) => !options.depthGrids?.[view]);
  if (missingViews.length) throw new Error(`style-preserving annealing requires depth grids for: ${missingViews.join(', ')}`);
  const reference = options.styleReference || initialLabels;
  const styleWeight = Number(options.styleWeight ?? 1);
  const iterations = Math.max(1, Number(options.iterations ?? 180));
  const random = seededRandom(options.seed ?? 17);
  const energy = (labels) => {
    const safety = fiveViewSafetyGate(labels, bounds, options.depthGrids, options);
    return { value: safety.objective + styleWeight * stylePreservationLoss(labels, reference, bounds, options), safety };
  };
  let current = clone(initialLabels);
  let currentEnergy = energy(current);
  const initialEnergy = currentEnergy.value;
  let best = clone(current);
  let bestEnergy = currentEnergy;
  let acceptedMoves = 0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const temperature = Math.max(0.035, 1 - iteration / iterations);
    const candidate = proposeLayout(current, bounds, random, temperature);
    const candidateEnergy = energy(candidate);
    if (options.hardSafetyGate && !candidateEnergy.safety.safe) continue;
    const delta = candidateEnergy.value - currentEnergy.value;
    if (delta <= 0 || random() < Math.exp(-delta / Math.max(temperature, 0.05))) {
      current = candidate;
      currentEnergy = candidateEnergy;
      acceptedMoves += 1;
    }
    const candidateIsBetter = candidateEnergy.safety.safe && (!bestEnergy.safety.safe || candidateEnergy.value < bestEnergy.value);
    if (candidateIsBetter) {
      best = clone(candidate);
      bestEnergy = candidateEnergy;
    }
  }
  const finalSafety = fiveViewSafetyGate(best, bounds, options.depthGrids, options);
  return {
    labels: best,
    accepted: finalSafety.safe,
    iterations,
    acceptedMoves,
    initialEnergy,
    finalEnergy: bestEnergy.value,
    styleLoss: stylePreservationLoss(best, reference, bounds, options),
    safety: finalSafety
  };
}



