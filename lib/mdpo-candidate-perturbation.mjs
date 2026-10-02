import { buildAnchorFrames } from './anchor-frame-features.mjs';
import { projectLabelToView } from './layout-optimizer.mjs';

export const MDPO_PERTURBATION_MODES = Object.freeze([
  'baseline', 'compact_leaders', 'spacious_leaders', 'small_panels', 'large_panels',
  'alternating_tangent', 'positive_tangent_bias', 'negative_tangent_bias',
  'hierarchy_panels', 'mixed_radial_rhythm', 'tangent_fan', 'normal_stagger',
  'increase_label_gap', 'long_leader_arc', 'short_leader_arc', 'arc_uniformity', 'repulsive_spacing'
]);

function seededRandom(seed) {
  let state = (Number(seed) >>> 0) || 17;
  return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 2 ** 32; };
}


function clampToMainViewport(label, bounds) {
  let output = { ...label, center: [...label.center], boxSize: [...label.boxSize] };
  for (let step = 0; step < 14; step += 1) {
    const projected = projectLabelToView(output, bounds, 'main');
    const overflow = Math.max(0, Math.abs(projected.center.x) + projected.width - 0.98) + Math.max(0, Math.abs(projected.center.y) + projected.height - 0.98);
    if (overflow <= 1e-5) return output;
    output.center = output.anchor.map((value, axis) => value + (output.center[axis] - value) * 0.88);
  }
  return output;
}
export function perturbMdpoCandidate(labels, geometry, bounds, { mode = 0, seed = 17 } = {}) {
  if (!Array.isArray(labels) || !labels.length) throw new Error('MDPO perturbation requires labels');
  const index = Math.max(0, Math.min(MDPO_PERTURBATION_MODES.length - 1, Math.floor(Number(mode) || 0)));
  const name = MDPO_PERTURBATION_MODES[index];
  if (index === 0) return { labels: structuredClone(labels), mode: name, bounded: true };
  const frames = buildAnchorFrames(labels, geometry, bounds);
  const radius = Math.max(Number(bounds.radius) || 0, 1e-8);
  const random = seededRandom(Number(seed) + index * 7919);
  const output = labels.map((label, labelIndex) => {
    const frame = frames[labelIndex];
    const local = frame.localCoordinates(label.center);
    let u = local.u, v = local.v, normal = local.normal;
    let sizeScale = 1;
    if (name === 'compact_leaders') { u *= 0.82; v *= 0.82; normal *= 0.88; }
    if (name === 'spacious_leaders') { u *= 1.14; v *= 1.14; normal *= 1.08; }
    if (name === 'small_panels') sizeScale = 0.68;
    if (name === 'large_panels') sizeScale = 1.22;
    if (name === 'alternating_tangent') { u += (labelIndex % 2 ? 1 : -1) * radius * (0.07 + random() * 0.025); v += ((labelIndex % 3) - 1) * radius * 0.025; }
    if (name === 'positive_tangent_bias') u += radius * (0.07 + random() * 0.025);
    if (name === 'negative_tangent_bias') u -= radius * (0.07 + random() * 0.025);
    if (name === 'hierarchy_panels') sizeScale = 0.72 + 0.48 * (1 - labelIndex / Math.max(1, labels.length - 1));
    if (name === 'mixed_radial_rhythm') { const scale = labelIndex % 2 ? 1.14 : 0.84; u *= scale; v *= scale; normal *= 0.94 + random() * 0.12; }
    if (name === 'tangent_fan') { const centered = labelIndex / Math.max(1, labels.length - 1) - 0.5; u += centered * radius * 0.20; v += Math.sin(labelIndex * 1.7) * radius * 0.035; }
    if (name === 'normal_stagger') normal += (labelIndex % 2 ? 1 : -1) * radius * (0.035 + random() * 0.015);
    if (name === 'increase_label_gap') { const centered = labelIndex / Math.max(1, labels.length - 1) - 0.5; u += Math.sign(centered || (random() - 0.5)) * radius * (0.16 + random() * 0.06); v += centered * radius * 0.12; }
    if (name === 'repulsive_spacing') { let repulsionU = 0, repulsionV = 0; for (let otherIndex = 0; otherIndex < labels.length; otherIndex += 1) { if (otherIndex === labelIndex) continue; const otherLocal = frames[labelIndex].localCoordinates(labels[otherIndex].center); const du = u - otherLocal.u, dv = v - otherLocal.v; const d2 = Math.max(du * du + dv * dv, radius * radius * 0.0025); const inv = 1 / Math.sqrt(d2); repulsionU += du * inv * (radius * 0.09 / Math.max(1, labels.length - 1)); repulsionV += dv * inv * (radius * 0.09 / Math.max(1, labels.length - 1)); } u += repulsionU; v += repulsionV; }
    if (name === 'long_leader_arc') { u *= 1.22; v *= 1.22; normal *= 1.10; }
    if (name === 'short_leader_arc') { u *= 0.74; v *= 0.74; normal *= 0.82; }
    if (name === 'arc_uniformity') { const angle = (labelIndex / Math.max(1, labels.length)) * Math.PI * 2; const planarLength = Math.max(radius * 0.22, Math.hypot(u, v)); u = Math.cos(angle) * planarLength; v = Math.sin(angle) * planarLength; }
    const center = frame.worldFromLocal({ u, v, normal });
    const boxSize = label.boxSize.map((value, axis) => axis < 2 ? value * sizeScale : value);
    return clampToMainViewport({ ...label, center, boxSize, mdpo_perturbation: { mode: name, mode_index: index, bounded_local_coordinate_change: true, main_viewport_clamped: true, qwen_generated_coordinates: false } }, bounds);
  });
  return { labels: output, mode: name, bounded: true };
}

