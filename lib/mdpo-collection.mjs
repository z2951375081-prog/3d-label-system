import { createHash } from 'node:crypto';
import { MDPO_DIMENSIONS } from './mdpo-continuous-policy.mjs';
import { assessSafetyEligibility, compareSafetyReference } from './preference-policy.mjs';

export const MDPO_VIEWS = Object.freeze(['before', 'main', 'right', 'left', 'up', 'down']);
export const MDPO_ZERO_GEOMETRY_FIELDS = Object.freeze([
  'leader_crossings', 'worst_view_leader_crossing_count',
  'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'worst_view_penetration_v10',
  'multi_view_worst_overflow'
]);

export function strictMdpoSafety(metrics) {
  const missing = MDPO_ZERO_GEOMETRY_FIELDS.filter((key) => !Number.isFinite(metrics?.[key]));
  const nonzero = MDPO_ZERO_GEOMETRY_FIELDS.filter((key) => Number.isFinite(metrics?.[key]) && metrics[key] > 1e-9);
  const basic = assessSafetyEligibility(metrics, metrics);
  return { eligible: basic.eligible && !missing.length && !nonzero.length, violations: [...basic.violations, ...missing.map((key) => `missing:${key}`), ...nonzero.map((key) => `nonzero:${key}`)] };
}

export function hashMdpoViews(visuals) {
  return Object.fromEntries(MDPO_VIEWS.map((view) => {
    const match = /^data:image\/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(visuals?.[view] || '');
    if (!match || match[1].length < 100 || match[1].length % 4 !== 0) throw new Error(`MDPO missing valid ${view} rendered image`);
    const bytes = Buffer.from(match[1], 'base64');
    if (bytes.length < 100 || bytes.toString('base64') !== match[1]) throw new Error(`MDPO invalid ${view} image bytes`);
    return [view, createHash('sha256').update(bytes).digest('hex')];
  }));
}

export function selectMdpoPairs(candidates, { count = 8, tieThreshold = 0.025 } = {}) {
  if (!Array.isArray(candidates) || candidates.length < 8 || count < 6 || count > 12) throw new Error('MDPO requires at least eight candidates and 6-12 pairs');
  if (new Set(candidates.map((item) => item.candidate_id)).size !== candidates.length
      || new Set(candidates.map((item) => JSON.stringify(item.local_layout))).size !== candidates.length) throw new Error('MDPO candidate IDs and layouts must be unique');
  const edges = [];
  for (let a = 0; a < candidates.length; a++) for (let b = a + 1; b < candidates.length; b++) {
    const left = candidates[a], right = candidates[b];
    if (!strictMdpoSafety(left.safety.metrics).eligible || !strictMdpoSafety(right.safety.metrics).eligible) continue;
    const [safer, other] = compareSafetyReference(left.safety.metrics, right.safety.metrics) <= 0 ? [left, right] : [right, left];
    if (!assessSafetyEligibility(other.safety.metrics, safer.safety.metrics).eligible) continue;
    const margins = Object.fromEntries(MDPO_DIMENSIONS.map((dimension) => [dimension, (left.scores[dimension] - right.scores[dimension]) / 4]));
    const informative = Object.values(margins).filter((value) => Math.abs(value) > tieThreshold);
    if (!informative.length) continue;
    edges.push({ a, b, margins, informativeness: informative.reduce((sum, value) => sum + Math.abs(value), 0) });
  }
  edges.sort((a, b) => b.informativeness - a.informativeness || a.a - b.a || a.b - b.b);
  const covered = new Set(), selected = [], taken = new Set();
  while (covered.size < 8 && selected.length < 12) {
    const available = edges.filter((item) => !taken.has(`${item.a}:${item.b}`) && (!covered.has(item.a) || !covered.has(item.b)));
    const maximumNewEndpoints = Math.max(0, ...available.map((item) => Number(!covered.has(item.a)) + Number(!covered.has(item.b))));
    const edge = available.find((item) => Number(!covered.has(item.a)) + Number(!covered.has(item.b)) === maximumNewEndpoints);
    if (!edge) break;
    selected.push(edge); taken.add(`${edge.a}:${edge.b}`);
    covered.add(edge.a); covered.add(edge.b);
  }
  if (covered.size < 8) throw new Error(`Insufficient informative safety-compatible MDPO pairs: only ${covered.size}/8 candidates covered`);
  for (const edge of edges) {
    if (selected.length >= count) break;
    const key = `${edge.a}:${edge.b}`;
    if (!taken.has(key)) { selected.push(edge); taken.add(key); }
  }
  if (selected.length < count) throw new Error(`Insufficient informative safety-compatible MDPO pairs: ${selected.length}/${count} edges`);
  return selected.map(({ a, b, margins }) => ({ candidate_a: candidates[a], candidate_b: candidates[b], dimension_margins: margins }));
}
