import assert from 'node:assert/strict';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';

const ids = Array.from({ length: 11 }, (_, index) => `C/${index}`);
const scores = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4, manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: .2, PCK_010: .4, OLR: .1, LCD: 0, overlap_pairs: 0, occluded_points: 0, intersections: 0, worst_view_intersections: 0, object_occlusion: .1, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: .02, text_clarity: 4, leader_line_clarity: 4 };
const base = { sample_count: 11, cohort: ids, scorer_model: 'qwen3-vl:4b-instruct', prompt_version: 'mdpo-v1', views: ['before', 'main', 'right', 'left', 'up', 'down'], metric_protocol: 'fixed-val11-v1', test_used_for_selection: false, score_means: scores, metrics };
const good = structuredClone(base); good.score_means.overall = 4.1; good.metrics.PCK_005 = .19; good.metrics.PCK_010 = .39; good.metrics.OLR = .11;
const accepted = evaluateMdpoVal11Gate({ baseline: base, candidate: good });
assert.equal(accepted.accepted, true);
const badPck = structuredClone(good); badPck.metrics.PCK_005 = .189;
assert.equal(evaluateMdpoVal11Gate({ baseline: base, candidate: badPck }).violations.some((name) => name.includes('pck_005')), false);
const badOverlap = structuredClone(good); badOverlap.metrics.overlap_pairs = 1;
assert.ok(evaluateMdpoVal11Gate({ baseline: base, candidate: badOverlap }).violations.includes('gate:overlap_pairs_non_degradation'));
assert.equal(accepted.policy.id, 'safety_priority_v2');
const unsafe = structuredClone(good); unsafe.metrics.worst_view_intersections = 1;
assert.ok(evaluateMdpoVal11Gate({ baseline: base, candidate: unsafe }).violations.includes('gate:zero_worst_view_intersections'));
const textDrop = structuredClone(good); textDrop.score_means.text_clarity = 3.99;
assert.ok(evaluateMdpoVal11Gate({ baseline: base, candidate: textDrop }).violations.includes('gate:text_clarity_non_degradation'));
const protocolLeak = structuredClone(good); protocolLeak.test_used_for_selection = true;
assert.ok(evaluateMdpoVal11Gate({ baseline: base, candidate: protocolLeak }).violations.includes('evidence:test_not_used'));
for (const missing of [null, undefined, '', false]) {
  const incomplete = structuredClone(good); incomplete.metrics.penetration = missing;
  assert.ok(evaluateMdpoVal11Gate({ baseline: base, candidate: incomplete }).violations.includes('evidence:candidate_complete'));
}
const duplicates = structuredClone(good); duplicates.cohort[1] = duplicates.cohort[0];
assert.ok(evaluateMdpoVal11Gate({ baseline: base, candidate: duplicates }).violations.includes('evidence:full_val11'));
console.log('v10-MDPO strict val11 activation gate tests passed.');
