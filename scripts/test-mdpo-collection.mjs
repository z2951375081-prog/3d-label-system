import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { hashMdpoViews, selectMdpoPairs, MDPO_VIEWS } from '../lib/mdpo-collection.mjs';
import { MDPO_DIMENSIONS } from '../lib/mdpo-continuous-policy.mjs';

const image = Buffer.alloc(120, 42);
const visuals = Object.fromEntries(MDPO_VIEWS.map((view) => [view, `data:image/png;base64,${image.toString('base64')}`]));
const hashes = hashMdpoViews(visuals);
assert.equal(Object.keys(hashes).length, 6);
assert.equal(hashes.main, createHash('sha256').update(image).digest('hex'));
assert.throws(() => hashMdpoViews({ ...visuals, right: 'data:image/png;base64,!' }), /right/);

const requiredMetrics = {
  objective_score: 1, label_label_occlusion_ratio: 0, label_object_occlusion_ratio: 0,
  object_label_occlusion_ratio: 0, object_penetration_ratio: 0, mesh_surface_intersection_ratio: 0,
  multi_view_worst_olr: 0, multi_view_worst_overflow: 0, text_clarity: 5,
  leader_crossings: 0, worst_view_leader_crossing_count: 0, weighted_leader_crossing_risk: 0,
  worst_view_leader_crossing_risk: 0, cvar_view_leader_crossing_risk: 0,
  worst_view_penetration_v10: 0, multi_view_worst_overflow: 0
};
const candidates = Array.from({ length: 8 }, (_, index) => ({
  candidate_id: `candidate_${index}`, local_layout: [[index / 20, 0, 0, 0, 0, 0]],
  safety: { metrics: { ...requiredMetrics }, reference_metrics: { ...requiredMetrics }, eligible: true },
  scores: Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, 1 + index * 0.4]))
}));
const pairs = selectMdpoPairs(candidates);
assert.equal(pairs.length, 8);
assert.equal(new Set(pairs.flatMap(({ candidate_a, candidate_b }) => [candidate_a.candidate_id, candidate_b.candidate_id])).size, 8);
assert.deepEqual(Object.keys(pairs[0].dimension_margins), MDPO_DIMENSIONS);
assert.throws(() => selectMdpoPairs(candidates.map((item) => ({ ...item, scores: candidates[0].scores }))), /Insufficient informative/);
assert.throws(() => selectMdpoPairs(candidates.map((item, index) => index === 1 ? { ...item, local_layout: candidates[0].local_layout } : item)), /unique/);
const sparse = candidates.map((item, index) => ({ ...item, scores: Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, index === 0 ? 4 : 3])) }));
assert.throws(() => selectMdpoPairs(sparse, { count: 12 }), /edges/);
const minimumPairs = selectMdpoPairs(candidates, { count: 6 });
assert.equal(minimumPairs.length, 6);
assert.equal(new Set(minimumPairs.flatMap(({ candidate_a, candidate_b }) => [candidate_a.candidate_id, candidate_b.candidate_id])).size, 8);
console.log('MDPO image-byte receipts and eight-candidate informative pairing passed.');
