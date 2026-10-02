import assert from 'node:assert/strict';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';
import { MDPO_DIMENSIONS } from '../lib/mdpo-continuous-policy.mjs';
const key = 'a'.repeat(64);
const metrics = {
  objective_score: 1, label_label_occlusion_ratio: 0, label_object_occlusion_ratio: 0,
  object_label_occlusion_ratio: 0, object_penetration_ratio: 0, mesh_surface_intersection_ratio: 0,
  multi_view_worst_olr: 0, multi_view_worst_overflow: 0, text_clarity: 5, leader_crossings: 0,
  worst_view_leader_crossing_count: 0, weighted_leader_crossing_risk: 0,
  worst_view_leader_crossing_risk: 0, cvar_view_leader_crossing_risk: 0,
  worst_view_penetration_v10: 0, multi_view_worst_overflow: 0
};
const scores = { overall: 5, composition_harmony: 5, visual_hierarchy: 5,
  spatial_balance: 5, manual_style_similarity: 5, text_clarity: 5, leader_line_clarity: 5 };
const candidate = (id, score) => ({ candidate_id: id, seed: id, label_ids: ['part'], local_layout: [[id / 100, 0, 0, 0, 0, 0]],
  scores: { ...scores, overall: score }, scorer_model: 'qwen3-vl:4b-instruct', prompt_version: 'mdpo-v1', response_id: `receipt-${id}`,
  view_sha256: Object.fromEntries(['before', 'main', 'right', 'left', 'up', 'down'].map((view) => [view, key])),
  safety: { eligible: true, metrics, reference_metrics: metrics }, label_contract_valid: true });
const pair = { split: 'train', category: 'Chair', sample_id: '1', clean_obj_source: 'data/Chair.obj',
  scorer_model: 'qwen3-vl:4b-instruct', prompt_version: 'mdpo-v1', candidate_a: candidate(17, 5), candidate_b: candidate(18, 4),
  dimension_margins: Object.fromEntries(MDPO_DIMENSIONS.map((name) => [name, name === 'overall' ? 0.25 : 0])) };
const manifest = { samples: [{ category: 'Chair', sample_id: '1', split: 'train', input: { source_obj: 'data/Chair.obj' } }] };
const dataset = { version: 'v10_mdpo_train_pairs_v1', pairs: [pair] };
assert.equal(validateMdpoDataset(dataset, manifest, { requireComplete: false }).pair_count, 1);
assert.throws(() => validateMdpoDataset(dataset, manifest), /Incomplete/);
assert.throws(() => validateMdpoDataset({ ...dataset, pairs: [{ ...pair, split: 'val' }] }, manifest, { requireComplete: false }), /val or test/);
assert.throws(() => validateMdpoDataset({ ...dataset, pairs: [{ ...pair, candidate_a: { ...pair.candidate_a, safety: { ...pair.candidate_a.safety, metrics: { ...metrics, leader_crossings: 1 } } } }] }, manifest, { requireComplete: false }), /safety gate/);
for (const field of ['mesh_surface_intersection_ratio', 'object_penetration_ratio', 'worst_view_penetration_v10']) {
  assert.throws(() => validateMdpoDataset({ ...dataset, pairs: [{ ...pair, candidate_a: { ...pair.candidate_a, safety: { ...pair.candidate_a.safety, metrics: { ...metrics, [field]: 0.001 } } } }] }, manifest, { requireComplete: false }), /strict MDPO/);
}
assert.throws(() => validateMdpoDataset({ ...dataset, pairs: [{ ...pair, candidate_a: { ...pair.candidate_a, local_layout: pair.candidate_b.local_layout } }] }, manifest, { requireComplete: false }), /duplicate/);
for (const dimension_margins of [undefined, { ...pair.dimension_margins, overall: -0.25 },
  { ...pair.dimension_margins, text_clarity: 0.25 }, { ...pair.dimension_margins, unused: 0 }]) {
  assert.throws(() => validateMdpoDataset({ ...dataset, pairs: [{ ...pair, dimension_margins }] }, manifest,
    { requireComplete: false }), /seven-dimensional Qwen preference margins/);
}
console.log('MDPO train-only complete provenance and deterministic safety validator passed.');
