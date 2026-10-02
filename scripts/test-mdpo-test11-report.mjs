import assert from 'node:assert/strict';
import { MDPO_VAL11_GROUPS } from '../lib/mdpo-four-group.mjs';
import { validateMdpoTest11Records } from '../lib/mdpo-test11-report.mjs';

const cohort = Array.from({ length: 11 }, (_, index) => `Category/${index}`), lockSha256 = 'a'.repeat(64), baselineSha256 = 'b'.repeat(64), activeSha256 = 'c'.repeat(64), preferenceSha256 = 'd'.repeat(64);
const views = ['before', 'main', 'right', 'left', 'up', 'down'], viewHash = 'e'.repeat(64);
const scores = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4, manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: .2, PCK_010: .3, OLR: .1, LCD: 0, avg_leader_length: 1, overlap_pairs: 0, occluded_points: 0, intersections: 0, quality_score: 4, worst_view_intersections: 0, object_occlusion: 0, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: 0 };
const groups = Object.fromEntries(Object.entries(MDPO_VAL11_GROUPS).map(([name, spec]) => [name, cohort.map((sample, index) => {
  const [category, sample_id] = sample.split('/');
  return { version: 'v10_mdpo_test11_sample_v1', split: 'test', group: name, role: spec.model_role, test_run_id: 'test11_locked_123', lock_sha256: lockSha256,
    sample: { category, sample_id }, scorer: { model: 'qwen', prompt_version: 'prompt-v1', response_id: `${name}-${index}` }, views, view_sha256: Object.fromEntries(views.map((view) => [view, viewHash])), metric_protocol: 'metrics-v1', scores, metrics,
    evaluation: { phase: 'test11', test_run_id: 'test11_locked_123', lock_sha256: lockSha256, group: name, role: spec.model_role, reference_model_sha256: baselineSha256, candidate_sha256: spec.model_role === 'candidate' ? activeSha256 : baselineSha256 },
    generation_strategy: { generator: 'annotation', viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', seed: 17017 + index * 4, iterations: 180, preferenceRerank: spec.preference_rerank },
    preference_model: { model_sha256: spec.preference_rerank ? preferenceSha256 : null }, security: { test_used: true, test_used_for_training: false, test_used_for_selection: false, train_preference_created: false, qwen_inference_input: false } };
})]));
const input = { groups, cohort, runId: 'test11_locked_123', lockSha256, baselineSha256, activeSha256, preferenceSha256, scorerModel: 'qwen', promptVersion: 'prompt-v1', metricProtocol: 'metrics-v1' };
assert.equal(validateMdpoTest11Records(input).record_count, 44);
const partial = Object.fromEntries(Object.entries(groups).map(([name, rows]) => [name, rows.slice(0, name === 'v10_no_rerank' ? 3 : 2)]));
assert.equal(validateMdpoTest11Records({ ...input, groups: partial, allowIncomplete: true }).record_count, 9);
assert.throws(() => validateMdpoTest11Records({ ...input, groups: partial }), /exactly 11|all 44/);
const leaked = structuredClone(groups); leaked.mdpo_no_rerank[0].security.test_used_for_selection = true;
assert.throws(() => validateMdpoTest11Records({ ...input, groups: leaked }), /leak/);
const wrongSeed = structuredClone(groups); wrongSeed.mdpo_safe_rerank[3].generation_strategy.seed++;
assert.throws(() => validateMdpoTest11Records({ ...input, groups: wrongSeed }), /different seeds/);
console.log('v10-MDPO locked four-group test11 provenance tests passed.');
