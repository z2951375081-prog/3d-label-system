import assert from 'node:assert/strict';
import { validateMdpoVal11Records } from '../lib/mdpo-val11-provenance.mjs';

const referenceSha256 = 'a'.repeat(64), candidateSha256 = 'b'.repeat(64), viewHash = 'c'.repeat(64);
const expected = Array.from({ length: 11 }, (_, index) => `Category${index}/${index}`);
const scores = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4, manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: .2, PCK_010: .3, OLR: .1, LCD: 0, avg_leader_length: 1, overlap_pairs: 0, occluded_points: 0, intersections: 0, quality_score: 4,
  worst_view_intersections: 0, object_occlusion: 0, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: 0 };
function row(sample, index, role) {
  const [category, sample_id] = sample.split('/');
  return { version: 'v10_mdpo_val11_sample_v1', split: 'val', group: role === 'baseline' ? 'v10_no_rerank' : 'mdpo_no_rerank', role, candidate_id: 'candidate_123', sample: { category, sample_id },
    scorer: { model: 'qwen3-vl:4b-instruct', response_id: `${role}-${index}`, prompt_version: 'fixed-prompt-v1' },
    evaluation: { candidate_id: 'candidate_123', role, candidate_file: role === 'candidate' ? 'experiments/mdpo/candidate.json' : null,
      candidate_sha256: role === 'candidate' ? candidateSha256 : referenceSha256, reference_model_sha256: referenceSha256 },
    views: ['before', 'main', 'right', 'left', 'up', 'down'], metric_protocol: 'fixed-metric-v1',
    view_sha256: Object.fromEntries(['before', 'main', 'right', 'left', 'up', 'down'].map((view) => [view, viewHash])),
    generation_strategy: { generator: 'annotation', viewPolicy: 'binocular', groupPolicy: 'all', sizePolicy: 'relative', optimizer: 'annealing', seed: 17 + index * 4, iterations: 180, preferenceRerank: false },
    scores, metrics, security: { test_used: false, train_preference_created: false, qwen_inference_input: false } };
}
const baseline = expected.map((sample, index) => row(sample, index, 'baseline'));
const candidate = expected.map((sample, index) => row(sample, index, 'candidate'));
const input = { baseline, candidate, expected, candidateId: 'candidate_123', referenceSha256, candidateSha256,
  scorerModel: 'qwen3-vl:4b-instruct', promptVersion: 'fixed-prompt-v1', metricProtocol: 'fixed-metric-v1' };
const valid = validateMdpoVal11Records(input);
assert.equal(valid.paired_samples, 11);
assert.match(valid.cohort_sha256, /^[a-f0-9]{64}$/);
const seedMismatch = structuredClone(candidate); seedMismatch[4].generation_strategy.seed++;
assert.throws(() => validateMdpoVal11Records({ ...input, candidate: seedMismatch }), /generation mismatch/);
const modelMismatch = structuredClone(candidate); modelMismatch[3].evaluation.candidate_sha256 = 'd'.repeat(64);
assert.throws(() => validateMdpoVal11Records({ ...input, candidate: modelMismatch }), /checkpoint hash mismatch/);
const leak = structuredClone(candidate); leak[0].security.test_used = true;
assert.throws(() => validateMdpoVal11Records({ ...input, candidate: leak }), /leak/);
console.log('v10-MDPO paired val11 scorer, seed, generator, checkpoint and split provenance passed.');
