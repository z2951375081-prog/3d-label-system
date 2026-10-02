import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { MDPO_VAL11_GROUPS, mdpoVal11Group, validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';

const cohort = Array.from({ length: 11 }, (_, index) => `C/${index}`), rewardHash = 'a'.repeat(64);
const referenceHash = 'b'.repeat(64), candidateHash = 'c'.repeat(64), viewHash = 'd'.repeat(64);
const scores = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4,
  manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: .2, PCK_010: .3, OLR: .1, LCD: 0, avg_leader_length: 1,
  overlap_pairs: 0, occluded_points: 0, intersections: 0, quality_score: 4, worst_view_intersections: 0,
  object_occlusion: 0, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: 0 };
const groups = Object.fromEntries(Object.entries(MDPO_VAL11_GROUPS).map(([name, spec]) => [name, {
  group: name, name: spec.label, label: spec.label, model_role: spec.model_role, preference_rerank: spec.preference_rerank,
  preference_model_sha256: spec.preference_rerank ? rewardHash : null, sample_count: 11, cohort, test_used_for_selection: false,
  scorer_model: 'qwen3-vl:4b-instruct', prompt_version: 'fixed-mdpo-v1', views: ['before', 'main', 'right', 'left', 'up', 'down'], metric_protocol: 'fixed-metric-v1',
  sample_scores: cohort.map((id, index) => ({ sample: { category: 'C', sample_id: String(index) }, response_id: `${name}-${index}`,
    scores, metrics, view_sha256: Object.fromEntries(['before', 'main', 'right', 'left', 'up', 'down'].map((view) => [view, viewHash])),
    generation_strategy: { generator: 'v10', viewPolicy: 'five_views', groupPolicy: 'all', sizePolicy: 'relative',
      optimizer: 'annealing', seed: 17017 + index * 4, iterations: 180, preferenceRerank: spec.preference_rerank },
    evaluation: { group: name, role: spec.model_role, reference_model_sha256: referenceHash,
      candidate_sha256: spec.model_role === 'baseline' ? referenceHash : candidateHash,
      candidate_file: spec.model_role === 'baseline' ? null : 'experiments/mdpo/candidate.json' },
    preference_model_sha256: spec.preference_rerank ? rewardHash : null,
    security: { test_used: false, train_preference_created: false, qwen_inference_input: false } }))
}]));
assert.equal(validateMdpoFourGroupSummaries(groups).complete, true);
assert.equal(validateMdpoFourGroupSummaries(groups).paired_sample_protocol.length, 11);
assert.equal(mdpoVal11Group('mdpo_no_rerank').core_gate_role, 'candidate');
const mixedReward = structuredClone(groups); mixedReward.mdpo_safe_rerank.preference_model_sha256 = 'b'.repeat(64);
assert.throws(() => validateMdpoFourGroupSummaries(mixedReward), /same historical reward model/);
const leaked = structuredClone(groups); leaked.v10_no_rerank.preference_model_sha256 = rewardHash;
assert.throws(() => validateMdpoFourGroupSummaries(leaked), /unexpectedly used/);
const wrongCohort = structuredClone(groups); wrongCohort.mdpo_safe_rerank.cohort = ['Other/0', ...wrongCohort.mdpo_safe_rerank.cohort.slice(1)];
assert.throws(() => validateMdpoFourGroupSummaries(wrongCohort), /protocol mismatch/);
const wrongSeed = structuredClone(groups); wrongSeed.mdpo_safe_rerank.sample_scores[3].generation_strategy.seed++;
assert.throws(() => validateMdpoFourGroupSummaries(wrongSeed), /paired generation/);
const wrongOptimizer = structuredClone(groups); wrongOptimizer.v10_historical_rerank.sample_scores[4].generation_strategy.optimizer = 'different';
assert.throws(() => validateMdpoFourGroupSummaries(wrongOptimizer), /paired generation/);
const wrongCohortOptimizer = structuredClone(groups);
for (const group of Object.values(wrongCohortOptimizer)) group.sample_scores[4].generation_strategy.optimizer = 'different';
assert.throws(() => validateMdpoFourGroupSummaries(wrongCohortOptimizer), /cohort changed/);
const wrongPrompt = structuredClone(groups); wrongPrompt.mdpo_safe_rerank.prompt_version = 'different';
assert.throws(() => validateMdpoFourGroupSummaries(wrongPrompt), /scorer\/prompt/);
const wrongCandidateHash = structuredClone(groups); wrongCandidateHash.mdpo_safe_rerank.sample_scores[2].evaluation.candidate_sha256 = 'e'.repeat(64);
assert.throws(() => validateMdpoFourGroupSummaries(wrongCandidateHash), /paired generation/);
const missingView = structuredClone(groups); delete missingView.mdpo_safe_rerank.sample_scores[2].view_sha256.down;
assert.throws(() => validateMdpoFourGroupSummaries(missingView), /sample evidence/);
const scratch = await fs.mkdtemp(path.join(os.tmpdir(), 'v10-mdpo-four-group-raw-test-'));
try {
  const candidateId = 'candidate_123';
  const fourGroup = { candidate_id: candidateId, groups: Object.values(groups), protocol: validateMdpoFourGroupSummaries(groups) };
  for (const [name, summary] of Object.entries(groups)) {
    const directory = path.join(scratch, candidateId, name);
    await fs.mkdir(directory, { recursive: true });
    for (const entry of summary.sample_scores) {
      const record = { version: 'v10_mdpo_val11_sample_v1', split: 'val', candidate_id: candidateId,
        group: name, role: summary.model_role, sample: entry.sample, scorer: { model: summary.scorer_model,
          prompt_version: summary.prompt_version, response_id: entry.response_id }, views: summary.views,
        metric_protocol: summary.metric_protocol, scores: entry.scores, metrics: entry.metrics,
        view_sha256: entry.view_sha256, generation_strategy: entry.generation_strategy,
        evaluation: entry.evaluation, security: entry.security,
        preference_model: { model_sha256: entry.preference_model_sha256 } };
      const filename = `${encodeURIComponent(`${entry.sample.category}__${entry.sample.sample_id}`)}.json`;
      await fs.writeFile(path.join(directory, filename), JSON.stringify(record));
    }
  }
  const input = { directory: scratch, candidateId, fourGroup };
  assert.equal((await validateMdpoFourGroupRawRecords(input)).immutable_records_verified, 44);
  const file = path.join(scratch, candidateId, 'mdpo_safe_rerank', 'C__2.json');
  const altered = JSON.parse(await fs.readFile(file, 'utf8'));
  altered.scores.overall = 1;
  await fs.writeFile(file, JSON.stringify(altered));
  await assert.rejects(() => validateMdpoFourGroupRawRecords(input), /differs from four-group report/);
  assert.equal(fourGroup.groups[3].sample_scores[2].scores.overall, 4);
} finally {
  const directory = path.resolve(scratch), prefix = path.resolve(os.tmpdir()) + path.sep;
  if (!directory.startsWith(prefix) || !path.basename(directory).startsWith('v10-mdpo-four-group-raw-test-')) throw new Error('Unsafe raw val11 test cleanup');
  await fs.rm(directory, { recursive: true, force: true });
}
console.log('v10-MDPO four-group model/reranker/cohort protocol tests passed.');
