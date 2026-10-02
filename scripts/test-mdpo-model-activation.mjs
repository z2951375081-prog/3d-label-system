import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { materializeActiveMdpoModel, validateMdpoActivationBundle } from '../lib/mdpo-model-activation.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const activeBytes = Buffer.from(JSON.stringify({ version: 'layout_model_v10_anchor_frame_heterogeneous_graph_moe' }));
const activeHash = hash(activeBytes), ids = Array.from({ length: 11 }, (_, index) => `C/${index}`);
const scores = { overall: 4, composition_harmony: 4, visual_hierarchy: 4, spatial_balance: 4, manual_style_similarity: 4, text_clarity: 4, leader_line_clarity: 4 };
const metrics = { PCK_005: .2, PCK_010: .4, OLR: .1, LCD: 0, overlap_pairs: 0, occluded_points: 0, intersections: 0, worst_view_intersections: 0, object_occlusion: .1, penetration: 0, mesh_surface_intersection: 0, worst_view_overflow: .02, text_clarity: 4, leader_line_clarity: 4 };
const baseline = { group: 'v10_no_rerank', name: 'v10 · reranker off', label: 'v10 · reranker off', model_role: 'baseline', preference_rerank: false, preference_model_sha256: null,
  sample_count: 11, cohort: ids, scorer_model: 'qwen3-vl:4b-instruct', prompt_version: 'mdpo-v1', views: ['before', 'main', 'right', 'left', 'up', 'down'], metric_protocol: 'fixed-val11-v1', test_used_for_selection: false, score_means: scores, metrics };
const candidateMetrics = structuredClone(baseline); candidateMetrics.score_means.overall = 4.1; candidateMetrics.metrics.PCK_005 = .19; candidateMetrics.metrics.PCK_010 = .39; candidateMetrics.metrics.OLR = .11;
Object.assign(candidateMetrics, { group: 'mdpo_no_rerank', name: 'v10-MDPO · reranker off', label: 'v10-MDPO · reranker off', model_role: 'candidate' });
const gate = evaluateMdpoVal11Gate({ baseline, candidate: candidateMetrics });
assert.equal(gate.accepted, true);
const candidate = { version: 'layout_model_v10_mdpo_candidate', status: 'diagnostic_only_requires_full_val11_gate', reference: { sha256: activeHash, frozen_verified: true },
  architecture: { qwen_inference_input: false }, mdpo: { dataset_audit: { sample_count: 33, pair_count: 198 }, trainable_parameters: 7676,
    preference_holdout: { heldout_pair_count: 33, gradient_pair_count: 165 },
    weight_update_evidence: { matrix_count: 19, trainable_parameters: 7676, ablation: { effective: true },
      modules: Object.fromEntries(Array.from({ length: 7 }, (_, index) => [`module_${index}`, { changed_matrices: 1 }])) },
    train_unified_metrics: { baseline: { sample_count: 33 }, candidate: { sample_count: 33 },
      delta: Object.fromEntries(['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'].map((name) => [name, 0])) } },
  split_policy: { train33_gradient_only: true, train33_preference_holdout_no_gradient: true, test11_not_loaded_or_used: true }, validation_gate: { status: 'pending_full_val11_visual_and_safety_gate', active: false } };
const candidateBytes = Buffer.from(JSON.stringify(candidate)), candidateHash = hash(candidateBytes);
const rewardHash = 'f'.repeat(64);
const historical = { ...structuredClone(baseline), group: 'v10_historical_rerank', name: 'v10 + historical reward MLP', label: 'v10 + historical reward MLP', preference_rerank: true, preference_model_sha256: rewardHash };
const safeRerank = { ...structuredClone(candidateMetrics), group: 'mdpo_safe_rerank', name: 'v10-MDPO + safe reward reranking', label: 'v10-MDPO + safe reward reranking', preference_rerank: true, preference_model_sha256: rewardHash };
const fourGroupRows = { v10_no_rerank: baseline, v10_historical_rerank: historical, mdpo_no_rerank: candidateMetrics, mdpo_safe_rerank: safeRerank };
for (const row of Object.values(fourGroupRows)) row.sample_scores = ids.map((id, index) => ({
  sample: { category: 'C', sample_id: String(index) }, response_id: `${row.group}-${index}`,
  scores: row.score_means, metrics: { ...row.metrics, avg_leader_length: 1, overlap_pairs: 0, occluded_points: 0, quality_score: 4 },
  view_sha256: Object.fromEntries(['before', 'main', 'right', 'left', 'up', 'down'].map((view) => [view, 'd'.repeat(64)])),
  generation_strategy: { generator: 'v10', viewPolicy: 'five_views', groupPolicy: 'all', sizePolicy: 'relative',
    optimizer: 'annealing', seed: 17017 + index * 4, iterations: 180, preferenceRerank: row.preference_rerank },
  evaluation: { group: row.group, role: row.model_role, reference_model_sha256: activeHash,
    candidate_sha256: row.model_role === 'baseline' ? activeHash : candidateHash,
    candidate_file: row.model_role === 'baseline' ? null : 'experiments/mdpo/candidate.json' },
  preference_model_sha256: row.preference_model_sha256,
  security: { test_used: false, train_preference_created: false, qwen_inference_input: false }
}));
const fourGroupProtocol = validateMdpoFourGroupSummaries(fourGroupRows);
const report = { version: 'v10_mdpo_val11_gate_report_v2', evaluation_policy: 'safety_priority_v2', candidate_id: 'candidate_123', candidate_model: { file: 'experiments/mdpo/candidate.json', sha256: candidateHash },
  paired_provenance: { paired_samples: 11 }, four_group_protocol: fourGroupProtocol, baseline, candidate: candidateMetrics, gate, test_not_used: true };
const reportBytes = Buffer.from(JSON.stringify(report)), reportHash = hash(reportBytes);
const fourGroupReport = { version: 'v10_mdpo_four_group_val11_v2', evaluation_policy: 'safety_priority_v2', candidate_id: 'candidate_123', candidate_model: report.candidate_model,
  protocol: fourGroupProtocol, groups: Object.values(fourGroupRows), core_gate: gate, test_not_used: true };
const fourGroupReportBytes = Buffer.from(JSON.stringify(fourGroupReport)), fourGroupReportHash = hash(fourGroupReportBytes);
const selection = { version: 'v10_mdpo_hyperparameter_selection_v1', status: 'selected_by_complete_authoritative_val11', val11_gate_complete: true, test_used_for_selection: false,
  evaluated_configuration_count: 81, required_grid: { total: 81 }, deployment_eligible: true,
  selected: { candidate_id: 'candidate_123', candidate_file: 'experiments/mdpo/candidate.json', candidate_sha256: candidateHash,
    report_file: 'experiments/mdpo/report.json', report_sha256: reportHash, four_group_report_file: 'experiments/mdpo/report.four_group.json', four_group_report_sha256: fourGroupReportHash, gate } };
const validated = validateMdpoActivationBundle({ selection, candidateBytes, activeBytes, val11ReportBytes: reportBytes, fourGroupReportBytes });
assert.equal(validated.candidateHash, candidateHash);
const active = materializeActiveMdpoModel(validated, { selectionFile: 'selection.json', reportFile: 'report.json', preferenceModelSha256: 'd'.repeat(64) });
assert.equal(active.version, 'layout_model_v10_mdpo');
assert.equal(active.validation_gate.active, true);
assert.equal(active.activation_provenance.test11_used_for_activation, false);
assert.equal(active.activation_provenance.four_group_report_sha256, fourGroupReportHash);
assert.throws(() => validateMdpoActivationBundle({ selection: { ...selection, deployment_eligible: false }, candidateBytes, activeBytes, val11ReportBytes: reportBytes, fourGroupReportBytes }), /did not pass every/);
assert.throws(() => validateMdpoActivationBundle({ selection: { ...selection, selected: { ...selection.selected, candidate_sha256: 'e'.repeat(64) } }, candidateBytes, activeBytes, val11ReportBytes: reportBytes, fourGroupReportBytes }), /SHA-256 mismatch/);
assert.throws(() => validateMdpoActivationBundle({ selection, candidateBytes, activeBytes, val11ReportBytes: reportBytes, fourGroupReportBytes: Buffer.from(JSON.stringify({ ...fourGroupReport, candidate_id: 'tampered' })) }), /SHA-256 mismatch|four-group report invalid/);
const incompleteCandidate = { ...candidate, mdpo: { dataset_audit: { sample_count: 32, pair_count: 192 } } };
assert.throws(() => validateMdpoActivationBundle({ selection: { ...selection, selected: { ...selection.selected, candidate_sha256: hash(Buffer.from(JSON.stringify(incompleteCandidate))) } }, candidateBytes: Buffer.from(JSON.stringify(incompleteCandidate)), activeBytes, val11ReportBytes: reportBytes, fourGroupReportBytes }), /SHA-256 mismatch|complete train33/);
console.log('v10-MDPO complete-val11 no-force atomic activation bundle tests passed.');
