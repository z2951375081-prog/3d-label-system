import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateExistingMdpoTest11Lock, validateMdpoTest11Prerequisites } from '../lib/mdpo-test11-lock.mjs';

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const baselineBytes = Buffer.from(JSON.stringify({ version: 'layout_model_v10_anchor_frame_heterogeneous_graph_moe' }));
const preferenceBytes = Buffer.from(JSON.stringify({ version: 'historical_reward_model' }));
const baselineSha256 = hash(baselineBytes), preferenceSha256 = hash(preferenceBytes), candidateSha256 = 'c'.repeat(64), val11Sha256 = 'd'.repeat(64), fourGroupSha256 = 'a'.repeat(64);
const active = { version: 'layout_model_v10_mdpo', status: 'active_val11_selected_v10_mdpo', architecture: { qwen_inference_input: false },
  validation_gate: { status: 'accepted', active: true }, activation_provenance: { qwen_inference_input: false, original_candidate_sha256: candidateSha256,
    val11_report_sha256: val11Sha256, four_group_report_sha256: fourGroupSha256, frozen_reference_sha256: baselineSha256, historical_reward_model_sha256: preferenceSha256,
    historical_reward_model_preserved: true, test11_used_for_activation: false } };
const activeBytes = Buffer.from(JSON.stringify(active)), activeSha256 = hash(activeBytes);
const activationReport = { version: 'v10_mdpo_activation_report_v1', status: 'activated', test11_pending: true, active_sha256: activeSha256,
  previous_active_sha256: baselineSha256, historical_reward_model_sha256: preferenceSha256, historical_reward_model_unchanged: true };
const selection = { version: 'v10_mdpo_hyperparameter_selection_v1', status: 'selected_by_complete_authoritative_val11', val11_gate_complete: true,
  deployment_eligible: true, test_used_for_selection: false, evaluated_configuration_count: 81, required_grid: { total: 81 },
  selected: { candidate_sha256: candidateSha256, report_sha256: val11Sha256, four_group_report_sha256: fourGroupSha256, gate: { accepted: true } } };
const cohort = Array.from({ length: 11 }, (_, index) => `Category/${index}`);
const input = { activeBytes, baselineBytes, preferenceBytes, activationReport, selection, cohort };
assert.equal(validateMdpoTest11Prerequisites(input).activeSha256, activeSha256);
assert.throws(() => validateMdpoTest11Prerequisites({ ...input, cohort: cohort.slice(0, 10) }), /11 unique/);
assert.throws(() => validateMdpoTest11Prerequisites({ ...input, activationReport: { ...activationReport, previous_active_sha256: 'e'.repeat(64) } }), /incomplete or changed/);
assert.throws(() => validateMdpoTest11Prerequisites({ ...input, selection: { ...selection, test_used_for_selection: true } }), /complete no-test/);
assert.throws(() => validateMdpoTest11Prerequisites({ ...input, activeBytes: Buffer.from(JSON.stringify({ ...active, status: 'diagnostic_only' })) }), /formally activated/);
const codeSha256 = { 'server.mjs': 'f'.repeat(64) };
const lock = { version: 'v10_mdpo_test11_lock_v1', status: 'locked_pending_single_test11_evaluation', sample_count: 11, cohort_split: 'test', cohort,
  scorer: { model: 'qwen3-vl:4b-instruct', prompt_version: 'aesthetic_safety_v3_fourteen_dimension_mdpo_teacher_v1', views: ['before', 'main', 'right', 'left', 'up', 'down'] },
  active_model: { sha256: activeSha256 }, baseline_model: { sha256: baselineSha256 }, historical_reward_model: { sha256: preferenceSha256, frozen: true },
  selection: { candidate_sha256: candidateSha256, val11_report_sha256: val11Sha256, four_group_report_sha256: fourGroupSha256 }, seed: 17017, code_sha256: codeSha256,
  policy: { one_final_test_only: true, test_not_used_for_training: true, test_not_used_for_selection: true, qwen_offline_evaluator_only: true, qwen_inference_input: false } };
const lockInput = { activeSha256, baselineSha256, preferenceSha256, candidateSha256, val11ReportSha256: val11Sha256, fourGroupReportSha256: fourGroupSha256, cohort, codeSha256,
  scorerModel: 'qwen3-vl:4b-instruct', promptVersion: 'aesthetic_safety_v3_fourteen_dimension_mdpo_teacher_v1' };
assert.equal(validateExistingMdpoTest11Lock(lock, lockInput).sample_count, 11);
assert.throws(() => validateExistingMdpoTest11Lock({ ...lock, cohort: [...cohort].reverse() }, lockInput), /does not match/);
assert.throws(() => validateExistingMdpoTest11Lock(lock, { ...lockInput, scorerModel: 'changed-qwen' }), /does not match/);
console.log('v10-MDPO locked one-time test11 prerequisite tests passed.');
