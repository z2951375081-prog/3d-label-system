import assert from 'node:assert/strict';
import { assessMdpoDeployment } from '../lib/mdpo-experiment-status.mjs';

const activeHash = 'a'.repeat(64), candidateHash = 'b'.repeat(64), reportHash = 'c'.repeat(64), referenceHash = 'd'.repeat(64), preferenceHash = 'e'.repeat(64), fourGroupHash = 'f'.repeat(64);
const gate = { accepted: true };
const selection = { version: 'v10_mdpo_hyperparameter_selection_v1', status: 'selected_by_complete_authoritative_val11', val11_gate_complete: true,
  deployment_eligible: true, evaluated_configuration_count: 81, required_grid: { total: 81 }, test_used_for_selection: false,
  selected: { candidate_sha256: candidateHash, report_sha256: reportHash, four_group_report_sha256: fourGroupHash, gate } };
const active = { version: 'layout_model_v10_mdpo', status: 'active_val11_selected_v10_mdpo', architecture: { qwen_inference_input: false },
  validation_gate: { status: 'accepted', active: true }, activation_provenance: { original_candidate_sha256: candidateHash,
    frozen_reference_sha256: referenceHash, val11_report_sha256: reportHash, four_group_report_sha256: fourGroupHash, historical_reward_model_sha256: preferenceHash,
    historical_reward_model_preserved: true, test11_used_for_activation: false, qwen_inference_input: false } };
const activationReport = { version: 'v10_mdpo_activation_report_v1', status: 'activated', active_sha256: activeHash,
  candidate_sha256: candidateHash, previous_active_sha256: referenceHash, four_group_report_sha256: fourGroupHash, historical_reward_model_sha256: preferenceHash,
  historical_reward_model_unchanged: true };
const valid = { active, activeSha256: activeHash, gate, selection, activationReport, preferenceSha256: preferenceHash,
  selectedCandidateSha256: candidateHash, selectedReportSha256: reportHash, selectedFourGroupSha256: fourGroupHash };
assert.equal(assessMdpoDeployment(valid).active, true);
assert.equal(assessMdpoDeployment({ ...valid, active: { ...active, version: 'layout_model_v10_mdpo_candidate' } }).active, false);
assert.equal(assessMdpoDeployment({ ...valid, selectedReportSha256: 'f'.repeat(64) }).checks.val11_artifact_hash_matches, false);
assert.equal(assessMdpoDeployment({ ...valid, selectedFourGroupSha256: 'a'.repeat(64) }).checks.four_group_artifact_hash_matches, false);
assert.equal(assessMdpoDeployment({ ...valid, selection: { ...selection, evaluated_configuration_count: 80 } }).checks.complete_authoritative_selection, false);
assert.equal(assessMdpoDeployment({ ...valid, preferenceSha256: 'f'.repeat(64) }).checks.historical_reward_model_preserved, false);
assert.equal(assessMdpoDeployment({ ...valid, active: { ...active, activation_provenance: { ...active.activation_provenance, test11_used_for_activation: true } } }).checks.test_not_used_for_activation, false);
console.log('v10-MDPO API deployment evidence-chain tests passed.');
