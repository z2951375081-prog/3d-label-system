import { createHash } from 'node:crypto';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const sameHash = (left, right) => /^[a-f0-9]{64}$/i.test(String(left || '')) && String(left).toLowerCase() === String(right || '').toLowerCase();

export function validateMdpoTest11Prerequisites({ activeBytes, baselineBytes, preferenceBytes, activationReport, selection, cohort } = {}) {
  const active = JSON.parse(activeBytes.toString('utf8'));
  const activeSha256 = digest(activeBytes), baselineSha256 = digest(baselineBytes), preferenceSha256 = digest(preferenceBytes);
  if (active.version !== 'layout_model_v10_mdpo' || active.status !== 'active_val11_selected_v10_mdpo'
      || active.validation_gate?.status !== 'accepted' || active.validation_gate?.active !== true
      || active.architecture?.qwen_inference_input !== false || active.activation_provenance?.qwen_inference_input !== false) throw new Error('test11 requires the formally activated v10-MDPO model with Qwen excluded from inference');
  if (activationReport?.version !== 'v10_mdpo_activation_report_v1' || activationReport.status !== 'activated' || activationReport.test11_pending !== true
      || !sameHash(activationReport.active_sha256, activeSha256) || !sameHash(activationReport.previous_active_sha256, baselineSha256)
      || activationReport.historical_reward_model_unchanged !== true || !sameHash(activationReport.historical_reward_model_sha256, preferenceSha256)) throw new Error('test11 activation/baseline/reward evidence is incomplete or changed');
  if (selection?.version !== 'v10_mdpo_hyperparameter_selection_v1' || selection.status !== 'selected_by_complete_authoritative_val11'
      || selection.val11_gate_complete !== true || selection.deployment_eligible !== true || selection.test_used_for_selection !== false
      || selection.evaluated_configuration_count !== 81 || selection.required_grid?.total !== 81 || selection.selected?.gate?.accepted !== true
      || !sameHash(selection.selected?.candidate_sha256, active.activation_provenance?.original_candidate_sha256)
      || !sameHash(selection.selected?.report_sha256, active.activation_provenance?.val11_report_sha256)
      || !sameHash(selection.selected?.four_group_report_sha256, active.activation_provenance?.four_group_report_sha256)
      || !sameHash(active.activation_provenance?.frozen_reference_sha256, baselineSha256)
      || !sameHash(active.activation_provenance?.historical_reward_model_sha256, preferenceSha256)
      || active.activation_provenance?.historical_reward_model_preserved !== true || active.activation_provenance?.test11_used_for_activation !== false) throw new Error('test11 requires complete no-test 81-grid val11 selection and matching activation provenance');
  if (!Array.isArray(cohort) || cohort.length !== 11 || new Set(cohort).size !== 11 || cohort.some((item) => typeof item !== 'string' || !item.includes('/'))) throw new Error('test11 requires exactly 11 unique manifest test samples');
  return { active, activeSha256, baselineSha256, preferenceSha256, cohort: [...cohort] };
}

export function validateExistingMdpoTest11Lock(lock, { activeSha256, baselineSha256, preferenceSha256, candidateSha256, val11ReportSha256, fourGroupReportSha256, cohort, codeSha256, scorerModel, promptVersion } = {}) {
  if (lock?.version !== 'v10_mdpo_test11_lock_v1' || lock.status !== 'locked_pending_single_test11_evaluation'
      || lock.sample_count !== 11 || lock.cohort_split !== 'test' || JSON.stringify(lock.cohort) !== JSON.stringify(cohort)
      || !sameHash(lock.active_model?.sha256, activeSha256) || !sameHash(lock.baseline_model?.sha256, baselineSha256)
      || !sameHash(lock.historical_reward_model?.sha256, preferenceSha256) || lock.historical_reward_model?.frozen !== true
      || !sameHash(lock.selection?.candidate_sha256, candidateSha256) || !sameHash(lock.selection?.val11_report_sha256, val11ReportSha256)
      || !sameHash(lock.selection?.four_group_report_sha256, fourGroupReportSha256)
      || typeof scorerModel !== 'string' || !scorerModel || lock.scorer?.model !== scorerModel
      || typeof promptVersion !== 'string' || !promptVersion || lock.scorer?.prompt_version !== promptVersion
      || JSON.stringify(lock.scorer?.views) !== JSON.stringify(['before', 'main', 'right', 'left', 'up', 'down'])
      || JSON.stringify(lock.code_sha256) !== JSON.stringify(codeSha256)
      || !Number.isInteger(lock.seed) || lock.seed < 0
      || lock.policy?.one_final_test_only !== true || lock.policy?.test_not_used_for_training !== true
      || lock.policy?.test_not_used_for_selection !== true || lock.policy?.qwen_offline_evaluator_only !== true
      || lock.policy?.qwen_inference_input !== false) throw new Error('Existing immutable test11 lock does not match the current activated model, code, cohort, or no-leak policy');
  return lock;
}
