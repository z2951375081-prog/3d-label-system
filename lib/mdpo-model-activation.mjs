import { createHash } from 'node:crypto';
import { evaluateMdpoVal11Gate } from './mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from './mdpo-four-group.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function validateMdpoActivationBundle({ selection, candidateBytes, activeBytes, val11ReportBytes, fourGroupReportBytes } = {}) {
  const candidate = JSON.parse(candidateBytes.toString('utf8'));
  const report = JSON.parse(val11ReportBytes.toString('utf8'));
  const fourGroup = JSON.parse(fourGroupReportBytes.toString('utf8'));
  const candidateHash = digest(candidateBytes), activeHash = digest(activeBytes), reportHash = digest(val11ReportBytes), fourGroupHash = digest(fourGroupReportBytes);
  if (selection?.version !== 'v10_mdpo_hyperparameter_selection_v1' || selection.status !== 'selected_by_complete_authoritative_val11'
      || selection.val11_gate_complete !== true || selection.test_used_for_selection !== false
      || selection.evaluated_configuration_count !== 81 || selection.required_grid?.total !== 81) throw new Error('MDPO activation requires complete 81-grid authoritative val11 selection');
  if (selection.deployment_eligible !== true || selection.selected?.gate?.accepted !== true
      || !Object.values(selection.selected.gate.evidence || {}).every(Boolean) || !Object.values(selection.selected.gate.checks || {}).every(Boolean)) throw new Error('Selected MDPO checkpoint did not pass every val11 activation gate');
  if (selection.selected.candidate_sha256 !== candidateHash || selection.selected.report_sha256 !== reportHash || selection.selected.four_group_report_sha256 !== fourGroupHash
      || report.candidate_model?.sha256 !== candidateHash) throw new Error('MDPO activation candidate/report SHA-256 mismatch');
  if (candidate.version !== 'layout_model_v10_mdpo_candidate' || candidate.status !== 'diagnostic_only_requires_full_val11_gate'
      || candidate.reference?.sha256 !== activeHash || candidate.reference?.frozen_verified !== true
      || candidate.architecture?.qwen_inference_input !== false) throw new Error('MDPO activation candidate/reference/inference contract invalid');
  if (candidate.mdpo?.dataset_audit?.sample_count !== 33 || candidate.mdpo?.dataset_audit?.pair_count < 198 || candidate.mdpo?.dataset_audit?.pair_count > 396
      || candidate.mdpo?.preference_holdout?.heldout_pair_count !== 33 || candidate.mdpo?.preference_holdout?.gradient_pair_count < 165
      || candidate.mdpo?.weight_update_evidence?.matrix_count !== 19
      || candidate.mdpo?.weight_update_evidence?.trainable_parameters !== candidate.mdpo?.trainable_parameters
      || Object.keys(candidate.mdpo?.weight_update_evidence?.modules || {}).length !== 7
      || Object.values(candidate.mdpo?.weight_update_evidence?.modules || {}).some((module) => module.changed_matrices < 1)
      || candidate.mdpo?.weight_update_evidence?.ablation?.effective !== true
      || candidate.mdpo?.train_unified_metrics?.baseline?.sample_count !== 33
      || candidate.mdpo?.train_unified_metrics?.candidate?.sample_count !== 33
      || Object.keys(candidate.mdpo?.train_unified_metrics?.delta || {}).length !== 9
      || Object.values(candidate.mdpo?.train_unified_metrics?.delta || {}).some((value) => !Number.isFinite(value))
      || candidate.split_policy?.train33_gradient_only !== true
      || candidate.split_policy?.train33_preference_holdout_no_gradient !== true
      || candidate.split_policy?.test11_not_loaded_or_used !== true) throw new Error('MDPO activation candidate was not trained from complete train33-only evidence');
  if (report.version !== 'v10_mdpo_val11_gate_report_v2' || report.evaluation_policy !== 'safety_priority_v2' || report.test_not_used !== true || report.candidate_model?.file !== selection.selected.candidate_file
      || report.candidate_id !== selection.selected.candidate_id || report.paired_provenance?.paired_samples !== 11
      || report.four_group_protocol?.complete !== true) throw new Error('MDPO activation val11/four-group provenance incomplete');
  if (fourGroup.version !== 'v10_mdpo_four_group_val11_v2' || fourGroup.evaluation_policy !== 'safety_priority_v2' || fourGroup.candidate_id !== report.candidate_id
      || fourGroup.candidate_model?.sha256 !== candidateHash || fourGroup.test_not_used !== true || !Array.isArray(fourGroup.groups) || fourGroup.groups.length !== 4) throw new Error('MDPO activation four-group report invalid');
  const byGroup = Object.fromEntries(fourGroup.groups.map((row) => [row.group, row]));
  const checkedFourGroup = validateMdpoFourGroupSummaries(byGroup);
  if (JSON.stringify(checkedFourGroup) !== JSON.stringify(report.four_group_protocol)
      || JSON.stringify(fourGroup.core_gate) !== JSON.stringify(report.gate)
      || JSON.stringify(byGroup.v10_no_rerank) !== JSON.stringify(report.baseline)
      || JSON.stringify(byGroup.mdpo_no_rerank) !== JSON.stringify(report.candidate)) throw new Error('MDPO activation four-group/core-gate evidence mismatch');
  const checked = evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate });
  if (!checked.accepted || JSON.stringify(checked.evidence) !== JSON.stringify(report.gate?.evidence)
      || JSON.stringify(checked.checks) !== JSON.stringify(report.gate?.checks)
      || JSON.stringify(checked.delta) !== JSON.stringify(report.gate?.delta)) throw new Error('MDPO activation gate cannot be reproduced from report evidence');
  return { candidate, report, fourGroup, candidateHash, activeHash, reportHash, fourGroupHash, checked };
}

export function materializeActiveMdpoModel(validated, { selectionFile, reportFile, preferenceModelSha256 } = {}) {
  if (!validated?.candidate || !validated?.checked?.accepted || !/^[a-f0-9]{64}$/i.test(preferenceModelSha256 || '')) throw new Error('Invalid validated MDPO activation inputs');
  const active = structuredClone(validated.candidate);
  active.version = 'layout_model_v10_mdpo';
  active.status = 'active_val11_selected_v10_mdpo';
  active.validation_gate = { ...active.validation_gate, status: 'accepted', active: true, authoritative_val11: validated.checked,
    selected_without_test: true, activated_at: new Date().toISOString() };
  active.activation_provenance = {
    policy: 'complete_81_grid_authoritative_val11_all_gates_required_no_force_override',
    original_candidate_version: validated.candidate.version, original_candidate_sha256: validated.candidateHash,
    frozen_reference_sha256: validated.activeHash, val11_report_sha256: validated.reportHash, four_group_report_sha256: validated.fourGroupHash,
    selection_file: selectionFile, val11_report_file: reportFile, historical_reward_model_sha256: preferenceModelSha256,
    historical_reward_model_preserved: true, test11_used_for_activation: false, qwen_inference_input: false
  };
  return active;
}
