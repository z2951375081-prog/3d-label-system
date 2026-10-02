const sameHash = (left, right) => /^[a-f0-9]{64}$/i.test(String(left || ''))
  && String(left).toLowerCase() === String(right || '').toLowerCase();

export function assessMdpoDeployment({
  active,
  activeSha256,
  gate,
  selection,
  activationReport,
  preferenceSha256,
  selectedCandidateSha256,
  selectedReportSha256,
  selectedFourGroupSha256
} = {}) {
  const checks = {
    formal_active_version: active?.version === 'layout_model_v10_mdpo',
    formal_active_status: active?.status === 'active_val11_selected_v10_mdpo',
    active_gate_accepted: active?.validation_gate?.status === 'accepted' && active?.validation_gate?.active === true,
    qwen_not_used_at_inference: active?.architecture?.qwen_inference_input === false && active?.activation_provenance?.qwen_inference_input === false,
    complete_authoritative_selection: selection?.version === 'v10_mdpo_hyperparameter_selection_v1'
      && selection?.status === 'selected_by_complete_authoritative_val11'
      && selection?.val11_gate_complete === true
      && selection?.deployment_eligible === true
      && selection?.evaluated_configuration_count === 81
      && selection?.required_grid?.total === 81
      && selection?.test_used_for_selection === false
      && selection?.selected?.gate?.accepted === true,
    independently_accepted_gate: gate?.accepted === true,
    activation_report_accepted: activationReport?.version === 'v10_mdpo_activation_report_v1' && activationReport?.status === 'activated',
    active_hash_matches_report: sameHash(activationReport?.active_sha256, activeSha256),
    candidate_artifact_hash_matches: sameHash(selection?.selected?.candidate_sha256, selectedCandidateSha256)
      && sameHash(selection?.selected?.candidate_sha256, activationReport?.candidate_sha256)
      && sameHash(selection?.selected?.candidate_sha256, active?.activation_provenance?.original_candidate_sha256),
    val11_artifact_hash_matches: sameHash(selection?.selected?.report_sha256, selectedReportSha256)
      && sameHash(selection?.selected?.report_sha256, active?.activation_provenance?.val11_report_sha256),
    four_group_artifact_hash_matches: sameHash(selection?.selected?.four_group_report_sha256, selectedFourGroupSha256)
      && sameHash(selection?.selected?.four_group_report_sha256, activationReport?.four_group_report_sha256)
      && sameHash(selection?.selected?.four_group_report_sha256, active?.activation_provenance?.four_group_report_sha256),
    frozen_reference_hash_matches: sameHash(activationReport?.previous_active_sha256, active?.activation_provenance?.frozen_reference_sha256),
    historical_reward_model_preserved: activationReport?.historical_reward_model_unchanged === true
      && active?.activation_provenance?.historical_reward_model_preserved === true
      && sameHash(activationReport?.historical_reward_model_sha256, preferenceSha256)
      && sameHash(active?.activation_provenance?.historical_reward_model_sha256, preferenceSha256),
    test_not_used_for_activation: active?.activation_provenance?.test11_used_for_activation === false
  };
  const violations = Object.entries(checks).filter(([, passed]) => !passed).map(([name]) => name);
  return { active: violations.length === 0, checks, violations };
}
