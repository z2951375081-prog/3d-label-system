export function buildMdpoRejectionReport(selection, { selectionSha256, referenceSha256, historicalRewardSha256 } = {}) {
  const hash = (value) => /^[a-f0-9]{64}$/i.test(String(value || ''));
  const rows = selection?.ranked_candidates;
  if (selection?.version !== 'v10_mdpo_hyperparameter_selection_v1'
      || selection?.val11_gate_complete !== true || selection?.deployment_eligible !== false
      || selection?.test_used_for_selection !== false || selection?.evaluated_configuration_count !== 81
      || selection?.required_grid?.total !== 81 || !Array.isArray(rows) || rows.length !== 81
      || new Set(rows.map((row) => row.id)).size !== 81
      || rows.some((row) => row.gate?.accepted !== false || row.gate?.status !== 'rejected'
        || row.gate?.deployment_status !== 'diagnostic_only' || !row.gate?.violations?.length)
      || selection?.selected?.id !== rows[0]?.id || selection?.selected?.gate?.accepted !== false
      || selection?.deployment_action !== 'retain_original_v10_selected_candidate_diagnostic_only'
      || ![selectionSha256, referenceSha256, historicalRewardSha256].every(hash)) {
    throw new Error('Immutable rejection evidence requires a complete 81-candidate all-rejected val11 selection and preserved model hashes');
  }
  return {
    version: 'v10_mdpo_val11_rejection_report_v1', status: 'all_candidates_rejected_original_v10_preserved',
    selection_file: 'experiments/mdpo/hyperparameter_selection.json', selection_sha256: selectionSha256,
    original_v10_sha256: referenceSha256, historical_reward_sha256: historicalRewardSha256,
    selected_candidate_id: selection.selected.id, selected_candidate_sha256: selection.selected.candidate_sha256,
    selected_status: 'diagnostic_only', candidate_count: 81, accepted_count: 0,
    test11_not_run: true, thresholds_unchanged: true,
    rejected_candidates: rows.map((row) => ({ id: row.id, candidate_sha256: row.candidate_sha256,
      val11_report_sha256: row.report_sha256, violations: row.gate.violations, delta: row.gate.delta }))
  };
}
