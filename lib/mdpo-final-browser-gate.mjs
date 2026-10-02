export function verifyMdpoFinalBrowserPrerequisites(report) {
  const selection = report?.hyperparameter_selection;
  const complete = report?.dataset?.complete === true && report.dataset.sample_count === 33
    && report.sweep?.trained === 81 && report.sweep?.evaluated === 81
    && selection?.evaluated_configuration_count === 81 && selection?.val11_gate_complete === true
    && report.ablations?.trained === 9 && report.ablations?.report?.ablations?.length === 9;
  const accepted = selection?.deployment_eligible === true
    && report.active_model?.mdpo_activated === true
    && report.test11?.status === 'complete_locked_single_test11';
  const rejected = selection?.deployment_eligible === false
    && report.val11_rejection?.verified === true
    && report.active_model?.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe'
    && report.active_model?.mdpo_activated === false
    && report.test11?.status === 'not_locked_before_successful_activation';
  if (!complete || !(accepted || rejected)) throw new Error('Final MDPO browser audit requires complete train33, 81-grid, nine val11 ablations and verified activation/test11 or all-rejected preservation');
  if (!selection.selected?.id || !report.active_model.sha256 || !report.historical_reward_model?.sha256)
    throw new Error('Final MDPO browser audit requires selected candidate and preserved model hashes');
  return { outcome: accepted ? 'activated_locked_test11_complete' : 'all_rejected_original_v10_preserved',
    expectedGate: accepted ? '所选 val11 通过' : '81 个候选全部被拒',
    expectedTest: accepted ? '已完成锁定单次 test11' : '不运行 test11' };
}

export function mdpoFinalPageReady(page, verified) {
  return Boolean(page?.dataset?.includes('33/33') && page?.sweep?.includes('81/81 已评估')
    && page?.ablations?.includes('9/9 已评估') && page?.gate?.includes(verified.expectedGate)
    && page?.test?.includes(verified.expectedTest)
    && page?.train_metrics?.includes('冻结原始 v10') && page?.train_metrics?.includes('v10-MDPO · reranker off')
    && page?.train_metrics?.includes('Δ MDPO − v10') && page?.weight_evidence?.includes('19 个矩阵'));
}
