import { ModelViewer, parseOBJ } from './webgl-viewer.js';
import { preferenceComposite } from './preference-scoring.js';
import { assessSafetyEligibility, compareSafetyReference } from './preference-policy.js';

const $ = (id) => document.getElementById(id);
const state = { catalog: null, splitManifest: null, sample: null, sampleRequestId: 0, rawObjText: '', objText: '', sourceName: '', annotation: null, model: null, modelInfo: null, modelConfig: null, labels: [], viewLabels: null, manualLabels: [], selectedVisualizationMethod: 'v10_historical_rerank', visualizationRequestId: 0, metrics: null, timing: null, comparison: null, qualityComparison: null, reproductionComparison: null, finalEightResults: null, llmVal11Comparison: null, mdpoLiveResults: null, reproductionView: 'test11_final', alignedDisplayCandidateId: null, generationInFlight: false, combinedObj: '', viewers: {}, multiViewers: {}, manualViewers: {}, preferenceCandidates: { A: null, B: null }, optimizerStrategyOverrides: {}, preferenceEvaluationOverride: null, layoutEvaluationOverride: null, preferenceModelInfo: null, validationGenerationReceipt: null };
let all55AveragesPromise;
let mdpoLiveResultsLoading = false;

function mdpoLossCurve(history, label, getter, color) {
  const values = history.map(getter).map(Number).filter(Number.isFinite);
  if (values.length < 2) return `<div class="mdpo-curve"><small>${escapeHtml(label)}：至少 2 个真实 epoch 才绘制曲线</small></div>`;
  const low = Math.min(...values), high = Math.max(...values), span = Math.max(1e-9, high - low);
  const points = values.map((value, index) => `${(index * 440 / (values.length - 1)).toFixed(2)},${(108 - (value - low) / span * 98).toFixed(2)}`).join(' ');
  return `<div class="mdpo-curve"><small>${escapeHtml(label)} · epoch ${values.length} · ${qualityNumber(values.at(-1), 5)}</small><svg viewBox="0 0 440 118" role="img" aria-label="${escapeHtml(label)} 随 epoch 变化"><polyline fill="none" stroke="${color}" stroke-width="2.5" points="${points}"/></svg></div>`;
}

const MDPO_UI_TRAINING_FIELDS = {
  epochs: 'mdpoTrainEpochs', learningRate: 'mdpoTrainLearningRate', beta: 'mdpoTrainBeta',
  lambdaMulti: 'mdpoTrainLambdaMulti', lambdaSafe: 'mdpoTrainLambdaSafe', rank: 'mdpoTrainRank',
  seed: 'mdpoTrainSeed', patience: 'mdpoTrainPatience', lambdaDpo: 'mdpoTrainLambdaDpo',
  lambdaKl: 'mdpoTrainLambdaKl', lambdaSup: 'mdpoTrainLambdaSup', lambdaCenterTail: 'mdpoTrainLambdaCenterTail',
  alpha: 'mdpoTrainAlpha', dropout: 'mdpoTrainDropout', clipNorm: 'mdpoTrainClipNorm',
  compositionWeight: 'mdpoTrainCompositionWeight', hierarchyWeight: 'mdpoTrainHierarchyWeight',
  balanceWeight: 'mdpoTrainBalanceWeight', manualStyleWeight: 'mdpoTrainManualStyleWeight',
  textWeight: 'mdpoTrainTextWeight', leaderWeight: 'mdpoTrainLeaderWeight'
};
function mdpoUiTrainingParameters() {
  const result = { model: $('mdpoTrainModel').value, learnableVariance: $('mdpoTrainLearnableVariance').checked,
    noReference: $('mdpoTrainNoReference').checked };
  for (const [name, id] of Object.entries(MDPO_UI_TRAINING_FIELDS)) {
    const field = $(id);
    if (!field.checkValidity() || !field.value.trim() || !Number.isFinite(Number(field.value))) {
      field.reportValidity(); throw new Error('训练参数无效：' + field.closest('label')?.textContent?.trim());
    }
    result[name] = Number(field.value);
  }
  return result;
}
async function loadMdpoUiTrainingStatus() {
  const button = $('mdpoStartTrainingBtn');
  if (!button) return;
  try {
    const response = await fetch('/api/mdpo-training-control', { cache: 'no-store' });
    if (!response.ok) throw new Error('训练状态接口 ' + response.status + '；请重启服务端后刷新页面');
    const status = await response.json(), job = status.job;
    const canSubmit = status.can_submit ?? status.can_start;
    button.disabled = !canSubmit;
    button.textContent = status.pipeline?.blocks_manual_training ? '提交训练（排队等待）' : '开始训练所选模型';
    $('mdpoManualTrainingStatus').textContent = job?.status === 'queued'
      ? '已提交 ' + job.run_id + '；正式流水线结束后自动开始独立候选训练。当前 ' + status.dataset.sample_count + '/33 样本、' + status.dataset.pair_count + ' 对。'
      : job?.status === 'running'
      ? '正在训练 ' + job.run_id + '（PID ' + job.pid + '）；仅生成独立候选。' + (status.blockers.length ? ' ' + status.blockers.join('；') : '')
      : job?.status === 'completed_diagnostic_candidate'
        ? '训练完成：' + job.candidate_file + '；仅供诊断，尚未通过正式 val11 激活门控。' + (status.blockers.length ? ' ' + status.blockers.join('；') : '')
        : job?.status === 'failed' ? '训练失败：' + (job.error || '查看日志') + '。' + (status.blockers.length ? ' ' + status.blockers.join('；') : '')
          : status.can_start ? 'train33 严格校验通过，可开始独立 v10-MDPO 候选训练。'
            : status.can_submit ? 'train33 严格校验通过；现在可提交参数，正式流水线结束后自动训练。'
            : '暂不可训练：' + status.blockers.join('；') + '。当前 ' + status.dataset.sample_count + '/33 样本、' + status.dataset.pair_count + ' 对。';
    $('mdpoManualTrainingStatus').className = 'tool-status ' + (canSubmit || job?.status === 'queued' || job?.status === 'completed_diagnostic_candidate' ? 'ready' : 'warn');
    $('mdpoManualTrainingLog').textContent = job ? [
      '训练批次：' + job.run_id, '状态：' + job.status, '模型：' + job.model,
      '输出目录：' + job.output_dir, '最近 stdout：', ...(job.stdout_tail || []),
      '最近 stderr：', ...(job.stderr_tail || [])
    ].join('\n') : '暂无界面训练日志。';
  } catch (error) {
    button.disabled = true;
    $('mdpoManualTrainingStatus').textContent = error.message;
    $('mdpoManualTrainingStatus').className = 'tool-status warn';
  }
}
async function startMdpoUiTraining() {
  const button = $('mdpoStartTrainingBtn');
  try {
    const parameters = mdpoUiTrainingParameters();
    button.disabled = true;
    $('mdpoManualTrainingStatus').textContent = '正在核验 train33 与冻结参考模型，提交训练请求…';
    const response = await fetch('/api/mdpo-training-control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(parameters) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '无法开始 v10-MDPO 训练');
    showToast(result.status === 'queued' ? '训练已排队；正式流水线结束后自动开始' : '候选训练已启动；不会自动激活模型', 'success');
  } catch (error) {
    $('mdpoManualTrainingStatus').textContent = error.message;
    showToast(error.message, 'error');
  } finally { await loadMdpoUiTrainingStatus(); }
}
function bindMdpoUiTrainingControls() {
  setInterval(() => { if (!document.hidden) loadMdpoLiveResults().catch(() => {}); }, 5_000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) loadMdpoLiveResults().catch(() => {});
  });
}

async function loadMdpoExperimentStatus() {
  const dimensions = ['overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance', 'manual_style_similarity', 'text_clarity', 'leader_line_clarity'];
  try {
    const response = await fetch('/api/mdpo-experiment');
    if (!response.ok) throw new Error(`MDPO 实验接口 ${response.status}`);
    const report = await response.json();
    const training = report.training || {}, dataset = report.dataset || {}, gate = report.val11_gate || {};
    const deployed = report.active_model?.mdpo_activated === true;
    const unverifiedMdpo = report.active_model?.version === 'layout_model_v10_mdpo' && !deployed;
    const sweepTrained = Number(report.sweep?.trained || 0), sweepEvaluated = Number(report.sweep?.evaluated || 0);
    $('mdpoStatus').textContent = deployed ? 'val11 与完整证据链已通过 · v10-MDPO 正式激活'
      : unverifiedMdpo ? '发现 v10-MDPO 文件，但部署证据未通过校验'
        : sweepTrained === 81 && sweepEvaluated < 81 ? `81-grid 训练已完成 · val11 六视角评估 ${sweepEvaluated}/81 · 当前仍为原始 v10`
          : 'v10-MDPO 未激活 · 当前仍为原始 v10';
    $('mdpoActiveModel').textContent = `${report.active_model?.version || '—'} · ${deployed ? 'MDPO ACTIVE' : unverifiedMdpo ? '证据不完整' : '原始 v10 ACTIVE'}`;
    $('mdpoDatasetCount').textContent = `${dataset.sample_count || 0}/33 样本 · ${dataset.pair_count || 0}/198–396 偏好对${dataset.complete ? ' · 完整' : ' · 未完成'}`;
    $('mdpoCandidateStatus').textContent = training.best_epoch ? `${training.status} · epoch ${training.best_epoch} · ${training.trainable_parameters} 参数`
      : sweepTrained ? `${sweepTrained}/81 候选已训练 · val11 ${sweepEvaluated}/81 · 正式选模未完成` : '尚无完整 train33 正式候选';
    $('mdpoGateStatus').textContent = report.hyperparameter_selection ? gate.accepted ? '所选 val11 通过 · 核对部署证据' : report.val11_rejection?.verified ? '81 个候选全部被拒 · 原始 v10 保留 · 不运行 test11' : '所选 val11 未通过 · 拒绝证据待核对' : '等待 81-grid 正式 val11 选择';
    $('mdpoFrozenHash').textContent = `冻结参考 SHA-256：${dataset.reference_model_sha256 || '尚未落盘'}；checkpoint：${training.checkpoint_count || 0}；Qwen 不作为推理输入。`;
    $('mdpoGateReasons').textContent = gate.accepted ? `正式 val11 已通过；Δ aesthetic ${qualityNumber(gate.delta?.aesthetic, 4)}，Δ composition ${qualityNumber(gate.delta?.composition_harmony, 4)}；部署验证：${deployed ? '通过' : (report.activation?.verification?.violations || ['尚未原子激活']).join('、')}。` : report.val11_rejection?.verified ? `最佳被拒候选 ${report.val11_rejection.selected_candidate_id} 仅供诊断；拒绝原因：${report.val11_rejection.selected_violations.join('；')}。原始 v10 保留；test11 不运行。` : `未激活：${(gate.violations || ['尚未完成 81-grid、完整 val11 六图和安全门控']).join('；')}`;
    $('mdpoSweepStatus').textContent = `${sweepTrained}/81 已训练 · ${sweepEvaluated}/81 已评估`;
    $('mdpoAblationStatus').textContent = `${report.ablations?.trained || 0}/9 已训练 · ${report.ablations?.report?.ablations?.length || 0}/9 已评估`;
    $('mdpoRewardHash').textContent = report.historical_reward_model?.sha256 || '未读取到历史奖励模型';
    $('mdpoTestStatus').textContent = report.test11?.status === 'complete_locked_single_test11' ? '已完成锁定单次 test11' : report.test11?.status === 'locked_pending_single_test11_evaluation' ? '已锁定 · 等待唯一 test11' : '未锁定 · 不运行 test11';
    const history = Array.isArray(training.history) ? training.history : [];
    $('mdpoTrainingCurves').innerHTML = history.length ? [mdpoLossCurve(history, 'train 总损失', (row) => row.train?.total, '#176cbd'),
      mdpoLossCurve(history, 'train MDPO 损失', (row) => row.train?.mdpo, '#7a4bc1'),
      mdpoLossCurve(history, 'train 原始 v10 监督损失（中心/尺寸/人工风格/引导线）', (row) => row.train?.original_v10_supervised, '#4179a0'),
      mdpoLossCurve(history, 'train 几何安全损失（λ safe · 主视角/worst-view/CVaR）', (row) => row.train?.geometry_safe, '#c04e38'),
      mdpoLossCurve(history, 'train33 留出偏好损失（无梯度，非 val11）', (row) => row.val?.preference_holdout?.total, '#b06223'),
      mdpoLossCurve(history, 'val11 人工中心几何诊断损失（仅选模）', (row) => row.val?.total, '#16876a'),
      ...dimensions.flatMap((name) => [
        mdpoLossCurve(history, `train ${name}`, (row) => row.train?.[`loss_${name}`], '#176cbd'),
        mdpoLossCurve(history, `train33 留出 ${name}（非 val11）`, (row) => row.val?.preference_holdout?.[`loss_${name}`], '#b06223')
      ])].join('') : sweepTrained === 81 ? `81-grid 的 81 个候选均已完成训练；当前 val11 ${sweepEvaluated}/81，待选出正式候选后展示该候选完整 loss 曲线。` : '尚未完成 train33 正式训练；不展示烟雾测试曲线。';
    const lastEpoch = training.history?.at(-1);
    $('mdpoDimensionRows').innerHTML = dimensions.map((name) => `<tr><td>${escapeHtml(name)}</td><td>${qualityNumber(lastEpoch?.train?.[`loss_${name}`], 6)}</td><td>${qualityNumber(lastEpoch?.val?.preference_holdout?.[`loss_${name}`], 6)}</td><td>${qualityNumber(gate.candidate?.scores?.[name], 4)}</td><td>${name === 'overall' ? '基础 DPO' : name === 'text_clarity' || name === 'leader_line_clarity' ? '独立 MDPO 维度 + 非退化门控' : '独立多维偏好梯度'}</td></tr>`).join('');
    const trainUnified = training.train_unified_metrics;
    const trainMetricNames = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];
    $('mdpoTrainMetricRows').innerHTML = trainUnified?.baseline?.sample_count === 33 && trainUnified?.candidate?.sample_count === 33
      ? [['冻结原始 v10', trainUnified.baseline.metrics], ['v10-MDPO · reranker off', trainUnified.candidate.metrics], ['Δ MDPO − v10', trainUnified.delta]].map(([label, values]) => `<tr><td>${escapeHtml(label)}</td>${trainMetricNames.map((name) => `<td>${qualityNumber(values?.[name], 4)}</td>`).join('')}</tr>`).join('')
      : `<tr><td colspan="10">${sweepTrained === 81 ? `81-grid 已训练完成，val11 ${sweepEvaluated}/81；待正式选模后展示所选候选与冻结 v10 的统一指标。` : '等待完整 train33 正式候选；不显示烟雾训练或代理指标。'}</td></tr>`;
    const updateEvidence = training.weight_update_evidence;
    $('mdpoWeightEvidence').textContent = updateEvidence?.ablation?.effective ? `LoRA 更新证据：${updateEvidence.matrix_count}/19 个矩阵；${updateEvidence.trainable_parameters} / ${updateEvidence.total_numeric_parameters_including_adapters} 可训练参数（${qualityRatio(updateEvidence.trainable_fraction)}）；消融后局部 6D 最大变化 ${Number(updateEvidence.ablation.maximum_absolute_local_6d_output_change).toExponential(3)}。` : '等待 LoRA 逐模块更新与输出消融证据。';
    const ranked = report.hyperparameter_selection?.ranked_candidates;
    const trainedConfigurations = report.sweep?.configurations;
    $('mdpoSweepRows').innerHTML = Array.isArray(ranked) && ranked.length === 81 ? ranked.map((row) => `<tr><td>${escapeHtml(row.id)}</td><td>${qualityNumber(row.hyperparameters?.learningRate, 6)}</td><td>${qualityNumber(row.hyperparameters?.beta, 4)}</td><td>${qualityNumber(row.hyperparameters?.lambdaMulti, 4)}</td><td>${qualityNumber(row.hyperparameters?.rank, 0)}</td><td>${qualityNumber(row.score?.aesthetic_delta, 4)}</td><td>${qualityNumber(row.score?.composition_delta, 4)}</td><td>${row.gate?.accepted ? '通过' : '拒绝'}</td></tr>`).join('')
      : Array.isArray(trainedConfigurations) && trainedConfigurations.length
        ? trainedConfigurations.map((row) => `<tr><td>${escapeHtml(row.id)}</td><td>${qualityNumber(row.hyperparameters?.learningRate, 6)}</td><td>${qualityNumber(row.hyperparameters?.beta, 4)}</td><td>${qualityNumber(row.hyperparameters?.lambdaMulti, 4)}</td><td>${qualityNumber(row.hyperparameters?.rank, 0)}</td><td>待完整 val11</td><td>待完整 val11</td><td>训练完成 · epoch ${qualityNumber(row.best_epoch, 0)}</td></tr>`).join('')
        : '<tr><td colspan="8">尚未生成 81-grid 训练候选。</td></tr>';
    const ablation = report.four_group_ablation?.groups;
    $('mdpoAblationRows').innerHTML = Array.isArray(ablation) && ablation.length === 4 ? ablation.map((group) => `<tr><td>${escapeHtml(group.name)}</td>${['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'].map((metric) => `<td>${qualityNumber(group.metrics?.[metric], 4)}</td>`).join('')}</tr>`).join('') : '<tr><td colspan="10">尚未完成四组 val11 同协议实验，不显示推测数值</td></tr>';
    const formalAblations = report.ablations?.report?.ablations;
    $('mdpoFormalAblationRows').innerHTML = Array.isArray(formalAblations) && formalAblations.length === 9 ? formalAblations.map((row) => `<tr><td>${escapeHtml(row.id)}</td><td>${qualityNumber(row.trainable_parameters, 0)}</td>${['aesthetic', 'composition_harmony', 'PCK_005', 'PCK_010', 'text_clarity', 'leader_line_clarity'].map((metric) => `<td>${qualityNumber(row.delta?.[metric], 4)}</td>`).join('')}<td>${row.gate_accepted ? '通过' : '拒绝'}</td></tr>`).join('') : '<tr><td colspan="9">等待真实消融训练与 val11 四组评估。</td></tr>';
    const sensitivity = report.ablations?.report?.sweep_sensitivity;
    $('mdpoSensitivityStatus').textContent = sensitivity ? ['rank', 'beta'].map((name) => `${name}：${(sensitivity[name] || []).map((item) => `${item.value}（${item.accepted}/${item.configurations} 通过，平均 Δ 美学 ${qualityNumber(item.mean_aesthetic_delta, 4)}）`).join('；')}`).join('。') : 'rank / β 敏感性：等待完整 81-grid 与九组消融报告。';
    const completeTest = report.test11?.status === 'complete_locked_single_test11';
    $('mdpoTest11Status').textContent = completeTest ? `test11 单次锁定报告完成；v10-MDPO 无重排 Δ 美学 ${qualityNumber(report.test11.core_no_rerank_delta?.aesthetic, 4)}，Δ 构图 ${qualityNumber(report.test11.core_no_rerank_delta?.composition_harmony, 4)}；测试结果未用于训练或选模。` : report.test11?.status === 'locked_pending_single_test11_evaluation' ? 'test11 已锁定，等待唯一一次正式四组评价；不展示不完整结果。' : 'test11 尚未锁定：val11 完成并正式激活前不运行测试。';
    const testGroups = completeTest ? Object.values(report.test11.groups || {}) : [];
    $('mdpoTest11Rows').innerHTML = testGroups.length === 4 ? testGroups.map((group) => `<tr><td>${escapeHtml(group.label)}</td>${['PCK_005', 'PCK_010', 'OLR', 'LCD', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'].map((metric) => `<td>${qualityNumber(group.metrics?.[metric], 4)}</td>`).join('')}</tr>`).join('') : '<tr><td colspan="10">未锁定并完成 test11，不显示测试指标。</td></tr>';
  } catch (error) { $('mdpoStatus').textContent = `MDPO 状态不可用：${error.message}`; }
}

function formatBytes(bytes) { if (!bytes) return '0 B'; const units = ['B', 'KB', 'MB', 'GB']; const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1); return `${(bytes / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`; }
function setAction(text) { $('lastAction').textContent = text; }
function showToast(message, kind = 'info') { let toast = document.querySelector('.toast'); if (!toast) { toast = document.createElement('div'); toast.className = 'toast'; document.body.appendChild(toast); } toast.textContent = message; toast.dataset.kind = kind; clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.remove(), 3600); }
function splitForSample(sample) { return state.splitManifest?.samples?.find((item) => item.category === sample?.category && item.sample_id === String(sample?.id ?? sample?.sample_id))?.split || null; }
function getSamples(category) { const samples = state.catalog?.categories.find((item) => item.name === category)?.samples || []; const split = $('splitSelect')?.value || 'test'; return split === 'all' ? samples : samples.filter((sample) => splitForSample(sample) === split); }
function populateCatalog() { const categories = state.catalog.categories; const split = $('splitSelect')?.value || 'test'; $('categorySelect').innerHTML = categories.map((item) => `<option value="${item.name}">${item.name}</option>`).join(''); $('categoryList').innerHTML = categories.map((item) => `<div class="category-item"><span>${item.name}</span><b>${split === 'all' ? item.sampleCount : getSamples(item.name).length}</b></div>`).join(''); $('datasetTotal').textContent = split === 'all' ? `${state.catalog.totals.samples} 样本` : `${state.splitManifest?.counts?.[split] || 0} ${split} 样本`; updateSamples(); }
function updateSamples() { const samples = getSamples($('categorySelect').value); $('sampleSelect').innerHTML = samples.map((item) => `<option value="${item.id}">${item.id} · ${splitForSample(item) || '未划分'}</option>`).join(''); if ($('reproductionComparisonStatus')) $('reproductionComparisonStatus').textContent = `正在读取 ${$('splitSelect')?.value || 'test'} 结果…`; if ($('reproductionMetricTable')) $('reproductionMetricTable').innerHTML = '<tr><td colspan="11">正在生成当前样本并读取统一指标…</td></tr>'; if (samples[0]) loadSample(samples[0]); }
function sampleDetails(sample) { const split = splitForSample(sample); return `<b>${sample.category} / ${sample.id}</b><span class="split-badge">${split || 'custom'}</span><br>输入：${sample.mainObj ? 'main-O.obj（自动清除标签层）' : '未找到'}<br>OBJ ${sample.counts.obj} · PNG ${sample.counts.png} · JSON ${sample.counts.json ? '有' : '无'}`; }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]); }
function qualityNumber(value, digits = 3) { return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—'; }
function qualityRatio(value) { return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—'; }
function sampleKey(sample = state.sample) { return sample ? `${sample.category}/${sample.id ?? sample.sample_id}` : null; }
function modelDisplayName(model = state.modelConfig?.model) { if (model?.version === 'layout_model_v10_mdpo_candidate') return '最新完成的安全对齐 v10-MDPO'; if (model?.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') return '锚点局部坐标异构图 Transformer-MoE v10'; if (model?.version === 'layout_model_v9_3d_human_style_moe') return '纯三维人类风格图 Transformer-MoE v9'; if (model?.version === 'layout_model_v8_dgcnn_cnn_fnn_relational_graph_transformer_moe') return 'DGCNN-CNN-FNN-关系图 Transformer-MoE v8'; if (model?.version === 'layout_model_v7_fnn_relational_graph_transformer_moe') return 'FNN-关系图 Transformer-MoE v7'; if (model?.version === 'layout_model_v6_relational_graph_transformer_moe') return '关系图 Transformer-MoE v6'; return model?.version || '未加载'; }
function renderArchitectureHierarchy(model) {
  const target = document.querySelector('.architecture-hierarchy');
  if (!target) return;
  const activeV10 = model?.architecture?.type === 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe';
  const architecture = activeV10 ? model.architecture : { dgcnn_edgeconv_layers: 2, message_passing_layers: 2, transformer_layers: 1, moe_expert_count: 4 };
  target.innerHTML = `<div class="architecture-level"><span class="level-label">L1 几何与局部编码</span><article><b>DGCNN Geometry</b><strong>1024 points → ${architecture.dgcnn_edgeconv_layers || 2}×EdgeConv → 64D</strong><small>只读取 clean OBJ 表面几何</small></article><article><b>Anchor Local Frame</b><strong>weighted PCA → t₁,t₂,n</strong><small>高曲率自适应 patch、法向平均、符号对齐与对称等价方向</small></article><article><b>Label FNN</b><strong>51→64→128→64</strong><small>局部坐标、曲率、尺寸、间距与三维风格</small></article></div><div class="architecture-down">↓ typed relations</div><div class="architecture-level"><span class="level-label">L2 异构关系</span><article><b>Anchor → Label</b><strong>18D · independent weights</strong><small>法向、局部位置、距离、引导线、曲率与标签尺寸</small></article><article><b>Label → Label</b><strong>10D · independent weights</strong><small>相对位置、距离、尺寸比例、方向夹角与共享语义</small></article></div><div class="architecture-down">↓ global coordination</div><div class="architecture-level"><span class="level-label">L3 全局与解码</span><article><b>${architecture.message_passing_layers || 2}-layer Relation GNN</b><strong>64D messages</strong><small>分别学习两种关系，再聚合到标签节点</small></article><article><b>${architecture.transformer_layers || 1}-layer Transformer</b><strong>global self-attention</strong><small>协调完整标签集合</small></article><article><b>Human-style MoE</b><strong>64→32→6 local decode</strong><small>${architecture.moe_expert_count || 4} 个风格专家输出局部 u、v、法向距离与三轴尺寸比例</small></article></div><div class="architecture-down">↓ safety-aligned generation</div><div class="architecture-level"><span class="level-label">L4 多视角安全</span><article><b>Five-view Objective</b><strong>main 0.40 · others 0.15</strong><small>silhouette overlap、depth、穿模、引导线交叉、字体与留白分布</small></article><article><b>Worst-view + CVaR</b><strong>no bad-view averaging</strong><small>避免单个极差视角被平均掩盖</small></article><article><b>Aligned MDPO LoRA</b><strong>GNN + Transformer + MoE</strong><small>展示最新完整候选；不使用 Qwen 在线输入或奖励重排</small></article></div>`;
}
function renderActiveModelConfig(config) {
  state.modelConfig = config;
  const model = config?.model || {};
  renderArchitectureHierarchy(model);
  const aligned = config?.aligned_candidate;
  const architectureCard = document.querySelector('.experiment-tools:not(.scoring-settings):not(.mdpo-experiment)');
  const architectureTitle = architectureCard?.querySelector('.tools-heading h3');
  const architectureNote = architectureCard?.querySelector('.tools-note');
  const architectureKicker = architectureCard?.querySelector('.tools-heading .eyebrow');
  if (architectureKicker) architectureKicker.textContent = 'LATEST COMPLETED SAFETY-ALIGNED MODEL';
  if (architectureTitle) architectureTitle.textContent = '最新安全对齐 v10-MDPO 架构';
  if (architectureNote) architectureNote.textContent = aligned
    ? aligned.id + ' · best epoch ' + (aligned.best_epoch ?? '—') + ' · diagnostic-only / 未正式部署'
    : '等待完整的 safety_priority_v3_aligned 候选';
  const hierarchy = architectureCard?.querySelector('.architecture-hierarchy');
  if (hierarchy && aligned) hierarchy.insertAdjacentHTML('afterbegin', '<div class="aligned-architecture-summary"><b>MDPO Safety Alignment</b><span>LoRA：GNN 第 2 层 + Transformer + MoE Router / 4 Experts + 64→32→6 解码</span><span>安全目标：overlap · occlusion · crossing · overflow · penetration · mesh intersection</span><span>超参数：lr ' + escapeHtml(aligned.hyperparameters?.learningRate) + ' · β ' + escapeHtml(aligned.hyperparameters?.beta) + ' · λ multi ' + escapeHtml(aligned.hyperparameters?.lambdaMulti) + ' · rank ' + escapeHtml(aligned.hyperparameters?.rank) + '</span></div>');
  if (model.architecture?.generation_input === 'pure_3d') {
    const isV10 = model.architecture.type === 'fixed_label_anchor_frame_heterogeneous_graph_transformer_moe';
    if ($('modelEncoder')) { const encoder = $('modelEncoder'); const value = 'anchor_frame_heterogeneous_graph_transformer_moe_v10'; if (!encoder.querySelector(`option[value="${value}"]`)) encoder.add(new Option('DGCNN-局部坐标-异构图 Transformer-MoE v10', value)); encoder.value = value; }
    if ($('trainLayoutBtn')) $('trainLayoutBtn').textContent = '训练 v10 局部坐标异构图模型';
    if ($('experimentStatus')) $('experimentStatus').textContent = isV10 ? '当前活动模型：v10 局部坐标异构图纯三维生成；主视角优先并加入 worst-view/CVaR/stereo；Qwen 仅安全候选重排。' : '当前活动模型仍为 v9；本区展示并训练目标 v10，完成 val11 安全门控后才替换活动模型。';
    if ($('layoutTrainStatus')) $('layoutTrainStatus').textContent = 'v10 仅用 train 更新；val 选择检查点和推理混合比例，test 只做最终确认；五视角辅助目标包含 worst-view/CVaR、引导线方向、不相交和字体清晰度。';
  }
  const architecture = model.architecture || {};
  const preference = config?.preference_model || {};
  const routing = model.training?.routing?.val?.average_weights || {};
  const accepted = model.validation_gate?.status === 'accepted';
  const status = $('activeModelStatus');
  if (status) { status.textContent = accepted ? '验证通过 · 当前生效' : (model.status || '未加载'); status.className = `active-model-status${accepted ? ' ready' : ''}`; }
  const version = $('activeModelVersion');
  if (version) version.textContent = modelDisplayName(model);
  const detail = $('activeModelDetail');
  if (detail) detail.textContent = model.version || '—';
  const values = {
    activeNodeDim: architecture.node_input_dim,
    activeSurfacePoints: architecture.obj_surface_points,
    activeEdgeConvLayers: architecture.dgcnn_edgeconv_layers,
    activeGeometryDim: architecture.geometry_feature_dim,
    activeVisualDim: architecture.visual_feature_dim,
    activeFusionDim: architecture.fused_feature_dim,
    activeGenerationInput: architecture.generation_input || 'legacy',
    activeVisualRole: architecture.visual_generation_input === false ? '仅评价 / 安全优化' : '生成器输入',
    activeSpatialGrid: architecture.spatial_grid?.grid_size ? `${architecture.spatial_grid.grid_size}³` : null,
    activeCvImpact: Number.isFinite(Number(model.training?.functional_evidence?.cv_fusion_ablation?.max_abs_output_change)) ? Number(model.training.functional_evidence.cv_fusion_ablation.max_abs_output_change).toExponential(2) : null,
    activeEdgeDim: architecture.edge_input_dim,
    activeFnnLayers: architecture.pre_gnn_fnn_layers,
    activeFnnImpact: Number.isFinite(Number(model.training?.functional_evidence?.pre_gnn_fnn_ablation?.max_abs_output_change)) ? Number(model.training.functional_evidence.pre_gnn_fnn_ablation.max_abs_output_change).toExponential(2) : null,
    activeGnnLayers: architecture.message_passing_layers,
    activeTransformerLayers: architecture.transformer_layers,
    activeMoeExperts: architecture.moe_expert_count,
    activeOutputHead: Array.isArray(architecture.output_head) ? architecture.output_head.join('→') : null,
    activePreferenceHead: preference.architecture ? `${preference.architecture.input_dim}→${preference.architecture.hidden_dim}→${preference.architecture.output_dim}` : null
  };
  for (const [id, value] of Object.entries(values)) if ($(id)) $(id).textContent = value ?? '—';
  const cvTags = $('activeCvTags');
  if (cvTags) cvTags.hidden = !architecture.obj_surface_points;
  const route = $('activeMoeRouting');
  if (route) route.innerHTML = Object.keys(routing).length
    ? Object.entries(routing).map(([name, weight]) => `<span><b>${escapeHtml(name)}</b>${(Number(weight) * 100).toFixed(1)}%</span>`).join('')
    : '<span>等待路由统计</span>';
  const preferenceStatus = $('activePreferenceStatus');
  if (preferenceStatus) preferenceStatus.textContent = preference.validation_gate?.status === 'accepted_visual_val11' ? 'Qwen 偏好奖励已通过真实 val11 激活' : preference.validation_gate?.status === 'accepted' ? '历史门控奖励模型已激活' : (preference.status || '偏好模型未加载');
  const preferenceDetail = $('activePreferenceDetail');
  if (preferenceDetail) preferenceDetail.textContent = preference.training?.run_id_filter ? `真实 Qwen 运行 ${preference.training.run_id_filter} · ${preference.training.examples || 0} 对偏好` : '等待真实 Qwen 训练运行';
  const preferenceValidation = $('activePreferenceValidation');
  if (preferenceValidation) {
    const gate = preference.validation_gate;
    const visual = gate?.visual_activation_gate;
    const candidate = gate?.candidate;
    const previous = gate?.previous_active;
    const baseline = gate?.baseline;
    preferenceValidation.textContent = visual
      ? `真实 val11 五维综合 ${qualityNumber(visual.baseline?.composite_score, 4)} → ${qualityNumber(visual.candidate?.composite_score, 4)}；Δ综合 ${qualityNumber(visual.constraints?.aesthetic_composite_gain, 4)}；Δ构图 ${qualityNumber(visual.constraints?.composition_harmony_change, 4)}；${visual.accepted ? '通过' : '拒绝'}`
      : gate?.status === 'accepted' && candidate && previous && baseline
        ? `历史门控：构图代理 ${qualityNumber(previous.aesthetic_composition_harmony, 4)} → ${qualityNumber(candidate.aesthetic_composition_harmony, 4)}；尚未按新版五维 val11 门控激活`
        : '等待真实 val11 六图五维验证';
  }
  renderModelPathResults();
}
async function loadActiveModelConfig() {
  try {
    const response = await fetch('/api/model-config?source=latest_aligned', { cache: 'no-store' });
    const config = await response.json();
    if (!response.ok) throw new Error(config.error || '活动模型信息读取失败');
    renderActiveModelConfig(config);
  } catch (error) {
    const status = $('activeModelStatus');
    if (status) { status.textContent = error.message; status.className = 'active-model-status warn'; }
  }
}
function renderLiveSampleQuality(metrics = state.metrics) {
  const target = $('qualityLiveCurrent');
  if (!target) return;
  if (!metrics || !state.sample) { target.innerHTML = '<div class="quality-empty">选择数据集样本后，将显示活动模型的实时生成质量。</div>'; return; }
  const quality = Number(metrics.multidimensional_quality_score);
  const gate = state.layoutModelInfo?.gate?.average_weights || {};
  const split = state.splitManifest?.samples?.find((item) => item.category === state.sample.category && String(item.sample_id) === String(state.sample.id))?.split || '—';
  target.innerHTML = `<article class="live-quality-summary"><div class="live-quality-score"><span>综合质量</span><strong>${qualityNumber(quality)}</strong><small>/ 5</small></div><div class="live-quality-details"><div><span>样本</span><b>${escapeHtml(sampleKey())}</b></div><div><span>split</span><b>${escapeHtml(split)}</b></div><div><span>目标能量 ↓</span><b>${qualityNumber(metrics.objective_score)}</b></div><div><span>文字清晰 ↑</span><b>${qualityNumber(metrics.text_clarity)}</b></div><div><span>标签重叠 ↓</span><b>${qualityRatio(metrics.label_label_occlusion_ratio)}</b></div><div><span>标签遮物 ↓</span><b>${qualityRatio(metrics.label_object_occlusion_ratio)}</b></div><div><span>穿模 ↓</span><b>${qualityRatio(metrics.object_penetration_ratio)}</b></div><div><span>网格相交 ↓</span><b>${qualityRatio(metrics.mesh_surface_intersection_ratio)}</b></div><div><span>偏好分</span><b>${qualityNumber(metrics.preference_score)}</b></div></div><div class="live-quality-model"><span>${escapeHtml(modelDisplayName())}</span><small>${escapeHtml(state.modelInfo?.version || state.modelConfig?.model?.version || '')}</small><div class="live-routing">${Object.entries(gate).map(([name, weight]) => `<span>${escapeHtml(name)} ${(Number(weight) * 100).toFixed(1)}%</span>`).join('') || '<span>MoE 路由待生成</span>'}</div></div></article>`;
}
async function readAll55AveragesForLegacyServer() {
  if (!all55AveragesPromise) all55AveragesPromise = (async () => {
    const response = await fetch('/quality-all55.json');
    if (!response.ok) throw new Error('全 55 样本比较证据暂不可用');
    const summary = await response.json();
    if (summary.protocol !== 'unified_dataset_camera_55_samples'
      || summary.split_counts?.train !== 33 || summary.split_counts?.val !== 11 || summary.split_counts?.test !== 11
      || summary.averages?.length !== 4 || !summary.averages.every((item) => item.sample_count === 55)) {
      throw new Error('全 55 样本比较样本数不完整');
    }
    return summary.averages;
  })().catch((error) => { all55AveragesPromise = undefined; throw error; });
  return all55AveragesPromise;
}
function renderQualityCards(targetId, rows, emptyMessage) {
  const target = $(targetId);
  if (!target) return;
  if (!rows?.length) { target.innerHTML = `<div class="quality-empty">${escapeHtml(emptyMessage)}</div>`; return; }
  target.innerHTML = rows.map((row) => {
    const quality = row.multidimensional_quality_score === null ? NaN : Number(row.multidimensional_quality_score);
    const width = Number.isFinite(quality) ? Math.max(0, Math.min(100, quality / 5 * 100)) : 0;
    const label = row.method === 'current_fixed_label_seed17' ? '历史基础布局（冻结对照）' : row.method === 'llm_preference_after' ? 'Qwen 偏好模型（test11）' : row.label;
    return `<article class="quality-method-card" data-method="${escapeHtml(row.method)}"><div class="quality-method-head"><span>${escapeHtml(label)}</span><strong>${qualityNumber(quality)}</strong></div><div class="quality-bar-track"><div class="quality-bar" style="width:${width.toFixed(1)}%"></div></div><div class="quality-method-meta"><span>文字 ${qualityNumber(row.text_clarity)}</span><span>重叠 ${qualityRatio(row.label_label_occlusion_ratio)}</span><span>风格距 ${qualityNumber(row.manual_style_distance)}</span></div></article>`;
  }).join('');
}
function renderQualityComparison(data) {
  state.qualityComparison = data;
  renderQualityCards('qualityCurrentSample', data.sample?.rows, data.sample?.category ? `${data.sample.category}/${data.sample.sample_id} 不属于冻结 test11，当前样本不参与跨方法排名。` : '请选择数据集样本。');
  renderQualityCards('qualityTestAverage', data.averages, '尚无 test11 平均结果。');
  renderQualityCards('qualityAll55Average', data.all55_averages, '全 55 样本均值待服务重启后展示；不把 test11 均值冒充全数据集均值。');
  const table = $('qualityAverageTable');
  if (table) table.innerHTML = data.averages?.length ? data.averages.map((row) => `<tr data-method="${escapeHtml(row.method)}"><td>${escapeHtml(row.label)}</td><td>${qualityNumber(row.multidimensional_quality_score)}</td><td>${qualityNumber(row.text_clarity)}</td><td>${qualityRatio(row.label_label_occlusion_ratio)}</td><td>${qualityRatio(row.label_object_occlusion_ratio)}</td><td>${qualityRatio(row.object_penetration_ratio)}</td><td>${qualityRatio(row.mesh_surface_intersection_ratio)}</td><td>${qualityNumber(row.mean_anchor_distance)}</td><td>${qualityRatio(row.leader_length_compliance_ratio)}</td><td>${qualityNumber(row.directional_allocation_mismatch)}</td><td>${qualityNumber(row.directional_uniformity)}</td><td>${qualityNumber(row.manual_style_distance)}</td></tr>`).join('') : '<tr><td colspan="12">尚无可展示数据</td></tr>';
  const paperTable = $('paperMetricTable');
  if (paperTable) paperTable.innerHTML = data.source_paper_metrics?.rows?.length ? data.source_paper_metrics.rows.map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${qualityNumber(row.PCK_005, 4)}</td><td>${qualityNumber(row.PCK_010, 4)}</td><td>${qualityNumber(row.OLR, 4)}</td><td>${qualityNumber(row.LCD, 5)}</td><td>${row.DBV == null ? 'N/A' : qualityNumber(row.DBV, 6)}</td><td>${qualityNumber(row.avg_leader_length, 4)}</td><td>${qualityNumber(row.quality_score, 4)}</td></tr>`).join('') : '<tr><td colspan="8">论文复现英文指标尚不可用</td></tr>';
  const definitions = $('paperMetricDefinitions');
  if (definitions) definitions.innerHTML = (data.source_paper_metrics?.definitions || []).map((item) => `<article><b>${escapeHtml(item.key)} ${item.direction === 'higher' ? '↑' : '↓'}</b><span>${escapeHtml(item.meaning)}</span></article>`).join('');
  const convergence = $('qualityConvergence');
  if (convergence) convergence.innerHTML = data.llm_available && data.convergence?.length ? `<div class="quality-convergence-grid">${data.convergence.map((row) => `<div class="quality-round"><span>第 ${row.round} 轮</span><strong>${qualityNumber(row.composite_score)}</strong><small>val11 美学综合 / 5</small></div>`).join('')}</div>` : '<span class="quality-pending">等待真实 8 轮实验；不会填入模拟值。</span>';
  const status = $('qualityComparisonStatus');
  if (status) status.textContent = `${data.protocol?.cohort || 'test11'} · ${data.llm_status}`;
}
async function loadQualityComparison(sample = state.sample) {
  if (!sample) return;
  const requestedSampleKey = sampleKey(sample);
  try {
    const response = await fetch(`/api/quality-comparison?category=${encodeURIComponent(sample.category)}&sampleId=${encodeURIComponent(sample.id)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '质量比较读取失败');
    if (!data.all55_averages?.length) data.all55_averages = await readAll55AveragesForLegacyServer().catch(() => []);
    if (requestedSampleKey === sampleKey()) renderQualityComparison(data);
  } catch (error) {
    if (requestedSampleKey !== sampleKey()) return;
    const status = $('qualityComparisonStatus');
    if (status) status.textContent = error.message;
    renderQualityCards('qualityCurrentSample', [], '质量比较暂不可用。');
    renderQualityCards('qualityTestAverage', [], '质量比较暂不可用。');
  }
}

const reproductionMetricKeys = ['PCK_005', 'PCK_010', 'OLR', 'LCD', 'DBV', 'avg_leader_length', 'overlap_pairs', 'occluded_points', 'intersections', 'quality_score'];
const reproductionMetricMeanings = {
  PCK_005: '与人工优化中心误差不超过图像对角线 5% 的标签比例',
  PCK_010: '与人工优化中心误差不超过图像对角线 10% 的标签比例',
  OLR: '标签—标签及标签—物体投影重叠比例',
  LCD: '引导线交叉的几何严重程度',
  DBV: '左右眼重叠面积差异；Hedgehog 为 N/A',
  avg_leader_length: '屏幕引导线长度除以图像对角线',
  overlap_pairs: '发生投影重叠的标签对数量',
  occluded_points: '被其他标签矩形覆盖的锚点次数',
  intersections: '发生交叉的引导线对数量',
  quality_score: '统一综合分：test11 最终表采用 v4，PCK 合计 35%、LCD 20%；历史 val11 门控保留 v3'
};
function reproductionValue(key, value) {
  if (value === null || value === undefined || value === '' || !Number.isFinite(Number(value))) return '<span class="na">N/A</span>';
  if (['overlap_pairs', 'occluded_points', 'intersections'].includes(key)) return Number(value).toFixed(Number.isInteger(Number(value)) ? 0 : 3);
  return Number(value).toFixed(key === 'quality_score' ? 3 : 5);
}
function renderModelOverview() {
  const live = state.mdpoLiveResults || {};
  const aligned = live.aligned || {};
  const training = aligned.training || {};
  const val11 = aligned.val11 || {};
  const selection = aligned.selection || {};
  const ablations = aligned.ablations || {};
  const rows = Array.isArray(state.finalEightResults?.rows) ? state.finalEightResults.rows : [];
  const resultSplit = state.finalEightResults?.split === 'test' ? 'test' : 'val';
  const setText = (id, value) => { const target = $(id); if (target) target.textContent = value; };
  const setCardState = (id, value) => {
    const card = $(id)?.closest('article');
    if (card) card.dataset.state = value || '';
  };
  const countText = (done, required) => Number.isFinite(Number(done)) && Number.isFinite(Number(required))
    ? String(Number(done)) + '/' + String(Number(required)) : '—';

  setText('modelOverviewTrain', countText(training.trained, training.required || 81));
  setCardState('modelOverviewTrain', Number(training.trained) >= Number(training.required || 81) && !Number(training.failed) ? 'complete' : aligned.status === 'training' ? 'running' : 'pending');
  setText('modelOverviewVal', countText(val11.evaluated, val11.required || 81));
  setCardState('modelOverviewVal', Number(val11.evaluated) >= Number(val11.required || 81) && !Number(val11.failed) ? 'complete' : aligned.status === 'evaluating' ? 'running' : 'pending');

  const selectedId = selection.selected_candidate_id || '尚未选定';
  setText('modelOverviewSelection', selectedId);
  const hp = selection.hyperparameters || {};
  const hpValue = (camel, snake) => hp[camel] ?? hp[snake];
  const hpParts = [
    ['lr', hpValue('learningRate', 'learning_rate')],
    ['β', hp.beta],
    ['λ multi', hpValue('lambdaMulti', 'lambda_multi')],
    ['rank', hp.rank]
  ].filter((item) => item[1] !== undefined && item[1] !== null && item[1] !== '');
  setText('modelOverviewHyperparameters', hpParts.length
    ? hpParts.map((item) => item[0] + ' ' + item[1]).join(' · ')
    : selection.selected_candidate_id ? '安全优先 v3 全量 val11 选参结果' : '等待选参结果');
  setCardState('modelOverviewSelection', selection.selected_candidate_id ? (selection.deployment_eligible ? 'complete' : 'warning') : 'pending');

  const violationLabels = {
    'gate:overlap_pairs_non_degradation': '重叠对非退化未通过',
    'gate:overflow_non_degradation': '越界非退化未通过',
    overlap_pairs_non_degradation: '重叠对非退化未通过',
    overflow_non_degradation: '越界非退化未通过'
  };
  const violations = Array.isArray(selection.violations) ? selection.violations : [];
  const gateRejected = Boolean(selection.selected_candidate_id) && selection.deployment_eligible !== true;
  setText('modelOverviewGate', selection.selected_candidate_id
    ? selection.deployment_eligible === true ? '已通过 / 可部署' : '未部署 / 安全门控拒绝'
    : '等待门控');
  setText('modelOverviewViolations', violations.length
    ? violations.map((item) => violationLabels[item] || String(item).replace(/^gate:/, '')).join('；')
    : gateRejected ? '选中候选仅用于诊断，正式活动模型未替换' : '所有部署门控已通过');
  setCardState('modelOverviewGate', selection.deployment_eligible === true ? 'complete' : gateRejected ? 'rejected' : 'pending');

  const completedAblations = Number(ablations.completed || 0);
  const requiredAblations = Number(ablations.required || 9);
  const currentAblation = ablations.current?.id || ablations.current?.name || null;
  setText('modelOverviewAblation', countText(completedAblations, requiredAblations) + (currentAblation ? ' · ' + currentAblation : ''));
  const ablationMetrics = ablations.current_metrics || {};
  const epoch = Number(ablationMetrics.epoch);
  const epochs = Number(ablationMetrics.epochs);
  const bestEpoch = Number(ablationMetrics.best_epoch);
  setText('modelOverviewAblationDetail', currentAblation
    ? (Number.isFinite(epoch) ? 'epoch ' + epoch + '/' + (Number.isFinite(epochs) ? epochs : 30) : '训练进度写入中')
      + (Number.isFinite(bestEpoch) ? ' · best epoch ' + bestEpoch : '')
    : completedAblations >= requiredAblations ? '全部消融实验已完成'
      : ablations.latest_completed?.id ? '最近完成 ' + ablations.latest_completed.id : '等待消融');
  setCardState('modelOverviewAblation', completedAblations >= requiredAblations ? 'complete' : currentAblation ? 'running' : 'pending');

  const body = $('modelOverviewRows');
  if (body) body.innerHTML = rows.length ? rows.map((row) => '<tr data-role="' + escapeHtml(row.role || '') + '" data-method="' + escapeHtml(row.id || '') + '"><td><span class="final-result-name">'
    + escapeHtml(row.label || row.id || '—') + '</span><small>' + resultSplit + ' ' + escapeHtml(String(row.sample_count ?? state.finalEightResults?.sample_count ?? 11)) + ' · 五视角平均</small></td>'
    + reproductionMetricKeys.map((key) => '<td>' + reproductionValue(key, row[key]) + '</td>').join('')
    + '<td><span class="final-result-status ' + escapeHtml(row.role || '') + '">' + escapeHtml(row.status || '—') + '</span></td></tr>').join('')
    : '<tr><td colspan="12">正在读取 8 种方法统一指标…</td></tr>';

  setText('modelOverviewProtocol', '统一指标协议 v4 · 0–100 · PCK 35% · LCD 20% · 其他安全与可读性 45% · ' + resultSplit + '11 五视角平均');
  setText('modelOverviewStatus', rows.length + '/8 项 · ' + resultSplit + '11 平均 · ' + (state.finalEightResults?.generated_at ? '结果更新 ' + formatExecutionTime(state.finalEightResults.generated_at) : '等待统一结果'));
}
function renderFinalEightResults(data = state.finalEightResults) {
  const body = $('finalEightResultsRows');
  if (!body) return;
  const rows = Array.isArray(data?.rows) ? data.rows : [];
  const resultSplit = data?.split === 'test' ? 'test' : 'val';
  body.innerHTML = rows.length ? rows.map((row) => '<tr data-role="' + escapeHtml(row.role || '') + '"><td><span class="final-result-name">'
    + escapeHtml(row.label) + '</span><small>' + resultSplit + ' ' + escapeHtml(String(row.sample_count ?? data.sample_count ?? 11)) + ' · 五视角平均</small></td>'
    + reproductionMetricKeys.map((key) => '<td>' + reproductionValue(key, row[key]) + '</td>').join('')
    + '<td><span class="final-result-status ' + escapeHtml(row.role || '') + '">' + escapeHtml(row.status || '—') + '</span></td></tr>').join('')
    : '<tr><td colspan="12">暂无8项最终实验结果。</td></tr>';
  const status = $('finalEightResultsStatus');
  if (status) status.textContent = rows.length + '/8 项 · ' + resultSplit + (data?.sample_count || 11) + ' · ' + (data?.views?.length || 5) + '视角 · ' + (data?.protocol?.id || '统一协议');
  renderModelOverview();
}
async function loadFinalEightResults() {
  const response = await fetch('/api/final-eight-model-results?ts=' + Date.now(), { cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '8项最终实验结果读取失败');
  state.finalEightResults = data;
  renderFinalEightResults(data);
  renderReproductionComparison();
}
function renderReproductionComparison() {
  const body = $('reproductionMetricTable');
  if (!body) return;
  const finalView = state.reproductionView === 'test11_final';
  const sampleView = state.reproductionView === 'sample';
  const finalRows = Array.isArray(state.finalEightResults?.rows) ? state.finalEightResults.rows : [];
  const rows = finalView
    ? finalRows.map((row) => ({ ...row, method: row.id }))
    : sampleView ? (state.reproductionComparison?.sample?.rows || []) : (state.reproductionComparison?.summary || []);
  const emptyMessage = finalView ? '正在读取并审计 8 种方法 test11 结果…' : sampleView ? '当前样本没有旧基线结果。' : '当前划分没有旧基线结果。';
  body.innerHTML = rows.length ? rows.map((row) => {
    const detail = finalView
      ? '<small class="method-source-tag">test ' + escapeHtml(String(row.sample_count || 11)) + ' · 五视角 · ' + escapeHtml(row.status || '统一最终报告') + '</small>'
      : '<small class="method-source-tag">' + (sampleView ? '当前样本 · 五视角均值' : escapeHtml(state.reproductionComparison?.selected_split || 'test') + ' ' + escapeHtml(String(row.sample_count || 0)) + ' · 五视角均值') + '</small>';
    return '<tr data-method="' + escapeHtml(row.method) + '"><td>' + escapeHtml(row.label || row.method) + detail + '</td>'
      + reproductionMetricKeys.map((key) => '<td>' + reproductionValue(key, row[key]) + '</td>').join('') + '</tr>';
  }).join('') : '<tr><td colspan="11">' + emptyMessage + '</td></tr>';
  const status = $('reproductionComparisonStatus');
  if (status) status.textContent = finalView
    ? finalRows.length + '/8 方法 · test11 · 5视角 · ' + (state.finalEightResults?.audit?.passed ? '440条视角记录审计通过' : '等待审计')
    : rows.length + '/4 旧基线 · ' + (sampleView ? '当前样本' : (state.reproductionComparison?.selected_split || 'test') + '划分平均');
  const title = $('reproductionComparisonTitle');
  if (title) title.textContent = finalView ? '8种方法统一 test11 平均质量' : sampleView ? '旧基线当前样本统一指标' : '旧基线当前划分统一指标';
  const headers = document.querySelectorAll('.reproduction-comparison-card:not(.llm-val11-card) .reproduction-table thead th');
  if (headers.length >= 11) { headers[1].textContent = finalView ? 'PCK@.05 ↑（15%）' : 'PCK@.05 ↑（10%）'; headers[2].textContent = finalView ? 'PCK@.10 ↑（20%）' : 'PCK@.10 ↑（15%）'; headers[10].textContent = '安全质量分 ↑'; }
  const note = $('reproductionProtocolNote');
  if (note) note.textContent = finalView
    ? '统一采用 v4：PCK 合计 35%，LCD 20%，其余安全与可读性指标 45%。每个方法先对每个 test 样本的 5 个视角求均值，再按该样本标签数归一化计算质量分，最后对 11 个样本等权平均。MDPO 两项是事后诊断结果，仍未部署。'
    : '旧基线视图保留 v3 历史口径，采用 750×500 五视角复现相机；不与上方 test11 v4 最终均值混算。';
  for (const [id, active] of [['showMdpoLiveComparisonBtn', finalView], ['showSampleComparisonBtn', sampleView], ['showSplitComparisonBtn', !finalView && !sampleView]]) $(id)?.classList.toggle('active', active);
  const definitions = $('reproductionMetricDefinitions');
  if (definitions && !definitions.childElementCount) definitions.innerHTML = reproductionMetricKeys.map((key) => '<article><b>' + key + ' ' + (['PCK_005', 'PCK_010', 'quality_score'].includes(key) ? '↑' : '↓') + '</b><span>' + escapeHtml(reproductionMetricMeanings[key]) + '</span></article>').join('');
}
function renderMdpoLiveResults(data = state.mdpoLiveResults) {
  renderMdpoExecutionStatus(data);
  renderModelPathResults(data);
  renderReproductionComparison();
  renderModelOverview();
}
function renderModelPathResults(data = state.mdpoLiveResults) {
  const target = $('modelPathResultRows');
  if (!target) return;
  const groups = Array.isArray(data?.groups) ? data.groups : [];
  const base = groups.find((row) => row.group === 'v10_historical_rerank');
  const mdpo = groups.find((row) => row.group === 'mdpo_no_rerank');
  const mdpoRerank = groups.find((row) => row.group === 'mdpo_safe_rerank');
  const latest = data?.latest || {};
  const gate = latest.gate || {};
  const preference = state.modelConfig?.preference_model || {};
  const aesthetic = (row) => row?.scores ? preferenceComposite(row.scores) : null;
  const metricCards = (row) => row ? [
    ['Qwen 美学', aesthetic(row), 4], ['构图和谐', row.scores?.composition_harmony, 4],
    ['OLR', row.metrics?.OLR, 4], ['安全质量分', row.metrics?.quality_score, 3]
  ].map(([label, value, digits]) => '<div><span>' + label + '</span><strong>' + qualityNumber(value, digits) + '</strong></div>').join('') : '<div><span>状态</span><strong>等待数据</strong></div>';
  $('baseRewardResultMetrics').innerHTML = metricCards(base);
  $('mdpoResultMetrics').innerHTML = metricCards(mdpo);
  const baseExamples = preference.training?.examples;
  $('baseRewardResultSummary').textContent = base
    ? '原始 v10 先通过确定性安全门控，再由基础奖励 MLP 对候选重排；本卡展示同协议 val11 平均。'
    : '尚未读取到 v10 + 基础奖励模型的同协议 val11 结果。';
  $('baseRewardResultDetail').textContent = '12→32→1 pairwise MLP · ' + (Number.isFinite(Number(baseExamples)) ? Number(baseExamples) + ' 对活动训练样本 · ' : '') + '仅在安全候选中重排';
  const mdpoAccepted = gate.accepted === true && latest.deployed === true;
  const mdpoRejected = gate.accepted === false || latest.status === 'diagnostic_only_rejected';
  const mdpoBadge = $('mdpoResultBadge');
  mdpoBadge.textContent = mdpoAccepted ? '已部署' : mdpoRejected ? 'val11 已拒绝' : '候选评估中';
  mdpoBadge.className = 'model-path-badge ' + (mdpoAccepted ? 'accepted' : mdpoRejected ? 'rejected' : 'diagnostic');
  $('mdpoResultSummary').textContent = mdpo
    ? 'MDPO 直接更新 v10 的 LoRA / bias；主结果关闭基础奖励重排，以隔离策略更新本身的效果。'
    : '尚未读取到完整的 v10-MDPO 同协议 val11 结果。';
  const hp = latest.hyperparameters || {};
  $('mdpoResultDetail').textContent = latest.candidate_id
    ? latest.candidate_id + ' · lr ' + qualityNumber(hp.learningRate, 6) + ' · β ' + qualityNumber(hp.beta, 2) + ' · λ multi ' + qualityNumber(hp.lambdaMulti, 2) + ' · rank ' + qualityNumber(hp.rank, 0)
    : 'LoRA 更新 GNN2、Transformer 与 MoE；Qwen 不参与在线推理';
  const rows = [
    { row: base, role: 'base', label: 'v10 + 基础奖励模型', note: '历史 12→32→1 MLP 重排', status: '活动基线', statusClass: 'active' },
    { row: mdpo, role: 'mdpo', label: 'v10 + MDPO', note: '关闭末端奖励重排', status: mdpoAccepted ? '已部署' : mdpoRejected ? '拒绝 / 诊断' : '评估中', statusClass: mdpoAccepted ? 'accepted' : 'rejected' },
    { row: mdpoRerank, role: 'mdpo-rerank', label: 'v10 + MDPO + 奖励重排', note: '补充对照，不代表 MDPO 单独效果', status: mdpoAccepted ? '部署对照' : '候选对照', statusClass: mdpoAccepted ? 'accepted' : 'rejected' }
  ].filter((item) => item.row);
  target.innerHTML = rows.length ? rows.map((item) => {
    const row = item.row, metrics = row.metrics || {}, scores = row.scores || {};
    return '<tr data-role="' + item.role + '"><td><span class="model-path-row-label">' + escapeHtml(item.label) + '<small>' + escapeHtml(item.note) + '</small></span></td>'
      + '<td>' + qualityNumber(aesthetic(row), 4) + '</td><td>' + qualityNumber(scores.composition_harmony, 4) + '</td><td>' + qualityNumber(scores.overall, 4) + '</td>'
      + '<td>' + qualityNumber(metrics.PCK_010, 4) + '</td><td>' + qualityNumber(metrics.OLR, 4) + '</td><td>' + qualityNumber(metrics.overlap_pairs, 4) + '</td>'
      + '<td>' + qualityNumber(metrics.occluded_points, 4) + '</td><td>' + qualityNumber(metrics.worst_view_overflow, 4) + '</td><td>' + qualityNumber(metrics.quality_score, 3) + '</td>'
      + '<td><span class="model-path-status ' + item.statusClass + '">' + escapeHtml(item.status) + '</span></td></tr>';
  }).join('') : '<tr><td colspan="11">尚无完整的同协议四组 val11 结果。</td></tr>';
  const updated = $('modelPathResultsUpdated');
  if (updated) updated.textContent = data?.generated_at ? '数据更新 ' + formatExecutionTime(data.generated_at) : '等待实时结果';
  const interpretation = $('modelPathResultInterpretation');
  if (interpretation) {
    const delta = gate.delta || {};
    const violations = Array.isArray(gate.violations) ? gate.violations.map((item) => String(item).replace(/^gate:/, '')).join('、') : '';
    interpretation.textContent = gate.accepted
      ? '最新 MDPO 候选已通过 val11 门控；相对无重排 v10 的 Δ美学为 ' + qualityNumber(delta.aesthetic, 4) + '，并满足几何安全约束。'
      : mdpo ? '当前结论：MDPO 相对无重排 v10 的 Δ美学为 ' + qualityNumber(delta.aesthetic, 4) + '、Δ构图为 ' + qualityNumber(delta.composition_harmony, 4) + '，但因 ' + (violations || '安全非退化条件') + ' 未通过而保持 diagnostic-only；正式活动模型未被替换。'
        : '等待完整 MDPO val11 门控结论。';
  }
}
function formatExecutionTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString('zh-CN', { hour12: false });
}
function renderMdpoExecutionStatus(data = state.mdpoLiveResults) {
  const execution = data?.execution || {};
  const progress = data?.progress || {};
  const badge = $('mdpoEvaluationBadge');
  const message = $('mdpoEvaluationMessage');
  const candidate = $('mdpoEvaluationCandidate');
  const stage = $('mdpoEvaluationStage');
  const completed = $('mdpoEvaluationProgress');
  const started = $('mdpoEvaluationStartedAt');
  const heartbeat = $('mdpoEvaluationHeartbeat');
  const card = $('mdpoEvaluationStatus');
  if (!badge || !message) return;
  const labels = { evaluating: '正在评估', waiting: '等待中', error: '异常', completed: '已完成', pending: '读取中' };
  const status = labels[execution.status] ? execution.status : 'pending';
  badge.textContent = labels[status];
  badge.className = 'mdpo-evaluation-badge ' + status;
  if (card) card.className = 'mdpo-evaluation-status ' + status;
  message.textContent = execution.explanation || (status === 'pending' ? '正在读取实时执行状态…' : '暂无状态说明');
  const current = execution.current_candidate;
  candidate.textContent = current?.id || progress.running?.id || '当前没有运行中的候选';
  stage.textContent = execution.stage_id ? execution.stage_id + (execution.child_alive ? ' · 子进程在线' : '') : '—';
  const aligned = data?.aligned;
  const groupProgress = aligned?.val11_group_progress;
  completed.textContent = aligned?.status === 'evaluating'
    ? '候选 ' + (aligned.val11?.evaluated || 0) + '/' + (aligned.val11?.required || 81) + ' · 当前 ' + (groupProgress?.completed || 0) + '/' + (groupProgress?.required || 44)
    : (progress.evaluated || 0) + '/' + (progress.required || 81) + ' · 通过 ' + (progress.accepted || 0) + ' · 拒绝 ' + (progress.rejected || 0);
  started.textContent = formatExecutionTime(current?.started_at);
  const metricWrittenAt = groupProgress?.updated_at || aligned?.current_metrics?.updated_at || aligned?.updated_at || execution.last_activity_at || data?.generated_at;
  heartbeat.textContent = formatExecutionTime(metricWrittenAt);
  const sync = $('mdpoLiveUpdatedAt');
  if (sync) {
    const active = ['training', 'evaluating'].includes(aligned?.status) || execution.status === 'evaluating';
    const metricAgeMs = Date.now() - Date.parse(metricWrittenAt || 0);
    const stale = active && (!Number.isFinite(metricAgeMs) || metricAgeMs > 120_000);
    sync.textContent = formatExecutionTime(data?.generated_at) + (stale ? ' · 指标超过 2 分钟未写入' : ' · 已同步');
    sync.className = stale ? 'stale' : 'live';
  }
  const alignedStatus = $('mdpoAlignedStatus');
  if (alignedStatus) {
    const training = aligned?.training || {}, val11 = aligned?.val11 || {};
    const currentAligned = aligned?.current?.id ? ' · 当前 ' + aligned.current.id : '';
    alignedStatus.textContent = aligned?.status === 'training'
      ? '训练中：' + (training.trained || 0) + '/' + (training.required || 81) + ' 组已完成' + currentAligned
      : aligned?.status === 'evaluating'
        ? 'val11 评估中：' + (val11.evaluated || 0) + '/' + (val11.required || 81) + ' 个候选已完成 · 当前候选 ' + (groupProgress?.completed || 0) + '/' + (groupProgress?.required || 44) + currentAligned
        : aligned?.status === 'completed'
          ? 'v3 对齐训练与 val11 已完成'
          : aligned?.status === 'error'
            ? '对齐链路异常：训练失败 ' + (training.failed || 0) + '，评估失败 ' + (val11.failed || 0)
            : '等待对齐版 81-grid 训练…';
  }
  const groupTarget = $('mdpoVal11GroupProgress');
  if (groupTarget) {
    const groups = Array.isArray(groupProgress?.groups) ? groupProgress.groups : [];
    const activeGroup = groupProgress?.active_group;
    groupTarget.innerHTML = groups.length
      ? '<div class="mdpo-group-progress-head"><span>当前候选四组内部进度</span><strong>' + escapeHtml(String(groupProgress.completed)) + '/' + escapeHtml(String(groupProgress.required)) + (groupProgress.complete ? ' · 已完成' : ' · 正在评估 ' + escapeHtml(groupProgress.active_group_label || '')) + '</strong></div><div class="mdpo-group-progress-grid">'
        + groups.map((group) => {
          const role = group.complete ? 'complete' : group.group === activeGroup ? 'running' : 'pending';
          const percentage = Math.max(0, Math.min(100, Number(group.completed || 0) / Number(group.required || 11) * 100));
          return '<div class="mdpo-group-progress-item ' + role + '"><div class="mdpo-group-progress-title"><span>' + escapeHtml(group.label) + '</span><strong>' + escapeHtml(String(group.completed)) + '/' + escapeHtml(String(group.required)) + '</strong></div><div class="mdpo-group-progress-track"><i style="width:' + percentage.toFixed(2) + '%"></i></div><small>' + (group.latest_sample ? '最新：' + escapeHtml(group.latest_sample) + ' · ' + escapeHtml(formatExecutionTime(group.updated_at)) : role === 'running' ? '等待首条结果写入' : '尚未开始') + '</small></div>';
        }).join('') + '</div>'
      : '<div class="mdpo-group-progress-head"><span>当前候选四组内部进度</span><strong>等待评估记录…</strong></div>';
  }
}
async function loadMdpoLiveResults() {
  if (mdpoLiveResultsLoading) return;
  mdpoLiveResultsLoading = true;
  try {
    let response = await fetch('/api/mdpo-live-results?ts=' + Date.now(), { cache: 'no-store' });
    if (!response.ok && response.status === 404) response = await fetch('/mdpo-live-results.json?ts=' + Date.now(), { cache: 'no-store' });
    if (!response.ok) throw new Error('最新 MDPO 实时结果 ' + response.status);
    const data = await response.json(); state.mdpoLiveResults = data;
    const completedAlignedId = data?.aligned?.latest_completed_training?.id || null;
    const alignedChanged = Boolean(state.alignedDisplayCandidateId && completedAlignedId && completedAlignedId !== state.alignedDisplayCandidateId);
    if (completedAlignedId) state.alignedDisplayCandidateId = completedAlignedId;
    renderMdpoLiveResults(data);
    if (alignedChanged && state.sample) {
      await loadActiveModelConfig();
      await generateLabels('annotation');
      await loadReproductionComparison(state.sample);
      showToast('最新完成的 aligned 模型已同步到标签放置结果', 'success');
    }
  } catch (error) {
    const sync = $('mdpoLiveUpdatedAt');
    if (sync) { sync.textContent = '同步失败：' + error.message; sync.className = 'error'; }
  } finally { mdpoLiveResultsLoading = false; }
}
async function loadReproductionComparison(sample = state.sample) {
  if (!sample) return;
  const requestedSampleKey = sampleKey(sample);
  try {
    const selectedSplit = $('splitSelect')?.value || splitForSample(sample) || 'test';
    const query = new URLSearchParams({ split: selectedSplit, category: sample.category, sampleId: sample.id });
    const response = await fetch('/api/reproduction-comparison?' + query.toString());
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '统一复现指标读取失败');
    if (requestedSampleKey !== sampleKey()) return;
    state.reproductionComparison = data;
    renderReproductionComparison();
  } catch (error) {
    if (requestedSampleKey !== sampleKey()) return;
    const status = $('reproductionComparisonStatus');
    if (status) status.textContent = error.message;
    if ($('reproductionMetricTable')) $('reproductionMetricTable').innerHTML = `<tr><td colspan="11">${escapeHtml(error.message)}</td></tr>`;
  }
}

function renderLLMVal11Comparison(data = state.llmVal11Comparison) {
  const roundBody = $('llmVal11RoundTable');
  const metricBody = $('llmVal11MetricTable');
  const status = $('llmVal11Status');
  if (roundBody) {
    roundBody.innerHTML = data?.rounds?.length ? data.rounds.map((row) => {
      const gate = row.activation_gate;
      const scores = row.score_means || {};
      const geometry = Number(row.round) === 0 ? '基线' : gate?.constraints?.geometry_safety_eligible ? '通过' : '未通过';
      const gateStatus = Number(row.round) === 0 ? '训练前基线' : gate?.accepted ? '已通过' : '已拒绝';
      return `<tr data-round="${Number(row.round)}"><td>${Number(row.round)}${Number(row.round) === 0 ? ' · 基线' : ''}</td><td>${qualityNumber(scores.manual_style_similarity, 4)}</td><td>${qualityNumber(scores.spatial_balance, 4)}</td><td>${qualityNumber(scores.visual_hierarchy, 4)}</td><td>${qualityNumber(scores.composition_harmony, 4)}</td><td>${qualityNumber(scores.overall, 4)}</td><td>${qualityNumber(row.composite_score, 4)}</td><td>${Number(row.round) === 0 ? '—' : qualityNumber(gate?.constraints?.aesthetic_composite_gain, 4)}</td><td>${Number(row.round) === 0 ? '—' : qualityNumber(gate?.constraints?.composition_harmony_change, 4)}</td><td>${geometry}</td><td>${gateStatus}</td></tr>`;
    }).join('') : '<tr><td colspan="11">没有可用的 val11 Qwen 轮次结果。</td></tr>';
  }
  if (metricBody) {
    if (state.mdpoLiveResults?.groups?.length) renderMdpoLiveResults(state.mdpoLiveResults);
    else metricBody.innerHTML = '<tr><td colspan="10">正在读取最新完成的 v10-MDPO val11 候选…</td></tr>';
  }
  if (status) {
    const accepted = data?.accepted_rounds || [];
    const best = data?.best_trained_round;
    status.textContent = accepted.length
      ? `val11 · ${data.sample_count || 0} 样本 · 通过轮次 ${accepted.join(', ')}`
      : best ? `val11 · ${data.sample_count || 0} 样本 · 最佳训练轮 round ${best.round}，Δ综合 ${qualityNumber(best.aesthetic_composite_gain, 4)}，未激活` : 'val11 结果不可用';
  }
}

async function loadLLMVal11Comparison() {
  try {
    const response = await fetch('/api/llm-val11-comparison');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'val11 指标读取失败');
    state.llmVal11Comparison = data;
    renderLLMVal11Comparison(data);
  } catch (error) {
    if ($('llmVal11Status')) $('llmVal11Status').textContent = error.message;
    if ($('llmVal11RoundTable')) $('llmVal11RoundTable').innerHTML = `<tr><td colspan="11">${escapeHtml(error.message)}</td></tr>`;
    if ($('llmVal11MetricTable')) $('llmVal11MetricTable').innerHTML = `<tr><td colspan="10">${escapeHtml(error.message)}</td></tr>`;
  }
}

function isPresentationGroupName(name) { return /^(label_|leader_)/i.test(String(name || '')); }
function isPresentationMaterialName(name) { return /^(label_|leader_)/i.test(String(name || '')); }
function stripExistingLabels(objText) {
  // The dataset's Obj-O exports contain object geometry plus label panels,
  // leader lines and anchor-region material styling. Rebuild a compact OBJ
  // from retained object faces only; presentation vertices are dropped and
  // anchor-region colors are normalized to the neutral object material.
  const vertices = [];
  const faces = [];
  let currentGroup = 'object';
  let skipGroup = false;
  let skipMaterial = false;
  for (const line of objText.split(/\r?\n/)) {
    const trimmed = line.trim();
    const parts = trimmed.split(/\s+/);
    if (parts[0] === 'v' && parts.length >= 4) {
      vertices.push(parts.slice(1, 4));
      continue;
    }
    if (parts[0] === 'g' || parts[0] === 'o') {
      const groupName = parts.slice(1).join('_') || 'object';
      if (parts[0] === 'g') currentGroup = groupName;
      skipGroup = parts.slice(1).some(isPresentationGroupName);
      skipMaterial = false;
      continue;
    }
    if (parts[0] === 'usemtl') {
      // anchor_region_* is part of the original object surface assignment,
      // but its material color is a manual anchor cue. We keep the face
      // geometry and later emit one neutral object material for all faces.
      skipMaterial = isPresentationMaterialName(parts[1] || '');
      continue;
    }
    if (parts[0] === 'f' && parts.length >= 4 && !skipGroup && !skipMaterial) {
      const refs = parts.slice(1).map((part) => Number(part.split('/')[0]));
      if (refs.every((ref) => Number.isInteger(ref) && ref !== 0)) faces.push({ refs, group: currentGroup });
    }
  }
  const used = new Map();
  const cleanVertices = [];
  const resolve = (ref) => {
    const oldIndex = ref > 0 ? ref - 1 : vertices.length + ref;
    if (!used.has(oldIndex)) { used.set(oldIndex, cleanVertices.length + 1); cleanVertices.push(vertices[oldIndex]); }
    return used.get(oldIndex);
  };
  const cleanFaces = faces.map((face) => ({ indices: face.refs.map(resolve), group: face.group }));
  const output = ['# clean input generated from Obj-O', '# presentation layers removed; anchor material styling normalized to object_default', 'o clean_model', 'g clean_model', 'usemtl object_default'];
  cleanVertices.forEach((vertex) => output.push(`v ${vertex.join(' ')}`));
  let previousGroup = null;
  cleanFaces.forEach((face) => { if (face.indices.every((index) => index !== null)) { if (face.group !== previousGroup) { output.push(`g ${face.group}`); previousGroup = face.group; } output.push(`f ${face.indices.join(' ')}`); } });
  return `${output.join('\n')}\n`;
}

async function loadSample(sample) { state.sample = sample; state.sampleRequestId += 1; state.rawObjText = ''; state.annotation = null; state.metrics = null; state.reproductionComparison = null; state.preferenceCandidates = { A: null, B: null }; state.llmScores = null; updateCandidateStatus(); renderLiveSampleQuality(null); $('preferenceStatus').textContent = '新样本已载入，请分别保存候选 A/B'; $('sampleMeta').innerHTML = sampleDetails(sample); state.sourceName = sample.mainObj.split('/').pop(); $('pageTitle').textContent = `${sample.category} · ${sample.id} · ${splitForSample(sample) || 'custom'}`; setAction(`正在载入 ${sample.category}/${sample.id} 的所选方法…`); await loadModelVisualization(); await loadReproductionComparison(sample); }

function updateMetrics(metrics, timing = state.timing, comparison = state.comparison) {
  state.metrics = metrics;
  state.timing = timing;
  state.comparison = comparison;
  const fixed = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : null;
  const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : null;
  const values = {
    mainReadability: metrics?.readability,
    mainCoverage: metrics?.coverage,
    mainOverlap: Number.isFinite(Number(metrics?.overlap_pairs)) ? `${metrics.overlap_pairs} 对` : null,
    mainObjectOcclusion: percent(metrics?.object_occlusion_ratio),
    mainOverflow: percent(metrics?.viewport_overflow_ratio),
    mainCrossings: Number.isFinite(Number(metrics?.leader_crossings)) ? `${metrics.leader_crossings} 条` : null,
    mainAnchorCoverage: percent(metrics?.anchor_coverage),
    mainDisparity: fixed(metrics?.binocular_disparity),
    mainLeaderLength: fixed(metrics?.mean_anchor_distance),
    mainLeaderCompliance: percent(metrics?.leader_length_compliance_ratio),
    mainDirectionalMismatch: fixed(metrics?.directional_allocation_mismatch),
    mainDirectionalUniformity: fixed(metrics?.directional_uniformity),
    v10SilhouetteOverlap: percent(metrics?.weighted_label_object_overlap),
    v10WorstDepth: percent(metrics?.worst_view_depth_occlusion),
    v10WorstPenetration: percent(metrics?.worst_view_penetration_v10),
    v10CvarOcclusion: fixed(metrics?.cvar_view_occlusion),
    v10CvarFreeSpace: fixed(metrics?.cvar_view_free_space_mismatch),
    v10WorstTextLoss: fixed(metrics?.worst_view_text_clarity_loss),
    v10WeightedLeaderCrossing: fixed(metrics?.weighted_leader_crossing_risk),
    v10WorstLeaderCrossing: fixed(metrics?.worst_view_leader_crossing_risk),
    v10CvarLeaderCrossing: fixed(metrics?.cvar_view_leader_crossing_risk),
    v10FreeSpaceMismatch: fixed(metrics?.weighted_free_space_mismatch),
    v10ViewObjective: fixed(metrics?.view_conditioned_objective),
    generationTime: Number.isFinite(Number(timing?.total_ms)) ? `${timing.total_ms} ms` : null,
    generationSpeed: Number.isFinite(Number(timing?.labels_per_second)) ? `${timing.labels_per_second} 标签/s` : null,
    manualScore: Number.isFinite(Number(comparison?.generated_score)) ? `${comparison.generated_score}` : null,
    manualOlr: fixed(comparison?.metrics?.olr?.generated),
    manualLcd: fixed(comparison?.metrics?.lcd?.generated),
    manualDbv: fixed(comparison?.metrics?.dbv?.generated)
  };
  for (const [id, value] of Object.entries(values)) {
    const element = $(id);
    if (element) element.textContent = value === null || value === undefined || Number.isNaN(value) ? '—' : value;
  }
  const note = $('metricsNote');
  if (note) note.textContent = metrics
    ? `综合质量 ${qualityNumber(metrics.multidimensional_quality_score)} / 5 · 五视角联合 · 几何目标 ${qualityNumber(metrics.objective_score, 2)} · v10 视角目标 ${qualityNumber(metrics.view_conditioned_objective, 2)}`
    : '生成标签后自动计算';
  const compareNote = $('comparisonNote');
  if (compareNote) compareNote.textContent = comparison ? `人工布局 = 100 · 当前模型 = ${qualityNumber(comparison.generated_score, 0)}` : '载入人工 JSON 后可比较';
  renderLiveSampleQuality(metrics);
  const displayCandidate = state.layoutModelInfo?.evaluation?.candidate_id;
  if (displayCandidate && $('afterBadge')) $('afterBadge').textContent = state.labels.length + ' 个标签 · ' + displayCandidate;
  renderReproductionComparison();
}
function updateFiveViews() { if (!state.model) return; for (const [view, viewer] of Object.entries(state.multiViewers)) { const labels = state.viewLabels?.[view] || state.labels; viewer.setContent(state.model, labels, getCombinedFrame(state.model, labels)); } const manualFrame = getCombinedFrame(state.model, state.manualLabels); for (const viewer of Object.values(state.manualViewers)) viewer.setContent(state.model, state.manualLabels, manualFrame); }

function applyModelVisualization(result) {
  state.objText = result.clean_obj;
  state.modelInfo = result.model || null;
  state.labels = result.labels || [];
  state.viewLabels = result.labels_by_view || null;
  state.labelContract = result.label_contract || null;
  state.layoutModelInfo = result.layout_model || null;
  state.preferenceModelInfo = result.preference_model || null;
  state.validationGenerationReceipt = result.validation_generation_receipt || null;
  state.manualLabels = result.manual_reference?.labels || [];
  state.metrics = result.metrics || null;
  state.timing = result.timing || null;
  state.comparison = result.manual_comparison || null;
  state.model = parseOBJ(state.objText);
  applyPartColors(state.labels);
  state.viewers.before.setContent(state.model, [], state.model.bounds);
  state.viewers.after.setContent(state.model, state.labels, getCombinedFrame(state.model, state.labels));
  updateFiveViews();
  buildLabelList();
  state.combinedObj = exportObjWithLabels(state.objText, state.labels);
  $('downloadBtn').disabled = false;
  $('downloadJsonBtn').disabled = false;
  $('labelCount').textContent = state.labels.length;
  $('beforeBadge').textContent = '干净输入';
  $('beforeBadge').classList.add('ready');
  $('beforeFile').textContent = state.sourceName + ' · clean';
  $('objStat').textContent = formatBytes(new Blob([state.objText]).size);
  $('afterBadge').textContent = state.labels.length + ' 个标签';
  $('afterFile').textContent = (result.method_label || result.method || 'selected') + ' · ' + (result.fixed_view_layout ? '主视角复现' : '可旋转 OBJ');
  if ($('afterResultTitle')) $('afterResultTitle').textContent = result.method_label || result.method || '所选方法';
  if ($('afterResultDescription')) $('afterResultDescription').textContent = result.method_note || '统一模型展示';
  if ($('selectedMethodFiveViewTitle')) $('selectedMethodFiveViewTitle').textContent = (result.method_label || result.method || '所选方法') + (result.fixed_view_layout ? ' · 各固定相机独立复现' : ' · 同一三维布局五视角');
  if ($('fiveViewComparisonTitle')) $('fiveViewComparisonTitle').textContent = (result.method_label || result.method || '所选方法') + ' vs 人工参考 · 五视角对比';
  updateMetrics(state.metrics, state.timing, state.comparison);
}

async function loadModelVisualization(method = $('modelVisualizationSelect')?.value || state.selectedVisualizationMethod) {
  if (!state.sample) { showToast('请先选择数据集样本', 'error'); return; }
  state.selectedVisualizationMethod = method;
  const requestId = ++state.visualizationRequestId;
  const requestSampleKey = sampleKey();
  const select = $('modelVisualizationSelect');
  const status = $('modelVisualizationStatus');
  if (select) select.disabled = true;
  if (status) { status.textContent = '正在生成并载入所选方法…'; status.className = 'loading'; }
  try {
    const response = await fetch('/api/model-visualization', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ method, sample: { category: state.sample.category, sample_id: state.sample.id } }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '模型可视化读取失败');
    if (requestId !== state.visualizationRequestId || requestSampleKey !== sampleKey()) return;
    applyModelVisualization(result);
    if (status) { status.textContent = (result.fixed_view_layout ? '固定视角复现' : '三维布局') + ' · ' + result.method_label + ' · 已同步 OBJ 与五视角'; status.className = 'ready'; }
    setAction('已切换到 ' + result.method_label);
  } catch (error) {
    if (requestId !== state.visualizationRequestId || requestSampleKey !== sampleKey()) return;
    if (status) { status.textContent = error.message; status.className = 'error'; }
    showToast(error.message, 'error');
  } finally {
    if (requestId === state.visualizationRequestId && select) select.disabled = false;
  }
}
async function prepareModel() { state.model = parseOBJ(state.objText); state.manualLabels = []; state.metrics = null; state.timing = null; state.comparison = null; updateMetrics(null, null, null); state.viewers.before.setContent(state.model, [], state.model.bounds); state.viewers.after.setContent(state.model, [], state.model.bounds); updateFiveViews(); $('beforeBadge').textContent = '干净输入'; $('beforeBadge').classList.add('ready'); $('beforeFile').textContent = `${state.sourceName} · clean`; $('objStat').textContent = formatBytes(new Blob([state.objText]).size); $('downloadBtn').disabled = true; $('downloadJsonBtn').disabled = true; $('afterBadge').textContent = '未生成'; $('afterFile').textContent = '—'; }
function annotationsToLabels(annotation) { if (!annotation?.groups?.length) return []; return annotation.groups.map((group, index) => { const label = group.label || {}; const anchor = group.anchor?.point || group.leader_line?.start || label.center || [0, 0, 0]; const center = label.center || group.leader_line?.end || anchor; return { id: group.group_id || `label-${index + 1}`, text: label.text || group.group_id || `label-${index + 1}`, anchor, center, boxSize: label.box_size || [0.25, 0.12, 0.02], bendPoints: group.leader_line?.bend_points || [], sourceObjs: group.source_objs || [], targetGroups: group.target_g || [] }; }); }
function bboxLabels(model) { const { center, max, radius } = model.bounds; return [{ id: 'model-bounding-box', text: 'OBJ MODEL', anchor: [center[0], max[1], center[2]], center: [center[0], max[1] + radius * 0.7, center[2]], boxSize: [radius * 1.55, radius * 0.22, radius * 0.025], bendPoints: [], sourceObjs: [] }]; }
function getCombinedFrame(model) { const { center, radius } = model.bounds; const fixedRadius = Math.max(radius * 1.7, 0.001); return { ...model.bounds, center: [...center], radius: fixedRadius, camera_frame_policy: 'fixed_per_object_across_candidates_and_views' }; }
async function runGenerator(type) { if (type === 'annotation') return annotationsToLabels(state.annotation); if (type === 'bbox') return bboxLabels(state.model); return []; }
function numericInput(id, fallback, min, max) { const value = Number($(id)?.value); return Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback; }
const labelPalette = [[0.93, 0.35, 0.25], [0.20, 0.55, 0.95], [0.95, 0.66, 0.16], [0.55, 0.34, 0.86], [0.14, 0.68, 0.56], [0.84, 0.27, 0.58], [0.42, 0.70, 0.25], [0.18, 0.72, 0.78]];
function applyPartColors(labels) { if (!state.model) return; const colors = {}; labels.forEach((label, index) => { const color = label.color || labelPalette[index % labelPalette.length]; label.color = [...color]; [...(label.targetGroups || []), ...(label.sourceObjs || [])].forEach((key) => { colors[key] = [...color]; colors[String(key).replace(/\.obj$/i, '')] = [...color]; }); }); state.model.partColors = colors; state.viewers.before?.draw(); }
function currentOptimizerOptions() { return { viewPolicy: $('viewPolicy')?.value || 'binocular', groupPolicy: $('groupPolicy')?.value || 'all', sizePolicy: $('sizePolicy')?.value || 'relative', optimizer: $('optimizerPolicy')?.value || 'annealing', seed: numericInput('seedInput', 17, 0, 2147483647), iterations: numericInput('iterationsInput', 180, 20, 2000), preferenceRerank: true, ...state.optimizerStrategyOverrides }; }
function setExperimentStatus(text, kind = '') { const element = $('experimentStatus'); if (!element) return; element.textContent = text; element.className = `tool-status${kind ? ` ${kind}` : ''}`; }
async function generateLabels(type = $('generatorSelect').value, strict = false) { if (type !== 'custom') $('generatorSelect').value = type; if (!state.sample && !state.rawObjText) return; const requestSampleKey = sampleKey(); const requestId = state.sampleRequestId; setExperimentStatus('本地管线正在计算标签布局…'); try { const response = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ generator: type === 'bbox' ? 'bbox' : 'annotation', sample: state.sample ? { category: state.sample.category, sample_id: state.sample.id } : null, source_obj_text: state.rawObjText || undefined, source_name: state.sourceName, annotation: state.annotation, display_model: state.layoutEvaluationOverride ? undefined : 'v10_historical_rerank', strategy: currentOptimizerOptions(), preference_evaluation: state.preferenceEvaluationOverride, mdpo_evaluation: state.layoutEvaluationOverride, model: { encoder: $('modelEncoder')?.value || 'fnn_graph_transformer_moe_v7' } }) }); const result = await response.json(); if (!response.ok) throw new Error(result.error || `本地生成失败 (${response.status})`); if (requestId !== state.sampleRequestId || requestSampleKey !== sampleKey()) return result; state.objText = result.clean_obj; state.modelInfo = result.model || null; state.labels = result.labels || []; state.viewLabels = null; state.llmScores = null; state.labelContract = result.label_contract || null; state.layoutModelInfo = result.layout_model || null; state.preferenceModelInfo = result.preference_model || null; state.validationGenerationReceipt = result.validation_generation_receipt || null; state.manualLabels = result.manual_reference?.labels || []; state.metrics = result.metrics || null; state.timing = result.timing || null; state.comparison = result.manual_comparison || null; state.model = parseOBJ(state.objText); applyPartColors(state.labels); state.viewers.before.setContent(state.model, [], state.model.bounds); state.viewers.after.setContent(state.model, state.labels, getCombinedFrame(state.model, state.labels)); updateFiveViews(); buildLabelList(); state.combinedObj = exportObjWithLabels(state.objText, state.labels); $('downloadBtn').disabled = false; $('downloadJsonBtn').disabled = false; $('labelCount').textContent = state.labels.length; $('beforeBadge').textContent = '干净输入'; $('beforeBadge').classList.add('ready'); $('beforeFile').textContent = `${state.sourceName} · clean`; $('objStat').textContent = formatBytes(new Blob([state.objText]).size); $('downloadBtn').disabled = false; $('downloadJsonBtn').disabled = false; $('afterBadge').textContent = `${state.labels.length} 个标签`; $('afterFile').textContent = `${state.sourceName.replace(/\.obj$/i, '')}_labeled.obj`; updateMetrics(state.metrics, state.timing, state.comparison); setExperimentStatus(`本地生成完成：${state.labels.length} 个标签 · ${state.timing?.total_ms ?? '—'} ms`, 'ready'); if (state.labelContract?.fixed) { $('groupPolicy').value = 'all'; $('groupPolicy').disabled = true; } else $('groupPolicy').disabled = false; setAction(`已完成本地标签生成 ${state.sourceName}`); return result; } catch (error) { setExperimentStatus('本地生成失败，请检查输入或服务状态', 'warn'); showToast(error.message || '标签生成失败', 'error'); setAction('生成失败'); if (strict) throw error; } }
function buildLabelList() { $('labelList').innerHTML = state.labels.length ? state.labels.map((label, index) => `<div class="label-row"><span class="label-index" style="--label-color:rgb(${Math.round((label.color?.[0] || 0.2) * 255)},${Math.round((label.color?.[1] || 0.5) * 255)},${Math.round((label.color?.[2] || 0.7) * 255)})">${index + 1}</span><span class="label-name" title="${label.text}">${label.text}</span><span class="label-coord">${label.sourceObjs?.length ? `${label.sourceObjs.length} obj` : 'generated'}</span></div>`).join('') : '<div class="empty-list">当前生成器没有输出标签</div>'; }
function exportObjWithLabels(original, labels) { const vertexCount = (original.match(/^v\s+/gm) || []).length; const out = [original.trimEnd(), '', '# --- Generated 3D labels ---']; let nextVertex = vertexCount + 1; labels.forEach((label, index) => { const [cx, cy, cz] = label.center; const [sx, sy, sz] = label.boxSize; const x = sx / 2, y = sy / 2, z = sz / 2; const verts = [[cx-x,cy-y,cz-z],[cx+x,cy-y,cz-z],[cx+x,cy+y,cz-z],[cx-x,cy+y,cz-z],[cx-x,cy-y,cz+z],[cx+x,cy-y,cz+z],[cx+x,cy+y,cz+z],[cx-x,cy+y,cz+z]]; out.push(`o label_${index + 1}_${label.text.replace(/\s+/g, '_')}`); verts.forEach((v) => out.push(`v ${v[0]} ${v[1]} ${v[2]}`)); [[0,1,2,0,2,3],[4,7,6,4,6,5],[0,4,5,0,5,1],[1,5,6,1,6,2],[2,6,7,2,7,3],[4,0,3,4,3,7]].forEach((face) => out.push(`f ${face.map((v) => nextVertex + v).join(' ')}`)); nextVertex += 8; const line = [label.anchor, ...(label.bendPoints || []), label.center]; const indices = []; line.forEach((point) => { out.push(`v ${point[0]} ${point[1]} ${point[2]}`); indices.push(nextVertex); nextVertex += 1; }); out.push(`l ${indices.join(' ')}`); }); return `${out.join('\n')}\n`; }
function exportLabelsJson() { return JSON.stringify({ version: 'generated_label_layout_v1', generated_at: new Date().toISOString(), sample: state.sample ? { category: state.sample.category, sample_id: state.sample.id } : null, source_obj: state.sourceName, input: { clean: true, cleaning: 'remove_existing_label_and_leader_groups_and_materials' }, model: state.modelInfo, strategy: { generator: $('generatorSelect')?.value, ...currentOptimizerOptions() }, metrics: state.metrics, timing: state.timing, manual_comparison: state.comparison, labels: state.labels }, null, 2) + '\n'; }
function downloadJson() { const blob = new Blob([exportLabelsJson()], { type: 'application/json;charset=utf-8' }); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${state.sourceName.replace(/\.obj$/i, '')}_labels.json`; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 500); setAction(`已下载 ${link.download}`); }
function downloadObj() { const blob = new Blob([state.combinedObj], { type: 'text/plain;charset=utf-8' }); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${state.sourceName.replace(/\.obj$/i, '')}_labeled.obj`; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 500); setAction(`已下载 ${link.download}`); }
async function handleObjFile(file) { state.sample = null; state.sampleRequestId += 1; state.metrics = null; renderLiveSampleQuality(null); state.preferenceCandidates = { A: null, B: null }; state.llmScores = null; updateCandidateStatus(); $('preferenceStatus').textContent = '本地 OBJ 已载入，请分别保存候选 A/B'; state.rawObjText = await file.text(); state.objText = ''; state.sourceName = file.name; state.annotation = null; $('pageTitle').textContent = file.name.replace(/\.obj$/i, ''); await generateLabels('bbox'); }
async function handleAnnotationFile(file) { state.annotation = JSON.parse(await file.text()); if (!state.sample && !state.rawObjText) { showToast('请先载入 OBJ，再载入标注 JSON', 'error'); return; } await generateLabels('annotation'); setAction(`已载入标注 ${file.name}`); }

function experimentPayload(extra = {}) {
  const splitSample = state.splitManifest?.samples?.find((item) => item.category === state.sample?.category && item.sample_id === state.sample?.id);
  return { sample: state.sample ? { category: state.sample.category, sample_id: state.sample.id } : null, split: splitSample?.split || null, model: { encoder: $('modelEncoder')?.value || 'graph_transformer_moe_v6' }, strategy: { view_policy: $('viewPolicy')?.value, group_policy: $('groupPolicy')?.value, size_policy: $('sizePolicy')?.value, optimizer: $('optimizerPolicy')?.value, seed: numericInput('seedInput', 17, 0, 2147483647), iterations: numericInput('iterationsInput', 180, 20, 2000), mdpo_candidate: state.optimizerStrategyOverrides.mdpoCandidate === true, mdpo_perturbation: Number(state.optimizerStrategyOverrides.mdpoPerturbation || 0), preference_rerank: state.optimizerStrategyOverrides.preferenceRerank !== false }, metrics: state.metrics, timing: state.timing, manual_comparison: state.comparison, label_contract: state.labelContract || null, layout_model: state.layoutModelInfo || null, preference_model: state.preferenceModelInfo || null, validation_generation_receipt: state.validationGenerationReceipt || null, llm_scores: state.llmScores || null, labels: state.labels, human: { readability: Number($('scoreReadability')?.value || 0), coverage: Number($('scoreCoverage')?.value || 0), occlusion: Number($('scoreOcclusion')?.value || 0), balance: Number($('scoreBalance')?.value || 0), comment: $('scoreComment')?.value || '' }, ...extra };
}
async function recordEvaluation() { if (!state.model || !state.labels.length) { showToast('请先生成标签，再记录评价', 'error'); return; } try { const response = await fetch('/api/evaluations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(experimentPayload({ type: 'human' })) }); if (!response.ok) throw new Error((await response.json()).error || '保存失败'); $('scoreStatus').textContent = '人工评价已保存'; showToast('人工评价已保存', 'success'); } catch (error) { $('scoreStatus').textContent = error.message; showToast(error.message, 'error'); } }
function updateCandidateStatus() { for (const slot of ['A', 'B']) { const button = $(`saveCandidate${slot}Btn`); if (!button) continue; const saved = Boolean(state.preferenceCandidates[slot]); button.classList.toggle('active', saved); button.textContent = saved ? `候选 ${slot} 已保存 · 覆盖` : `保存当前为候选 ${slot}`; } }
function savePreferenceCandidate(slot) { if (!state.model || !state.labels.length) { showToast('请先生成标签，再保存候选', 'error'); return; } state.preferenceCandidates[slot] = JSON.parse(JSON.stringify(experimentPayload({ type: 'preference_candidate', candidate: slot }))); updateCandidateStatus(); $('preferenceStatus').textContent = `候选 ${slot} 已保存：${state.preferenceCandidates[slot].labels.length} 个标签；切换策略后可覆盖它`; showToast(`候选 ${slot} 已保存`, 'success'); }
function resetExperimentInputs() { $('viewPolicy').value = 'binocular'; $('groupPolicy').value = 'all'; $('sizePolicy').value = 'relative'; $('optimizerPolicy').value = 'annealing'; $('modelEncoder').value = 'graph_transformer_moe_v6'; $('seedInput').value = '17'; $('iterationsInput').value = '180'; $('scoreReadability').value = '3'; $('scoreCoverage').value = '3'; $('scoreOcclusion').value = '3'; $('scoreBalance').value = '3'; $('scoreComment').value = ''; state.preferenceCandidates = { A: null, B: null }; state.llmScores = null; updateCandidateStatus(); $('preferenceStatus').textContent = '已清空候选 A/B'; setExperimentStatus('已重置为推荐配置'); if (state.model) generateLabels(); }
async function applyPreset(name) { const presets = { balanced: ['binocular', 'all', 'relative', 'annealing'], rules: ['binocular', 'all', 'relative', 'rules'], 'all-labels': ['binocular', 'all', 'relative', 'annealing'], 'fixed-size': ['binocular', 'all', 'fixed', 'annealing'] }; const preset = presets[name]; if (!preset) return; [$('viewPolicy').value, $('groupPolicy').value, $('sizePolicy').value, $('optimizerPolicy').value] = preset; document.querySelectorAll('[data-preset]').forEach((button) => button.classList.toggle('active', button.dataset.preset === name)); setExperimentStatus(`已应用预设：${name === 'balanced' ? '五视角 + 固定人工标签' : name === 'rules' ? '规则式对照' : name === 'all-labels' ? '全部原始标签消融' : '固定尺寸消融'}`); if (state.model) await generateLabels(); }
async function recordPreference() { const chosenSlot = $('preferredCandidate').value; const rejectedSlot = chosenSlot === 'A' ? 'B' : 'A'; const chosen = state.preferenceCandidates[chosenSlot]; const rejected = state.preferenceCandidates[rejectedSlot]; if (!chosen || !rejected) { showToast('请先分别保存候选 A 和 B', 'error'); return; } try { const body = { type: 'human_pairwise_preference', sample: chosen.sample, split: chosen.split, prompt: { strategy_comparison: true }, chosen, rejected, human_scores: chosen.human, comment: $('scoreComment')?.value || '' }; const response = await fetch('/api/preferences', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); if (!response.ok) throw new Error((await response.json()).error || '偏好保存失败'); $('preferenceStatus').textContent = `已记录：${chosenSlot} > ${rejectedSlot}`; showToast('两两偏好已保存', 'success'); } catch (error) { $('preferenceStatus').textContent = error.message; showToast(error.message, 'error'); } }
function captureViewerImage(viewer) { try { return viewer?.toDataURL('image/jpeg', 0.86) || null; } catch { return null; } }
function setScoringReadiness(config) {
  const button = $('runLLMPreferenceBtn');
  const status = $('llmRoundStatus');
  if (!button || !status) return;
  const ready = Boolean(config?.configured && config?.tested);
  button.disabled = !ready;
  button.title = ready ? '本地 Qwen 已通过六图/JSON 自检' : '请启动 Ollama 并通过“测试六图与 JSON”';
  if (!config?.configured) {
    status.textContent = '等待本地模型：请启动 Ollama 并确认 qwen3-vl:4b-instruct 已安装。';
    status.className = 'tool-status warn';
  } else if (!config?.tested) {
    status.textContent = '连接已保存但尚未通过六图/JSON 自检；通过后才会启用真实 LLM 偏好学习。';
    status.className = 'tool-status warn';
  } else {
    const samples = numericInput('llmSamplesInput', 1, 1, 33);
    const candidates = numericInput('llmCandidatesInput', 4, 2, 8);
    const rounds = numericInput('llmRoundsInput', 1, 1, 10);
    status.textContent = `连接已就绪：预计 ${samples * candidates * rounds} 次本地六图评分（${samples} 样本 × ${candidates} 候选 × ${rounds} 轮）。`;
    status.className = 'tool-status ready';
  }
}
async function loadScoringConfig() {
  const status = $('scoringConfigStatus');
  try {
    const response = await fetch('/api/scoring-config');
    if (!response.ok) throw new Error('连接设置读取失败');
    const config = await response.json();
    $('scoringApiUrl').value = config.baseUrl || 'http://127.0.0.1:11434';
    $('scoringModel').value = config.model || 'qwen3-vl:4b-instruct';
    const protocol = 'Ollama /api/chat';
    status.textContent = config.configured
      ? `本地模型：${config.model} · ${protocol} · ${config.tested ? '六图/JSON 已测试' : '尚未测试视觉能力'}`
      : '本地视觉模型尚未配置。';
    status.className = `tool-status ${config.configured && config.tested ? 'ready' : 'warn'}`;
    $('llmScoreBtn').textContent = config.model ? `请求 ${config.model} 评分` : '请求本地 Qwen 评分';
    setScoringReadiness(config);
  } catch (error) { status.textContent = error.message; status.className = 'tool-status warn'; }
}
async function saveScoringConfig() {
  const button = $('saveScoringConfigBtn');
  button.disabled = true;
  try {
    const response = await fetch('/api/scoring-config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ baseUrl: $('scoringApiUrl').value.trim(), model: $('scoringModel').value.trim() }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '保存失败');
    await loadScoringConfig();
    showToast('本地 Ollama 配置已保存', 'success');
  } catch (error) { $('scoringConfigStatus').textContent = error.message; $('scoringConfigStatus').className = 'tool-status warn'; showToast(error.message, 'error'); }
  finally { button.disabled = false; }
}
async function testScoringConfig() {
  const button = $('testScoringConfigBtn');
  const saveButton = $('saveScoringConfigBtn');
  const status = $('scoringConfigStatus');
  button.disabled = true;
  saveButton.disabled = true;
  status.textContent = '正在测试本地 Qwen：六张图片输入与十四维 JSON 返回…';
  status.className = 'tool-status';
  try {
    const configResponse = await fetch('/api/scoring-config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ baseUrl: $('scoringApiUrl').value.trim(), model: $('scoringModel').value.trim() }) });
    const config = await configResponse.json();
    if (!configResponse.ok) throw new Error(config.error || '连接设置保存失败');
    if (!config.configured) throw new Error('尚未配置本地视觉模型');
    const response = await fetch('/api/scoring-test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `连接测试失败 (${response.status})`);
    status.textContent = `本地测试通过：${result.model} · Ollama /api/chat · 已接受 6 张图片并返回完整十四维 JSON；5 个美学维度用于偏好，9 个安全维度用于诊断与门控。`;
    status.className = 'tool-status ready';
    $('llmScoreBtn').textContent = `请求 ${result.model} 评分`;
    setScoringReadiness({ configured: true, tested: true });
    showToast('视觉评分连接测试通过', 'success');
  } catch (error) {
    status.textContent = `连接测试失败：${error.message}`;
    status.className = 'tool-status warn';
    try { setScoringReadiness(await fetch('/api/scoring-config').then((response) => response.json())); } catch {}
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
    saveButton.disabled = false;
  }
}
async function requestLLMScore() {
  if (!state.model || !state.labels.length) { showToast('请先生成标签，再请求评分', 'error'); return; }
  const visuals = captureFiveViewVisuals();
  if (Object.values(visuals).some((image) => !image)) { showToast('五视角尚未全部渲染，请生成布局后重试', 'error'); return; }
  const button = $('llmScoreBtn'); button.disabled = true; $('scoreStatus').textContent = '正在请求视觉评分…';
  try {
    const visualSummary = { views: ['main', 'right', 'left', 'up', 'down'], label_count: state.labels.length, source: 'webgl_five_view_with_text', images_attached: true };
    const requestPayload = experimentPayload({ type: 'local_qwen_vision_score', visual_summary: visualSummary, visuals });
    const response = await fetch('/api/score', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestPayload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `评分失败 (${response.status})`);
    const scores = data.result?.scores || {};
    state.llmScores = scores;
    $('scoreStatus').textContent = `Qwen 评分：${scores.overall ?? '—'} / 5；${data.result?.rationale || ''}`;
    showToast('视觉评分完成', 'success');
    const save = await fetch('/api/evaluations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(experimentPayload({ type: 'local_qwen_vision_score', llm: data, visual_summary: visualSummary })) });
    if (!save.ok) throw new Error('评分已返回，但评价日志写入失败');
  } catch (error) { $('scoreStatus').textContent = error.message; showToast(error.message, 'error'); }
  finally { button.disabled = false; }
}

function captureFiveViewVisuals() {
  const visuals = { before: captureViewerImage(state.viewers.before), ...Object.fromEntries(['main', 'right', 'left', 'up', 'down'].map((view) => [view, captureViewerImage(state.multiViewers[view])])) };
  if (state.metrics) state.metrics.rendered_text_measurements = {
    method: 'canvas_measureText_at_dataset_resolution',
    views: Object.fromEntries(['main','right','left','up','down'].map(view=>[view,structuredClone(state.multiViewers[view]?.lastCaptureTextMeasurements || [])]))
  };
  return visuals;
}

async function evaluatePreferenceTest() {
  const button = $('evalPrefTestBtn'), status = $('prefTestStatus');
  button.disabled = true; status.textContent = '正在比较 test11：固定 seed17 与偏好模型四候选重排…'; status.className = 'tool-status';
  try {
    const result = await postExperimentJson('/api/evaluate-preference-test', {});
    const objective = result.summary.objective_score;
    status.textContent = `test11 五视角能量：${objective.baseline} → ${objective.preferred}（越低越好）；${result.candidate_activated ? 'val 门控通过，奖励模型已激活' : '候选仅供诊断，val 门控未通过，奖励模型未激活'}；明细：${result.report_file}`;
    status.className = 'tool-status ready';
    return result;
  } catch (error) { status.textContent = error.message; status.className = 'tool-status warn'; showToast(error.message, 'error'); return null; }
  finally { button.disabled = false; }
}

function preferenceRank(candidate) {
  return preferenceComposite(candidate.llm_scores || {});
}

function safetyCompatiblePair(left, right) {
  if (!assessSafetyEligibility(left.metrics, left.metrics).eligible || !assessSafetyEligibility(right.metrics, right.metrics).eligible) return false;
  const safer = compareSafetyReference(left.metrics, right.metrics) <= 0 ? left : right;
  const other = safer === left ? right : left;
  return assessSafetyEligibility(other.metrics, safer.metrics).eligible;
}

function pairwiseSafeCandidatePool(drafts, limit) {
  const ordered = [...drafts].sort((left, right) => compareSafetyReference(left.metrics, right.metrics));
  const pool = [];
  for (const candidate of ordered) {
    if (!assessSafetyEligibility(candidate.metrics, candidate.metrics).eligible) continue;
    if (pool.every((selected) => safetyCompatiblePair(candidate, selected))) pool.push(candidate);
    if (pool.length >= limit) break;
  }
  return pool;
}

async function postExperimentJson(endpoint, body) {
  const maxAttempts = endpoint === '/api/score' ? 30 : 1;
  let lastError;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const result = await response.json();
      if (response.ok) return result;
      const message = result.error || `${endpoint} failed (${response.status})`;
      const retryable = [429, 500, 502, 503, 504].includes(response.status) || /busy|try again|overload|timeout|temporar/i.test(message);
      if (!retryable || attempt === maxAttempts) throw Object.assign(new Error(message), { code: result.code || null, httpStatus: response.status });
      lastError = new Error(message);
    } catch (error) {
      const retryable = /busy|try again|overload|timeout|temporar|fetch|network/i.test(error.message || '');
      if (!retryable || attempt === maxAttempts) throw error;
      lastError = error;
    }
    const delayMs = Math.min(60000, 4000 * 2 ** (attempt - 1));
    const status = $('llmRoundStatus');
    if (status) status.textContent = `本地 Ollama 暂时繁忙：${lastError.message}；${Math.round(delayMs / 1000)} 秒后自动重试 ${attempt + 1}/${maxAttempts}…`;
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  throw lastError || new Error(`${endpoint} failed`);
}

async function finalizeLLMPreferenceRun({ runId, savedPairs, roundCount, candidateCount, samples, testIterations = 180, testLimit = 0 }) {
  if (!savedPairs) return { testNote: '', reportNote: '', testResult: null };
  const status = $('llmRoundStatus');
  status.textContent = '偏好数据收集与候选奖励训练完成，正在读取 test11 最终确认…';
  const testResult = await postExperimentJson('/api/evaluate-preference-test', { force: true, runId, iterations: testIterations, limit: testLimit });
  const testNote = `；val选择第${testResult.selected_round}轮，test11 五视角能量 ${testResult.summary.objective_score.baseline} → ${testResult.summary.objective_score.preferred}；奖励模型${testResult.candidate_activated ? '已通过 val 并激活' : '未通过 val，仅诊断未激活'}`;
  $('prefTestStatus').textContent = `完整报告：${testResult.report_file}`;
  $('prefTestStatus').className = 'tool-status ready';
  const runReport = await postExperimentJson('/api/llm-run-report', { runId, sample: samples, rounds: roundCount, candidates: candidateCount });
  await loadQualityComparison();
  return { testNote, reportNote: `；运行报告 ${runReport.report_file}`, testResult };
}

async function trainLLMRewardForRun(runId, baseSeed, gateOptions = {}) {
  const trained = await postExperimentJson('/api/train', { epochs: numericInput('trainEpochsInput', 80, 1, 1000), seed: baseSeed, source: 'llm', runId, skipTest: true, ...gateOptions });
  if (trained.training?.source_filter !== 'llm' || trained.training?.run_id_filter !== runId || trained.training?.source_counts?.human) throw new Error('偏好训练来源审计失败：自动 LLM 轮不得混入其他运行或人工偏好');
  return trained;
}

async function saveLLMPreferenceCheckpoint(runId, round) {
  return postExperimentJson('/api/llm-checkpoint', { runId, round });
}

async function runLLMPreferenceRounds(runOptions = {}) {
  const status = $('llmRoundStatus'), button = $('runLLMPreferenceBtn');
  const split = experimentPayload().split;
  if (split !== 'train') { status.textContent = '偏好训练只允许 train 样本；val/test 仅可评分和评估。请先选择 train 样本。'; status.className = 'tool-status warn'; return { ok: false, error: status.textContent }; }
  const config = await fetch('/api/scoring-config').then((response) => response.json());
  if (!config.configured) { status.textContent = '请先启动本机 Ollama 并配置 qwen3-vl:4b-instruct。'; status.className = 'tool-status warn'; return { ok: false, error: status.textContent }; }
  if (!config.tested) { status.textContent = '连接尚未通过六图与 JSON 自检；请先点击“测试六图与 JSON”。'; status.className = 'tool-status warn'; return { ok: false, error: status.textContent }; }
  const candidateCount = numericInput('llmCandidatesInput', 4, 2, 8), roundCount = Number.isInteger(runOptions.roundCountOverride) ? runOptions.roundCountOverride : numericInput('llmRoundsInput', 1, 1, 10);
  const baseSeed = numericInput('seedInput', 17, 0, 2147483647);
  const runId = runOptions.runId || crypto.randomUUID();
  const finalize = runOptions.finalize !== false;
  const trainAfterRounds = runOptions.trainAfterRounds !== false;
  const sampleOrdinal = Number(runOptions.sampleOrdinal || 1), sampleTotal = Number(runOptions.sampleTotal || 1);
  const samplePrefix = sampleTotal > 1 ? `样本 ${sampleOrdinal}/${sampleTotal} · ` : '';
  const lockedControls = Array.from(document.querySelectorAll('button, select, input')).filter((control) => !control.disabled);
  lockedControls.forEach((control) => { control.disabled = true; });
  status.className = 'tool-status';
  let savedPairs = 0;
  let rewardActivatedInThisRun = Boolean(runOptions.rewardActivatedInThisRun);
  const roundOffset = Math.max(0, Number(runOptions.roundOffset || 0));
  const checkpoints = [];
  let outcome = null;
  try {
    for (let round = 0; round < roundCount; round += 1) {
      const actualRound = roundOffset + round + 1;
      const drafts = [];
      const maxLocalAttempts = candidateCount * 10;
      let safeDrafts = [];
      for (let index = 0; index < maxLocalAttempts; index += 1) {
        $('seedInput').value = String(baseSeed + (actualRound - 1) * maxLocalAttempts + index);
        status.textContent = `${samplePrefix}第 ${actualRound} 轮：本地生成候选 ${index + 1}/${maxLocalAttempts}，正在筛选互相可比的安全布局（尚未调用外部模型）…`;
        state.optimizerStrategyOverrides = { preferenceRerank: rewardActivatedInThisRun };
        try { await generateLabels(undefined, true); }
        finally { state.optimizerStrategyOverrides = {}; }
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        drafts.push(JSON.parse(JSON.stringify(experimentPayload({ type: 'llm_preference_candidate', run_id: runId, visuals: captureFiveViewVisuals() }))));
        safeDrafts = pairwiseSafeCandidatePool(drafts, candidateCount);
        if (safeDrafts.length >= candidateCount) break;
      }
      if (safeDrafts.length < 2) throw new Error(`第 ${actualRound} 轮本地生成 ${drafts.length} 个布局后仍不足 2 个互相可比的安全候选；未调用本地 Qwen 评分，也未制造偏好`);
      const candidates = [];
      for (let index = 0; index < safeDrafts.length; index += 1) {
        status.textContent = `${samplePrefix}第 ${actualRound} 轮：安全候选 ${index + 1}/${safeDrafts.length}，正在提交 6 张图片进行美学评分…`;
        const data = await postExperimentJson('/api/score', safeDrafts[index]);
        state.llmScores = data.result.scores;
        if (!data.score_receipt) throw new Error('服务端未签发评分凭据；本次结果不能用于 LLM 偏好训练');
        const { visuals, ...draft } = safeDrafts[index];
        const candidate = JSON.parse(JSON.stringify({ ...draft, llm_scores: data.result.scores, llm_score_receipt: data.score_receipt, llm_rationale: data.result.rationale }));
        candidates.push(candidate);
        if (!runOptions.diagnosticOnly) await postExperimentJson('/api/evaluations', { ...candidate, llm: data, round: actualRound });
      }
      candidates.sort((left, right) => preferenceRank(right) - preferenceRank(left));
      if (runOptions.diagnosticOnly) {
        const diagnostic = candidates.map((candidate) => ({
          score: Number(preferenceRank(candidate).toFixed(4)),
          aesthetic_scores: Object.fromEntries(['manual_style_similarity', 'spatial_balance', 'visual_hierarchy', 'composition_harmony', 'overall'].map((name) => [name, candidate.llm_scores[name]])),
          sample: candidate.sample,
          seed: candidate.strategy?.seed,
          objective: candidate.metrics?.objective_score,
          leader_crossings: candidate.metrics?.leader_crossings,
          worst_view_leader_crossing_count: candidate.metrics?.worst_view_leader_crossing_count
        }));
        outcome = { ok: true, diagnostic_only: true, scored_candidates: diagnostic, non_tied: diagnostic.some((row) => diagnostic[0].score - row.score >= 0.01) };
        break;
      }
      const chosen = candidates[0];
      let roundPairs = 0;
      for (const rejected of candidates.slice(1)) {
        if (!safetyCompatiblePair(chosen, rejected)) continue;
        if (preferenceRank(chosen) - preferenceRank(rejected) < 0.01) continue;
        await postExperimentJson('/api/preferences', { type: 'llm_pairwise_preference', run_id: runId, sample: chosen.sample, split: 'train', round: actualRound, chosen, rejected, ranking_rule: 'aesthetic-only: 0.30 overall + 0.25 composition harmony + 0.20 visual hierarchy + 0.15 spatial balance + 0.10 manual style; deterministic safety gate is separate' });
        roundPairs += 1; savedPairs += 1;
      }
      if (!roundPairs) { status.textContent = `第 ${actualRound} 轮没有同时通过安全门控且美学分不同的候选对；仅保存评分，不伪造偏好。`; break; }
      if (trainAfterRounds) {
        status.textContent = `${samplePrefix}第 ${actualRound} 轮：${roundPairs} 条有效偏好对，正在累计训练候选奖励模型并执行 val 门控…`;
        const checkpoint = await trainLLMRewardForRun(runId, baseSeed);
        rewardActivatedInThisRun ||= Boolean(checkpoint.activated);
        checkpoints.push((await saveLLMPreferenceCheckpoint(runId, actualRound)).checkpoint);
        await generateLabels(undefined, true);
      } else status.textContent = `${samplePrefix}第 ${actualRound} 轮：已保存 ${roundPairs} 条有效偏好对，等待本轮跨样本累计训练。`;
    }
    if (runOptions.diagnosticOnly) return outcome;
    let testNote = '', reportNote = '';
    if (savedPairs && finalize) {
      const finalized = await finalizeLLMPreferenceRun({ runId, savedPairs, roundCount, candidateCount, samples: [experimentPayload().sample] });
      testNote = finalized.testNote; reportNote = finalized.reportNote;
    }
    status.textContent = finalize ? `偏好流程结束：run_id=${runId}，保存 ${savedPairs} 条非平局偏好${testNote}${reportNote}；奖励模型仅重排，基础图 Transformer-MoE 权重未改变。` : `${samplePrefix}数据收集结束：本样本保存 ${savedPairs} 条偏好，run_id=${runId}。`;
    status.className = 'tool-status ready';
    outcome = { ok: true, runId, savedPairs, rewardActivatedInThisRun, checkpoints, status: status.textContent };
  } catch (error) { status.textContent = error.message; status.className = 'tool-status warn'; showToast(error.message, 'error'); outcome = { ok: false, runId, savedPairs, error: error.message }; }
  finally { lockedControls.forEach((control) => { control.disabled = false; }); if (state.labelContract?.fixed) $('groupPolicy').disabled = true; button.disabled = false; }
  return outcome;
}

async function waitForWorkspaceReady(timeoutMs = 120000) {
  const started = performance.now();
  while ((!state.catalog || !state.splitManifest) && performance.now() - started < timeoutMs) await new Promise((resolve) => setTimeout(resolve, 100));
  if (!state.catalog || !state.splitManifest) throw new Error('数据目录或 split manifest 未就绪');
}

async function selectAutomationSample(category, sampleId, requiredSplit = 'train') {
  await waitForWorkspaceReady();
  const categoryEntry = state.catalog.categories.find((item) => item.name === category);
  const sample = categoryEntry?.samples?.find((item) => String(item.id) === String(sampleId));
  const split = state.splitManifest.samples.find((item) => item.category === category && String(item.sample_id) === String(sampleId))?.split;
  if (!sample) throw new Error(`找不到样本 ${category}/${sampleId}`);
  if (split !== requiredSplit) throw new Error(`自动流程要求 ${requiredSplit} 样本，${category}/${sampleId} 属于 ${split || 'unknown'}`);
  $('categorySelect').value = category;
  $('sampleSelect').innerHTML = categoryEntry.samples.map((item) => `<option value="${item.id}">${item.id}</option>`).join('');
  $('sampleSelect').value = String(sample.id);
  await loadSample(sample);
  return { category, sampleId: String(sample.id), split, labelCount: state.labels.length };
}

function stratifiedTrainSamples(count) {
  const groups = new Map();
  for (const item of state.splitManifest.samples.filter((sample) => sample.split === 'train')) {
    if (!groups.has(item.category)) groups.set(item.category, []);
    groups.get(item.category).push(item);
  }
  const output = [];
  for (let round = 0; output.length < count; round += 1) {
    let added = false;
    for (const rows of groups.values()) {
      if (rows[round]) { output.push({ category: rows[round].category, sampleId: String(rows[round].sample_id) }); added = true; if (output.length >= count) break; }
    }
    if (!added) break;
  }
  return output;
}

function stratifiedValidationSamples(count) {
  const groups = new Map();
  for (const item of state.splitManifest.samples.filter((sample) => sample.split === 'val')) {
    if (!groups.has(item.category)) groups.set(item.category, []);
    groups.get(item.category).push(item);
  }
  const output = [];
  for (let round = 0; output.length < count; round += 1) {
    let added = false;
    for (const rows of groups.values()) {
      if (rows[round]) { output.push({ category: rows[round].category, sampleId: String(rows[round].sample_id) }); added = true; if (output.length >= count) break; }
    }
    if (!added) break;
  }
  return output;
}

async function evaluateVisualValidationCheckpoint({ runId, round, samples, seed = 17 }) {
  const status = $('llmRoundStatus');
  const rows = [];
  state.preferenceEvaluationOverride = { run_id: runId, round };
  state.optimizerStrategyOverrides = { preferenceRerank: true };
  try {
    for (let index = 0; index < samples.length; index += 1) {
      const sample = samples[index];
      $('seedInput').value = String(Math.max(0, Number(seed) + index * 4));
      status.textContent = `真实视觉验证第 ${round} 轮：val 六图评分 ${index + 1}/${samples.length}（不会写入 train 偏好）…`;
      await selectAutomationSample(sample.category, sample.sampleId, 'val');
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      if (!state.validationGenerationReceipt) throw new Error(`第 ${round} 轮 ${sample.category}/${sample.sampleId} 缺少冻结检查点生成凭据`);
      const scored = await postExperimentJson('/api/score', experimentPayload({ type: 'llm_validation_checkpoint', run_id: runId, round, visuals: captureFiveViewVisuals() }));
      if (scored.score_receipt) throw new Error('val 视觉评分错误地获得了 train 偏好凭据，流程已中止');
      if (!scored.visual_validation_record) throw new Error('服务端未保存 val 视觉评分来源记录');
      rows.push({ sample, scores: scored.result.scores, record: scored.visual_validation_record });
    }
    const finalized = await postExperimentJson('/api/llm-visual-validation/finalize', { runId, round, samples });
    return { ...finalized, rows };
  } finally {
    state.preferenceEvaluationOverride = null;
    state.optimizerStrategyOverrides = {};
    state.validationGenerationReceipt = null;
  }
}

async function runLLMPreferenceFromUI() {
  await waitForWorkspaceReady();
  const sampleCount = numericInput('llmSamplesInput', 1, 1, 33);
  const candidates = numericInput('llmCandidatesInput', 4, 2, 8);
  const rounds = numericInput('llmRoundsInput', 1, 1, 10);
  const visualValSamples = numericInput('llmVisualValSamplesInput', 11, 1, 11);
  const trainRequests = sampleCount * candidates * rounds;
  const validationRequests = visualValSamples * (rounds + 1);
  $('llmRoundStatus').textContent = `准备运行：train 候选评分 ${trainRequests} 次 + 独立 val 六图评分 ${validationRequests} 次（0 轮基线及每轮检查点）= 共 ${trainRequests + validationRequests} 次本地请求。`;
  if (!window.confirm(`本次将调用本地 Qwen3-VL ${trainRequests + validationRequests} 次，每次上传 6 张渲染图片（其中 ${validationRequests} 次为独立 val 验证），将在本机运行，耗时取决于显卡/CPU。确认开始吗？`)) return { ok: false, cancelled: true };
  return window.__labelStudioAutomation.runLLMPreferenceDataset({ samples: stratifiedTrainSamples(sampleCount), candidates, rounds, visualValSamples, seed: numericInput('seedInput', 17, 0, 2147483647), epochs: numericInput('trainEpochsInput', 80, 1, 1000) });
}

window.__labelStudioAutomation = {
  runMDPOTrainDataset: async ({ samples = [], seed = 17, pairCount = 8, runId = crypto.randomUUID(), continueOnInsufficientPairs = false } = {}) => {
    await waitForWorkspaceReady();
    const config = await fetch('/api/scoring-config').then((response) => response.json());
    if (!config.tested) throw new Error('MDPO requires a successfully tested local Qwen six-image scorer');
    const reference = await fetch('/api/mdpo-reference').then((response) => response.json());
    const experiment = await fetch('/api/mdpo-experiment').then((response) => response.json());
    const samplePairCounts = experiment.dataset?.sample_pair_counts || {};
    if (!reference.version?.startsWith('layout_model_v10_') || !reference.reference_model_sha256) throw new Error('Active frozen v10 reference missing');
    const requested = samples.length ? samples : state.splitManifest.samples.filter((item) => item.split === 'train').map((item) => ({ category: item.category, sampleId: item.sample_id }));
    if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw new Error('Invalid MDPO run ID');
    const rows = [], insufficientSamples = [];
    const mdpoDimensions = ['overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance', 'manual_style_similarity', 'text_clarity', 'leader_line_clarity'];
    const informativePairStats = (candidates) => {
      const covered = new Set(); let edgeCount = 0;
      for (let left = 0; left < candidates.length; left += 1) for (let right = left + 1; right < candidates.length; right += 1) {
        if (safetyCompatiblePair(candidates[left], candidates[right]) && mdpoDimensions.some((name) => Math.abs((candidates[left].llm_scores[name] - candidates[right].llm_scores[name]) / 4) > 0.025)) { covered.add(left); covered.add(right); edgeCount += 1; }
      }
      return { candidateCount: covered.size, edgeCount };
    };
    for (let sampleIndex = 0; sampleIndex < requested.length; sampleIndex += 1) {
      const item = requested[sampleIndex];
      const requestedKey = `${item.category}/${String(item.sampleId ?? item.sample_id)}`;
      try {
      if (Number(samplePairCounts[requestedKey] || 0) >= 6) { rows.push({ sample: requestedKey, skipped: true, reason: 'already_atomically_persisted_with_6_to_12_pairs' }); continue; }
      const selected = await selectAutomationSample(item.category, String(item.sampleId ?? item.sample_id), 'train');
      const historyResponse = await fetch(`/api/mdpo-train-candidate-summaries?category=${encodeURIComponent(selected.category)}&sample_id=${encodeURIComponent(selected.sampleId)}`);
      if (!historyResponse.ok) throw new Error(`Failed to read persisted MDPO candidate summaries for ${requestedKey}`);
      const history = (await historyResponse.json()).candidates || [];
      const candidates = [], layouts = new Set();
      // Keep the original full-train33 seed namespace on targeted recovery.
      // Already staged Qwen-scored seeds must never be re-scored as new data.
      const manifestIndex = stratifiedTrainSamples(33)
        .findIndex((sample) => sample.category === selected.category && String(sample.sampleId) === String(selected.sampleId));
      if (manifestIndex < 0) throw new Error('MDPO recovery sample missing from train33 manifest');
      const seedBase = Number(seed) + manifestIndex * 10000;
      const firstAttempt = Math.max(0, ...history.map((row) => Number(row.seed) - seedBase + 1)
        .filter((offset) => Number.isInteger(offset) && offset >= 0 && offset < 10000));
      const maxAttempts = 288, maxNewCandidates = 24;
      let initialStats = informativePairStats(history);
      for (let attempt = firstAttempt; attempt < maxAttempts && candidates.length < maxNewCandidates && (history.length + candidates.length < 8 || initialStats.candidateCount < 8 || initialStats.edgeCount < Math.max(6, Number(pairCount) || 8)); attempt += 1) {
        const perturbationIndex = attempt % 12;
        $('seedInput').value = String(seedBase + attempt);
        $('llmRoundStatus').textContent = `MDPO train ${sampleIndex + 1}/${requested.length} ${selected.category}/${selected.sampleId}: generating safe candidate ${candidates.length + 1}/12 until 8 enter informative pairs (attempt ${attempt + 1}/${maxAttempts})`;
        state.optimizerStrategyOverrides = { preferenceRerank: false, mdpoCandidate: true, mdpoPerturbation: perturbationIndex };
        try { await generateLabels(undefined, true); } finally { state.optimizerStrategyOverrides = {}; }
        if (state.layoutModelInfo?.version !== reference.version) throw new Error('MDPO candidate not generated by frozen active v10');
        if (!assessSafetyEligibility(state.metrics, state.metrics).eligible) continue;
        const layout = JSON.stringify(state.labels.map((label) => [label.center, label.boxSize]));
        if (layouts.has(layout)) continue;
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const candidate = JSON.parse(JSON.stringify(experimentPayload({ type: 'mdpo_train_candidate', run_id: runId, visuals: captureFiveViewVisuals(), manual_comparison: null })));
        candidate.strategy.mdpo_candidate = true;
        candidate.strategy.mdpo_perturbation = perturbationIndex;
        candidate.strategy.preference_rerank = false;
        let scored;
        try { scored = await postExperimentJson('/api/score', candidate); }
        catch (error) {
          if (['MDPO_UNSAFE_MEASURED_GEOMETRY', 'MDPO_UNSAFE_PRECHECK'].includes(error.code)) continue;
          throw error;
        }
        if (!scored.score_receipt) throw new Error('MDPO Qwen score missing verifiable receipt');
        layouts.add(layout);
        candidate.llm_scores = scored.result.scores;
        candidate.llm_score_receipt = scored.score_receipt;
        await postExperimentJson('/api/mdpo-train-sample', { sample: { category: selected.category, sample_id: selected.sampleId }, split: 'train', run_id: runId, reference_model_sha256: reference.reference_model_sha256, candidates: [candidate], stage_only: true });
        candidates.push(candidate);
        const stats = informativePairStats([...history, ...candidates]);
        initialStats = stats;
        if (history.length + candidates.length >= 8 && stats.candidateCount >= 8 && stats.edgeCount >= Math.max(6, Number(pairCount) || 8)) break;
      }
      const informative = informativePairStats([...history, ...candidates]);
      if (history.length + candidates.length < 8 || informative.candidateCount < 8 || informative.edgeCount < Math.max(6, Number(pairCount) || 8)) throw new Error(`MDPO train ${selected.category}/${selected.sampleId}: ${history.length} persisted + ${candidates.length} new safe candidates, ${informative.candidateCount}/8 participate, ${informative.edgeCount}/${Math.max(6, Number(pairCount) || 8)} informative edges; no synthetic scores or pairs were created`);
      $('llmRoundStatus').textContent = `MDPO train ${sampleIndex + 1}/${requested.length}: verifying and storing seven-dimensional Qwen preferences…`;
      const stored = await postExperimentJson('/api/mdpo-train-sample', { sample: { category: selected.category, sample_id: selected.sampleId }, split: 'train', run_id: runId, reference_model_sha256: reference.reference_model_sha256, pair_count: pairCount, candidates: candidates.length ? [candidates.at(-1)] : [], finalize_only: candidates.length === 0 });
      rows.push({ ...stored, candidate_count: candidates.length });
      } catch (error) {
        if (!continueOnInsufficientPairs || !/^MDPO train [^:]+: \d+ persisted \+ \d+ new safe candidates, \d+\/8 participate, \d+\/\d+ informative edges; no synthetic scores or pairs were created$/.test(error.message || '')) throw error;
        const failure = { sample: requestedKey, status: 'needs_distinct_safe_qwen_candidates', error: error.message };
        insufficientSamples.push(failure);
        rows.push(failure);
      }
    }
    return { ok: insufficientSamples.length === 0, partial: insufficientSamples.length > 0, insufficient_samples: insufficientSamples, runId, reference_model_sha256: reference.reference_model_sha256, rows, candidate_count: rows.reduce((sum, item) => sum + Number(item.candidate_count || 0), 0), pair_count: rows.reduce((sum, item) => sum + Number(item.pair_count || 0), 0), skipped_count: rows.filter((item) => item.skipped).length };
  },
  runMDPOVal11: async ({ candidateFile, candidateSha256, candidateId, samples = [], seed = 17 } = {}) => {
    await waitForWorkspaceReady();
    const config = await fetch('/api/scoring-config').then((response) => response.json());
    if (!config.tested) throw new Error('MDPO val11 requires a successfully tested local Qwen scorer');
    if (!candidateFile || !candidateSha256 || !candidateId) throw new Error('MDPO val11 requires frozen candidate file, SHA-256 and candidate ID');
    const requested = samples.length ? samples : stratifiedValidationSamples(11);
    if (requested.length !== 11) throw new Error('MDPO val11 requires exactly 11 validation samples');
    const rows = [];
    const groups = [
      { id: 'v10_no_rerank', role: 'baseline', preferenceRerank: false },
      { id: 'v10_historical_rerank', role: 'baseline', preferenceRerank: true },
      { id: 'mdpo_no_rerank', role: 'candidate', preferenceRerank: false },
      { id: 'mdpo_safe_rerank', role: 'candidate', preferenceRerank: true }
    ];
    const progress = await postExperimentJson('/api/mdpo-val11-progress', { candidate_id: candidateId, candidate_file: candidateFile, candidate_sha256: candidateSha256, samples: requested });
    const completed = new Map((progress.completed || []).map((item) => [item.key, item]));
    try {
      for (const group of groups) for (let index = 0; index < requested.length; index += 1) {
        const sample = requested[index];
        const sampleKey = `${sample.category}/${String(sample.sampleId ?? sample.sample_id)}`;
        const resumeKey = `${group.id}|${sampleKey}`;
        if (completed.has(resumeKey)) { rows.push({ ...completed.get(resumeKey), resumed: true }); continue; }
        state.optimizerStrategyOverrides = { preferenceRerank: group.preferenceRerank };
        state.layoutEvaluationOverride = { candidate_id: candidateId, group: group.id, role: group.role, candidate_file: group.role === 'candidate' ? candidateFile : null, candidate_sha256: group.role === 'candidate' ? candidateSha256 : null };
        $('seedInput').value = String(Number(seed) + index * 4);
        $('llmRoundStatus').textContent = `v10-MDPO val11 ${group.id}：${index + 1}/${requested.length} 六图与确定性指标…`;
        await selectAutomationSample(sample.category, String(sample.sampleId ?? sample.sample_id), 'val');
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const payload = JSON.parse(JSON.stringify(experimentPayload({ type: 'mdpo_val11_checkpoint', candidate_id: candidateId, group: group.id, role: group.role, split: 'val', visuals: captureFiveViewVisuals() })));
        const scored = await postExperimentJson('/api/score', payload);
        if (!scored.visual_validation_record) throw new Error(`MDPO val11 ${group.id} ${sample.category}/${sample.sampleId} was not persisted`);
        rows.push({ group: group.id, role: group.role, sample: sampleKey, file: scored.visual_validation_record, resumed: false });
      }
    } finally { state.layoutEvaluationOverride = null; state.optimizerStrategyOverrides = {}; }
    const report = await postExperimentJson('/api/mdpo-val11-finalize', { candidate_id: candidateId, samples: requested });
    return { ok: true, candidate_id: candidateId, rows, group_count: groups.length, gate: report.gate, report_file: 'experiments/mdpo/val11_gate_report.json', four_group_report: 'experiments/mdpo/four_group_ablation.json' };
  },
  runMDPOTest11: async () => {
    await waitForWorkspaceReady();
    const config = await fetch('/api/scoring-config').then((response) => response.json());
    if (!config.tested) throw new Error('MDPO test11 requires a successfully tested local Qwen scorer');
    const progress = await postExperimentJson('/api/mdpo-test11-progress', {});
    const requested = progress.cohort.map((sample) => { const separator = sample.indexOf('/'); return { category: sample.slice(0, separator), sampleId: sample.slice(separator + 1) }; });
    if (requested.length !== 11) throw new Error('Locked MDPO test11 requires exactly 11 test samples');
    const completed = new Map((progress.completed || []).map((item) => [item.key, item]));
    const groups = [
      { id: 'v10_no_rerank', role: 'baseline', preferenceRerank: false },
      { id: 'v10_historical_rerank', role: 'baseline', preferenceRerank: true },
      { id: 'mdpo_no_rerank', role: 'candidate', preferenceRerank: false },
      { id: 'mdpo_safe_rerank', role: 'candidate', preferenceRerank: true }
    ];
    const rows = [];
    try {
      for (const group of groups) for (let index = 0; index < requested.length; index += 1) {
        const sample = requested[index], sampleKey = `${sample.category}/${sample.sampleId}`, resumeKey = `${group.id}|${sampleKey}`;
        if (completed.has(resumeKey)) { rows.push({ ...completed.get(resumeKey), resumed: true }); continue; }
        state.optimizerStrategyOverrides = { preferenceRerank: group.preferenceRerank };
        state.layoutEvaluationOverride = { phase: 'test11', test_run_id: progress.test_run_id, lock_sha256: progress.lock_sha256, group: group.id, role: group.role };
        $('seedInput').value = String(Number(progress.seed) + index * 4);
        $('llmRoundStatus').textContent = `锁定 v10-MDPO test11 ${group.id}：${index + 1}/${requested.length} 六图与确定性指标…`;
        await selectAutomationSample(sample.category, sample.sampleId, 'test');
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        if (!state.validationGenerationReceipt) throw new Error(`MDPO test11 ${group.id} ${sampleKey} missing locked generation receipt`);
        const payload = JSON.parse(JSON.stringify(experimentPayload({ type: 'mdpo_test11_checkpoint', test_run_id: progress.test_run_id,
          lock_sha256: progress.lock_sha256, group: group.id, role: group.role, split: 'test', visuals: captureFiveViewVisuals() })));
        const scored = await postExperimentJson('/api/score', payload);
        if (!scored.visual_validation_record) throw new Error(`MDPO test11 ${group.id} ${sampleKey} was not persisted`);
        rows.push({ group: group.id, role: group.role, sample: sampleKey, file: scored.visual_validation_record, resumed: false });
      }
    } finally { state.layoutEvaluationOverride = null; state.optimizerStrategyOverrides = {}; state.validationGenerationReceipt = null; }
    const report = await postExperimentJson('/api/mdpo-test11-finalize', {});
    return { ok: true, test_run_id: progress.test_run_id, row_count: rows.length, rows, report_file: 'experiments/mdpo/test11_final_report.json', report };
  },
  ready: () => Boolean(state.catalog && state.splitManifest),
  listTrainSamples: async () => { await waitForWorkspaceReady(); return state.splitManifest.samples.filter((item) => item.split === 'train').map((item) => ({ category: item.category, sampleId: String(item.sample_id), split: item.split })); },
  selectTrainSample: selectAutomationSample,
  diagnoseQwenScoreResolution: async ({ category, sampleId, candidates = 4, seed = 17 } = {}) => {
    await selectAutomationSample(category, sampleId, 'train');
    $('llmCandidatesInput').value = String(candidates);
    $('seedInput').value = String(seed);
    return runLLMPreferenceRounds({ runId: crypto.randomUUID(), finalize: false, trainAfterRounds: false, roundCountOverride: 1, diagnosticOnly: true });
  },
  runLLMPreference: async ({ category, sampleId, candidates = 4, rounds = 1, visualValSamples = 11, seed = 17, epochs = 80 } = {}) => {
    const selected = await selectAutomationSample(category, sampleId, 'train');
    const result = await window.__labelStudioAutomation.runLLMPreferenceDataset({ samples: [selected], candidates, rounds, visualValSamples, seed, epochs });
    return { selected, ...result };
  },
  runLLMPreferenceDataset: async ({ samples = [], candidates = 4, rounds = 1, visualValSamples = 11, seed = 17, epochs = 80, gateIterations = 180, gateLimit = 0, runId: requestedRunId = null, skipBaseline = false } = {}) => {
    await waitForWorkspaceReady();
    const config = await fetch('/api/scoring-config').then((response) => response.json());
    if (!config.configured) throw new Error('请先启动本机 Ollama 并配置 qwen3-vl:4b-instruct。');
    if (!config.tested) throw new Error('连接尚未通过六图与 JSON 自检；请先点击“测试六图与 JSON”。');
    const runId = requestedRunId || crypto.randomUUID();
    if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw new Error('续跑 run_id 无效');
    const requested = samples.length ? samples : state.splitManifest.samples.filter((item) => item.split === 'train').slice(0, 1).map((item) => ({ category: item.category, sampleId: String(item.sample_id) }));
    const normalized = requested.map((item) => ({ category: item.category, sampleId: String(item.sampleId ?? item.sample_id) }));
    $('llmCandidatesInput').value = String(Math.max(2, Math.min(8, Number(candidates) || 4)));
    $('llmRoundsInput').value = String(Math.max(1, Math.min(10, Number(rounds) || 1)));
    $('trainEpochsInput').value = String(Math.max(1, Math.min(1000, Number(epochs) || 80)));
    let savedPairs = 0;
    const processed = normalized.map((sample) => ({ selected: null, savedPairs: 0, rounds: [] }));
    const checkpointHistory = [];
    let rewardActivatedInThisRun = false;
    const totalRounds = Math.max(1, Math.min(10, Number(rounds) || 1));
    const candidatesPerRound = Math.max(2, Math.min(8, Number(candidates) || 4));
    const valSamples = stratifiedValidationSamples(Math.max(1, Math.min(11, Number(visualValSamples) || 11)));
    const visualValidationHistory = [];
    if (skipBaseline) {
      $('llmRoundStatus').textContent = `续跑 ${runId}：复用已完成并落盘的第 0 轮 val11 六图评分。`;
    } else {
      $('llmRoundStatus').textContent = `正在执行 0 轮无奖励基线：${valSamples.length} 个 val 样本，每个样本提交原始模型图和五视角布局图…`;
      visualValidationHistory.push((await evaluateVisualValidationCheckpoint({ runId, round: 0, samples: valSamples, seed: Number(seed) || 17 })).summary);
    }
    for (let round = 1; round <= totalRounds; round += 1) {
      let roundPairs = 0;
      for (let index = 0; index < normalized.length; index += 1) {
        const sample = normalized[index];
        const selected = await selectAutomationSample(sample.category, sample.sampleId);
        processed[index].selected = selected;
        $('seedInput').value = String(Math.max(0, Number(seed) + index * candidatesPerRound * totalRounds));
        let result;
        try {
          result = await runLLMPreferenceRounds({ runId, finalize: false, trainAfterRounds: false, roundCountOverride: 1, roundOffset: round - 1, rewardActivatedInThisRun, sampleOrdinal: index + 1, sampleTotal: normalized.length });
        } catch (error) {
          if (/不足\s*\d+\s*个互相可比的安全候选|安全候选/.test(error.message || '')) {
            processed[index].skipped = { reason: 'insufficient_safe_comparable_candidates', message: error.message };
            processed[index].rounds.push({ round, savedPairs: 0, skipped: true });
            $('llmRoundStatus').textContent = `跳过 ${sample.category}/${sample.sampleId}：安全候选不足，继续后续样本（不会调用 Qwen 或制造偏好）…`;
            continue;
          }
          throw error;
        }
        if (!result?.ok) throw new Error(result?.error || `第 ${round} 轮样本 ${sample.category}/${sample.sampleId} 偏好收集失败`);
        roundPairs += result.savedPairs;
        savedPairs += result.savedPairs;
        processed[index].savedPairs += result.savedPairs;
        processed[index].rounds.push({ round, savedPairs: result.savedPairs });
      }
      if (!roundPairs) throw new Error(`第 ${round} 轮所有样本均未产生非平局偏好，无法继续累计训练`);
      $('llmRoundStatus').textContent = `第 ${round}/${totalRounds} 轮跨样本收集完成：本轮 ${roundPairs} 条、累计 ${savedPairs} 条，正在累计训练并执行 val 联合门控…`;
      const trained = await trainLLMRewardForRun(runId, Number(seed) || 17, { gateIterations, gateLimit });
      rewardActivatedInThisRun ||= Boolean(trained.activated);
      const savedCheckpoint = await saveLLMPreferenceCheckpoint(runId, round);
      checkpointHistory.push(savedCheckpoint.checkpoint);
      const visualResult = await evaluateVisualValidationCheckpoint({ runId, round, samples: valSamples, seed: Number(seed) || 17 });
      visualValidationHistory.push(visualResult.summary);
      rewardActivatedInThisRun ||= Boolean(visualResult.activated);
    }
    if (!savedPairs) throw new Error('所有样本的 Qwen 评分均未产生非平局偏好，未训练奖励模型');
    const finalized = await finalizeLLMPreferenceRun({ runId, savedPairs, roundCount: totalRounds, candidateCount: candidatesPerRound, samples: processed.map((item) => item.selected), testIterations: gateIterations, testLimit: gateLimit });
    $('llmRoundStatus').textContent = `跨样本偏好流程结束：run_id=${runId}，样本 ${processed.length}，轮次 ${totalRounds}，偏好 ${savedPairs}${finalized.testNote}${finalized.reportNote}。`;
    $('llmRoundStatus').className = 'tool-status ready';
    return { ok: true, runId, savedPairs, processed, checkpoints: checkpointHistory, visualValidation: visualValidationHistory, activated: rewardActivatedInThisRun, status: $('llmRoundStatus').textContent };
  },
  captureVisualSummary: () => {
    const visuals = captureFiveViewVisuals();
    return { labelCount: state.labels.length, images: Object.fromEntries(Object.entries(visuals).map(([name, image]) => [name, { valid: /^data:image\/(?:jpeg|png|webp);base64,/.test(image || ''), bytes: image?.length || 0 }])) };
  },
  v10VisualizationSummary: () => {
    const tools = document.querySelector('.experiment-tools:not(.scoring-settings)');
    const metricIds = ['v10SilhouetteOverlap', 'v10WorstDepth', 'v10WorstPenetration', 'v10CvarOcclusion', 'v10CvarFreeSpace', 'v10WorstTextLoss', 'v10FreeSpaceMismatch', 'v10WeightedLeaderCrossing', 'v10WorstLeaderCrossing', 'v10CvarLeaderCrossing', 'v10ViewObjective'];
    return {
      active_model: state.modelConfig?.model?.version || null,
      architecture_heading: tools?.querySelector('h3')?.textContent || null,
      architecture_text: tools?.querySelector('.architecture-hierarchy')?.textContent?.replace(/\s+/g, ' ').trim() || null,
      train_button: $('trainLayoutBtn')?.textContent || null,
      experiment_status: $('experimentStatus')?.textContent || null,
      metric_cards: Object.fromEntries(metricIds.map((id) => [id, { exists: Boolean($(id)), value: $(id)?.textContent || null }])),
      requirements: {
        anchor_label_18d: Boolean(tools?.textContent?.includes('18D')),
        label_label_10d: Boolean(tools?.textContent?.includes('10D')),
        local_decode_6d: Boolean(tools?.textContent?.includes('64→32→6 local decode')),
        worst_view_cvar: Boolean(tools?.textContent?.includes('Worst-view + CVaR')),
        safe_qwen_only: Boolean(tools?.textContent?.includes('safe candidates only')),
        all_metric_cards_present: metricIds.every((id) => Boolean($(id)))
      }
    };
  },
  status: () => ({ experiment: $('experimentStatus')?.textContent, scoring: $('scoringConfigStatus')?.textContent, llm: $('llmRoundStatus')?.textContent })
};

async function trainPreferenceModel() { const button = $('trainModelBtn'); const status = $('modelTrainStatus'); if (!button || !status) return; button.disabled = true; status.textContent = '正在训练候选奖励模型并执行 val 门控…'; status.className = 'tool-status'; try { const response = await fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: numericInput('trainEpochsInput', 80, 1, 1000), learningRate: 0.01, seed: numericInput('seedInput', 17, 0, 2147483647) }) }); const result = await response.json(); if (!response.ok) throw new Error(result.error || '训练失败'); status.textContent = result.activated ? `候选通过 val 门控并已激活：${result.model_file} · ${result.timing?.elapsed_ms ?? '—'} ms` : `候选已训练但 val 门控拒绝，活动奖励模型未改变；候选：${result.candidate_model_file}`; status.className = `tool-status ${result.activated ? 'ready' : 'warn'}`; showToast(result.activated ? '偏好模型已通过验证并激活' : '候选未通过验证，已保留旧模型', result.activated ? 'success' : 'info'); await loadActiveModelConfig(); if (state.model) await generateLabels(); } catch (error) { status.textContent = error.message.includes('偏好文件') ? '暂无有效 A/B 偏好，请先保存候选并记录偏好' : error.message; status.className = 'tool-status warn'; showToast(error.message, 'error'); } finally { button.disabled = false; } }
async function trainLayoutModel() {
  const button = $('trainLayoutBtn');
  const status = $('layoutTrainStatus');
  if (!button || !status) return;
  button.disabled = true;
  status.textContent = '正在训练 v10：局部坐标 51D → 关系类型感知异构 GNN → Transformer → 人类风格 MoE，并优化 worst-view/CVaR、引导线与字体清晰度…';
  status.className = 'tool-status';
  try {
    const response = await fetch('/api/train-layout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: numericInput('trainEpochsInput', 80, 1, 1000), learningRate: 0.002, styleWeight: 0.2, directionWeight: 1.25, viewWeight: 0.25, worstViewWeight: 2, cvarViewWeight: 1, stereoWeight: 1, textClarityWeight: 3, preGnnFnnLayers: 1, messageLayers: 2, transformerLayers: 1, expertCount: 4, seed: numericInput('seedInput', 17, 0, 2147483647) })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'v10 布局模型训练失败');
    status.textContent = `v10 锚点局部坐标异构图模型已训练、完成 val 诊断并激活：${result.model_file} · ${result.timing?.elapsed_ms ?? '—'} ms`;
    status.className = 'tool-status ready';
    showToast('v10 局部坐标异构图模型训练完成并已激活', 'success');
    await loadActiveModelConfig();
    if (state.model) await generateLabels();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'tool-status warn';
    showToast(error.message, 'error');
  } finally {
    button.disabled = false;
  }
}

function upgradeV10Visualization() {
  const comparison = document.querySelector('.quality-comparison-card');
  const comparisonTitle = comparison?.querySelector('h3');
  const comparisonNote = comparison?.querySelector('.quality-protocol-note');
  if (comparisonTitle) comparisonTitle.textContent = '历史方法、v10 局部坐标异构图与最新 test11 对照';
  if (comparisonNote) comparisonNote.textContent = 'v10 仅用 clean OBJ 三维几何生成初始坐标；五视角用于主视角优先的遮挡、字体清晰度、worst-view/CVaR 与双目安全评价。Qwen 只在安全候选中进行美学重排。';
  const tools = document.querySelector('.experiment-tools:not(.scoring-settings)');
  const heading = tools?.querySelector('.tools-heading');
  if (heading) heading.innerHTML = '<div><span class="eyebrow">DGCNN + ANCHOR FRAME + HETEROGENEOUS GNN + TRANSFORMER + MOE</span><h3>v10 纯三维局部坐标异构图与安全候选偏好实验</h3></div><span class="tools-note">clean OBJ 1024 点经 2 层 EdgeConv 得到 64D；51D 局部标签特征经 FNN 后进入 18D anchor→label 与 10D label→label 独立关系通道</span>';
  const encoder = $('modelEncoder');
  if (encoder) encoder.innerHTML = '<option value="anchor_frame_heterogeneous_graph_transformer_moe_v10">DGCNN-局部坐标-异构图 Transformer-MoE v10</option>';
  if ($('trainLayoutBtn')) $('trainLayoutBtn').textContent = '训练 v10 布局模型';
  if ($('experimentStatus')) $('experimentStatus').textContent = 'v10 候选记录 val11 主视角优先、worst-view/CVaR、安全与关系参数更新；test11 只做最终确认';
  const hierarchy = tools?.querySelector('.architecture-hierarchy');
  if (hierarchy) hierarchy.innerHTML = '<div class="architecture-level"><span class="level-label">L1 几何与局部编码</span><article><b>DGCNN Geometry</b><strong>1024 points → 2×EdgeConv → 64D</strong><small>只读取 clean OBJ 表面几何</small></article><article><b>Anchor Local Frame</b><strong>weighted PCA → t₁,t₂,n</strong><small>高曲率自适应 patch、法向平均、符号对齐与对称等价方向</small></article><article><b>Label FNN</b><strong>51→64→128→64</strong><small>局部坐标、曲率、尺寸、间距与三维风格</small></article></div><div class="architecture-down">↓ typed relations</div><div class="architecture-level"><span class="level-label">L2 异构关系</span><article><b>Anchor → Label</b><strong>18D · independent weights</strong><small>法向、局部位置、距离、引导线、曲率与标签尺寸</small></article><article><b>Label → Label</b><strong>10D · independent weights</strong><small>相对位置、距离、尺寸比例、方向夹角与共享语义</small></article></div><div class="architecture-down">↓ global coordination</div><div class="architecture-level"><span class="level-label">L3 全局与解码</span><article><b>2-layer Relation GNN</b><strong>64D messages</strong><small>分别学习两种关系，再聚合到标签节点</small></article><article><b>1-layer Transformer</b><strong>global self-attention</strong><small>协调完整标签集合</small></article><article><b>Human-style MoE</b><strong>64→32→6 local decode</strong><small>输出局部 u、v、法向距离与三轴尺寸比例</small></article></div><div class="architecture-down">↓ safety-first reranking</div><div class="architecture-level"><span class="level-label">L4 多视角安全</span><article><b>Five-view Objective</b><strong>main 0.40 · others 0.15</strong><small>silhouette overlap、depth、穿模、引导线交叉、字体与留白分布</small></article><article><b>Worst-view + CVaR</b><strong>no bad-view averaging</strong><small>避免单个极差视角被平均掩盖</small></article><article><b>Local Qwen Reward</b><strong>safe candidates only</strong><small>不参与初始三维坐标生成</small></article></div>';
  const directionalLabel = $('mainDirectionalUniformity')?.closest('.metric-item');
  if (directionalLabel) directionalLabel.querySelector('span').textContent = '逐视角留白分布均衡';
  const metricsGrid = document.querySelector('.metrics-card .metrics-grid');
  if (metricsGrid && !$('v10SilhouetteOverlap')) metricsGrid.insertAdjacentHTML('beforeend', '<div class="metric-item v10-metric"><span>加权 silhouette 交叠</span><strong id="v10SilhouetteOverlap">—</strong><small>主视角优先</small></div><div class="metric-item v10-metric"><span>最差视角深度遮挡</span><strong id="v10WorstDepth">—</strong><small>worst-view</small></div><div class="metric-item v10-metric"><span>最差视角穿模</span><strong id="v10WorstPenetration">—</strong><small>worst-view</small></div><div class="metric-item v10-metric"><span>遮挡 CVaR</span><strong id="v10CvarOcclusion">—</strong><small>尾部风险</small></div><div class="metric-item v10-metric"><span>留白失配 CVaR</span><strong id="v10CvarFreeSpace">—</strong><small>尾部风险</small></div><div class="metric-item v10-metric"><span>最差字体损失</span><strong id="v10WorstTextLoss">—</strong><small>越低越清晰</small></div><div class="metric-item v10-metric"><span>加权留白失配</span><strong id="v10FreeSpaceMismatch">—</strong><small>per-view</small></div><div class="metric-item v10-metric"><span>v10 五视角目标</span><strong id="v10ViewObjective">—</strong><small>越低越好</small></div>');
  if (metricsGrid && !$('v10WeightedLeaderCrossing')) metricsGrid.insertAdjacentHTML('beforeend', '<div class="metric-item v10-metric"><span>加权引导线交叉风险</span><strong id="v10WeightedLeaderCrossing">—</strong><small>主视角优先</small></div><div class="metric-item v10-metric"><span>最差视角引导线交叉</span><strong id="v10WorstLeaderCrossing">—</strong><small>worst-view</small></div><div class="metric-item v10-metric"><span>引导线交叉 CVaR</span><strong id="v10CvarLeaderCrossing">—</strong><small>尾部风险</small></div>');
}

function bindExperimentControls() { document.querySelectorAll('[data-preset]').forEach((button) => button.addEventListener('click', () => applyPreset(button.dataset.preset))); ['seedInput', 'iterationsInput', 'modelEncoder'].forEach((id) => $(id)?.addEventListener('change', () => state.model && generateLabels())); $('resetExperimentBtn')?.addEventListener('click', resetExperimentInputs); $('trainModelBtn')?.addEventListener('click', trainPreferenceModel); $('trainLayoutBtn')?.addEventListener('click', trainLayoutModel); $('saveScoringConfigBtn')?.addEventListener('click', saveScoringConfig); $('testScoringConfigBtn')?.addEventListener('click', testScoringConfig); $('runLLMPreferenceBtn')?.addEventListener('click', runLLMPreferenceFromUI); $('evalPrefTestBtn')?.addEventListener('click', evaluatePreferenceTest); updateCandidateStatus(); loadScoringConfig(); }

async function init() { try { state.viewers.before = new ModelViewer($('beforeViewport')); state.viewers.after = new ModelViewer($('afterViewport')); } catch (error) { showToast(error.message, 'error'); return; } await Promise.all([loadActiveModelConfig(), loadLLMVal11Comparison()]); try { const response = await fetch('/api/catalog'); if (!response.ok) throw new Error('数据目录服务不可用'); state.catalog = await response.json(); $('serverStatus').classList.add('ready'); $('serverStatus').innerHTML = '<i></i> 数据服务已连接'; populateCatalog(); } catch (error) { showToast(error.message, 'error'); $('serverStatus').innerHTML = '<i></i> 服务连接失败'; } try { const splitResponse = await fetch('/api/dataset-split'); state.splitManifest = await splitResponse.json(); $('splitBadge').textContent = `train ${state.splitManifest.counts.train} · val ${state.splitManifest.counts.val} · test ${state.splitManifest.counts.test}`; if (state.metrics) renderLiveSampleQuality(state.metrics); } catch { $('splitBadge').textContent = 'split 不可用'; } $('categorySelect').addEventListener('change', updateSamples); $('sampleSelect').addEventListener('change', () => loadSample(getSamples($('categorySelect').value).find((item) => item.id === $('sampleSelect').value))); $('generateBtn').addEventListener('click', () => generateLabels()); $('downloadBtn').addEventListener('click', downloadObj); $('downloadJsonBtn').addEventListener('click', downloadJson); $('recordEvaluationBtn').addEventListener('click', recordEvaluation); $('llmScoreBtn').addEventListener('click', requestLLMScore); $('saveCandidateABtn').addEventListener('click', () => savePreferenceCandidate('A')); $('saveCandidateBBtn').addEventListener('click', () => savePreferenceCandidate('B')); $('recordPreferenceBtn').addEventListener('click', recordPreference); ['viewPolicy', 'groupPolicy', 'sizePolicy', 'optimizerPolicy'].forEach((id) => $(id)?.addEventListener('change', () => state.model && generateLabels())); $('objInput').addEventListener('change', (event) => event.target.files[0] && handleObjFile(event.target.files[0])); $('annotationInput').addEventListener('change', (event) => event.target.files[0] && handleAnnotationFile(event.target.files[0])); $('resetViewBtn').addEventListener('click', () => { state.viewers.before.resetCamera(); state.viewers.after.resetCamera(); }); document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => state.viewers[button.dataset.view].resetCamera())); }
function initFiveViewers() { for (const preset of ['main', 'right', 'left', 'up', 'down']) { const viewer = new ModelViewer($(`view${preset[0].toUpperCase()}${preset.slice(1)}`)); viewer.setPreset(preset); state.multiViewers[preset] = viewer; const manualViewer = new ModelViewer($(`manualView${preset[0].toUpperCase()}${preset.slice(1)}`)); manualViewer.setPreset(preset); state.manualViewers[preset] = manualViewer; } }
try { initFiveViewers(); } catch (error) { showToast(`五视角初始化失败：${error.message}`, 'error'); }
upgradeV10Visualization();
initLatestResultsConsole();
bindMdpoUiTrainingControls();

async function initLatestResultsConsole() {
  try {
    state.viewers.before = new ModelViewer($('beforeViewport'));
    state.viewers.after = new ModelViewer($('afterViewport'));
  } catch (error) { showToast(error.message, 'error'); return; }
  await Promise.all([loadActiveModelConfig(), loadMdpoLiveResults(), loadFinalEightResults().catch((error) => {
    const status = $('finalEightResultsStatus');
    if (status) status.textContent = error.message;
    const body = $('finalEightResultsRows');
    if (body) body.innerHTML = '<tr><td colspan="12">' + escapeHtml(error.message) + '</td></tr>';
  })]);
  try {
    const [catalogResponse, splitResponse] = await Promise.all([fetch('/api/catalog'), fetch('/api/dataset-split')]);
    if (!catalogResponse.ok) throw new Error('数据目录服务不可用');
    if (!splitResponse.ok) throw new Error('数据划分服务不可用');
    state.catalog = await catalogResponse.json();
    state.splitManifest = await splitResponse.json();
    if ($('splitBadge')) $('splitBadge').textContent = `train ${state.splitManifest.counts.train} · val ${state.splitManifest.counts.val} · test ${state.splitManifest.counts.test}`;
    $('serverStatus').classList.add('ready');
    $('serverStatus').innerHTML = '<i></i> 数据服务已连接';
    populateCatalog();
  } catch (error) {
    showToast(error.message, 'error');
    $('serverStatus').innerHTML = '<i></i> 服务连接失败';
    return;
  }
  $('splitSelect')?.addEventListener('change', () => { populateCatalog(); });
  $('categorySelect')?.addEventListener('change', updateSamples);
  $('sampleSelect')?.addEventListener('change', () => loadSample(getSamples($('categorySelect').value).find((item) => item.id === $('sampleSelect').value)));
  $('modelVisualizationSelect')?.addEventListener('change', (event) => loadModelVisualization(event.target.value));
  $('showMdpoLiveComparisonBtn')?.addEventListener('click', () => { state.reproductionView = 'test11_final'; renderReproductionComparison(); });
  $('showSampleComparisonBtn')?.addEventListener('click', () => { state.reproductionView = 'sample'; renderReproductionComparison(); });
  $('showSplitComparisonBtn')?.addEventListener('click', () => { state.reproductionView = 'summary'; renderReproductionComparison(); });
  $('resetViewBtn')?.addEventListener('click', () => { state.viewers.before.resetCamera(); state.viewers.after.resetCamera(); });
  document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => state.viewers[button.dataset.view]?.resetCamera()));
}


