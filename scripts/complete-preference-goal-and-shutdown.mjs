import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const baseUrl = 'http://127.0.0.1:5173';
const logFile = path.join(experiments, 'complete_preference_goal.log');
const sha256 = async (file) => createHash('sha256').update(await fs.readFile(file)).digest('hex').toUpperCase();
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const exists = async (file) => fs.stat(file).then((item) => item.isFile(), () => false);
const baselineLayoutHash = await sha256(path.join(experiments, 'layout_model.json'));
const resumeIndex = process.argv.indexOf('--resumeRunId');
const resumeRunId = resumeIndex >= 0 ? String(process.argv[resumeIndex + 1] || '') : '';
if (resumeRunId && !/^[a-zA-Z0-9_-]{8,80}$/.test(resumeRunId)) throw new Error('续跑 run_id 无效');
if (!resumeRunId) await fs.writeFile(logFile, '', 'utf8');

async function stamp(message) {
  const line = `[${new Date().toISOString()}] ${message}`;
  console.log(line);
  await fs.appendFile(logFile, `${line}\n`, 'utf8');
}

async function runNode(args, label, timeoutMinutes = 300) {
  await stamp(`开始：${label}`);
  const child = spawn(process.execPath, args, { cwd: root, windowsHide: true });
  let tail = '';
  const append = (chunk, stream) => {
    const text = chunk.toString();
    tail = `${tail}${text}`.slice(-12000);
    process[stream].write(text);
    fs.appendFile(logFile, text, 'utf8').catch(() => {});
  };
  child.stdout.on('data', (chunk) => append(chunk, 'stdout'));
  child.stderr.on('data', (chunk) => append(chunk, 'stderr'));
  const code = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { child.kill(); reject(new Error(`${label} 超过 ${timeoutMinutes} 分钟；最后输出：${tail}`)); }, timeoutMinutes * 60 * 1000);
    child.once('error', (error) => { clearTimeout(timer); reject(error); });
    child.once('close', (value) => { clearTimeout(timer); resolve(value); });
  });
  if (code !== 0) throw new Error(`${label} 失败，退出码 ${code}；最后输出：${tail}`);
  await stamp(`完成：${label}`);
}

async function scoringConfig() {
  const response = await fetch(`${baseUrl}/api/scoring-config`, { signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error(`评分配置接口返回 ${response.status}`);
  return response.json();
}

const number = (value, digits = 4) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const tableRow = (values) => `| ${values.join(' | ')} |`;

async function buildReport(runId) {
  const convergence = await readJson(path.join(experiments, 'llm_preference_convergence.json'));
  const run = await readJson(path.join(experiments, `llm_preference_run_${runId}.json`));
  const ledger = await readJson(path.join(experiments, `llm_preference_checkpoints_${runId}.json`));
  const comparison = await readJson(path.join(experiments, 'comparisons', 'round1', 'comparison.json'));
  const layoutModel = await readJson(path.join(experiments, 'layout_model.json'));
  const rewardModel = await readJson(path.join(experiments, 'preference_model_candidate.json'));
  const leaderPrior = await readJson(path.join(experiments, 'manual_leader_length_prior.json'));
  const densityPolicy = await readJson(path.join(experiments, 'directional_density_policy.json'));
  const testReport = await readJson(path.join(experiments, 'preference_test11_report.json'));
  const layoutHashAfter = await sha256(path.join(experiments, 'layout_model.json'));
  if (convergence.run_id !== runId || run.run_id !== runId || testReport.run_id !== runId || rewardModel.training?.run_id_filter !== runId) throw new Error('报告证据的 run_id 不一致，拒绝生成最终报告');
  const verifiedRounds = convergence.checkpoints?.filter((item) => [0, 1, 2, 4, 8].includes(Number(item.round))) || [];
  if (!convergence.evidence_sufficient || verifiedRounds.length !== 5 || !verifiedRounds.every((item) => item.status === 'evaluated' && item.checkpoint_verified)) throw new Error('0/1/2/4/8 真实六图评分或检查点证据不完整，拒绝生成最终报告');
  if (testReport.test_samples !== 11 || testReport.rows?.length !== 11 || !testReport.rows.every((item) => item.split === 'test' && ['multidimensional_quality_score', 'directional_allocation_mismatch', 'leader_length_compliance_ratio'].every((key) => item.preferred?.[key] !== null && item.preferred?.[key] !== undefined && Number.isFinite(Number(item.preferred[key]))))) throw new Error('test11 新协议指标不完整，拒绝生成最终报告');
  if (baselineLayoutHash !== layoutHashAfter) throw new Error('基础布局模型哈希发生变化，拒绝生成最终报告');
  const activePreferenceFile = path.join(experiments, 'preference_model.json');
  const activePreferenceHash = await exists(activePreferenceFile) ? await sha256(activePreferenceFile) : null;
  if (!ledger.checkpoints?.some((item) => item.activated && item.validation_status === 'accepted')) throw new Error('没有通过 val 门控并激活的偏好检查点，拒绝标记全部完成和自动关机');
  const activePreference = activePreferenceHash ? await readJson(activePreferenceFile) : null;
  if (activePreference?.training?.run_id_filter !== runId) throw new Error('活动偏好模型不属于本次运行，拒绝标记全部完成和自动关机');
  const standard = convergence.checkpoints.filter((item) => [0, 1, 2, 4, 8].includes(item.round));
  const aesthetics = ['overall', 'composition_harmony', 'visual_hierarchy', 'spatial_balance', 'manual_style_similarity'];
  const safety = ['text_clarity', 'coverage', 'label_label_occlusion', 'object_occlusion', 'object_penetration', 'leader_line_clarity', 'multiview_consistency', 'binocular_consistency', 'size_consistency'];
  const methods = ['current_fixed_label_seed17', 'hedgehog_1d', 'hedgehog_3d', 'BinoForce_final_snapshot'];
  const benchmark = (comparison.unified_snapshot_evaluation?.summary || []).filter((item) => methods.includes(item.method));
  const accepted = ledger.checkpoints.filter((item) => item.activated && item.validation_status === 'accepted');
  const output = path.join(experiments, `最终LLM美学偏好实验报告_${runId}.md`);
  const lines = [
    '# 最终 LLM 美学偏好实验报告', '',
    `- run_id：\`${runId}\``,
    `- 本地评分模型：\`${convergence.scorer_model || '未记录'}\``,
    `- train 偏好对：${run.evidence?.pair_count ?? '—'}`,
    `- 证据是否完整：${convergence.evidence_sufficient ? '是' : '否'}`,
    `- 是否先提升后稳定：${convergence.rise_then_stable ? '是' : '否'}`,
    `- 第 4→8 轮是否稳定：${convergence.plateau_round4_to_round8 ? '是' : '否'}`,
    `- 通过 val 并激活的检查点：${accepted.length ? accepted.map((item) => item.round).join('、') : '无；未强制覆盖活动模型'}`,
    `- 活动美学奖励模型 SHA-256：${activePreferenceHash || '无活动模型'}`,
    `- 基础布局模型是否保持：${baselineLayoutHash === layoutHashAfter ? '是' : '否'}`,
    `- 基础布局模型 SHA-256：\`${layoutHashAfter}\``, '',
    '## 0/1/2/4/8 轮真实 val11 美学评分', '',
    tableRow(['轮次', '综合分', ...aesthetics]),
    tableRow(['---:', '---:', ...aesthetics.map(() => '---:')]),
    ...standard.map((item) => tableRow([item.round, number(item.composite_score), ...aesthetics.map((name) => number(item.score_means?.[name]))])), '',
    '## 九个安全诊断维度', '',
    tableRow(['轮次', ...safety]),
    tableRow(['---:', ...safety.map(() => '---:')]),
    ...standard.map((item) => tableRow([item.round, ...safety.map((name) => number(item.score_means?.[name]))])), '',
    '安全维度不参与 chosen/rejected 美学加权；确定性五视角能量与安全门控负责穿模、遮挡、越界、文字和引导线约束。', '',
    '## 与 BinoForce / Hedgehog 的统一相机比较', '',
    tableRow(['方法', '质量↑', '能量↓', '文字↑', '标签重叠↓', '标签遮物↓', '深度穿模↓', '网格相交↓', '引导线长度', '长度合规↑', '方向失配↓', '方向均匀↑', '人工风格距离↓']),
    tableRow(['---', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---:']),
    ...benchmark.map((item) => tableRow([item.method, number(item.multidimensional_quality_score), number(item.objective_score), number(item.text_clarity), number(item.label_label_occlusion_ratio), number(item.label_object_occlusion_ratio), number(item.object_penetration_ratio, 5), number(item.mesh_surface_intersection_ratio), number(item.mean_anchor_distance), number(item.leader_length_compliance_ratio), number(item.directional_allocation_mismatch), number(item.directional_uniformity), number(item.manual_style_distance)])), '',
    '### LLM 偏好后 test11', '',
    tableRow(['指标', '基础 seed17', 'LLM 安全门控后']),
    tableRow(['---', '---:', '---:']),
    ...['multidimensional_quality_score', 'objective_score', 'text_clarity', 'label_label_occlusion_ratio', 'label_object_occlusion_ratio', 'object_penetration_ratio', 'mesh_surface_intersection_ratio', 'mean_anchor_distance', 'leader_length_compliance_ratio', 'directional_allocation_mismatch', 'directional_uniformity', 'manual_style_distance'].map((field) => tableRow([field, number(testReport.summary?.[field]?.baseline), number(testReport.summary?.[field]?.preferred)])), '',
    '比较使用同一 test11、750×500、50 mm、36×24 mm、相机距离 10、主方向 [1,1,1] 和主/右/左/俯/仰五视角。外部 LLM 的 1–5 美学分与几何指标量纲不同，只并列报告。', '',
    '## 模型完整说明', '',
    '### 基础布局网络', '',
    `- 活动版本：\`${layoutModel.version}\`；类型：\`${layoutModel.architecture?.type}\`。`,
    `- 四个确定性风格专家：${layoutModel.architecture?.expert_styles?.join('、') || 'balanced、elongated、dense、symmetric'}；每个 MLP 为 \`${layoutModel.architecture?.expert_mlp_layers?.join('→') || '51→128→64→32→6'}\`。`,
    '- 51 维输入：16 维三维几何/固定标签契约特征，加主、右、左、俯、仰五视角各 7 维投影特征。',
    '- 6 维输出：标签中心相对固定锚点的 xyz 偏移，以及面板 xyz 尺寸。标签 ID、文字、数量、锚点和目标组均锁定。',
    `- 训练超参数：epochs=${layoutModel.hyperparameters?.epochs}，learning rate=${layoutModel.hyperparameters?.learning_rate}，weight decay=${layoutModel.hyperparameters?.weight_decay}，seed=${layoutModel.hyperparameters?.seed}，每轮打乱=${layoutModel.hyperparameters?.shuffle_each_epoch ? '是' : '否'}。`,
    `- 数据：train/val/test=${layoutModel.split_policy?.train}/${layoutModel.split_policy?.val}/${layoutModel.split_policy?.test} 个样本；对应 ${layoutModel.training?.train_examples}/${layoutModel.training?.val_examples}/${layoutModel.training?.test_examples} 个标签监督样本。`,
    `- 推理融合：center_blend=${layoutModel.inference?.center_blend}，size_blend=${layoutModel.inference?.size_blend}，size_ratio_range=${JSON.stringify(layoutModel.inference?.size_ratio_range)}；退火默认 180 次、seed 17。`, '',
    '### LLM 美学奖励网络', '',
    `- 候选奖励网络：\`${rewardModel.architecture?.input_dim}→${rewardModel.architecture?.hidden_dim}→${rewardModel.architecture?.output_dim}\`，隐藏层 tanh，pairwise logistic loss。`,
    `- 训练超参数：epochs=${rewardModel.training?.epochs}，learning rate=${rewardModel.training?.learning_rate}，weight decay=0.0001，seed=${rewardModel.training?.seed}；仅使用 run_id=${runId} 的 train 偏好对。`,
    '- 12 维推理输入只描述候选自身美学统计：标签/语义数量、引导线长度、风格平衡、尺寸一致性、径向均值/离散度、引导线均值/离散度、间距一致性、文字尺寸适配、构图和谐。',
    '- 本地 Qwen 十四维输出中只有 5 个美学维度决定 chosen/rejected：overall 0.30、composition_harmony 0.25、visual_hierarchy 0.20、spatial_balance 0.15、manual_style_similarity 0.10。',
    '- 其余 9 个维度只作诊断：文字清晰、覆盖、标签重叠、遮挡、穿模、引导线清晰、多视角一致、双目一致、尺寸一致；它们不进入美学偏好加权。', '',
    '### 确定性能量与惩罚', '',
    tableRow(['惩罚项', '权重', '职责']),
    tableRow(['---', '---:', '---']),
    tableRow(['标签重叠 OLR', '3.2', '降低标签互相遮挡']),
    tableRow(['视野越界', '4.5', '约束面板留在可见范围']),
    tableRow(['引导线交叉 LCD', '1.8', '减少线线交叉']),
    tableRow(['物体遮挡标签', '3.5', '减少物体挡住标签']),
    tableRow(['标签遮挡物体', '3.5', '减少标签挡住模型']),
    tableRow(['穿模', '5.0', '抑制面板进入物体深度范围']),
    tableRow(['文字清晰度损失', '2.0', '保持文字像素高度、适配和不裁切']),
    tableRow(['方向自由空间失配', '1.6', '标签方向占比匹配各方向非物体空间']),
    tableRow(['方向集中度', '1.2', '避免标签集中在少数方向']),
    tableRow(['引导线过短', '4.0', '强约束低于人工 train P10']),
    tableRow(['引导线过长', '0.35', '限制超过人工 train P90/P95']),
    tableRow(['引导线目标偏离', '0.04', '软引导到人工 train P50']),
    tableRow(['跨视角方差', '2.4', '保持五视角稳定']),
    tableRow(['双目不一致', '2.8', '降低左右眼布局重影']), '',
    `引导线先验只来自 train33 的 ${leaderPrior.global?.count} 个人工调整后标签：P10=${number(leaderPrior.global?.p10, 6)}、P50=${number(leaderPrior.global?.p50, 6)}、P90=${number(leaderPrior.global?.p90, 6)}、硬上限 P95=${number(leaderPrior.global?.p95, 6)}。方向密度策略由 val11 选择为 strong（1.6/1.2），test11 只作冻结确认。`, '',
    '### 安全门控与写入规则', '',
    '- 候选先按网格相交、穿模和总能量确定安全参考，再检查显式阈值；只有安全合格候选进入美学奖励重排。',
    '- 相对安全阈值包括：总能量最多增加 30%，标签/标签和标签/物体遮挡各最多增加 0.01，穿模和网格相交各最多增加 0.005，最差 OLR 增加不超过 0.03，最差越界增加不超过 0.01，文字清晰度下降不超过 0.10。',
    '- 每轮奖励模型按 run_id 冻结并记录 SHA-256；只有 val11 美学提升且安全不退化的检查点才可写入活动模型。test11 不参与训练、选权重或激活。', '',
    '## 结论', '', convergence.interpretation, '',
    `test11 最终选择轮次：${convergence.final_test11?.selected_round ?? '—'}；test11 只作最终确认。`, '',
    '## 主要证据文件', '',
    `- \`experiments/llm_visual_validation_${runId}.json\``,
    `- \`experiments/llm_preference_checkpoints_${runId}.json\``,
    `- \`experiments/llm_preference_run_${runId}.json\``,
    '- `experiments/llm_preference_convergence.json`',
    '- `experiments/preference_validation_selection.json`',
    '- `experiments/preference_test11_report.json`',
    '- `experiments/comparisons/round1/comparison.json`'
  ];
  await fs.writeFile(output, `${lines.join('\n')}\n`, 'utf8');
  await fs.writeFile(path.join(experiments, '最终LLM美学偏好实验报告_latest.md'), `${lines.join('\n')}\n`, 'utf8');
  await fs.writeFile(path.join(experiments, `最终模型完整说明报告_${runId}.md`), `${lines.join('\n')}\n`, 'utf8');
  await fs.writeFile(path.join(experiments, '最终模型完整说明报告_latest.md'), `${lines.join('\n')}\n`, 'utf8');
  return { output, convergence, layoutHashAfter };
}

try {
  await stamp('等待本地页面完成十四维六图连接测试；不需要 API Key。');
  while (true) {
    const config = await scoringConfig().catch(() => null);
    if (config?.configured && config?.tested && config?.scoringProtocol === 'aesthetic_safety_v3_fourteen_dimension') {
      await stamp(`连接测试已通过：model=${config.model}，testedAt=${config.testedAt}`);
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
  await runNode(['scripts/preflight-llm-preference.mjs', '--samples', '3', '--candidates', '4', '--rounds', '8', '--visualValSamples', '11'], '正式实验预检', 5);
  const preflight = await readJson(path.join(experiments, 'llm_preference_preflight.json'));
  if (!preflight.ready) throw new Error(`预检未通过：${preflight.blocking_failures.join(', ')}`);
  const runArgs = ['scripts/run-llm-preference-headless.mjs', '--samples', '3', '--candidates', '4', '--rounds', '8', '--visualValSamples', '11', '--seed', '17', '--epochs', '80', '--gateIterations', '180', '--timeoutMinutes', '240'];
  if (resumeRunId) runArgs.push('--runId', resumeRunId, '--skipBaseline', 'true');
  await runNode(runArgs, resumeRunId ? `续跑真实 8 轮 LLM 美学偏好实验（复用第 0 轮 ${resumeRunId}）` : '真实 8 轮 LLM 美学偏好实验', 250);
  const latest = await readJson(path.join(experiments, 'llm_preference_run_latest.json'));
  const runId = String(latest.run_id || '');
  if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw new Error('正式运行完成但没有有效 run_id');
  await runNode(['scripts/analyze-preference-convergence.mjs', '--runId', runId], '0/1/2/4/8 收敛分析', 10);
  for (const [script, label] of [
    ['test-preference-policy.mjs', '安全门控回归'], ['test-preference-training.mjs', '美学奖励训练回归'],
    ['test-preference-gate.mjs', 'val 门控回归'], ['test-layout-provenance.mjs', '人工调整后监督来源审计'],
    ['test-model-preservation.mjs', '模型保护回归'], ['test-camera-calibration.mjs', '数据集相机审计'],
    ['test-display-camera.mjs', '3D 正置相机回归'], ['test-directional-density.mjs', '方向自由空间惩罚回归'],
    ['test-adaptive-directional-rerank.mjs', '自适应方向安全回退回归']
  ]) await runNode([`scripts/${script}`], label, 5);
  const report = await buildReport(runId);
  const standard = report.convergence.checkpoints.filter((item) => [0, 1, 2, 4, 8].includes(item.round));
  if (!report.convergence.evidence_sufficient || standard.length !== 5 || !standard.every((item) => item.status === 'evaluated' && item.checkpoint_verified)) throw new Error('0/1/2/4/8 评分或冻结检查点不完整');
  if (baselineLayoutHash !== report.layoutHashAfter) throw new Error('基础布局模型哈希发生变化');
  await stamp(`全部完成。最终报告：${path.relative(root, report.output)}；60 秒后自动关机。`);
  if (process.platform === 'win32') {
    const shutdown = spawn('shutdown.exe', ['/s', '/t', '60', '/c', 'LLM preference experiment completed and verified.'], { windowsHide: true, detached: true, stdio: 'ignore' });
    shutdown.unref();
  }
} catch (error) {
  await stamp(`停止且不关机：${error.stack || error.message}`);
  process.exitCode = 1;
}
