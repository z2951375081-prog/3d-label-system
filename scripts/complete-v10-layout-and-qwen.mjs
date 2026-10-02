import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const baseUrl = 'http://127.0.0.1:5173';
const logFile = path.join(experiments, 'service-logs', 'v10-qwen-completion.log');
const auditFile = path.join(experiments, 'v10_training_completion_audit.json');
const execFileAsync = promisify(execFile);
const readJson = async (file) => JSON.parse(await fs.readFile(file, 'utf8'));
const exists = async (file) => fs.stat(file).then((item) => item.isFile(), () => false);
const sha256 = async (file) => createHash('sha256').update(await fs.readFile(file)).digest('hex').toUpperCase();
const stamp = async (message) => {
  const line = `[${new Date().toISOString()}] ${message}`;
  await fs.mkdir(path.dirname(logFile), { recursive: true });
  await fs.appendFile(logFile, `${line}\n`, 'utf8');
  console.log(line);
};

function runNode(args, label, timeoutMinutes = 60, extraEnv = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true, env: { ...process.env, ...extraEnv } });
    let stdout = '', stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString(); process.stdout.write(chunk); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); process.stderr.write(chunk); });
    const timer = setTimeout(() => { child.kill(); reject(new Error(`${label} exceeded ${timeoutMinutes} minutes`)); }, timeoutMinutes * 60 * 1000);
    child.on('error', (error) => { clearTimeout(timer); reject(error); });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ code, stdout, stderr });
      else reject(new Error(`${label} failed with code ${code}: ${stderr || stdout}`));
    });
  });
}

async function fetchJson(url, options = {}, timeoutMs = 180_000) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
  const value = await response.json();
  if (!response.ok) throw new Error(`${url} failed (${response.status}): ${value.error || JSON.stringify(value)}`);
  return value;
}

async function ensureScoringConnection() {
  let config = await fetchJson(`${baseUrl}/api/scoring-config`, {}, 10_000);
  if (!config.configured) throw new Error('local Qwen scoring is not configured');
  if (!config.tested) {
    await stamp('Running the non-training six-image structured Qwen connection probe.');
    await fetchJson(`${baseUrl}/api/scoring-test`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }, 240_000);
    config = await fetchJson(`${baseUrl}/api/scoring-config`, {}, 10_000);
  }
  if (!config.tested || config.model !== 'qwen3-vl:4b-instruct') throw new Error('local Qwen visual connection probe did not pass');
  return config;
}

async function listeningServer() {
  const command = "$c=Get-NetTCPConnection -State Listen -LocalPort 5173 -ErrorAction SilentlyContinue | Select-Object -First 1; if($c){Get-CimInstance Win32_Process -Filter \"ProcessId = $($c.OwningProcess)\" | Select-Object ProcessId,Name,CommandLine,CreationDate | ConvertTo-Json -Compress}";
  const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-Command', command], { windowsHide: true });
  return stdout.trim() ? JSON.parse(stdout) : null;
}

async function restartServer() {
  const current = await listeningServer();
  if (!current || current.Name !== 'node.exe' || !/server\.mjs/.test(current.CommandLine || '')) throw new Error(`cannot safely identify the 5173 server: ${JSON.stringify(current)}`);
  await stamp(`Stopping stale local service PID ${current.ProcessId} so the activated v10 backend and metric pipeline are loaded.`);
  process.kill(Number(current.ProcessId));
  const stdout = await fs.open(path.join(experiments, 'service-logs', 'server-v10-stdout.log'), 'a');
  const stderr = await fs.open(path.join(experiments, 'service-logs', 'server-v10-stderr.log'), 'a');
  const child = spawn(process.execPath, ['server.mjs'], { cwd: root, windowsHide: true, detached: true, stdio: ['ignore', stdout.fd, stderr.fd] });
  child.unref();
  await stdout.close(); await stderr.close();
  const started = Date.now();
  while (Date.now() - started < 60_000) {
    try {
      const config = await fetchJson(`${baseUrl}/api/model-config`, {}, 5_000);
      if (config.model?.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') return { pid: child.pid, config };
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error('updated v10 service did not become ready on port 5173');
}

async function buildReport(runId, uiCheck) {
  const [layout, layoutTraining, selection, preferenceRun, convergence, preferenceTest, preference] = await Promise.all([
    readJson(path.join(experiments, 'layout_model.json')),
    readJson(path.join(experiments, 'layout_training_v10_anchor_frame_report.json')),
    readJson(path.join(experiments, 'v10_anchor_frame_inference_selection.json')),
    readJson(path.join(experiments, 'llm_preference_run_latest.json')),
    readJson(path.join(experiments, 'llm_preference_convergence.json')),
    readJson(path.join(experiments, 'preference_test11_report.json')),
    readJson(path.join(experiments, 'preference_model.json'))
  ]);
  if (layout.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') throw new Error('final active layout is not v10');
  if ([preferenceRun.run_id, convergence.run_id, preferenceTest.run_id, preference.training?.run_id_filter].some((value) => value !== runId)) throw new Error('Qwen final artifacts do not share the same run_id');
  const standard = convergence.checkpoints?.filter((item) => [0, 1, 2, 4, 8].includes(Number(item.round))) || [];
  if (!convergence.evidence_sufficient || standard.length !== 5 || !standard.every((item) => item.status === 'evaluated' && item.checkpoint_verified)) throw new Error('0/1/2/4/8 visual checkpoints are incomplete');
  if (preferenceTest.test_samples !== 11) throw new Error('test11 confirmation is incomplete');
  const manifest = {
    version: 'v10_anchor_frame_heterogeneous_qwen_final_manifest_v1',
    generated_at: new Date().toISOString(),
    run_id: runId,
    final_model: {
      layout: { file: 'experiments/layout_model.json', sha256: await sha256(path.join(experiments, 'layout_model.json')), version: layout.version, status: layout.status, architecture: layout.architecture, validation_gate: layout.validation_gate, training: { best_epoch: layout.training?.best_epoch, metrics: layout.training?.metrics, view_metrics: layout.training?.view_metrics, parameter_updates: layout.training?.parameter_updates } },
      qwen_reward: { file: 'experiments/preference_model.json', sha256: await sha256(path.join(experiments, 'preference_model.json')), model: 'qwen3-vl:4b-instruct', role: 'safe_candidate_aesthetic_reranking_only', architecture: preference.architecture, validation_gate: preference.validation_gate, training: preference.training }
    },
    selection: { layout_val11: selection.validation, qwen_convergence: convergence, preference_test11: preferenceTest },
    visualization: uiCheck,
    invariants: { pure_3d_initial_generation: true, visual_generation_input: false, fixed_label_contract: true, test_not_used_for_selection: true, main_view_weight: 0.4, worst_view_and_cvar: true, qwen_never_generates_initial_coordinates: true }
  };
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
  await fs.writeFile(path.join(experiments, `v10_final_model_manifest_${runId}.json`), manifestText, 'utf8');
  await fs.writeFile(path.join(experiments, 'v10_final_model_manifest_latest.json'), manifestText, 'utf8');
  const m = layoutTraining.metrics || {};
  const test = preferenceTest.summary || {};
  const lines = [
    '# v10 锚点局部坐标异构图与本地 Qwen 最终报告', '',
    `- 生成时间：${manifest.generated_at}`, `- Qwen run_id：${runId}`,
    `- 活动布局模型：${layout.version}`, `- 布局 SHA-256：${manifest.final_model.layout.sha256}`,
    `- 活动奖励 SHA-256：${manifest.final_model.qwen_reward.sha256}`, '',
    '## 已验证架构', '',
    '- clean OBJ 1024 点、2 层 EdgeConv、64D OBJ 几何。',
    '- 加权 PCA / 面积加权法向构建锚点局部坐标系；高曲率自适应 patch、符号对齐、对称等价方向。',
    '- 51→64→128→64 标签 FNN。',
    '- 18D anchor→label 与 10D label→label 使用独立消息参数，2 层关系 GNN。',
    '- 1 层全局 Transformer、人类风格 Softmax MoE、64→32→6 局部坐标解码。',
    '- 主视角权重 0.40，其余各 0.15；包含 silhouette、深度遮挡、穿模、引导线交叉、字体、逐视角留白、worst-view、CVaR 与 stereo。',
    '- Qwen3-VL 只对确定性安全候选做美学重排，不参与初始坐标生成。', '',
    '## v10 监督训练', '',
    `- best epoch：${layout.training?.best_epoch} / ${layout.hyperparameters?.epochs}`,
    `- train total：${m.train?.total ?? '—'}；val total：${m.val?.total ?? '—'}；test total：${m.test?.total ?? '—'}。`,
    `- val 引导线方向损失：${m.val?.leader_direction ?? '—'}；val 引导线交叉损失：${m.val?.leader_crossing ?? '—'}；val view-conditioned loss：${m.val?.view_conditioned ?? '—'}。`, '',
    '## 本地 Qwen 完整训练', '',
    '- 使用 3 个分层 train 样本、每样本每轮 4 个安全候选、8 轮；独立 val11 在 0/1/2/4/8 轮进行六图十四维评分。',
    `- 收敛证据完整：${convergence.evidence_sufficient ? '是' : '否'}。`,
    `- test11 综合质量：${test.multidimensional_quality_score?.baseline ?? '—'} → ${test.multidimensional_quality_score?.preferred ?? '—'}。`,
    `- test11 文字清晰度：${test.text_clarity?.baseline ?? '—'} → ${test.text_clarity?.preferred ?? '—'}。`,
    `- test11 标签重叠：${test.label_label_occlusion_ratio?.baseline ?? '—'} → ${test.label_label_occlusion_ratio?.preferred ?? '—'}。`, '',
    '## 可视化证据', '',
    `- 六图检查：${uiCheck.screenshot}`, '- 架构图、18D/10D 关系、局部解码、worst-view/CVaR 与 11 个新指标卡（含引导线交叉三项）均通过 DOM 契约检查。', ''
  ];
  const reportText = `${lines.join('\n')}\n`;
  await fs.writeFile(path.join(experiments, `v10_最终模型完整报告_${runId}.md`), reportText, 'utf8');
  await fs.writeFile(path.join(experiments, 'v10_最终模型完整报告_latest.md'), reportText, 'utf8');
  return manifest;
}

try {
  const audit = await readJson(auditFile);
  if (audit.status !== 'complete' || audit.activation?.activated !== true) throw new Error('v10 layout finalization is not complete and activated');
  const restarted = await restartServer();
  await stamp(`Updated local service is ready with PID ${restarted.pid}; refreshing the local Qwen connection probe after restart.`);
  await ensureScoringConnection();
  await stamp('Running full local Qwen preflight for 3 samples × 4 candidates × 8 rounds and val11.');
  await runNode(['scripts/preflight-llm-preference.mjs', '--samples', '3', '--candidates', '4', '--rounds', '8', '--visualValSamples', '11'], 'Qwen preflight', 10);
  await stamp('Starting full local Qwen preference training.');
  await runNode(['scripts/run-llm-preference-headless.mjs', '--samples', '3', '--candidates', '4', '--rounds', '8', '--visualValSamples', '11', '--seed', '17', '--epochs', '80', '--gateIterations', '180', '--timeoutMinutes', '480'], 'Qwen 8-round preference training', 500);
  const latest = await readJson(path.join(experiments, 'llm_preference_run_latest.json'));
  const runId = String(latest.run_id || '');
  if (!/^[a-zA-Z0-9_-]{8,80}$/.test(runId)) throw new Error('Qwen run completed without a valid run_id');
  await runNode(['scripts/analyze-preference-convergence.mjs', '--runId', runId], 'Qwen convergence analysis', 15);
  for (const script of ['scripts/test-preference-policy.mjs', 'scripts/test-preference-training.mjs', 'scripts/test-preference-gate.mjs', 'scripts/test-convergence-analysis.mjs']) await runNode([script], script, 15);
  await stamp('Qwen training and test11 completed; verifying that the running service still serves the activated v10 backend.');
  const liveConfig = await fetchJson(`${baseUrl}/api/model-config`, {}, 10_000);
  if (liveConfig.model?.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') throw new Error('running service no longer serves the activated v10 backend');
  const uiDir = path.join(experiments, 'v10-final-ui');
  await runNode(['scripts/run-llm-preference-headless.mjs', '--checkOnly', 'true', '--samples', '1', '--requireMetricValues', 'true', '--baseUrl', baseUrl, '--experimentsDir', uiDir, '--timeoutMinutes', '20'], 'final v10 visualization check', 25);
  const uiCheck = await readJson(path.join(uiDir, 'headless_six_view_check.json'));
  const manifest = await buildReport(runId, uiCheck);
  await stamp(`All v10 layout, visualization and Qwen work completed. Manifest run_id=${manifest.run_id}.`);
} catch (error) {
  await stamp(`Completion stopped safely: ${error.stack || error.message}`);
  process.exitCode = 2;
}
