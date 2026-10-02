import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { writeImmutableFileBundle } from '../lib/immutable-json-bundle.mjs';
import { mdpoFinalPageReady, verifyMdpoFinalBrowserPrerequisites } from '../lib/mdpo-final-browser-gate.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experimentsDir = path.join(root, 'experiments');

function parseArgs(argv) {
  const options = { baseUrl: 'http://127.0.0.1:5173', experimentsDir: path.join(root, 'experiments'), candidates: 4, rounds: 1, samples: 1, visualValSamples: 11, seed: 17, epochs: 80, gateIterations: 180, gateLimit: 0, timeoutMinutes: 60 };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

function edgeExecutable() {
  const candidates = process.platform === 'win32'
    ? ['C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', 'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe']
    : ['/usr/bin/microsoft-edge', '/usr/bin/microsoft-edge-stable', '/usr/bin/chromium', '/usr/bin/google-chrome'];
  return candidates;
}

async function firstExisting(files) {
  for (const file of files) try { await fs.access(file); return file; } catch {}
  throw new Error('未找到 Microsoft Edge/Chromium，无法执行带文字 WebGL 六图的无头偏好轮');
}

async function waitForFile(file, timeoutMs = 15000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try { return await fs.readFile(file, 'utf8'); } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`等待浏览器调试端口超时：${file}`);
}

async function waitForJson(url, timeoutMs = 15000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try { const response = await fetch(url); if (response.ok) return response.json(); } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`等待浏览器调试接口超时：${url}`);
}

async function connectCdp(url) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', () => reject(new Error('浏览器 CDP WebSocket 连接失败')), { once: true }); });
  let nextId = 1;
  const pending = new Map();
  const eventListeners = new Set();
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(String(event.data));
    if (!message.id) { for (const listener of eventListeners) listener(message); return; }
    if (!pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message || 'CDP command failed'));
    else resolve(message.result);
  });
  socket.addEventListener('close', () => { for (const { reject } of pending.values()) reject(new Error('浏览器 CDP 连接已关闭')); pending.clear(); });
  return {
    send(method, params = {}) {
      const id = nextId++;
      const promise = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
      socket.send(JSON.stringify({ id, method, params }));
      return promise;
    },
    onEvent(listener) { eventListeners.add(listener); return () => eventListeners.delete(listener); },
    close() { socket.close(); }
  };
}

async function evaluate(cdp, expression, awaitPromise = false) {
  const result = await cdp.send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true, userGesture: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text || '浏览器执行失败');
  return result.result?.value;
}

async function waitForAutomation(cdp, timeoutMs = 120000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try { if (await evaluate(cdp, 'Boolean(window.__labelStudioAutomation?.ready())')) return; } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error('网页数据和自动化接口未在规定时间内就绪');
}

async function withTimeout(promise, timeoutMs, fallback = null) {
  if (!promise) return fallback;
  let timer;
  try {
    return await Promise.race([promise, new Promise((resolve) => { timer = setTimeout(() => resolve(fallback), timeoutMs); })]);
  } finally {
    clearTimeout(timer);
  }
}

async function removeTemporaryBrowserDir(directory) {
  let lastError;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try { await fs.rm(directory, { recursive: true, force: true }); return; }
    catch (error) { lastError = error; await new Promise((resolve) => setTimeout(resolve, 250)); }
  }
  throw lastError;
}

function stratifiedSamples(samples, count) {
  const groups = new Map();
  for (const sample of samples) {
    if (!groups.has(sample.category)) groups.set(sample.category, []);
    groups.get(sample.category).push(sample);
  }
  const output = [];
  for (let round = 0; output.length < count; round += 1) {
    let added = false;
    for (const rows of groups.values()) {
      if (rows[round]) { output.push(rows[round]); added = true; if (output.length >= count) break; }
    }
    if (!added) break;
  }
  return output;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const checkOnly = options.checkOnly === true || options.checkOnly === 'true';
  const finalMdpoAudit = options.finalMdpoAudit === true || options.finalMdpoAudit === 'true';
  if (finalMdpoAudit && !checkOnly) throw new Error('Final MDPO browser audit requires read-only --checkOnly true');
  const mdpoMode = options.mdpo === true || String(options.mdpo).toLowerCase() === 'true';
  const mdpoVal11Mode = options.mdpoVal11 === true || String(options.mdpoVal11).toLowerCase() === 'true';
  const mdpoTest11Mode = options.mdpoTest11 === true || String(options.mdpoTest11).toLowerCase() === 'true';
  if ([mdpoMode, mdpoVal11Mode, mdpoTest11Mode].filter(Boolean).length > 1) throw new Error('Choose exactly one MDPO train, val11, or locked test11 mode');
  const baseUrl = String(options.baseUrl).replace(/\/+$/, '');
  const runExperimentsDir = path.resolve(String(options.experimentsDir));
  if (finalMdpoAudit && runExperimentsDir !== path.join(experimentsDir, 'mdpo')) throw new Error('Final MDPO browser audit artifacts must remain within experiments/mdpo');
  await fs.mkdir(runExperimentsDir, { recursive: true });
  const scoring = await fetch(`${baseUrl}/api/scoring-config`).then(async (response) => { if (!response.ok) throw new Error(`本地服务不可用 (${response.status})`); return response.json(); });
  if (!checkOnly && !scoring.configured) throw new Error('本地 Qwen 视觉评分尚未配置：请先启动 Ollama');
  if (!checkOnly && !scoring.tested) throw new Error('本地 Qwen 视觉评分尚未通过“测试六图与 JSON”');
  const manifest = await fetch(`${baseUrl}/api/dataset-split`).then((response) => response.json());
  const trainSamples = manifest.samples.filter((item) => item.split === 'train');
  const valSamples = manifest.samples.filter((item) => item.split === 'val');
  const testSamples = manifest.samples.filter((item) => item.split === 'test');
  const selectionPool = mdpoTest11Mode ? testSamples : mdpoVal11Mode ? valSamples : trainSamples;
  const selected = options.category && options.sampleId
    ? selectionPool.find((item) => item.category === options.category && String(item.sample_id) === String(options.sampleId))
    : selectionPool[0];
  if (!selected) throw new Error(`找不到指定的 ${mdpoTest11Mode ? 'test' : mdpoVal11Mode ? 'val' : 'train'} 样本`);
  const sampleCount = mdpoVal11Mode || mdpoTest11Mode ? 11 : Math.max(1, Math.min(trainSamples.length, Number(options.samples) || 1));
  const startIndex = Math.max(0, Number(options.start || 0));
  const selectedSamples = mdpoTest11Mode ? stratifiedSamples(testSamples, 11) : mdpoVal11Mode ? stratifiedSamples(valSamples, 11)
    : options.category && options.sampleId ? [selected] : stratifiedSamples(trainSamples, Math.min(trainSamples.length, startIndex + sampleCount)).slice(startIndex, startIndex + sampleCount);
  if (mdpoVal11Mode && selectedSamples.length !== 11) throw new Error(`MDPO val11 requires exactly 11 manifest validation samples, found ${selectedSamples.length}`);
  if (mdpoTest11Mode && selectedSamples.length !== 11) throw new Error(`MDPO test11 requires exactly 11 manifest test samples, found ${selectedSamples.length}`);
  let mdpoVal11Input = null;
  if (mdpoVal11Mode) {
    const candidateFile = path.resolve(String(options.candidateFile || ''));
    const mdpoPrefix = path.resolve(runExperimentsDir, 'mdpo') + path.sep;
    if (!candidateFile.startsWith(mdpoPrefix) || path.extname(candidateFile).toLowerCase() !== '.json') throw new Error('MDPO val11 candidateFile must be a JSON file inside experiments/mdpo');
    const candidateBytes = await fs.readFile(candidateFile);
    const candidate = JSON.parse(candidateBytes.toString('utf8'));
    const candidateSha256 = createHash('sha256').update(candidateBytes).digest('hex');
    if (candidate.version !== 'layout_model_v10_mdpo_candidate' || candidate.status !== 'diagnostic_only_requires_full_val11_gate') throw new Error('MDPO val11 requires a frozen diagnostic candidate model');
    if (options.candidateSha256 && String(options.candidateSha256).toLowerCase() !== candidateSha256) throw new Error('MDPO val11 candidate SHA-256 argument mismatch');
    const candidateId = String(options.candidateId || `mdpo_${candidateSha256.slice(0, 20)}`);
    if (!/^[a-zA-Z0-9_-]{8,100}$/.test(candidateId)) throw new Error('Invalid MDPO val11 candidateId');
    mdpoVal11Input = { candidateFile: path.relative(root, candidateFile).split(path.sep).join('/'), candidateSha256, candidateId, alignedSafety: candidate.mdpo?.safety_alignment?.protocol === 'safety_priority_v3_aligned',
      samples: selectedSamples.map((item) => ({ category: item.category, sampleId: item.sample_id })), seed: Number(options.seed) };
  }
  const edge = await firstExisting(edgeExecutable());
  // Each runner owns only the directory it creates below. Another live
  // collection/evaluation process may be using older .llm-headless-* paths.
  const temporary = await fs.mkdtemp(path.join(runExperimentsDir, '.llm-headless-'));
  const browser = spawn(edge, [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${temporary}`, '--no-first-run', '--disable-extensions',
    '--enable-webgl', '--ignore-gpu-blocklist', '--use-angle=swiftshader', '--window-size=1600,1400', baseUrl
  ], { windowsHide: true, stdio: 'ignore' });
  let cdp;
  let statusTimer;
  try {
    const activePort = await waitForFile(path.join(temporary, 'DevToolsActivePort'));
    const port = Number(activePort.split(/\r?\n/)[0]);
    const targets = await waitForJson(`http://127.0.0.1:${port}/json/list`);
    const page = targets.find((item) => item.type === 'page' && item.url.startsWith(baseUrl)) || targets.find((item) => item.type === 'page');
    if (!page?.webSocketDebuggerUrl) throw new Error('未找到工作台浏览器页面');
    cdp = await connectCdp(page.webSocketDebuggerUrl);
    const browserErrors = { console_errors: [], page_errors: [] };
    cdp.onEvent((message) => {
      if (message.method === 'Runtime.consoleAPICalled' && message.params?.type === 'error') browserErrors.console_errors.push({
        text: (message.params.args || []).map((arg) => String(arg.value ?? arg.description ?? '')).join(' '),
        url: message.params.stackTrace?.callFrames?.[0]?.url || null
      });
      if (message.method === 'Runtime.exceptionThrown') browserErrors.page_errors.push({
        text: message.params.exceptionDetails?.exception?.description || message.params.exceptionDetails?.text || 'Unknown page error',
        url: message.params.exceptionDetails?.url || null
      });
    });
    await cdp.send('Runtime.enable');
    await cdp.send('Page.enable');
    await waitForAutomation(cdp);
    const input = { category: selected.category, sampleId: selected.sample_id, samples: selectedSamples.map((item) => ({ category: item.category, sampleId: item.sample_id })), candidates: Number(options.candidates), pairCount: mdpoMode ? Math.max(6, Math.min(12, Number(options.pairCount) || 6)) : undefined, continueOnInsufficientPairs: mdpoMode && selectedSamples.length > 1, rounds: Number(options.rounds), visualValSamples: Number(options.visualValSamples), seed: Number(options.seed), epochs: Number(options.epochs), gateIterations: Number(options.gateIterations), gateLimit: Number(options.gateLimit), runId: options.runId || null, skipBaseline: options.skipBaseline === true || String(options.skipBaseline).toLowerCase() === 'true' };
    if (options.diagnoseScores === true || String(options.diagnoseScores).toLowerCase() === 'true') {
      const result = await evaluate(cdp, `(async () => window.__labelStudioAutomation.diagnoseQwenScoreResolution(${JSON.stringify(input)}))()`, true);
      await fs.writeFile(path.join(runExperimentsDir, 'qwen_score_resolution_diagnostic.json'), JSON.stringify({ generated_at: new Date().toISOString(), result }, null, 2) + '\n', 'utf8');
      console.log(JSON.stringify(result, null, 2));
      if (!result?.ok || !result.non_tied) throw new Error('Four-candidate visual diagnostic remains tied; no preference training started');
      const candidates = Array.isArray(result.scored_candidates) ? result.scored_candidates : [];
      if (candidates.length !== Number(options.candidates)) throw new Error(`Expected ${Number(options.candidates)} safe candidates, received ${candidates.length}; no preference training started`);
      if (candidates.some((candidate) => candidate.leader_crossings === null || candidate.leader_crossings === undefined
        || candidate.worst_view_leader_crossing_count === null || candidate.worst_view_leader_crossing_count === undefined
        || Number(candidate.leader_crossings) !== 0 || Number(candidate.worst_view_leader_crossing_count) !== 0)) {
        throw new Error('A diagnostic candidate is missing exact zero-crossing evidence; no preference training started');
      }
      return;
    }
    if (checkOnly) {
      const checks = [];
      for (const sample of selectedSamples) {
        const selectedResult = await evaluate(cdp, `(async () => window.__labelStudioAutomation.selectTrainSample(${JSON.stringify(sample.category)}, ${JSON.stringify(String(sample.sample_id))}))()`, true);
        const visualSummary = await evaluate(cdp, 'window.__labelStudioAutomation.captureVisualSummary()');
        const v10Visualization = await evaluate(cdp, 'window.__labelStudioAutomation.v10VisualizationSummary()');
        if (Object.values(visualSummary.images || {}).length !== 6 || Object.values(visualSummary.images || {}).some((item) => !item.valid || item.bytes < 1000)) throw new Error(`${sample.category}/${sample.sample_id} 未生成完整的六张带文字视图`);
        if (!v10Visualization?.requirements?.anchor_label_18d || !v10Visualization?.requirements?.label_label_10d || !v10Visualization?.requirements?.local_decode_6d || !v10Visualization?.requirements?.worst_view_cvar || !v10Visualization?.requirements?.safe_qwen_only || !v10Visualization?.requirements?.all_metric_cards_present) throw new Error(`v10 架构或多视角指标可视化不完整：${JSON.stringify(v10Visualization)}`);
        const requireMetricValues = options.requireMetricValues === true || String(options.requireMetricValues).toLowerCase() === 'true';
        if (requireMetricValues && Object.values(v10Visualization.metric_cards || {}).some((item) => !item?.value || item.value === '—')) throw new Error(`v10 指标卡没有全部显示真实数值：${JSON.stringify(v10Visualization.metric_cards)}`);
        checks.push({ selected: selectedResult, visuals: visualSummary, v10_visualization: v10Visualization });
      }
      let finalStatus = null;
      let finalVerified = null;
      if (finalMdpoAudit) {
        const response = await fetch(`${baseUrl}/api/mdpo-experiment`);
        if (!response.ok) throw new Error(`MDPO final experiment API unavailable: HTTP ${response.status}`);
        finalStatus = await response.json();
        finalVerified = verifyMdpoFinalBrowserPrerequisites(finalStatus);
        const ready = await (async () => {
          const deadline = Date.now() + 30000;
          while (Date.now() < deadline) {
            const value = await evaluate(cdp, `({ dataset: document.getElementById('mdpoDatasetCount')?.textContent, sweep: document.getElementById('mdpoSweepStatus')?.textContent, ablations: document.getElementById('mdpoAblationStatus')?.textContent, gate: document.getElementById('mdpoGateStatus')?.textContent, test: document.getElementById('mdpoTestStatus')?.textContent, train_metrics: document.getElementById('mdpoTrainMetricRows')?.textContent, weight_evidence: document.getElementById('mdpoWeightEvidence')?.textContent })`);
            if (mdpoFinalPageReady(value, finalVerified)) return value;
            await new Promise((resolve) => setTimeout(resolve, 500));
          }
          throw new Error('Final MDPO page did not render the verified experiment/activation results in 30 seconds');
        })();
        finalStatus = { api: finalStatus, page: ready };
      }
      const screenshot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, fromSurface: true });
      const screenshotFile = path.join(runExperimentsDir, finalMdpoAudit ? 'final_browser_audit.png' : 'headless_six_view_check.png');
      const screenshotBytes = Buffer.from(screenshot.data, 'base64');
      const check = { version: finalMdpoAudit ? 'v10_mdpo_final_browser_audit_v1' : 'headless_six_view_check_v3_browser_audit', generated_at: new Date().toISOString(), sample_count: checks.length, checks,
        browser_audit: browserErrors, browser_error_free: browserErrors.console_errors.length === 0 && browserErrors.page_errors.length === 0,
        local_requests: 0, screenshot: path.relative(root, screenshotFile).split(path.sep).join('/'),
        ...(finalMdpoAudit ? { screenshot_sha256: createHash('sha256').update(screenshotBytes).digest('hex'),
          active_model_sha256: finalStatus.api.active_model.sha256,
          historical_reward_sha256: finalStatus.api.historical_reward_model.sha256,
          selected_candidate_id: finalStatus.api.hyperparameter_selection.selected.id,
          outcome: finalVerified.outcome,
          page_status: finalStatus.page } : {}) };
      if (!check.browser_error_free) throw new Error(`Browser console/page audit found errors: ${JSON.stringify(browserErrors)}`);
      if (finalMdpoAudit) await writeImmutableFileBundle([
        { file: screenshotFile, bytes: screenshotBytes },
        { file: path.join(runExperimentsDir, 'final_browser_audit.json'), text: JSON.stringify(check, null, 2) + '\n' }
      ]);
      else {
        await fs.writeFile(screenshotFile, screenshotBytes);
        await fs.writeFile(path.join(runExperimentsDir, 'headless_six_view_check.json'), JSON.stringify(check, null, 2) + '\n', 'utf8');
      }
      console.log(JSON.stringify({ check_only: true, ...check }, null, 2));
      return;
    }
    if (mdpoTest11Mode) {
      console.log('开始锁定 v10-MDPO test11：11 个 test 样本 × 四组 × 六图；结果不参与训练、选模或激活。');
      const runPromise = evaluate(cdp, '(async () => window.__labelStudioAutomation.runMDPOTest11())()', true);
      statusTimer = setInterval(async () => {
        try { const status = await evaluate(cdp, 'window.__labelStudioAutomation.status()'); console.log(`[状态] ${status?.llm || status?.experiment || 'test11 运行中'}`); } catch {}
      }, 60 * 60 * 1000);
      const timeoutMs = Math.max(1, Number(options.timeoutMinutes)) * 60 * 1000;
      let deadline;
      const result = await Promise.race([runPromise, new Promise((_, reject) => {
        deadline = setTimeout(() => reject(new Error(`MDPO test11 超过 ${options.timeoutMinutes} 分钟`)), timeoutMs);
      })]).finally(() => clearTimeout(deadline));
      clearInterval(statusTimer);
      if (!result?.ok) throw new Error(result?.error || 'MDPO test11 未成功完成');
      const reportFile = path.join(runExperimentsDir, 'mdpo', 'test11_final_report.json');
      const report = JSON.parse(await fs.readFile(reportFile, 'utf8'));
      if (report.status !== 'complete_locked_single_test11' || report.provenance?.record_count !== 44
          || report.test_used_for_training !== false || report.test_used_for_selection !== false) throw new Error('MDPO test11 persisted final report provenance mismatch');
      console.log(JSON.stringify({ result: { ...result, report: undefined }, report: path.relative(root, reportFile).split(path.sep).join('/'),
        test_run_id: report.test_run_id, core_no_rerank_delta: report.core_no_rerank_delta }, null, 2));
      return;
    }
    if (mdpoVal11Mode) {
      console.log(`开始 v10-MDPO authoritative val11：候选=${mdpoVal11Input.candidateId}，11 个 val 样本 × baseline/candidate × 六图，不运行 test11。`);
      const runPromise = evaluate(cdp, `(async () => window.__labelStudioAutomation.runMDPOVal11(${JSON.stringify(mdpoVal11Input)}))()`, true);
      statusTimer = setInterval(async () => {
        try { const status = await evaluate(cdp, 'window.__labelStudioAutomation.status()'); console.log(`[状态] ${status?.llm || status?.experiment || 'val11 运行中'}`); } catch {}
      }, 60 * 60 * 1000);
      const timeoutMs = Math.max(1, Number(options.timeoutMinutes)) * 60 * 1000;
      let deadline;
      const result = await Promise.race([runPromise, new Promise((_, reject) => {
        deadline = setTimeout(() => reject(new Error(`MDPO val11 超过 ${options.timeoutMinutes} 分钟`)), timeoutMs);
      })]).finally(() => clearTimeout(deadline));
      clearInterval(statusTimer);
      if (!result?.ok) throw new Error(result?.error || 'MDPO val11 未成功完成');
      const reportDirectory = mdpoVal11Input.alignedSafety ? 'val11_reports_safety_priority_v3_aligned' : 'val11_reports_safety_priority_v2';
      const reportFile = path.join(runExperimentsDir, 'mdpo', reportDirectory, mdpoVal11Input.candidateId + '.json');
      const report = JSON.parse(await fs.readFile(reportFile, 'utf8'));
      if (report.candidate_model?.sha256 !== mdpoVal11Input.candidateSha256 || report.test_not_used !== true || Boolean(report.evaluation_policy === 'safety_priority_v3_aligned') !== Boolean(mdpoVal11Input.alignedSafety)) throw new Error('MDPO val11 persisted report provenance mismatch');
      console.log(JSON.stringify({ result, report: path.relative(root, reportFile).split(path.sep).join('/'), gate: report.gate }, null, 2));
      return;
    }
    console.log(mdpoMode ? `开始 v10-MDPO train33 教师数据采集：train 样本=${input.samples.length}，每样本目标=${input.pairCount} 对且至少 8 个安全候选，不运行历史奖励模型。` : `开始真实 LLM 偏好轮：train 样本=${input.samples.length}，候选/样本/轮=${input.candidates}，轮数=${input.rounds}，val 视觉样本=${input.visualValSamples}，预期本地六图请求=${input.samples.length * input.candidates * input.rounds + input.visualValSamples * (input.rounds + 1)}`);
    const automation = mdpoMode ? 'runMDPOTrainDataset' : 'runLLMPreferenceDataset';
    const runPromise = evaluate(cdp, `(async () => window.__labelStudioAutomation.${automation}(${JSON.stringify(input)}))()`, true);
    statusTimer = setInterval(async () => {
      try { const status = await evaluate(cdp, 'window.__labelStudioAutomation.status()'); console.log(`[状态] ${status?.llm || status?.experiment || '运行中'}`); } catch {}
    }, 60 * 60 * 1000);
    const timeoutMs = Math.max(1, Number(options.timeoutMinutes)) * 60 * 1000;
    let deadline;
    let result;
    try {
      result = await Promise.race([runPromise, new Promise((_, reject) => {
        deadline = setTimeout(() => reject(new Error(`LLM 偏好轮超过 ${options.timeoutMinutes} 分钟`)), timeoutMs);
      })]);
    } finally {
      clearTimeout(deadline);
    }
    clearInterval(statusTimer);
    if (!result?.ok && !(mdpoMode && result?.partial)) throw new Error(result?.error || 'LLM 偏好轮未成功完成');
    if (mdpoMode) {
      const pairsFile = path.join(runExperimentsDir, 'mdpo', 'train_pairs.json');
      const dataset = JSON.parse(await fs.readFile(pairsFile, 'utf8'));
      if (dataset.reference_model_sha256 !== result.reference_model_sha256 || dataset.pairs.length < result.pair_count) throw new Error('MDPO persisted train data/active reference hash mismatch');
      console.log(JSON.stringify({ result, dataset: path.relative(root, pairsFile).split(path.sep).join('/'), total_pairs: dataset.pairs.length }, null, 2));
      if (result.partial) { console.error(`MDPO teacher collection incomplete: ${result.insufficient_samples.length} train samples still lack eight informative safety-compatible candidates`); process.exitCode = 2; }
      return;
    }
    const latest = JSON.parse(await fs.readFile(path.join(runExperimentsDir, 'llm_preference_run_latest.json'), 'utf8'));
    const preferenceLines = await fs.readFile(path.join(runExperimentsDir, 'preferences.jsonl'), 'utf8').then((text) => text.split(/\r?\n/).filter(Boolean), () => []);
    const runTrainPairCount = preferenceLines.reduce((count, line) => {
      try { const record = JSON.parse(line); return count + Number(record.run_id === result.runId && record.type === 'llm_pairwise_preference' && record.split === 'train'); }
      catch { return count; }
    }, 0);
    if (latest.run_id !== result.runId || latest.evidence?.pair_count !== runTrainPairCount || runTrainPairCount < result.savedPairs) throw new Error('Run report and persisted preference evidence do not match');
    console.log(JSON.stringify({ result, report: path.relative(root, path.join(runExperimentsDir, 'llm_preference_run_latest.json')).split(path.sep).join('/'), test11: latest.test11?.summary || null }, null, 2));
  } finally {
    if (statusTimer) clearInterval(statusTimer);
    try { await withTimeout(cdp?.send('Browser.close'), 3000); } catch {}
    try { cdp?.close(); } catch {}
    if (browser.exitCode === null) { try { browser.kill(); } catch {} }
    const resolved = path.resolve(temporary), prefix = path.resolve(runExperimentsDir) + path.sep;
    if (!resolved.startsWith(prefix) || !path.basename(resolved).startsWith('.llm-headless-')) throw new Error('拒绝清理不安全的浏览器临时目录');
    await removeTemporaryBrowserDir(resolved);
  }
}

main().catch((error) => { console.error(error.message || error); process.exitCode = 1; });
