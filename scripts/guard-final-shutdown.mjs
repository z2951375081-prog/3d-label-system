// Independent guard for a finalizer that was started before the strengthened
// completion checks were written. Never requests shutdown without fresh proof.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const exp = path.join(root, 'experiments');
const runId = '333ecca6-9abd-46b0-baa1-f67b6ec9ba7e';
const finalizerPid = 35764;
const oldServerPid = 18204;
const logFile = path.join(exp, 'complete_preference_goal.log');
const auditFile = path.join(exp, 'guard_final_shutdown.log');
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };
const audit = async (message) => fs.appendFile(auditFile, `[${new Date().toISOString()}] ${message}\n`, 'utf8');
const read = async (name) => JSON.parse(await fs.readFile(path.join(exp, name), 'utf8'));
const digest = async (file) => createHash('sha256').update(await fs.readFile(file)).digest('hex').toUpperCase();
const execFileAsync = promisify(execFile);
const baselineHash = 'DA82CDA630E44F26EBBDAB5C6F1F1463A8EE647FEFC95BEFAF9AF1883991424F';
const have = async (name) => fs.stat(path.join(exp, name)).then((s) => s.isFile(), () => false);
async function alignReportWithSelectedReward() {
  const manifestName = 'final_model_manifest_latest.json';
  if (!await have(manifestName)) return;
  const [manifest, selected, active] = await Promise.all([
    read(manifestName), read('preference_test11_report.json'), read('preference_model.json')
  ]);
  const hash = await digest(path.join(exp, 'preference_model.json'));
  if (manifest.run_id !== runId || selected.run_id !== runId || hash !== selected.candidate_model_sha256)
    throw new Error('最终清单或 test11 与 val 所选活动模型不一致');
  if (manifest.final_model?.llm_reward?.sha256 !== hash) {
    manifest.final_model.llm_reward.sha256 = hash;
    manifest.final_model.llm_reward.architecture = active.architecture;
    manifest.final_model.llm_reward.validation_gate = active.validation_gate;
    manifest.final_model.llm_reward.training = {
      run_id_filter: active.training.run_id_filter, examples: active.training.examples,
      epochs: active.training.epochs, learning_rate: active.training.learning_rate, seed: active.training.seed
    };
    manifest.final_model.llm_reward.val_selected_round = selected.selected_round;
    const text = `${JSON.stringify(manifest, null, 2)}\n`;
    await fs.writeFile(path.join(exp, manifestName), text, 'utf8');
    await fs.writeFile(path.join(exp, `final_model_manifest_${runId}.json`), text, 'utf8');
    await audit('最终清单已与真实 val 所选、test11 评估的活动权重同步。');
  }
  for (const name of [
    `最终LLM美学偏好实验报告_${runId}.md`, '最终LLM美学偏好实验报告_latest.md',
    `最终模型完整说明报告_${runId}.md`, '最终模型完整说明报告_latest.md'
  ]) {
    if (!await have(name)) continue;
    const original = await fs.readFile(path.join(exp, name), 'utf8');
    const line = `- 最终推理奖励权重：val11 选择第 ${selected.selected_round} 轮、SHA-256 ${hash}；与 test11 评估权重及活动模型相同，test11 不参与选择。`;
    const corrected = original.replace(/- 活动美学奖励模型 SHA-256：[^\r\n]*/u, `- 活动美学奖励模型 SHA-256：${hash}`);
    if (!corrected.includes(line)) await fs.writeFile(path.join(exp, name), `${corrected.trimEnd()}\n\n${line}\n`, 'utf8');
  }
}
async function verifyCompletion() {
  const required = ['llm_preference_convergence.json', 'preference_test11_report.json',
    'preference_model.json', `llm_preference_checkpoints_${runId}.json`,
    'final_model_manifest_latest.json', `最终模型完整说明报告_${runId}.md`,
    `final_reward_activation_${runId}.json`];
  for (const file of required) if (!(await have(file))) return `缺少 ${file}`;
  const [convergence, test, active, ledger, manifest, activation] = await Promise.all([
    read(required[0]), read(required[1]), read(required[2]), read(required[3]), read(required[4]), read(required[6])
  ]);
  if ([convergence.run_id, test.run_id, active.training?.run_id_filter, manifest.run_id].some((id) => id !== runId)) return '运行 ID 不一致';
  const rounds = convergence.checkpoints?.filter((c) => [0, 1, 2, 4, 8].includes(Number(c.round))) || [];
  if (!convergence.evidence_sufficient || rounds.length !== 5 ||
      !rounds.every((c) => c.status === 'evaluated' && c.checkpoint_verified)) return '五个真实视觉检查点不完整';
  if (!ledger.checkpoints?.some((c) => c.activated && c.validation_status === 'accepted') ||
      active.validation_gate?.status !== 'accepted' || !test.candidate_activated ||
      test.test_samples !== 11 || test.rows?.length !== 11 ||
      !test.rows.every((row) => row.split === 'test')) return '奖励模型尚未通过 val 并在 test11 真实评估';
  const activeHash = await digest(path.join(exp, 'preference_model.json'));
  if (activeHash !== test.candidate_model_sha256 || activation.frozen_model_sha256 !== activeHash ||
      activation.selected_round !== Number(test.selected_round) || activation.test_used_for_selection !== false)
    return 'test11 评估的模型不是当前活动奖励权重';
  if ((await digest(path.join(exp, 'layout_model.json'))) !== baselineHash ||
      manifest.model_1?.sha256 !== baselineHash ||
      manifest.final_model?.llm_reward?.sha256 !== activeHash)
    return '模型哈希与最终清单不匹配';
  const report = await fs.readFile(path.join(exp, required[5]), 'utf8');
  if (!report.includes('<!-- adaptive-directional-and-llm-final-model-v1 -->')) return '模型一/最终模型报告尚未补全';
  const summary = await fs.readFile(path.join(root, 'public', 'quality-all55.json'), 'utf8').then(JSON.parse);
  if (summary.averages?.length !== 4 || !summary.averages.every((item) => item.sample_count === 55)) return '全 55 样本汇总不完整';
  return null;
}
async function qualityApi() {
  const response = await fetch('http://127.0.0.1:5173/api/quality-comparison?category=Chair&sampleId=38635', { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`质量 API ${response.status}`);
  return response.json();
}
async function confirmedOldService() {
  if (process.platform !== 'win32' || !alive(oldServerPid)) return false;
  const script = `$owner = @(Get-NetTCPConnection -State Listen -LocalPort 5173 -ErrorAction SilentlyContinue | Where-Object { $_.LocalAddress -eq '127.0.0.1' -and $_.OwningProcess -eq ${oldServerPid} }); $p = Get-CimInstance Win32_Process -Filter 'ProcessId = ${oldServerPid}'; if ($owner.Count -eq 1 -and $p.Name -eq 'node.exe' -and $p.CommandLine -match '(^|[\\\\ /])server\\.mjs( |$)') { 'VERIFIED' }`;
  const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { windowsHide: true, timeout: 10000 });
  return stdout.trim() === 'VERIFIED';
}
await audit(`开始监视本次运行的收尾器 PID ${finalizerPid}；仅核验本次 run ${runId}。`);
let completed = false;
while (alive(finalizerPid)) {
  const log = await fs.readFile(logFile, 'utf8');
  if (log.includes(`全部完成。最终报告：experiments\\最终LLM美学偏好实验报告_${runId}.md`) ||
      log.includes(`全部完成。最终报告：experiments/最终LLM美学偏好实验报告_${runId}.md`)) { completed = true; break; }
  await pause(3000);
}
if (!completed) {
  const log = await fs.readFile(logFile, 'utf8');
  completed = log.includes(`全部完成。最终报告：experiments\\最终LLM美学偏好实验报告_${runId}.md`) ||
    log.includes(`全部完成。最终报告：experiments/最终LLM美学偏好实验报告_${runId}.md`);
}
if (!completed) {
  await audit('收尾器未报告本次运行全部完成；不触碰系统关机设置。');
  process.exit(0);
}
await audit('检测到旧收尾器报告完成；等待其发出计划关机，然后撤销并独立核验报告及新服务。');
await pause(15000);
if (process.platform === 'win32') {
  const aborted = spawn('shutdown.exe', ['/a'], { windowsHide: true });
  const code = await new Promise((resolve) => aborted.once('close', resolve));
  await audit(`旧收尾器计划关机撤销退出码 ${code}（非 0 表示当时没有待撤销的关机）。`);
}
try {
  const { stdout } = await execFileAsync(process.execPath, ['scripts/activate-selected-reward-checkpoint.mjs', '--runId', runId],
    { cwd: root, windowsHide: true, timeout: 60000 });
  await audit(`val 所选奖励权重核验/激活：${stdout.trim().slice(0, 1200)}`);
} catch (error) {
  await audit(`冻结权重与活动权重无法安全对齐：${error.stderr || error.message}；不关机。`);
  process.exit(1);
}
let why;
for (let index = 0; index < 7200; index += 1) {
  try { await alignReportWithSelectedReward(); why = await verifyCompletion(); } catch (error) { why = error.message; }
  if (!why) break;
  if (index % 120 === 0) await audit(`等待完整证据：${why}`);
  await pause(5000);
}
if (why) { await audit(`最终证据不完整：${why}；不关机。`); process.exit(1); }
let api = await qualityApi().catch(() => null);
if (!api?.all55_averages?.length || !api.llm_available) {
  if (!await confirmedOldService()) { await audit('原服务不再由预期的 PID 监听，不能安全自动重启；不关机。'); process.exit(1); }
  await audit('训练、验证与模型报告均已完成；正在重启旧版服务，以加载全 55 样本和最终模型推理。评分 Key 只在旧进程内存中且后续不再调用。');
  process.kill(oldServerPid);
  for (let index = 0; index < 30 && alive(oldServerPid); index += 1) await pause(1000);
  if (alive(oldServerPid)) { await audit('旧服务没有退出；拒绝启动第二个服务或关机。'); process.exit(1); }
  const out = await fs.open(path.join(exp, 'server.stdout.log'), 'a');
  const err = await fs.open(path.join(exp, 'server.stderr.log'), 'a');
  const server = spawn(process.execPath, ['server.mjs'], { cwd: root, detached: true, windowsHide: true, stdio: ['ignore', out.fd, err.fd] });
  server.unref(); await out.close(); await err.close();
  await audit(`新服务已启动 PID ${server.pid}`);
  for (let index = 0; index < 45; index += 1) {
    api = await qualityApi().catch(() => null);
    if (api?.all55_averages?.length === 4 && api.llm_available) break;
    await pause(1000);
  }
}
if (api?.all55_averages?.length !== 4 || !api.llm_available || await verifyCompletion()) {
  await audit('新服务或最终证据未通过联合核验；不关机。');
  process.exit(1);
}
await audit('模型、报告、全 55 样本与新服务均验证通过，开始 60 秒关机倒计时。');
if (process.platform === 'win32') {
  const shutdown = spawn('shutdown.exe', ['/s', '/t', '60', '/c', 'Verified final LLM layout run and report completed.'], { windowsHide: true });
  const code = await new Promise((resolve) => shutdown.once('close', resolve));
  await audit(`最终关机指令退出码 ${code}`);
  if (code !== 0) process.exitCode = 1;
}
