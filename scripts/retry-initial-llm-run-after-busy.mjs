// One bounded recovery from a transient scorer outage in the first train round.
// Does not restart after any preference pair/checkpoint exists: replaying a
// partially trained run without a complete cursor would duplicate evidence.
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const experiments = path.join(root, 'experiments');
const runId = '333ecca6-9abd-46b0-baa1-f67b6ec9ba7e';
const watchedProcessId = 35764;
const logFile = path.join(experiments, 'complete_preference_goal.log');
const auditFile = path.join(experiments, 'initial_busy_recovery.log');
const stamp = async (text) => fs.appendFile(auditFile, `[${new Date().toISOString()}] ${text}\n`, 'utf8');
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
await stamp(`监视已运行的收尾器 PID ${watchedProcessId}；只允许在首次 train 轮没有偏好对时续跑一次。`);
while (alive(watchedProcessId)) await pause(5000);
await pause(2000);
const log = await fs.readFile(logFile, 'utf8');
const tail = log.slice(-16000);
if (!tail.includes('停止且不关机：') || !/servers are currently busy|try again in a moment/i.test(tail)) {
  await stamp('不是评分服务繁忙导致的终止，或者原运行已成功；不重启。');
  process.exit(0);
}
const preferences = await fs.readFile(path.join(experiments, 'preferences.jsonl'), 'utf8').catch(() => '');
const pairCount = preferences.split(/\r?\n/).filter(Boolean).reduce((count, line) => {
  try { return count + (JSON.parse(line).run_id === runId ? 1 : 0); } catch { return count; }
}, 0);
const ledgerExists = await fs.stat(path.join(experiments, `llm_preference_checkpoints_${runId}.json`)).then(() => true, () => false);
if (pairCount || ledgerExists) {
  await stamp(`存在 ${pairCount} 条偏好或冻结检查点；不能从第一轮重放，停止且不关机。`);
  process.exit(0);
}
const visual = JSON.parse(await fs.readFile(path.join(experiments, `llm_visual_validation_${runId}.json`), 'utf8'));
if (visual.run_id !== runId || visual.rounds?.length !== 1 || visual.rounds[0].round !== 0 || visual.rounds[0].sample_count !== 11 || visual.rounds[0].status !== 'evaluated') {
  await stamp('第 0 轮完整 val11 证据不匹配；停止且不关机。');
  process.exit(0);
}
const configResponse = await fetch('http://127.0.0.1:5173/api/scoring-config', { signal: AbortSignal.timeout(5000) }).catch(() => null);
const config = configResponse?.ok ? await configResponse.json() : null;
if (!config?.configured || !config?.tested) {
  await stamp('连接已失效；停止且不关机，等待用户重新输入有效 Key。');
  process.exit(0);
}
await stamp('首次运行仅因评分服务繁忙停止；等待 5 分钟后，使用当前代码和同一 run_id 复用第 0 轮，只自动重启一次。');
await pause(5 * 60 * 1000);
const confirmResponse = await fetch('http://127.0.0.1:5173/api/scoring-config', { signal: AbortSignal.timeout(5000) }).catch(() => null);
const confirm = confirmResponse?.ok ? await confirmResponse.json() : null;
if (!confirm?.configured || !confirm?.tested) { await stamp('等待结束后连接失效；停止且不关机。'); process.exit(0); }
const out = await fs.open(path.join(experiments, 'busy_recovery_finalizer.stdout.log'), 'a');
const err = await fs.open(path.join(experiments, 'busy_recovery_finalizer.stderr.log'), 'a');
const child = spawn(process.execPath, ['scripts/complete-preference-goal-and-shutdown.mjs', '--resumeRunId', runId], {
  cwd: root, detached: true, windowsHide: true, stdio: ['ignore', out.fd, err.fd]
});
child.unref();
await stamp(`一次性续跑已启动，PID=${child.pid}；不清空既有日志、不重复第 0 轮。`);
await out.close();
await err.close();
