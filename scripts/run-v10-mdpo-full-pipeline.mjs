import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';
import { validateMdpoFinalDatasetAudit } from '../lib/mdpo-final-dataset-audit.mjs';
import { replaceFileWithRetry } from '../lib/atomic-file-replace.mjs';
import { MDPO_ABLATION_IDS } from '../lib/mdpo-ablation-evidence.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo');
const pipelineDir = path.join(mdpo, 'pipeline');
const ledgerFile = path.join(pipelineDir, 'full_pipeline_ledger.json');
const collectorPid = Number(process.argv[process.argv.indexOf('--collectorPid') + 1]);
const rebindFailedCollector = process.argv.includes('--rebindFailedCollector');
const baseUrl = 'http://127.0.0.1:5173';
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const expectedGridIds = [5e-5, 1e-4, 2e-4].flatMap((learningRate) =>
  [0.05, 0.10, 0.20].flatMap((beta) =>
    [0.1, 0.3, 0.5].flatMap((lambdaMulti) =>
      [2, 4, 8].map((rank) => `lr_${learningRate}_beta_${beta}_multi_${lambdaMulti}_rank_${rank}`.replaceAll('.', 'p')))));
const sameSet = (actual, expected) => actual.size === expected.length && expected.every((value) => actual.has(value));

if (!Number.isInteger(collectorPid) || collectorPid <= 0) throw new Error('A positive --collectorPid is required');
await fs.mkdir(pipelineDir, { recursive: true });

const readJson = (file) => fs.readFile(file, 'utf8').then(JSON.parse).catch((error) => {
  if (error?.code === 'ENOENT') return null;
  throw error;
});
const exists = (file) => fs.stat(file).then((stat) => stat.isFile(), (error) => {
  if (error?.code === 'ENOENT') return false;
  throw error;
});
const processAlive = (pid) => {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error?.code === 'ESRCH') return false; throw error; }
};

let ledger = await readJson(ledgerFile) || {
  version: 'v10_mdpo_full_pipeline_ledger_v1', created_at: new Date().toISOString(), collector_pid: collectorPid,
  policy: { train33_only: true, complete_81_grid: true, no_test_for_selection: true, no_force_activation: true,
    activate_only_after_val11_and_nine_ablations: true, all_rejected_skips_test11: true }, stages: []
};
if (ledger.collector_pid !== collectorPid) {
  const wait = ledger.stages?.find((row) => row.id === 'wait_for_train33_collection');
  const onlyFailedCollectionWait = ledger.stages?.length === 1 && wait?.status === 'failed'
    && /^Incomplete MDPO train33:/.test(String(wait.error || ''));
  if (!rebindFailedCollector || !onlyFailedCollectionWait || processAlive(ledger.collector_pid) || !processAlive(collectorPid))
    throw new Error('Pipeline ledger is bound to collector PID ' + ledger.collector_pid + '; failed collection recovery requires a live replacement PID and --rebindFailedCollector');
  ledger.collector_history ||= [];
  ledger.collector_history.push({ pid: ledger.collector_pid, rebound_at: new Date().toISOString(), terminal_stage: wait.status, error: wait.error });
  ledger.collector_pid = collectorPid;
}

async function saveLedger() {
  ledger.updated_at = new Date().toISOString();
  const temporary = path.join(pipelineDir, `.full_pipeline_ledger.${randomUUID()}.tmp`);
  await fs.writeFile(temporary, JSON.stringify(ledger, null, 2) + '\n', { flag: 'wx' });
  await replaceFileWithRetry(temporary, ledgerFile);
}

function stageRow(id) {
  let row = ledger.stages.find((item) => item.id === id);
  if (!row) { row = { id, status: 'pending' }; ledger.stages.push(row); }
  return row;
}

async function datasetState(requireComplete = false) {
  const [bytes, manifest] = await Promise.all([
    fs.readFile(path.join(mdpo, 'train_pairs.json')),
    readJson(path.join(root, 'experiments', 'dataset_manifest.json'))
  ]);
  const dataset = JSON.parse(bytes.toString('utf8'));
  const audit = validateMdpoDataset(dataset, manifest, { requireComplete });
  return { bytes, dataset, audit, sha256: digest(bytes) };
}

async function waitForCollector() {
  const row = stageRow('wait_for_train33_collection');
  if (row.status === 'completed') return;
  if (row.status === 'failed') {
    row.attempts ||= [];
    row.attempts.push({ started_at: row.started_at, failed_at: row.failed_at, error: row.error });
  }
  row.status = 'running'; row.started_at = new Date().toISOString();
  delete row.failed_at; delete row.error; delete row.completed_at; delete row.result;
  await saveLedger();
  try {
    let nextHeartbeat = Date.now() + 60 * 60_000;
    while (processAlive(collectorPid)) {
      if (Date.now() >= nextHeartbeat) {
        const state = await datasetState(false);
        row.last_heartbeat = { checked_at: new Date().toISOString(), collector_alive: true,
          sample_count: state.audit.sample_count, pair_count: state.audit.pair_count, dataset_sha256: state.sha256 };
        await saveLedger();
        process.stdout.write(`${JSON.stringify({ stage: row.id, ...row.last_heartbeat })}\n`);
        nextHeartbeat = Date.now() + 60 * 60_000;
      }
      await sleep(60_000);
    }
    const state = await datasetState(true);
    row.status = 'completed'; row.completed_at = new Date().toISOString();
    row.result = { collector_exit_observed: true, sample_count: state.audit.sample_count,
      pair_count: state.audit.pair_count, dataset_sha256: state.sha256 };
    await saveLedger();
  } catch (error) {
    row.status = 'failed'; row.failed_at = new Date().toISOString(); row.error = error.message; await saveLedger(); throw error;
  }
}

async function ensureScorerReady() {
  const configResponse = await fetch(`${baseUrl}/api/scoring-config`, { signal: AbortSignal.timeout(10_000) });
  if (!configResponse.ok) throw new Error(`Project service scoring config returned ${configResponse.status}`);
  let config = await configResponse.json();
  if (!config.configured) throw new Error('Local Qwen scorer is not configured');
  if (!config.tested) {
    const test = await fetch(`${baseUrl}/api/scoring-test`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}', signal: AbortSignal.timeout(10 * 60_000) });
    if (!test.ok) throw new Error(`Local Qwen six-image self-test failed: ${test.status} ${await test.text()}`);
    config = await fetch(`${baseUrl}/api/scoring-config`, { signal: AbortSignal.timeout(10_000) }).then((response) => response.json());
  }
  if (!config.tested) throw new Error('Local Qwen scorer did not enter tested state');
  return { model: config.model, protocol: config.scoringProtocol, tested_at: config.testedAt };
}

async function runStage({ id, script, args = [], command = null, complete, verifyAfter = complete, qwen = false }) {
  const row = stageRow(id);
  if (await complete()) {
    row.status = 'completed'; row.verified_existing = true; row.completed_at ||= new Date().toISOString();
    await saveLedger(); return;
  }
  if (qwen) row.scorer = await ensureScorerReady();
  row.status = 'running'; row.started_at = new Date().toISOString(); row.verified_existing = false; delete row.error;
  const stdoutFile = path.join(pipelineDir, `${id}.stdout.log`), stderrFile = path.join(pipelineDir, `${id}.stderr.log`);
  const executable = command?.executable || process.execPath;
  const invocationArgs = command?.args || [script, ...args];
  row.command = [executable, ...invocationArgs]; row.stdout = path.relative(root, stdoutFile).split(path.sep).join('/');
  row.stderr = path.relative(root, stderrFile).split(path.sep).join('/'); await saveLedger();
  const stdout = await fs.open(stdoutFile, 'a'), stderr = await fs.open(stderrFile, 'a');
  let child;
  try {
    child = spawn(executable, invocationArgs, { cwd: root, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd] });
    row.pid = child.pid; await saveLedger();
    const heartbeat = setInterval(async () => {
      try { row.last_heartbeat = { checked_at: new Date().toISOString(), pid: child.pid, alive: child.exitCode === null }; await saveLedger(); }
      catch {}
    }, 60 * 60_000);
    heartbeat.unref();
    const code = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', resolve); });
    clearInterval(heartbeat);
    if (code !== 0) throw new Error(`${id} exited with code ${code}`);
    if (!(await verifyAfter())) throw new Error(`${id} exited successfully but its authoritative completion evidence is missing`);
    row.status = 'completed'; row.completed_at = new Date().toISOString(); row.exit_code = code;
    await saveLedger();
  } catch (error) {
    row.status = 'failed'; row.failed_at = new Date().toISOString(); row.error = error.message; await saveLedger(); throw error;
  } finally { await stdout.close(); await stderr.close(); }
}

const completeDatasetAudit = async () => {
  const audit = await readJson(path.join(mdpo, 'dataset_audit_final.json'));
  if (!audit) return false;
  const state = await datasetState(true);
  validateMdpoFinalDatasetAudit({ datasetBytes: state.bytes, audit, referenceSha256: state.dataset.reference_model_sha256 });
  return true;
};
const completeSweep = async () => {
  const value = await readJson(path.join(mdpo, 'sweep', 'sweep_ledger.json'));
  const trained = new Set((value?.configurations || []).filter((row) => row.status === 'trained').map((row) => row.id));
  return value?.required_grid?.total === 81 && value?.test_used_for_selection === false && sameSet(trained, expectedGridIds);
};
const completeValSweep = async () => {
  const value = await readJson(path.join(mdpo, 'val11_sweep_ledger.json'));
  const evaluated = new Set((value?.configurations || []).filter((row) => row.status === 'evaluated').map((row) => row.id));
  return value?.required_configuration_count === 81 && value?.test_used === false && sameSet(evaluated, expectedGridIds);
};
const completeSelection = async () => {
  const value = await readJson(path.join(mdpo, 'hyperparameter_selection.json'));
  return value?.status === 'selected_by_complete_authoritative_val11' && value?.evaluated_configuration_count === 81
    && value?.required_grid?.total === 81 && value?.test_used_for_selection === false && value?.selected?.id
    && typeof value?.deployment_eligible === 'boolean';
};
const completeAblationTrain = async () => {
  const value = await readJson(path.join(mdpo, 'ablations', 'ablation_ledger.json'));
  const trained = new Set((value?.configurations || []).filter((row) => row.status === 'trained').map((row) => row.id));
  return value?.test_used_for_selection === false && sameSet(trained, MDPO_ABLATION_IDS);
};
const completeAblationEval = async () => {
  const value = await readJson(path.join(mdpo, 'ablations', 'ablation_report.json'));
  const evaluated = new Set((value?.ablations || []).map((row) => row.id));
  return value?.status === 'complete_authoritative_val11' && value?.test_used === false
    && sameSet(evaluated, MDPO_ABLATION_IDS) && await exists(path.join(mdpo, 'ablations', 'ablation_report.csv'));
};

await waitForCollector();
await runStage({ id: 'audit_train33', script: 'scripts/audit-v10-mdpo-dataset.mjs', complete: completeDatasetAudit });
await runStage({ id: 'train_81_grid', script: 'scripts/run-v10-mdpo-sweep.mjs', complete: completeSweep });
await runStage({ id: 'evaluate_81_grid_val11', script: 'scripts/run-v10-mdpo-val11-sweep.mjs', complete: completeValSweep, qwen: true });
await runStage({ id: 'select_by_val11', script: 'scripts/select-v10-mdpo-hyperparameters.mjs', complete: completeSelection });
await runStage({ id: 'train_nine_ablations', script: 'scripts/run-v10-mdpo-ablations.mjs', complete: completeAblationTrain });
await runStage({ id: 'evaluate_nine_ablations_val11', script: 'scripts/run-v10-mdpo-ablation-val11.mjs', complete: completeAblationEval, qwen: true });

const selection = await readJson(path.join(mdpo, 'hyperparameter_selection.json'));
if (selection.deployment_eligible === true) {
  await runStage({ id: 'activate_mdpo', script: 'scripts/activate-v10-mdpo.mjs', complete: async () => {
    const [report, active] = await Promise.all([readJson(path.join(mdpo, 'activation_report.json')), readJson(path.join(root, 'experiments', 'layout_model.json'))]);
    return report?.status === 'activated' && active?.version === 'layout_model_v10_mdpo' && active?.validation_gate?.status === 'accepted';
  }});
  await runStage({ id: 'lock_test11', script: 'scripts/lock-v10-mdpo-test11.mjs', complete: async () => (await readJson(path.join(mdpo, 'test11_lock.json')))?.status === 'locked_pending_single_test11_evaluation' });
  await runStage({ id: 'evaluate_single_locked_test11', script: 'scripts/run-llm-preference-headless.mjs', args: ['--mdpoTest11', 'true', '--timeoutMinutes', '720'], qwen: true,
    complete: async () => (await readJson(path.join(mdpo, 'test11_final_report.json')))?.status === 'complete_locked_single_test11' });
} else {
  const row = stageRow('preserve_original_v10_all_rejected');
  const [rejection, active] = await Promise.all([readJson(path.join(mdpo, 'val11_rejection_report.json')), readJson(path.join(root, 'experiments', 'layout_model.json'))]);
  if (!rejection || active?.version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe'
      || await exists(path.join(mdpo, 'activation_report.json')) || await exists(path.join(mdpo, 'test11_final_report.json'))) {
    throw new Error('Rejected val11 outcome did not preserve original v10 or incorrectly produced activation/test11 evidence');
  }
  row.status = 'completed'; row.completed_at = new Date().toISOString(); row.result = { activation_skipped: true, test11_skipped: true }; await saveLedger();
}

await runStage({ id: 'final_browser_audit', script: 'scripts/run-llm-preference-headless.mjs',
  args: ['--checkOnly', 'true', '--finalMdpoAudit', 'true', '--samples', '1', '--requireMetricValues', 'true', '--experimentsDir', 'experiments/mdpo', '--timeoutMinutes', '30'],
  complete: async () => {
    const report = await readJson(path.join(mdpo, 'final_browser_audit.json'));
    return report?.version === 'v10_mdpo_final_browser_audit_v1' && report?.browser_error_free === true
      && report?.browser_audit?.console_errors?.length === 0 && report?.browser_audit?.page_errors?.length === 0
      && await exists(path.join(mdpo, 'final_browser_audit.png'));
  } });
await runStage({ id: 'full_policy_tests', command: { executable: process.env.ComSpec || 'cmd.exe', args: ['/d', '/s', '/c', 'npm run test:mdpo-policy'] },
  complete: async () => stageRow('full_policy_tests').status === 'completed', verifyAfter: async () => true });
await runStage({ id: 'final_readonly_audit', script: 'scripts/audit-v10-mdpo-final.mjs',
  complete: async () => stageRow('final_readonly_audit').status === 'completed', verifyAfter: async () => true });

ledger.status = 'complete'; ledger.completed_at = new Date().toISOString(); await saveLedger();
process.stdout.write(`${JSON.stringify({ status: ledger.status, completed_at: ledger.completed_at, ledger: path.relative(root, ledgerFile).split(path.sep).join('/') })}\n`);
