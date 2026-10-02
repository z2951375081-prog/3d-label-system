import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo'), sweep = path.join(mdpo, 'sweep');
const reports = path.join(mdpo, 'val11_reports_safety_priority_v2'), ledgerFile = path.join(mdpo, 'val11_sweep_ledger.json');

function parseArgs(argv) {
  const options = { baseUrl: 'http://127.0.0.1:5173', timeoutMinutes: 240, seed: 17017, start: 0, limit: 81 };
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--')) continue;
    const key = argv[index].slice(2);
    options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const idFor = ({ learningRate, beta, lambdaMulti, rank }) => `lr_${learningRate}_beta_${beta}_multi_${lambdaMulti}_rank_${rank}`.replaceAll('.', 'p');
const expectedGrid = [5e-5, 1e-4, 2e-4].flatMap((learningRate) => [0.05, 0.10, 0.20].flatMap((beta) => [0.1, 0.3, 0.5].flatMap((lambdaMulti) => [2, 4, 8].map((rank) => ({ learningRate, beta, lambdaMulti, rank }))))) ;

async function validateExistingReport({ id, candidateFile, candidateHash }) {
  const reportFile = path.join(reports, `${id}.json`), fourGroupFile = path.join(reports, `${id}.four_group.json`);
  const [reportBytes, fourGroupBytes] = await Promise.all([fs.readFile(reportFile), fs.readFile(fourGroupFile)]);
  const report = JSON.parse(reportBytes.toString('utf8')), fourGroup = JSON.parse(fourGroupBytes.toString('utf8'));
  const relativeCandidate = path.relative(root, candidateFile).split(path.sep).join('/');
  if (report.version !== 'v10_mdpo_val11_gate_report_v2' || report.evaluation_policy !== 'safety_priority_v2' || report.candidate_id !== id || report.candidate_model?.file !== relativeCandidate
      || report.candidate_model?.sha256 !== candidateHash || report.test_not_used !== true || report.paired_provenance?.paired_samples !== 11
      || report.four_group_protocol?.complete !== true) throw new Error(`${id} val11 report provenance is incomplete or mismatched`);
  const checked = evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate });
  if (JSON.stringify(checked) !== JSON.stringify(report.gate)) throw new Error(`${id} val11 gate is not independently reproducible`);
  for (const role of ['baseline', 'candidate']) {
    const evaluation = report[role];
    if (evaluation?.sample_scores?.length !== 11 || new Set(evaluation.sample_scores.map((row) => row.sample)).size !== 11
        || evaluation.sample_scores.some((row) => !row.response_id || !Object.values(row.scores || {}).every(Number.isFinite))) throw new Error(`${id}/${role} does not contain 11 real seven-dimensional Qwen responses`);
  }
  if (fourGroup.version !== 'v10_mdpo_four_group_val11_v2' || fourGroup.evaluation_policy !== 'safety_priority_v2' || fourGroup.candidate_id !== id || fourGroup.candidate_model?.sha256 !== candidateHash
      || fourGroup.test_not_used !== true || !Array.isArray(fourGroup.groups) || fourGroup.groups.length !== 4) throw new Error(`${id} four-group report is incomplete`);
  validateMdpoFourGroupSummaries(Object.fromEntries(fourGroup.groups.map((row) => [row.group, row])));
  await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: id, fourGroup });
  return { reportFile, fourGroupFile, reportSha256: digest(reportBytes), fourGroupSha256: digest(fourGroupBytes), accepted: checked.accepted };
}

async function reportArtifactState(id) {
  const files = [path.join(reports, `${id}.json`), path.join(reports, `${id}.four_group.json`)];
  const exists = await Promise.all(files.map((file) => fs.stat(file).then(() => true, (error) => {
    if (error?.code === 'ENOENT') return false;
    throw error;
  })));
  if (exists.every(Boolean)) return 'complete';
  if (exists.some(Boolean)) throw new Error(`${id} has a partial immutable val11 report pair; preserve it for audit and repair explicitly before rerunning`);
  return 'missing';
}

async function runHeadless({ id, candidateFile, candidateHash, options }) {
  const directory = path.dirname(candidateFile), stdoutFile = path.join(directory, 'val11.stdout.log'), stderrFile = path.join(directory, 'val11.stderr.log');
  const stdout = await fs.open(stdoutFile, 'a'), stderr = await fs.open(stderrFile, 'a');
  try {
    const args = ['scripts/run-llm-preference-headless.mjs', '--mdpoVal11', 'true', '--candidateFile', candidateFile,
      '--candidateSha256', candidateHash, '--candidateId', id, '--seed', String(options.seed), '--baseUrl', String(options.baseUrl),
      '--timeoutMinutes', String(options.timeoutMinutes)];
    const code = await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, args, { cwd: root, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd] });
      child.once('error', reject);
      child.once('exit', resolve);
    });
    if (code !== 0) throw new Error(`${id} authoritative val11 runner exited ${code}; inspect ${path.relative(root, stderrFile)}`);
  } finally {
    await stdout.close();
    await stderr.close();
  }
}

const options = parseArgs(process.argv.slice(2));
const sweepLedger = await fs.readFile(path.join(sweep, 'sweep_ledger.json'), 'utf8').then(JSON.parse).catch((error) => {
  if (error?.code === 'ENOENT') throw new Error('Authoritative val11 sweep is unavailable until npm run sweep:mdpo completes all 81 training configurations');
  throw error;
});
const trained = new Map((sweepLedger.configurations || []).filter((row) => row.status === 'trained').map((row) => [row.id, row]));
const missing = expectedGrid.map(idFor).filter((id) => !trained.has(id));
if (missing.length || trained.size !== 81 || sweepLedger.required_grid?.total !== 81 || sweepLedger.test_used_for_selection !== false) throw new Error(`Authoritative val11 sweep requires the complete frozen 81-grid training ledger; missing ${missing.length}`);
const activeBytes = await fs.readFile(path.join(root, 'experiments', 'layout_model.json')), activeHash = digest(activeBytes);
const start = Math.max(0, Number(options.start) || 0), limit = Math.max(1, Math.min(81, Number(options.limit) || 81));
const selected = expectedGrid.slice(start, start + limit);
const ledger = await fs.readFile(ledgerFile, 'utf8').then(JSON.parse).catch(() => ({ version: 'v10_mdpo_authoritative_val11_sweep_v1', configurations: [] }));
ledger.fixed_seed = Number(options.seed);
ledger.base_url = String(options.baseUrl);
ledger.required_configuration_count = 81;
ledger.test_used = false;
await fs.mkdir(reports, { recursive: true });

for (const configuration of selected) {
  const id = idFor(configuration), candidateFile = path.join(sweep, id, 'layout_model_v10_mdpo_candidate.json');
  const candidateBytes = await fs.readFile(candidateFile), candidateHash = digest(candidateBytes), candidate = JSON.parse(candidateBytes.toString('utf8'));
  if (candidate.version !== 'layout_model_v10_mdpo_candidate' || candidate.status !== 'diagnostic_only_requires_full_val11_gate'
      || candidate.reference?.sha256 !== activeHash || candidate.architecture?.qwen_inference_input !== false) throw new Error(`${id} candidate/reference/inference contract invalid before val11`);
  const row = { id, hyperparameters: configuration, candidate_file: path.relative(root, candidateFile).split(path.sep).join('/'), candidate_sha256: candidateHash,
    started_at: new Date().toISOString(), status: 'running', test_used: false };
  ledger.configurations = (ledger.configurations || []).filter((item) => item.id !== id).concat(row);
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  try {
    let evidence;
    const artifactState = await reportArtifactState(id);
    if (artifactState === 'complete') {
      evidence = await validateExistingReport({ id, candidateFile, candidateHash });
      row.resumed_from_verified_immutable_report = true;
    } else {
      await runHeadless({ id, candidateFile, candidateHash, options });
      evidence = await validateExistingReport({ id, candidateFile, candidateHash });
      row.resumed_from_verified_immutable_report = false;
    }
    Object.assign(row, { status: 'evaluated', completed_at: new Date().toISOString(), gate_accepted: evidence.accepted,
      report_file: path.relative(root, evidence.reportFile).split(path.sep).join('/'), report_sha256: evidence.reportSha256,
      four_group_report_file: path.relative(root, evidence.fourGroupFile).split(path.sep).join('/'), four_group_report_sha256: evidence.fourGroupSha256 });
  } catch (error) {
    Object.assign(row, { status: 'failed', completed_at: new Date().toISOString(), error: error.message });
    await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
    throw error;
  }
  await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  console.log(JSON.stringify({ id, status: row.status, gate_accepted: row.gate_accepted, completed: ledger.configurations.filter((item) => item.status === 'evaluated').length }, null, 2));
}

console.log(JSON.stringify({ requested: selected.length, evaluated_total: ledger.configurations.filter((item) => item.status === 'evaluated').length,
  accepted_total: ledger.configurations.filter((item) => item.status === 'evaluated' && item.gate_accepted).length,
  ledger: path.relative(root, ledgerFile).split(path.sep).join('/') }, null, 2));
