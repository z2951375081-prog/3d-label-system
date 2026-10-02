import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { evaluateMdpoVal11AlignedGate } from '../lib/mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mdpo = path.join(root, 'experiments', 'mdpo');
const sweep = path.join(mdpo, 'aligned_sweep');
const reports = path.join(mdpo, 'val11_reports_safety_priority_v3_aligned');
const ledgerFile = path.join(mdpo, 'aligned_val11_sweep_ledger.json');
const policy = 'safety_priority_v3_aligned';
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const idFor = ({ learningRate, beta, lambdaMulti, rank }) => ('lr_' + learningRate + '_beta_' + beta + '_multi_' + lambdaMulti + '_rank_' + rank).replaceAll('.', 'p');
const expectedGrid = [5e-5, 1e-4, 2e-4].flatMap((learningRate) => [0.05, 0.10, 0.20].flatMap((beta) => [0.1, 0.3, 0.5].flatMap((lambdaMulti) => [2, 4, 8].map((rank) => ({ learningRate, beta, lambdaMulti, rank })))));
function parseArgs(argv) { const out = { baseUrl: 'http://127.0.0.1:5173', timeoutMinutes: 240, seed: 17017, start: 0, limit: 81 }; for (let i = 0; i < argv.length; i += 1) if (argv[i].startsWith('--')) { const key = argv[i].slice(2); out[key] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true; } return out; }
const options = parseArgs(process.argv.slice(2));
const trainedLedger = JSON.parse(await fs.readFile(path.join(sweep, 'aligned_sweep_ledger.json'), 'utf8'));
const trained = new Map((trainedLedger.configurations || []).filter((row) => row.status === 'trained').map((row) => [row.id, row]));
const missing = expectedGrid.map(idFor).filter((id) => !trained.has(id));
if (missing.length || trained.size !== 81 || trainedLedger.test_used_for_selection !== false) throw new Error('Aligned val11 requires all 81 safety-aligned training candidates');
const activeHash = digest(await fs.readFile(path.join(root, 'experiments', 'layout_model.json')));
await fs.mkdir(reports, { recursive: true });
let ledger = await fs.readFile(ledgerFile, 'utf8').then(JSON.parse).catch(() => ({ version: 'v10_mdpo_aligned_val11_sweep_v1', evaluation_policy: policy, required_configuration_count: 81, configurations: [] }));

async function validateExisting(id, candidateFile, candidateHash) {
  const reportFile = path.join(reports, id + '.json'), fourGroupFile = path.join(reports, id + '.four_group.json');
  const [reportBytes, fourBytes] = await Promise.all([fs.readFile(reportFile), fs.readFile(fourGroupFile)]);
  const report = JSON.parse(reportBytes.toString('utf8')), four = JSON.parse(fourBytes.toString('utf8'));
  if (report.version !== 'v10_mdpo_val11_gate_report_v3' || report.evaluation_policy !== policy || report.candidate_id !== id || report.candidate_model?.file !== path.relative(root, candidateFile).split(path.sep).join('/') || report.candidate_model?.sha256 !== candidateHash || report.test_not_used !== true || report.gate?.policy?.id !== policy || report.gate?.evidence?.training_alignment !== true) throw new Error(id + ' aligned val11 report provenance mismatch');
  const checked = evaluateMdpoVal11AlignedGate({ baseline: report.baseline, candidate: report.candidate, trainingAlignment: report.training_alignment });
  if (JSON.stringify(checked) !== JSON.stringify(report.gate)) throw new Error(id + ' aligned gate is not reproducible');
  for (const role of ['baseline', 'candidate']) if (report[role]?.sample_scores?.length !== 11 || report[role].sample_scores.some((row) => !row.response_id || !Object.values(row.scores || {}).every(Number.isFinite))) throw new Error(id + '/' + role + ' missing aligned val11 responses');
  if (four.version !== 'v10_mdpo_four_group_val11_v3' || four.evaluation_policy !== policy || four.candidate_id !== id || four.candidate_model?.sha256 !== candidateHash || four.test_not_used !== true || four.core_gate?.policy?.id !== policy || four.core_gate?.evidence?.training_alignment !== true) throw new Error(id + ' aligned four-group report mismatch');
  validateMdpoFourGroupSummaries(Object.fromEntries(four.groups.map((row) => [row.group, row])));
  await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: id, fourGroup: four });
  return { reportFile, fourGroupFile, reportSha256: digest(reportBytes), fourGroupSha256: digest(fourBytes), accepted: checked.accepted };
}
async function runHeadless(id, candidateFile, candidateHash) {
  const dir = path.dirname(candidateFile), stdout = await fs.open(path.join(dir, 'aligned-val11.stdout.log'), 'a'), stderr = await fs.open(path.join(dir, 'aligned-val11.stderr.log'), 'a');
  try {
    const args = ['scripts/run-llm-preference-headless.mjs', '--mdpoVal11', 'true', '--candidateFile', candidateFile, '--candidateSha256', candidateHash, '--candidateId', id, '--seed', String(options.seed), '--baseUrl', String(options.baseUrl), '--timeoutMinutes', String(options.timeoutMinutes)];
    const code = await new Promise((resolve, reject) => { const child = spawn(process.execPath, args, { cwd: root, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd] }); child.once('error', reject); child.once('exit', resolve); });
    if (code !== 0) throw new Error(id + ' aligned val11 runner exited ' + code);
  } finally { await stdout.close(); await stderr.close(); }
}
const start = Math.max(0, Number(options.start) || 0), limit = Math.max(1, Math.min(81, Number(options.limit) || 81));
const selected = expectedGrid.slice(start, start + limit);
for (const configuration of selected) {
  const id = idFor(configuration), candidateFile = path.join(sweep, id, 'layout_model_v10_mdpo_candidate.json'), candidateBytes = await fs.readFile(candidateFile), candidateHash = digest(candidateBytes);
  const row = { id, hyperparameters: configuration, candidate_file: path.relative(root, candidateFile).split(path.sep).join('/'), candidate_sha256: candidateHash, started_at: new Date().toISOString(), status: 'running', test_used: false, evaluation_policy: policy };
  ledger.configurations = (ledger.configurations || []).filter((item) => item.id !== id).concat(row); ledger.updated_at = new Date().toISOString(); await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  try {
    const reportState = await Promise.all([fs.access(path.join(reports, id + '.json')).then(() => true, () => false), fs.access(path.join(reports, id + '.four_group.json')).then(() => true, () => false)]);
    const evidence = reportState.every(Boolean) ? await validateExisting(id, candidateFile, candidateHash) : (await runHeadless(id, candidateFile, candidateHash), await validateExisting(id, candidateFile, candidateHash));
    Object.assign(row, { status: 'evaluated', completed_at: new Date().toISOString(), gate_accepted: evidence.accepted, report_file: path.relative(root, evidence.reportFile).split(path.sep).join('/'), report_sha256: evidence.reportSha256, four_group_report_file: path.relative(root, evidence.fourGroupFile).split(path.sep).join('/'), four_group_report_sha256: evidence.fourGroupSha256 });
  } catch (error) { Object.assign(row, { status: 'failed', completed_at: new Date().toISOString(), error: error.message }); await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n'); throw error; }
  ledger.configurations = (ledger.configurations || []).filter((item) => item.id !== id).concat(row); ledger.updated_at = new Date().toISOString(); await fs.writeFile(ledgerFile, JSON.stringify(ledger, null, 2) + '\n');
  console.log(JSON.stringify({ id, status: row.status, gate_accepted: row.gate_accepted, evaluated: ledger.configurations.filter((item) => item.status === 'evaluated').length }));
}
console.log(JSON.stringify({ policy, requested: selected.length, evaluated: ledger.configurations.filter((item) => item.status === 'evaluated').length, ledger: path.relative(root, ledgerFile).split(path.sep).join('/') }, null, 2));
