import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { evaluateMdpoVal11AlignedGate, evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';
import { writeImmutableFileBundle } from '../lib/immutable-json-bundle.mjs';
import { MDPO_ABLATION_IDS } from '../lib/mdpo-ablation-evidence.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), mdpo = path.join(root, 'experiments', 'mdpo');
const args = Object.fromEntries(process.argv.slice(2).reduce((rows, item, index, all) => item.startsWith('--') ? [...rows, [item.slice(2), all[index + 1] && !all[index + 1].startsWith('--') ? all[index + 1] : true]] : rows, []));
const aligned = args.aligned === true || String(args.aligned).toLowerCase() === 'true';
const ablations = path.join(mdpo, aligned ? 'aligned_ablations' : 'ablations'), reports = path.join(mdpo, aligned ? 'val11_reports_safety_priority_v3_aligned' : 'val11_reports_safety_priority_v2');
const ledgerFile = path.join(ablations, 'ablation_ledger.json'), outputFile = path.join(ablations, aligned ? 'aligned_ablation_report.json' : 'ablation_report.json'), csvFile = path.join(ablations, aligned ? 'aligned_ablation_report.csv' : 'ablation_report.csv');
const required = MDPO_ABLATION_IDS;

function parseArgs(argv) {
  const options = { baseUrl: 'http://127.0.0.1:5173', timeoutMinutes: 240, seed: 17017 };
  for (let index = 0; index < argv.length; index += 1) if (argv[index].startsWith('--')) {
    const key = argv[index].slice(2); options[key] = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[++index] : true;
  }
  return options;
}
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function artifactState(candidateId) {
  const files = [path.join(reports, `${candidateId}.json`), path.join(reports, `${candidateId}.four_group.json`)];
  const state = await Promise.all(files.map((file) => fs.stat(file).then(() => true, (error) => { if (error?.code === 'ENOENT') return false; throw error; })));
  if (state.every(Boolean)) return 'complete';
  if (state.some(Boolean)) throw new Error(`${candidateId} has a partial immutable val11 report pair`);
  return 'missing';
}

async function validateReport(candidateId, candidateFile, candidateHash) {
  const reportFile = path.join(reports, `${candidateId}.json`), fourGroupFile = path.join(reports, `${candidateId}.four_group.json`);
  const [reportBytes, fourBytes] = await Promise.all([fs.readFile(reportFile), fs.readFile(fourGroupFile)]);
  const report = JSON.parse(reportBytes.toString('utf8')), four = JSON.parse(fourBytes.toString('utf8'));
  if ((aligned && (report.version !== 'v10_mdpo_val11_gate_report_v3' || report.evaluation_policy !== 'safety_priority_v3_aligned')) || (!aligned && (report.version !== 'v10_mdpo_val11_gate_report_v2' || report.evaluation_policy !== 'safety_priority_v2')) || report.candidate_id !== candidateId
      || report.candidate_model?.file !== path.relative(root, candidateFile).split(path.sep).join('/') || report.candidate_model?.sha256 !== candidateHash
      || report.paired_provenance?.paired_samples !== 11 || report.test_not_used !== true) throw new Error(`${candidateId} ablation val11 provenance mismatch`);
  const checked = aligned ? evaluateMdpoVal11AlignedGate({ baseline: report.baseline, candidate: report.candidate, trainingAlignment: report.training_alignment }) : evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate });
  if (JSON.stringify(checked) !== JSON.stringify(report.gate)) throw new Error(`${candidateId} ablation gate is not reproducible`);
  if ((aligned && (four.version !== 'v10_mdpo_four_group_val11_v3' || four.evaluation_policy !== 'safety_priority_v3_aligned')) || (!aligned && (four.version !== 'v10_mdpo_four_group_val11_v2' || four.evaluation_policy !== 'safety_priority_v2')) || four.candidate_id !== candidateId || four.candidate_model?.sha256 !== candidateHash
      || four.test_not_used !== true || !Array.isArray(four.groups) || four.groups.length !== 4) throw new Error(`${candidateId} ablation four-group report invalid`);
  const protocol = validateMdpoFourGroupSummaries(Object.fromEntries(four.groups.map((row) => [row.group, row])));
  await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId, fourGroup: four });
  if (JSON.stringify(protocol) !== JSON.stringify(report.four_group_protocol) || JSON.stringify(four.core_gate) !== JSON.stringify(report.gate)) throw new Error(`${candidateId} ablation four-group evidence mismatch`);
  return { report, reportFile, fourGroupFile, reportSha256: digest(reportBytes), fourGroupSha256: digest(fourBytes) };
}

async function runHeadless(candidateId, candidateFile, candidateHash, options) {
  const directory = path.dirname(candidateFile), stdout = await fs.open(path.join(directory, 'val11.stdout.log'), 'a'), stderr = await fs.open(path.join(directory, 'val11.stderr.log'), 'a');
  try {
    const args = ['scripts/run-llm-preference-headless.mjs', '--mdpoVal11', 'true', '--candidateFile', candidateFile, '--candidateSha256', candidateHash,
      '--candidateId', candidateId, '--seed', String(options.seed), '--baseUrl', String(options.baseUrl), '--timeoutMinutes', String(options.timeoutMinutes)];
    const code = await new Promise((resolve, reject) => { const child = spawn(process.execPath, args, { cwd: root, windowsHide: true, stdio: ['ignore', stdout.fd, stderr.fd] }); child.once('error', reject); child.once('exit', resolve); });
    if (code !== 0) throw new Error(`${candidateId} val11 runner exited ${code}`);
  } finally { await stdout.close(); await stderr.close(); }
}

const options = { ...parseArgs(process.argv.slice(2)), ...args };
const [ledger, selection, activeBytes] = await Promise.all([
  fs.readFile(ledgerFile, 'utf8').then(JSON.parse).catch((error) => { if (error?.code === 'ENOENT') throw new Error('Run aligned ablation training before authoritative ablation val11'); throw error; }),
  fs.readFile(path.join(mdpo, aligned ? 'aligned_hyperparameter_selection.json' : 'hyperparameter_selection.json'), 'utf8').then(JSON.parse), fs.readFile(path.join(root, 'experiments', 'layout_model.json'))
]);
if ((aligned && selection.status !== 'selected_by_complete_aligned_val11') || (!aligned && selection.status !== 'selected_by_complete_authoritative_val11') || selection.test_used_for_selection !== false || selection.evaluated_configuration_count !== 81) throw new Error('Ablation val11 requires complete 81-grid no-test selection');
const trained = new Map((ledger.configurations || []).filter((row) => row.status === 'trained').map((row) => [row.id, row]));
const missing = required.filter((id) => !trained.has(id));
if (missing.length) throw new Error(`Ablation val11 requires all nine trained candidates; missing ${missing.join(', ')}`);
const activeHash = digest(activeBytes), evaluationRows = [];
await fs.mkdir(reports, { recursive: true });
for (const id of required) {
  const trainedRow = trained.get(id), candidateFile = path.resolve(root, trainedRow.candidate_file), candidateBytes = await fs.readFile(candidateFile), candidateHash = digest(candidateBytes);
  const candidate = JSON.parse(candidateBytes.toString('utf8')), candidateId = `ablation_${id}`;
  if (candidate.version !== 'layout_model_v10_mdpo_candidate' || candidate.status !== 'diagnostic_only_requires_full_val11_gate'
      || candidate.reference?.sha256 !== activeHash || candidate.architecture?.qwen_inference_input !== false) throw new Error(`${id} ablation candidate contract invalid`);
  if (await artifactState(candidateId) === 'missing') await runHeadless(candidateId, candidateFile, candidateHash, options);
  const evidence = await validateReport(candidateId, candidateFile, candidateHash), report = evidence.report;
  evaluationRows.push({ id, candidate_id: candidateId, candidate_file: trainedRow.candidate_file, candidate_sha256: candidateHash,
    trainable_parameters: trainedRow.trainable_parameters, hyperparameters: trainedRow.hyperparameters, gate_accepted: report.gate.accepted,
    delta: report.gate.delta, candidate_scores: report.candidate.score_means, candidate_metrics: report.candidate.metrics,
    report_file: path.relative(root, evidence.reportFile).split(path.sep).join('/'), report_sha256: evidence.reportSha256,
    four_group_report_file: path.relative(root, evidence.fourGroupFile).split(path.sep).join('/'), four_group_report_sha256: evidence.fourGroupSha256 });
  console.log(JSON.stringify({ id, gate_accepted: report.gate.accepted, evaluated: evaluationRows.length, total: required.length }));
}

const sensitivity = {};
for (const key of ['rank', 'beta']) {
  sensitivity[key] = Object.entries((selection.ranked_candidates || []).reduce((groups, row) => {
    const value = String(row.hyperparameters?.[key]); (groups[value] ||= []).push(row); return groups;
  }, {})).map(([value, rows]) => ({ value: Number(value), configurations: rows.length, accepted: rows.filter((row) => row.gate?.accepted).length,
    mean_aesthetic_delta: rows.reduce((sum, row) => sum + Number(row.score?.aesthetic_delta || 0), 0) / rows.length,
    best_aesthetic_delta: Math.max(...rows.map((row) => Number(row.score?.aesthetic_delta ?? -Infinity))) })).sort((a, b) => a.value - b.value);
}
const report = { version: 'v10_mdpo_ablation_report_v1', generated_at: new Date().toISOString(), status: 'complete_authoritative_val11',
  base_selection_file: 'experiments/mdpo/hyperparameter_selection.json', base_selected_id: selection.selected.id, sample_count: 11,
  fixed_seed: Number(options.seed), test_used: false, ablations: evaluationRows, sweep_sensitivity: sensitivity };
const fields = ['id', 'candidate_sha256', 'trainable_parameters', 'gate_accepted', 'delta_aesthetic', 'delta_composition_harmony', 'delta_PCK_005', 'delta_PCK_010', 'delta_text_clarity', 'delta_leader_line_clarity'];
const csvRows = evaluationRows.map((row) => ({ id: row.id, candidate_sha256: row.candidate_sha256, trainable_parameters: row.trainable_parameters,
  gate_accepted: row.gate_accepted, delta_aesthetic: row.delta.aesthetic, delta_composition_harmony: row.delta.composition_harmony,
  delta_PCK_005: row.delta.PCK_005, delta_PCK_010: row.delta.PCK_010, delta_text_clarity: row.delta.text_clarity, delta_leader_line_clarity: row.delta.leader_line_clarity }));
const cell = (value) => { const text = String(value ?? ''); return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; };
const csv = `${fields.join(',')}\n${csvRows.map((row) => fields.map((field) => cell(row[field])).join(',')).join('\n')}\n`;
const existing = await fs.readFile(outputFile, 'utf8').then(JSON.parse).catch((error) => { if (error?.code === 'ENOENT') return null; throw error; });
if (existing) {
  if (existing.version !== report.version || existing.status !== report.status || JSON.stringify(existing.ablations.map((row) => row.candidate_sha256)) !== JSON.stringify(report.ablations.map((row) => row.candidate_sha256))) throw new Error('Existing immutable ablation report does not match current candidates');
} else {
  await writeImmutableFileBundle([{ file: outputFile, text: JSON.stringify(report, null, 2) + '\n' }, { file: csvFile, text: csv }]);
}
console.log(JSON.stringify({ status: report.status, ablations: report.ablations.length, report: path.relative(root, outputFile).split(path.sep).join('/'), csv: path.relative(root, csvFile).split(path.sep).join('/') }, null, 2));
