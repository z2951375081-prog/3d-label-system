import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { writeImmutableJsonBundle } from '../lib/immutable-json-bundle.mjs';
import { buildMdpoRejectionReport } from '../lib/mdpo-rejection-report.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';
import { validateMdpoFinalDatasetAudit } from '../lib/mdpo-final-dataset-audit.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sweepDirectory = path.join(root, 'experiments', 'mdpo', 'sweep');
const reportDirectory = path.join(root, 'experiments', 'mdpo', 'val11_reports_safety_priority_v2');
const outputFile = path.join(root, 'experiments', 'mdpo', 'hyperparameter_selection.json');
const rejectionFile = path.join(root, 'experiments', 'mdpo', 'val11_rejection_report.json');
const ledger = await fs.readFile(path.join(sweepDirectory, 'sweep_ledger.json'), 'utf8').then(JSON.parse).catch((error) => {
  if (error?.code === 'ENOENT') throw new Error('MDPO selection is unavailable until the 81-grid sweep is complete');
  throw error;
});
const expectedGrid = [5e-5, 1e-4, 2e-4].flatMap((learningRate) => [0.05, 0.10, 0.20].flatMap((beta) => [0.1, 0.3, 0.5].flatMap((lambdaMulti) => [2, 4, 8].map((rank) => ({ learningRate, beta, lambdaMulti, rank }))))) ;
const [frozenReferenceBytes, datasetBytes, finalDataAudit] = await Promise.all([
  fs.readFile(path.join(root, 'experiments', 'layout_model.json')),
  fs.readFile(path.join(root, 'experiments', 'mdpo', 'train_pairs.json')),
  fs.readFile(path.join(root, 'experiments', 'mdpo', 'dataset_audit_final.json'), 'utf8').then(JSON.parse)
]);
const frozenReferenceSha256 = createHash('sha256').update(frozenReferenceBytes).digest('hex');
const finalDataset = validateMdpoFinalDatasetAudit({ datasetBytes, audit: finalDataAudit, referenceSha256: frozenReferenceSha256 });
const idFor = (configuration) => `lr_${configuration.learningRate}_beta_${configuration.beta}_multi_${configuration.lambdaMulti}_rank_${configuration.rank}`.replaceAll('.', 'p');
const trained = new Map((ledger.configurations || []).filter((row) => row.status === 'trained').map((row) => [row.id, row]));
const missingConfigurations = expectedGrid.filter((configuration) => !trained.has(idFor(configuration)));
if (missingConfigurations.length || trained.size !== 81) throw new Error(`Authoritative selection requires all 81 trained sweep configurations; missing ${missingConfigurations.length}`);
const reportFiles = (await fs.readdir(reportDirectory).catch((error) => {
  if (error?.code === 'ENOENT') throw new Error('MDPO selection requires 81 immutable authoritative val11 reports');
  throw error;
})).filter((name) => name.endsWith('.json') && !name.endsWith('.four_group.json'));
const reports = [];
for (const name of reportFiles) {
  const file = path.join(reportDirectory, name), bytes = await fs.readFile(file), report = JSON.parse(bytes.toString('utf8'));
  if (report.version !== 'v10_mdpo_val11_gate_report_v2' || report.evaluation_policy !== 'safety_priority_v2' || !report.candidate_model?.file) continue;
  const fourGroupFile = path.join(reportDirectory, `${report.candidate_id}.four_group.json`), fourGroupBytes = await fs.readFile(fourGroupFile);
  const fourGroup = JSON.parse(fourGroupBytes.toString('utf8'));
  if (fourGroup.version !== 'v10_mdpo_four_group_val11_v2' || fourGroup.evaluation_policy !== 'safety_priority_v2' || fourGroup.candidate_id !== report.candidate_id
      || fourGroup.candidate_model?.file !== report.candidate_model.file || fourGroup.candidate_model?.sha256 !== report.candidate_model.sha256
      || fourGroup.test_not_used !== true || !Array.isArray(fourGroup.groups) || fourGroup.groups.length !== 4) throw new Error(`Invalid four-group val11 report for ${report.candidate_id}`);
  const byGroup = Object.fromEntries(fourGroup.groups.map((row) => [row.group, row]));
  const checkedProtocol = validateMdpoFourGroupSummaries(byGroup);
  await validateMdpoFourGroupRawRecords({ directory: path.join(root, 'experiments', 'mdpo', 'val11_visual'), candidateId: report.candidate_id, fourGroup });
  if (JSON.stringify(checkedProtocol) !== JSON.stringify(report.four_group_protocol)
      || JSON.stringify(fourGroup.core_gate) !== JSON.stringify(report.gate)
      || JSON.stringify(byGroup.v10_no_rerank) !== JSON.stringify(report.baseline)
      || JSON.stringify(byGroup.mdpo_no_rerank) !== JSON.stringify(report.candidate)) throw new Error(`Four-group/core-gate evidence mismatch for ${report.candidate_id}`);
  reports.push({ ...report, report_file: path.relative(root, file).split(path.sep).join('/'), report_sha256: createHash('sha256').update(bytes).digest('hex'),
    four_group_report_file: path.relative(root, fourGroupFile).split(path.sep).join('/'), four_group_report_sha256: createHash('sha256').update(fourGroupBytes).digest('hex') });
}
const byCandidateFile = new Map(reports.map((report) => [report.candidate_model.file, report]));
const rows = [];
for (const configuration of expectedGrid) {
  const id = idFor(configuration), trainedRow = trained.get(id);
  const candidateFile = path.relative(root, path.join(sweepDirectory, id, 'layout_model_v10_mdpo_candidate.json')).split(path.sep).join('/');
  const report = byCandidateFile.get(candidateFile);
  if (!report) throw new Error(`Missing authoritative val11 report for ${id}`);
  if (report.test_not_used !== true || report.gate?.test_not_used_for_activation !== true || !Object.values(report.gate?.evidence || {}).every(Boolean)) throw new Error(`Incomplete or leaked val11 evidence for ${id}`);
  const independentlyChecked = evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate });
  if (!Object.values(independentlyChecked.evidence).every(Boolean) || JSON.stringify(independentlyChecked.checks) !== JSON.stringify(report.gate.checks)
      || JSON.stringify(independentlyChecked.delta) !== JSON.stringify(report.gate.delta) || independentlyChecked.accepted !== report.gate.accepted) throw new Error(`Invalid or incomplete authoritative val11 gate: ${id}`);
  for (const role of ['baseline', 'candidate']) {
    const evaluation = report[role];
    if (evaluation?.sample_scores?.length !== 11 || new Set(evaluation.sample_scores.map((row) => row.sample)).size !== 11
        || evaluation.sample_scores.some((row) => !row.response_id || !Object.values(row.scores || {}).every(Number.isFinite))) throw new Error(`Missing 11 real Qwen val responses for ${id}/${role}`);
  }
  const candidatePath = path.join(root, candidateFile), candidateBytes = await fs.readFile(candidatePath);
  if (createHash('sha256').update(candidateBytes).digest('hex') !== report.candidate_model.sha256) throw new Error(`Candidate hash mismatch for ${id}`);
  const candidate = JSON.parse(candidateBytes.toString('utf8'));
  if (candidate.mdpo?.preference_holdout?.heldout_pair_count !== 33
      || candidate.mdpo?.preference_holdout?.gradient_pair_count < 165
      || candidate.mdpo?.weight_update_evidence?.matrix_count !== 19
      || candidate.mdpo?.weight_update_evidence?.trainable_parameters !== candidate.mdpo?.trainable_parameters
      || Object.keys(candidate.mdpo?.weight_update_evidence?.modules || {}).length !== 7
      || candidate.mdpo?.weight_update_evidence?.ablation?.effective !== true
      || candidate.mdpo?.train_unified_metrics?.baseline?.sample_count !== 33
      || candidate.mdpo?.train_unified_metrics?.candidate?.sample_count !== 33
      || Object.keys(candidate.mdpo?.train_unified_metrics?.delta || {}).length !== 9
      || Object.values(candidate.mdpo?.train_unified_metrics?.delta || {}).some((value) => !Number.isFinite(value))
      || Object.values(candidate.mdpo?.weight_update_evidence?.modules || {}).some((module) => module.changed_matrices < 1))
    throw new Error(`Incomplete MDPO train33 holdout or effective LoRA update evidence for ${id}`);
  if (candidate.reference?.sha256 !== frozenReferenceSha256 || candidate.mdpo?.dataset_sha256 !== finalDataset.datasetSha256)
    throw new Error(`MDPO candidate changed frozen reference or audited train33 dataset for ${id}`);
  const hyperparameters = candidate.mdpo?.hyperparameters || trainedRow.hyperparameters;
  for (const key of ['learningRate', 'beta', 'lambdaMulti', 'rank']) if (Number(hyperparameters?.[key]) !== Number(configuration[key])) throw new Error(`Hyperparameter/report mismatch for ${id}/${key}`);
  const reportRelative = `experiments/mdpo/val11_reports_safety_priority_v2/${id}.json`;
  const fourGroupRelative = `experiments/mdpo/val11_reports_safety_priority_v2/${id}.four_group.json`;
  rows.push({ id, candidate_id: report.candidate_id, candidate_file: candidateFile, candidate_sha256: report.candidate_model.sha256,
    report_file: reportRelative, report_sha256: createHash('sha256').update(bytes).digest('hex'), four_group_report_file: fourGroupRelative, four_group_report_sha256: createHash('sha256').update(fourGroupBytes).digest('hex'),
    four_group_report_sha256: report.four_group_report_sha256, hyperparameters, gate: report.gate,
    score: { accepted: report.gate.accepted, safety_penalty: report.gate.safety_penalty, aesthetic_delta: report.gate.delta.aesthetic, composition_delta: report.gate.delta.composition_harmony,
      pck_005_delta: report.gate.delta.PCK_005, pck_010_delta: report.gate.delta.PCK_010, quality_score: report.candidate.metrics.quality_score } });
}
rows.sort((left, right) => Number(right.score.accepted) - Number(left.score.accepted)
  || left.score.safety_penalty - right.score.safety_penalty
  || right.score.aesthetic_delta - left.score.aesthetic_delta
  || right.score.composition_delta - left.score.composition_delta
  || right.score.quality_score - left.score.quality_score
  || left.id.localeCompare(right.id));
const selected = rows[0];
const output = {
  version: 'v10_mdpo_hyperparameter_selection_v1', generated_at: new Date().toISOString(), status: 'selected_by_complete_authoritative_val11',
  selected, val11_gate_complete: true, deployment_eligible: selected.gate.accepted === true,
  deployment_action: selected.gate.accepted ? 'candidate_may_proceed_to_atomic_activation_audit' : 'retain_original_v10_selected_candidate_diagnostic_only',
  required_grid: ledger.required_grid, evaluated_configuration_count: rows.length, fixed_seed: ledger.fixed_seed,
  ranking_policy: ['accepted_gate_first', 'safety_penalty_asc', 'aesthetic_delta_desc', 'composition_delta_desc', 'PCK_advisory_only', 'quality_score_desc'],
  test_used_for_selection: false, ranked_candidates: rows
};
if (!output.deployment_eligible) {
  output.rejection_report_file = path.relative(root, rejectionFile).split(path.sep).join('/');
  const selectionBytes = Buffer.from(JSON.stringify(output, null, 2) + '\n');
  const [activeBytes, rewardBytes, datasetBytes] = await Promise.all([
    fs.readFile(path.join(root, 'experiments', 'layout_model.json')),
    fs.readFile(path.join(root, 'experiments', 'preference_model.json')),
    fs.readFile(path.join(root, 'experiments', 'mdpo', 'train_pairs.json'))
  ]);
  if (JSON.parse(activeBytes.toString('utf8')).version !== 'layout_model_v10_anchor_frame_heterogeneous_graph_moe') {
    throw new Error('Rejected selection must preserve original active v10');
  }
  const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
  if (JSON.parse(datasetBytes.toString('utf8')).reference_model_sha256 !== sha256(activeBytes)) {
    throw new Error('Original v10 no longer matches the frozen reference used for MDPO train33');
  }
  const rejection = buildMdpoRejectionReport(output, { selectionSha256: sha256(selectionBytes),
    referenceSha256: sha256(activeBytes), historicalRewardSha256: sha256(rewardBytes) });
  await writeImmutableJsonBundle([{ file: outputFile, value: output }, { file: rejectionFile, value: rejection }]);
} else await fs.writeFile(outputFile, JSON.stringify(output, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ selected: selected.id, deployment_eligible: output.deployment_eligible, output: path.relative(root, outputFile).split(path.sep).join('/') }, null, 2));
