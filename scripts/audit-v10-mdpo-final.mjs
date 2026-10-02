import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { validateMdpoDataset } from '../lib/mdpo-dataset.mjs';
import { evaluateMdpoVal11Gate } from '../lib/mdpo-activation-gate.mjs';
import { validateMdpoFourGroupSummaries } from '../lib/mdpo-four-group.mjs';
import { assessMdpoDeployment } from '../lib/mdpo-experiment-status.mjs';
import { buildMdpoRejectionReport } from '../lib/mdpo-rejection-report.mjs';
import { validateMdpoTest11Records } from '../lib/mdpo-test11-report.mjs';
import { MDPO_VAL11_GROUPS } from '../lib/mdpo-four-group.mjs';
import { validateMdpoFourGroupRawRecords } from '../lib/mdpo-four-group-raw.mjs';
import { REPRODUCTION_METRIC_PROTOCOL } from '../lib/reproduction-metrics.mjs';
import { validateCompletedMdpoTrainingRun } from '../lib/mdpo-training-resume.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), experiments = path.join(root, 'experiments'), mdpo = path.join(experiments, 'mdpo');
const readJson = (file) => fs.readFile(file, 'utf8').then(JSON.parse).catch((error) => { if (error?.code === 'ENOENT') return null; throw error; });
const readBytes = (file) => fs.readFile(file).catch((error) => { if (error?.code === 'ENOENT') return null; throw error; });
const digest = (bytes) => bytes ? createHash('sha256').update(bytes).digest('hex') : null;
const checks = {};
const evidence = {};
const record = (name, complete, detail) => { checks[name] = Boolean(complete); evidence[name] = detail; };
const inside = (relative) => {
  const file = path.resolve(root, String(relative || ''));
  if (!file.startsWith(mdpo + path.sep) || path.extname(file).toLowerCase() !== '.json') throw new Error(`Refusing unsafe MDPO audit artifact path: ${relative}`);
  return file;
};

const [manifest, dataset, activeBytes, rewardBytes, sweep, selection, valSweep, ablationTrain, ablations, activation, testLock, test,
  screenshotAudit, selectionBytes, rejection, persistedDataAudit, datasetBytes] = await Promise.all([
  readJson(path.join(experiments, 'dataset_manifest.json')), readJson(path.join(mdpo, 'train_pairs.json')),
  readBytes(path.join(experiments, 'layout_model.json')), readBytes(path.join(experiments, 'preference_model.json')),
  readJson(path.join(mdpo, 'sweep', 'sweep_ledger.json')), readJson(path.join(mdpo, 'hyperparameter_selection.json')),
  readJson(path.join(mdpo, 'val11_sweep_ledger.json')), readJson(path.join(mdpo, 'ablations', 'ablation_ledger.json')),
  readJson(path.join(mdpo, 'ablations', 'ablation_report.json')), readJson(path.join(mdpo, 'activation_report.json')),
  readJson(path.join(mdpo, 'test11_lock.json')), readJson(path.join(mdpo, 'test11_final_report.json')),
  readJson(path.join(mdpo, 'final_browser_audit.json')),
  readBytes(path.join(mdpo, 'hyperparameter_selection.json')),
  readJson(path.join(mdpo, 'val11_rejection_report.json')),
  readJson(path.join(mdpo, 'dataset_audit_final.json')),
  readBytes(path.join(mdpo, 'train_pairs.json'))
]);
const active = activeBytes ? JSON.parse(activeBytes.toString('utf8')) : null, activeSha256 = digest(activeBytes), rewardSha256 = digest(rewardBytes);
const expectedReferenceSha256 = activation?.status === 'activated' ? activation.previous_active_sha256 : activeSha256;
let dataAudit = null;
try { if (dataset && manifest) dataAudit = validateMdpoDataset(dataset, manifest, { requireComplete: true }); } catch (error) { dataAudit = { error: error.message }; }
const currentUniqueCandidateCount = dataset ? new Set(dataset.pairs.flatMap((pair) => [pair.candidate_a?.candidate_id, pair.candidate_b?.candidate_id]).filter(Boolean)).size : 0;
const dataAuditVerified = dataAudit?.sample_count === 33 && dataAudit?.pair_count >= 198 && dataAudit?.pair_count <= 396
  && persistedDataAudit?.version === 'v10_mdpo_dataset_audit_v1' && persistedDataAudit?.complete === true
  && persistedDataAudit?.dataset_sha256 === digest(datasetBytes)
  && persistedDataAudit?.reference_model_sha256 === expectedReferenceSha256
  && persistedDataAudit?.sample_count === dataAudit.sample_count && persistedDataAudit?.pair_count === dataAudit.pair_count
  && persistedDataAudit?.unique_candidate_count === currentUniqueCandidateCount
  && persistedDataAudit?.scorer_model === dataAudit.scorer_model && persistedDataAudit?.prompt_version === dataAudit.prompt_version
  && persistedDataAudit?.split_integrity?.train_only === true && persistedDataAudit?.split_integrity?.val_pairs === 0
  && persistedDataAudit?.split_integrity?.test_pairs === 0;
record('train33_complete_strict_data', dataAuditVerified,
  { sample_count: dataAudit?.sample_count ?? null, pair_count: dataAudit?.pair_count ?? null,
    dataset_sha256: digest(datasetBytes), persisted_audit_verified: dataAuditVerified, error: dataAudit?.error ?? null });
const expectedGridIds = [5e-5, 1e-4, 2e-4].flatMap((learningRate) =>
  [0.05, 0.10, 0.20].flatMap((beta) => [0.1, 0.3, 0.5].flatMap((lambdaMulti) =>
    [2, 4, 8].map((rank) => `lr_${learningRate}_beta_${beta}_multi_${lambdaMulti}_rank_${rank}`.replaceAll('.', 'p')))));
let verifiedSweepTrainingRuns = 0, sweepTrainingError = null;
try {
  const trainedRows = (sweep?.configurations || []).filter((row) => row.status === 'trained');
  if (sweep?.required_grid?.total !== 81 || sweep?.test_used_for_selection !== false
      || trainedRows.length !== 81 || new Set(trainedRows.map((row) => row.id)).size !== 81
      || expectedGridIds.some((id) => !trainedRows.some((row) => row.id === id)))
    throw new Error('Incomplete, duplicate, or unexpected 81-grid training ledger');
  for (const row of trainedRows) {
    await validateCompletedMdpoTrainingRun({ outputDir: path.join(mdpo, 'sweep', row.id), root,
      referenceHash: expectedReferenceSha256, datasetHash: digest(datasetBytes), options: row.hyperparameters || {} });
    verifiedSweepTrainingRuns++;
  }
} catch (error) { sweepTrainingError = error.message; }
record('81_grid_trained', verifiedSweepTrainingRuns === 81,
  { trained: (sweep?.configurations || []).filter((row) => row.status === 'trained').length,
    independently_verified_training_runs: verifiedSweepTrainingRuns, error: sweepTrainingError });
let verifiedVal11Rows = 0, val11EvidenceError = null;
if (selection?.ranked_candidates?.length === 81) {
  try {
    const evaluated = new Set((valSweep?.configurations || []).filter((row) => row.status === 'evaluated').map((row) => row.id));
    const ranked = selection.ranked_candidates;
    if (evaluated.size !== 81 || new Set(ranked.map((row) => row.id)).size !== 81) throw new Error('Incomplete or duplicate 81-grid ledger/selection');
    for (const row of ranked) {
      if (!evaluated.has(row.id) || !/^lr_[a-z0-9p_]+$/.test(row.id)
          || row.report_file !== `experiments/mdpo/val11_reports_safety_priority_v2/${row.id}.json`
          || row.four_group_report_file !== `experiments/mdpo/val11_reports_safety_priority_v2/${row.id}.four_group.json`)
        throw new Error(`Invalid val11 row identity or paths: ${row.id}`);
      const [candidateBytes, reportBytes, fourBytes] = await Promise.all([
        fs.readFile(inside(row.candidate_file)), fs.readFile(inside(row.report_file)), fs.readFile(inside(row.four_group_report_file))
      ]);
      const candidate = JSON.parse(candidateBytes.toString('utf8')), report = JSON.parse(reportBytes.toString('utf8')), four = JSON.parse(fourBytes.toString('utf8'));
      if (digest(candidateBytes) !== row.candidate_sha256 || digest(reportBytes) !== row.report_sha256
          || digest(fourBytes) !== row.four_group_report_sha256 || report.version !== 'v10_mdpo_val11_gate_report_v2' || report.evaluation_policy !== 'safety_priority_v2'
          || report.candidate_id !== row.id || report.candidate_model?.file !== row.candidate_file
          || report.candidate_model?.sha256 !== row.candidate_sha256 || report.paired_provenance?.paired_samples !== 11
          || report.test_not_used !== true || four.version !== 'v10_mdpo_four_group_val11_v2' || four.evaluation_policy !== 'safety_priority_v2'
          || four.candidate_id !== row.id || four.candidate_model?.sha256 !== row.candidate_sha256
          || four.test_not_used !== true || four.groups?.length !== 4
          || JSON.stringify(evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate })) !== JSON.stringify(report.gate)
          || JSON.stringify(report.gate) !== JSON.stringify(row.gate)
          || JSON.stringify(validateMdpoFourGroupSummaries(Object.fromEntries(four.groups.map((group) => [group.group, group])))) !== JSON.stringify(report.four_group_protocol)
          || JSON.stringify(four.core_gate) !== JSON.stringify(report.gate)
          || candidate.reference?.sha256 !== dataset?.reference_model_sha256
          || candidate.mdpo?.dataset_sha256 !== digest(await readBytes(path.join(mdpo, 'train_pairs.json')))
          || candidate.mdpo?.preference_holdout?.heldout_pair_count !== 33
          || candidate.mdpo?.preference_holdout?.gradient_pair_count < 165
          || candidate.mdpo?.weight_update_evidence?.matrix_count !== 19
          || candidate.mdpo?.weight_update_evidence?.trainable_parameters !== candidate.mdpo?.trainable_parameters
          || Object.keys(candidate.mdpo?.weight_update_evidence?.modules || {}).length !== 7
          || candidate.mdpo?.weight_update_evidence?.ablation?.effective !== true
          || Object.values(candidate.mdpo?.weight_update_evidence?.modules || {}).some((module) => module.changed_matrices < 1)) throw new Error(`Val11 immutable evidence changed or failed gate: ${row.id}`);
      for (const role of ['baseline', 'candidate']) {
        const scores = report[role]?.sample_scores;
        if (scores?.length !== 11 || new Set(scores.map((sample) => sample.sample)).size !== 11
            || scores.some((sample) => !sample.response_id || !Object.values(sample.scores || {}).every(Number.isFinite)))
          throw new Error(`Incomplete 11-sample visual Qwen evidence: ${row.id}/${role}`);
      }
      await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: row.id, fourGroup: four });
      verifiedVal11Rows++;
    }
  } catch (error) { val11EvidenceError = error.message; }
}
record('81_grid_real_val11_evaluated', valSweep?.test_used === false && verifiedVal11Rows === 81,
  { evaluated: (valSweep?.configurations || []).filter((row) => row.status === 'evaluated').length, independently_verified: verifiedVal11Rows, error: val11EvidenceError });
record('formal_val11_selection', selection?.version === 'v10_mdpo_hyperparameter_selection_v1' && selection?.val11_gate_complete === true
  && selection?.test_used_for_selection === false && selection?.evaluated_configuration_count === 81
  && selection?.ranked_candidates?.length === 81, { selected: selection?.selected?.id ?? null, eligible: selection?.deployment_eligible ?? null });
let candidateSha256 = null, reportSha256 = null, fourSha256 = null, independentlyChecked = null;
try {
  if (selection?.selected) {
    const [candidateBytes, reportBytes, fourBytes] = await Promise.all([
      fs.readFile(inside(selection.selected.candidate_file)), fs.readFile(inside(selection.selected.report_file)), fs.readFile(inside(selection.selected.four_group_report_file))
    ]);
    candidateSha256 = digest(candidateBytes); reportSha256 = digest(reportBytes); fourSha256 = digest(fourBytes);
    const report = JSON.parse(reportBytes.toString('utf8')), four = JSON.parse(fourBytes.toString('utf8'));
    independentlyChecked = evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate });
    const protocol = validateMdpoFourGroupSummaries(Object.fromEntries(four.groups.map((group) => [group.group, group])));
    record('selected_four_group_hash_and_val11_gate', candidateSha256 === selection.selected.candidate_sha256
      && reportSha256 === selection.selected.report_sha256 && fourSha256 === selection.selected.four_group_report_sha256
      && report.paired_provenance?.paired_samples === 11 && four.groups?.length === 4
      && JSON.stringify(independentlyChecked) === JSON.stringify(report.gate)
      && JSON.stringify(protocol) === JSON.stringify(report.four_group_protocol),
    { candidate_sha256: candidateSha256, report_sha256: reportSha256, four_group_sha256: fourSha256, gate_accepted: independentlyChecked.accepted });
  } else record('selected_four_group_hash_and_val11_gate', false, { reason: 'No formally selected val11 candidate' });
} catch (error) { record('selected_four_group_hash_and_val11_gate', false, { error: error.message }); }
const expectedAblationIds = ['full_fixed_variance', 'no_multi_only_overall_dpo', 'no_overall_only_multidimensional',
  'no_reference_kl', 'no_text_clarity', 'no_leader_line_clarity', 'equal_multidimensional_weights',
  'clarity_emphasis_weights', 'learnable_variance'];
let verifiedAblationTrainingRuns = 0, ablationTrainingError = null;
try {
  const trainedRows = (ablationTrain?.configurations || []).filter((row) => row.status === 'trained');
  if (ablationTrain?.test_used_for_selection !== false || trainedRows.length !== 9
      || new Set(trainedRows.map((row) => row.id)).size !== 9
      || expectedAblationIds.some((id) => !trainedRows.some((row) => row.id === id)))
    throw new Error('Incomplete, duplicate, or unexpected nine-ablation training ledger');
  for (const row of trainedRows) {
    await validateCompletedMdpoTrainingRun({ outputDir: path.join(mdpo, 'ablations', row.id), root,
      referenceHash: expectedReferenceSha256, datasetHash: digest(datasetBytes), options: row.hyperparameters || {} });
    verifiedAblationTrainingRuns++;
  }
} catch (error) { ablationTrainingError = error.message; }
record('nine_ablation_training_runs', verifiedAblationTrainingRuns === 9,
  { trained: (ablationTrain?.configurations || []).filter((row) => row.status === 'trained').length,
    independently_verified_training_runs: verifiedAblationTrainingRuns, error: ablationTrainingError });
let verifiedAblations = 0, ablationError = null, csvVerified = false;
if (ablations?.ablations?.length === 9) {
  try {
    const ids = expectedAblationIds;
    if (ablations.base_selected_id !== selection?.selected?.id || ablations.sample_count !== 11
        || ablations.sweep_sensitivity?.rank?.length !== 3 || ablations.sweep_sensitivity?.beta?.length !== 3
        || JSON.stringify(ablations.ablations.map((row) => row.id)) !== JSON.stringify(ids))
      throw new Error('Nine-ablation configuration, cohort or 81-grid sensitivity incomplete');
    const fields = ['id', 'candidate_sha256', 'trainable_parameters', 'gate_accepted', 'delta_aesthetic',
      'delta_composition_harmony', 'delta_PCK_005', 'delta_PCK_010', 'delta_text_clarity', 'delta_leader_line_clarity'];
    const cell = (value) => { const text = String(value ?? ''); return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; };
    const csvRows = [];
    for (const row of ablations.ablations) {
      if (row.candidate_id !== `ablation_${row.id}`
          || row.report_file !== `experiments/mdpo/val11_reports_safety_priority_v2/${row.candidate_id}.json`
          || row.four_group_report_file !== `experiments/mdpo/val11_reports_safety_priority_v2/${row.candidate_id}.four_group.json`)
        throw new Error(`Ablation artifact identity invalid: ${row.id}`);
      const [candidateBytes, reportBytes, fourBytes] = await Promise.all([
        fs.readFile(inside(row.candidate_file)), fs.readFile(inside(row.report_file)), fs.readFile(inside(row.four_group_report_file))
      ]);
      const report = JSON.parse(reportBytes.toString('utf8')), four = JSON.parse(fourBytes.toString('utf8'));
      if (digest(candidateBytes) !== row.candidate_sha256 || digest(reportBytes) !== row.report_sha256
          || digest(fourBytes) !== row.four_group_report_sha256 || report.candidate_id !== row.candidate_id
          || report.candidate_model?.sha256 !== row.candidate_sha256 || report.test_not_used !== true
          || report.paired_provenance?.paired_samples !== 11 || four.candidate_id !== row.candidate_id
          || four.test_not_used !== true || four.groups?.length !== 4
          || JSON.stringify(evaluateMdpoVal11Gate({ baseline: report.baseline, candidate: report.candidate })) !== JSON.stringify(report.gate)
          || JSON.stringify(four.core_gate) !== JSON.stringify(report.gate)
          || JSON.stringify(validateMdpoFourGroupSummaries(Object.fromEntries(four.groups.map((group) => [group.group, group])))) !== JSON.stringify(report.four_group_protocol)
          || row.gate_accepted !== report.gate.accepted || JSON.stringify(row.delta) !== JSON.stringify(report.gate.delta))
        throw new Error(`Ablation val11 evidence invalid or changed: ${row.id}`);
      for (const role of ['baseline', 'candidate']) if (report[role]?.sample_scores?.length !== 11
          || new Set(report[role].sample_scores.map((item) => item.sample)).size !== 11
          || report[role].sample_scores.some((item) => !item.response_id || !Object.values(item.scores || {}).every(Number.isFinite)))
        throw new Error(`Missing Qwen ablation val11 receipts: ${row.id}/${role}`);
      const csvRow = { id: row.id, candidate_sha256: row.candidate_sha256, trainable_parameters: row.trainable_parameters,
        gate_accepted: row.gate_accepted, delta_aesthetic: row.delta.aesthetic, delta_composition_harmony: row.delta.composition_harmony,
        delta_PCK_005: row.delta.PCK_005, delta_PCK_010: row.delta.PCK_010, delta_text_clarity: row.delta.text_clarity,
        delta_leader_line_clarity: row.delta.leader_line_clarity };
      csvRows.push(fields.map((field) => cell(csvRow[field])).join(','));
      await validateMdpoFourGroupRawRecords({ directory: path.join(mdpo, 'val11_visual'), candidateId: row.candidate_id, fourGroup: four });
      verifiedAblations++;
    }
    const expectedCsv = `${fields.join(',')}\n${csvRows.join('\n')}\n`;
    csvVerified = (await fs.readFile(path.join(mdpo, 'ablations', 'ablation_report.csv'), 'utf8')) === expectedCsv;
    if (!csvVerified) throw new Error('Nine-ablation CSV differs from authoritative JSON');
  } catch (error) { ablationError = error.message; }
}
record('nine_real_val11_ablations_json_csv', ablations?.status === 'complete_authoritative_val11'
  && ablations?.test_used === false && verifiedAblations === 9 && csvVerified,
  { evaluated: ablations?.ablations?.length ?? 0, rank_sensitivity: ablations?.sweep_sensitivity?.rank?.length ?? 0,
    beta_sensitivity: ablations?.sweep_sensitivity?.beta?.length ?? 0, independently_verified: verifiedAblations, csv_verified: csvVerified, error: ablationError });
const deployment = assessMdpoDeployment({ active, activeSha256, gate: independentlyChecked, selection, activationReport: activation,
  preferenceSha256: rewardSha256, selectedCandidateSha256: candidateSha256, selectedReportSha256: reportSha256, selectedFourGroupSha256: fourSha256 });
if (selection?.deployment_eligible === true) record('val11_accepted_atomic_activation', deployment.active && rejection === null, { active_version: active?.version, active_sha256: activeSha256, violations: deployment.violations, unexpected_rejection_report: rejection !== null });
else {
  let rejectionVerified = false, rejectionError = null;
  try {
    if (selection?.rejection_report_file !== 'experiments/mdpo/val11_rejection_report.json'
        || !rejection || !selectionBytes || independentlyChecked?.accepted !== false
        || JSON.stringify(independentlyChecked) !== JSON.stringify(selection.selected.gate)) throw new Error('Missing or inconsistent selected val11 rejection evidence');
    const expected = buildMdpoRejectionReport(selection, { selectionSha256: digest(selectionBytes),
      referenceSha256: activeSha256, historicalRewardSha256: rewardSha256 });
    rejectionVerified = JSON.stringify(expected) === JSON.stringify(rejection)
      && dataset?.reference_model_sha256 === activeSha256
      && selection.selected.candidate_sha256 === candidateSha256
      && selection.selected.report_sha256 === reportSha256
      && selection.selected.four_group_report_sha256 === fourSha256;
    if (!rejectionVerified) rejectionError = 'Immutable rejection report does not match all 81 val11 decisions and original-model hashes';
  } catch (error) { rejectionError = error.message; }
  record('val11_rejection_keeps_original_v10', rejectionVerified && active?.version === 'layout_model_v10_anchor_frame_heterogeneous_graph_moe'
    && activation === null, { active_version: active?.version, selected_gate: selection?.selected?.gate?.status ?? null, rejection_report_verified: rejectionVerified, error: rejectionError });
}
const testJsonBytes = await readBytes(path.join(mdpo, 'test11_unified_metrics.json'));
const testCsvBytes = await readBytes(path.join(mdpo, 'test11_unified_metrics.csv'));
const testExpected = selection?.deployment_eligible === true;
let testRecordsVerified = false, testEvidenceError = null;
if (testExpected && test?.version === 'v10_mdpo_test11_final_report_v1' && testLock) {
  try {
    const runId = String(test.test_run_id || '');
    if (!/^[a-zA-Z0-9_-]{8,100}$/.test(runId) || testLock.cohort?.length !== 11 || test.lock_sha256 !== digest(await fs.readFile(path.join(mdpo, 'test11_lock.json')))
        || test.models?.active_v10_mdpo_sha256 !== activeSha256
        || test.models?.original_v10_sha256 !== activation?.previous_active_sha256
        || test.models?.historical_reward_model_sha256 !== rewardSha256 || test.model_changed_after_lock !== false)
      throw new Error('Locked test11 model/run/reference identities invalid');
    const groups = {};
    for (const group of Object.keys(MDPO_VAL11_GROUPS)) {
      const directory = path.join(mdpo, 'test11_visual', runId, group);
      const files = (await fs.readdir(directory)).filter((file) => file.endsWith('.json'));
      if (files.length !== 11) throw new Error(`Locked test11 ${group} has ${files.length}/11 raw records`);
      const rows = await Promise.all(files.map((file) => fs.readFile(path.join(directory, file), 'utf8').then(JSON.parse)));
      const bySample = new Map(rows.map((row) => [`${row.sample?.category}/${row.sample?.sample_id}`, row]));
      groups[group] = testLock.cohort.map((sample) => bySample.get(sample));
    }
    const provenance = validateMdpoTest11Records({ groups, cohort: testLock.cohort, runId,
      lockSha256: test.lock_sha256, baselineSha256: test.models.original_v10_sha256,
      activeSha256, preferenceSha256: rewardSha256, scorerModel: testLock.scorer.model,
      promptVersion: testLock.scorer.prompt_version, metricProtocol: REPRODUCTION_METRIC_PROTOCOL.id });
    if (JSON.stringify(provenance) !== JSON.stringify(test.provenance)) throw new Error('Locked test11 44-record evidence checksum changed');
    for (const [group, rows] of Object.entries(groups)) {
      const summary = test.groups?.[group];
      if (summary?.sample_count !== 11 || summary?.sample_scores?.length !== 11
          || rows.some((row, index) => JSON.stringify(summary.sample_scores[index]) !== JSON.stringify({ sample: row.sample, scores: row.scores,
            metrics: row.metrics, response_id: row.scorer.response_id }))) throw new Error(`Locked test11 ${group} summary differs from original Qwen receipts`);
    }
    const unified = JSON.parse(testJsonBytes.toString('utf8'));
    if (unified.test_run_id !== runId || unified.lock_sha256 !== test.lock_sha256 || unified.split !== 'test'
        || unified.sample_count !== 11 || unified.metric_protocol !== REPRODUCTION_METRIC_PROTOCOL.id || unified.rows?.length !== 4)
      throw new Error('Locked test11 unified JSON protocol differs from original report');
    testRecordsVerified = true;
  } catch (error) { testEvidenceError = error.message; }
}
record('locked_single_test11_and_unified_json_csv', !testExpected ? checks.val11_rejection_keeps_original_v10 === true
  && testLock === null && test === null && testJsonBytes === null && testCsvBytes === null : testLock?.version === 'v10_mdpo_test11_lock_v1'
  && test?.version === 'v10_mdpo_test11_final_report_v1' && test?.status === 'complete_locked_single_test11'
  && test?.provenance?.record_count === 44 && test?.test_used_for_training === false && test?.test_used_for_selection === false
  && testRecordsVerified
  && test?.lock_sha256 === digest(await readBytes(path.join(mdpo, 'test11_lock.json')))
  && test?.artifacts?.unified_json_sha256 === digest(testJsonBytes) && test?.artifacts?.unified_csv_sha256 === digest(testCsvBytes),
  { required_after_activation: testExpected, status: test?.status ?? null, records: test?.provenance?.record_count ?? 0,
    raw_records_verified: testRecordsVerified, error: testEvidenceError });
const imageFile = screenshotAudit?.screenshot ? path.resolve(root, screenshotAudit.screenshot) : null;
const imageBytes = imageFile?.startsWith(mdpo + path.sep) ? await readBytes(imageFile) : null;
record('final_browser_screenshot_and_console_audit', screenshotAudit?.version === 'v10_mdpo_final_browser_audit_v1'
  && screenshotAudit?.browser_error_free === true
  && screenshotAudit?.browser_audit?.console_errors?.length === 0 && screenshotAudit?.browser_audit?.page_errors?.length === 0
  && screenshotAudit?.sample_count >= 1 && screenshotAudit?.checks?.every((row) => Object.keys(row.visuals?.images || {}).length === 6
    && Object.values(row.visuals.images).every((view) => view.valid === true && view.bytes >= 1000))
  && imageBytes?.length > 1000 && screenshotAudit?.screenshot_sha256 === digest(imageBytes)
  && screenshotAudit?.active_model_sha256 === activeSha256
  && screenshotAudit?.historical_reward_sha256 === rewardSha256
  && screenshotAudit?.selected_candidate_id === selection?.selected?.id
  && screenshotAudit?.outcome === (selection?.deployment_eligible === true ? 'activated_locked_test11_complete' : 'all_rejected_original_v10_preserved')
  && screenshotAudit?.page_status?.dataset?.includes('33/33')
  && screenshotAudit?.page_status?.sweep?.includes('81/81 已评估')
  && screenshotAudit?.page_status?.ablations?.includes('9/9 已评估'),
  { screenshot: screenshotAudit?.screenshot ?? null, browser_error_free: screenshotAudit?.browser_error_free ?? null });

const pending = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name);
console.log(JSON.stringify({ version: 'v10_mdpo_final_readonly_audit_v1', audited_at: new Date().toISOString(),
  status: pending.length ? 'incomplete' : 'all_artifact_checks_passed_not_a_substitute_for_browser_or_scientific_review',
  active_version: active?.version ?? null, active_sha256: activeSha256, historical_reward_sha256: rewardSha256, checks, evidence, pending }, null, 2));
if (pending.length) process.exitCode = 2;
